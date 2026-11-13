#!/usr/bin/env python3
"""Read-only checks for source-package contents before publishing on GitHub."""
import ast
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

REQUIRED=['README.md','README_CN.md','LICENSE','requirements.txt','environment.yml',
          'config.example.env','run_all.sh','train.py','predict.py','scripts/setup_models.sh',
          'scripts/example_run.py','docs/INSTALL.md','docs/METHODS.md','docs/REPRODUCIBILITY.md',
          'exampleInput/Insect.fasta','exampleInput/Plant.fasta','exampleInput/Other.fasta',
          'exampleOutput/README.md','src/vh3/embedding_lucavirus.py','src/vh3/embedding_esm2.py']

def main():
    problems=[]
    for rel in REQUIRED:
        if not (ROOT/rel).is_file():problems.append(f'Missing: {rel}')
    paths=[]
    for path in ROOT.rglob('*'):
        if path.is_dir():continue
        rel=path.relative_to(ROOT)
        if any(p in ('.git','__pycache__','.pytest_cache','work','external','logs') for p in rel.parts):
            problems.append(f'Runtime folder in release tree: {rel}')
            continue
        if path.suffix in ('.npy','.pt','.joblib','.pyc'):
            problems.append(f'Unexpected model/cache file: {rel}')
        if path.suffix in ('.py','.md','.sh','.env','.json','.txt','.yml','.tsv','.gitignore') or path.name in ('README','LICENSE'):
            if path.suffix=='.py':
                try:ast.parse(path.read_text(encoding='utf-8'),filename=str(rel))
                except SyntaxError as exc:problems.append(f'Python parse error {rel}: {exc}')
            if path.resolve()!=Path(__file__).resolve() and path.suffix in ('.py','.sh','.env','.md'):
                text=path.read_text(encoding='utf-8')
                if re.search(r'/public/home/|/d3/scratch/|[A-Za-z]:\\\\Users\\\\',text):
                    problems.append(f'Local machine path in {rel}')
            paths.append(path)
    for path in (ROOT/'README.md',ROOT/'README_CN.md'):
        content=path.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^]]+\]\(([^)]+)\)',content):
            if target.startswith(('http://','https://','#')):continue
            if not (ROOT/target).exists():problems.append(f'Broken README link {path.name}: {target}')
    if problems:
        for item in problems:print('[FAIL]',item)
        raise SystemExit(f'{len(problems)} release audit errors')
    print(f'[PASS] Release tree: {len(paths)} text/source files audited; no runtime artifacts found.')
if __name__=='__main__':main()
