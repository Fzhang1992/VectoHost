"""Protect resumable runs from mixing distinct source/model configurations."""
import argparse
import json
from pathlib import Path
from .common import sha256sum

def verify(config,inputs,luca,output):
    sources={'config.env':Path(config), 'insect_fasta':Path(inputs[0]),
             'plant_fasta':Path(inputs[1]),'other_fasta':Path(inputs[2]),
             'lucavirus_inference_script':Path(luca)/'src/llm/lucavirus/get_embedding.py'}
    digest={name:sha256sum(path) for name,path in sources.items()}
    output=Path(output)
    if output.is_file():
        existing=json.loads(output.read_text())
        if existing.get('sha256')!=digest:
            raise SystemExit('Input/config/model inference source changed since the existing work/ run; use a NEW work directory before continuing.')
        print('[RESUME] Original input, config and LucaVirus script hashes match')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix('.tmp')
        tmp.write_text(json.dumps({'sha256':digest},indent=2)+'\n')
        tmp.replace(output)
        print('[RUN] Input/config/model script hashes recorded in',output)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--insect',required=True);p.add_argument('--plant',required=True)
    p.add_argument('--other',required=True);p.add_argument('--luca',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    verify(a.config,(a.insect,a.plant,a.other),a.luca,a.output)
if __name__=='__main__':main()
