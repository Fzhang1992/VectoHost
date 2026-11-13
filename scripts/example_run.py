#!/usr/bin/env python3
"""Run a small, sequence-to-prediction example without model checkpoints.

Artificial sequences are designed only to exercise the software. Metrics are
not representative of biological classification or published Figure 5 results.
"""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vh3.prepare import prepare
from vh3.features import extract_kmer
from vh3.train import train
from vh3.evaluate import evaluate_table
from vh3.final_model import fit_deploy_model
from vh3.predict import predict

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Destination already exists: {out}')
    input_dir=ROOT/'exampleInput'
    files={c:input_dir/f'{c}.fasta' for c in ('Insect','Plant','Other')}
    out.mkdir(parents=True)
    prepare(files,out/'sequences','regenerate',seed='EXAMPLE_ONLY_NOT_PAPER',expect_published=False)
    extract_kmer(out/'sequences',out/'features','1000bp',3)
    train(out/'features','1000bp',out/'results',['3mer'],['GaussianNB'],mode='strict_exclude')
    s,cm,audit=evaluate_table(out/'results'/'oof_predictions.tsv','1000bp')
    s.to_csv(out/'feature_model_summary.tsv',sep='\t',index=False)
    cm.to_csv(out/'confusion_matrix.tsv',sep='\t',index=False)
    fit_deploy_model(out/'features','1000bp','3mer','GaussianNB',out/'model')
    predict(input_dir/'query.fasta',out/'model',out/'predictions.tsv','1000bp',slice_seed='EXAMPLE_ONLY_NOT_PAPER')
    print('Example complete:',out)
    print('These are illustrative outputs from ARTIFICIAL sequences; NOT real virus prediction scores.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='work/example',help='New directory for all example outputs')
    a=p.parse_args()
    run(a.output)
if __name__=='__main__':main()
