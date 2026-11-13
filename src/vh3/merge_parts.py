"""Validate and combine independently trained feature/model OOF components.

Keeping one result per feature/model allows a long 96-combination run to resume
without repeating previously completed classifiers. Outputs are merged in a
fixed order and never mix stale or incomplete model combinations.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import pandas as pd
from .common import LENGTHS,FEATURE_DIMS,CROSS_HOST,load_ids
from .train import MODELS

HEADER=['Feature','Model','Fold','taxid','true_class','eval_true_class',
        'pred_class','is_correct','allowed_classes','prob_Insect','prob_Plant','prob_Other']

def validate_one(directory,length,feature,model,expected_ids=None):
    directory=Path(directory)
    config=directory/'run_config.json'
    oof=directory/'oof_predictions.tsv'
    fold_audit=directory/'fold_audit.tsv'
    if not all(p.is_file() for p in (config,oof,fold_audit)):
        raise ValueError(f'Incomplete model output directory: {directory}')
    cfg=json.loads(config.read_text())
    if cfg.get('length')!=length or cfg.get('features')!=[feature] or cfg.get('models')!=[model]:
        raise ValueError(f'OOF run configuration differs from requested combo: {directory}')
    df=pd.read_csv(oof,sep='\t',dtype={'taxid':str})
    if len(df)==0 or not set(HEADER).issubset(df.columns):
        raise ValueError(f'Empty/invalid OOF file: {oof}')
    if set(df['Feature'])!={feature} or set(df['Model'])!={model}:
        raise ValueError(f'Mislabelled OOF file: {oof}')
    if df['taxid'].isna().any() or df['taxid'].duplicated().any():
        raise ValueError(f'Duplicate or blank TaxID in: {oof}')
    if set(df['Fold'].astype(int))!={1,2,3,4,5}:
        raise ValueError(f'Incomplete five folds: {oof}')
    audit=pd.read_csv(fold_audit,sep='\t')
    if len(audit)!=5 or set(audit.Fold.astype(int))!={1,2,3,4,5}:
        raise ValueError(f'Incomplete fold audit: {fold_audit}')
    if expected_ids is not None:
        should=set(expected_ids)
        if feature in ('1mer','2mer','4mer'):
            should-=CROSS_HOST
        if set(df.taxid)!=should:
            raise ValueError(f'{length}/{feature}/{model}: OOF TaxID coverage != input')
    return len(df)

def merge(parts_root,length,output,features_root):
    parts_root=Path(parts_root)
    output=Path(output)
    if output.exists():raise FileExistsError(f'Merged output exists: {output}')
    if length not in LENGTHS:raise ValueError(f'Unexpected sequence length {length}')
    expected_ids={f:set(tid for cl in ('Insect','Plant','Other')
                        for tid,_ in load_ids(Path(features_root)/length/f/cl/'ids.tsv'))
                  for f in FEATURE_DIMS}
    for f in expected_ids:
        if len(expected_ids[f])!=sum(len(load_ids(Path(features_root)/length/f/cl/'ids.tsv')) for cl in ('Insect','Plant','Other')):
            raise ValueError(f'Duplicated TaxID in classes for {length}/{f}')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'.merging-',dir=output.parent))
    total=0
    try:
        with (staging/'oof_predictions.tsv').open('w',encoding='utf-8',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=HEADER,delimiter='\t')
            writer.writeheader()
            for feature in FEATURE_DIMS:
                for model in MODELS:
                    part=parts_root/length/f'{feature}_{model}'
                    n=validate_one(part,length,feature,model,expected_ids[feature])
                    with (part/'oof_predictions.tsv').open('r',encoding='utf-8',newline='') as fh:
                        reader=csv.DictReader(fh,delimiter='\t')
                        for row in reader:writer.writerow({k:row[k] for k in HEADER})
                    total+=n
        (staging/'run_manifest.json').write_text(json.dumps({'length':length,'feature_count':len(FEATURE_DIMS),
             'model_count':len(MODELS),'combinations':len(FEATURE_DIMS)*len(MODELS),
             'rows':total,'source':'independent full fivefold OOF per feature/model'},indent=2)+'\n')
        os.replace(staging,output)
    finally:
        if staging.exists():
            for p in staging.iterdir():p.unlink()
            staging.rmdir()
    print(f'[MERGE] {length}: {len(FEATURE_DIMS)*len(MODELS)} combinations, {total} rows',flush=True)
    return output

def validate_merge(directory,length):
    d=Path(directory)
    if not (d/'oof_predictions.tsv').is_file() or not (d/'run_manifest.json').is_file():
        raise ValueError('Incomplete merged OOF output')
    meta=json.loads((d/'run_manifest.json').read_text())
    if meta.get('length')!=length or meta.get('combinations')!=24:
        raise ValueError('Merged OOF manifest invalid')
    df=pd.read_csv(d/'oof_predictions.tsv',sep='\t',usecols=['Feature','Model','Fold','taxid'],dtype={'taxid':str})
    if len(df)!=meta['rows'] or len(df[['Feature','Model']].drop_duplicates())!=24:
        raise ValueError('Merged OOF row count or combinations mismatch')
    if df.duplicated(['Feature','Model','taxid']).any():
        raise ValueError('Duplicate OOF TaxID within combination')
    return len(df)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    one=sub.add_parser('check-one')
    one.add_argument('--directory',required=True);one.add_argument('--length',required=True)
    one.add_argument('--feature',required=True);one.add_argument('--model',required=True)
    one.add_argument('--features-dir',required=True)
    combine=sub.add_parser('merge')
    combine.add_argument('--parts-root',required=True);combine.add_argument('--length',required=True)
    combine.add_argument('--output',required=True);combine.add_argument('--features-dir',required=True)
    completed=sub.add_parser('check-merged')
    completed.add_argument('--directory',required=True);completed.add_argument('--length',required=True)
    a=p.parse_args(argv)
    if a.command=='check-one':
        expected=set(tid for cl in ('Insect','Plant','Other')
                     for tid,_ in load_ids(Path(a.features_dir)/a.length/a.feature/cl/'ids.tsv'))
        print('[CHECK]',validate_one(a.directory,a.length,a.feature,a.model,expected))
    elif a.command=='merge':merge(a.parts_root,a.length,a.output,a.features_dir)
    elif a.command=='check-merged':print('[CHECK MERGED]',validate_merge(a.directory,a.length))
if __name__=='__main__':main()
