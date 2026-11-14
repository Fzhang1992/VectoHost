"""Predict new FASTA sequences using a fitted deployment model.

No previous embeddings are imported. k-mer/ESM2/LucaVirus features are generated
freshly for every query, using the same extraction functions as training.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import tempfile
import joblib
import numpy as np
from .common import CLASSES, LENGTHS, FEATURE_DIMS, choose_fragment, write_fasta_record, write_ids, sha256sum
from .features import kmer_vector
from .embedding_esm2 import infer_class as infer_esm2
from .embedding_lucavirus import infer_class as infer_luca


def query_records(path):
    """Query ID = first token of FASTA header; doesn't require numeric TaxID."""
    path=Path(path)
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8-sig') as f:
        head=None; data=[]
        for n,line in enumerate(f,1):
            s=line.strip()
            if not s:continue
            if s.startswith('>'):
                if head is not None:yield head,''.join(data).upper()
                head=s[1:].strip();data=[]
                if not head:raise ValueError(f'Empty FASTA header {path}:{n}')
            elif head is None:
                raise ValueError(f'Sequence before first header {path}:{n}')
            else:data.append(s)
        if head is not None:yield head,''.join(data).upper()


def predict(input_fasta, model_dir, output, length, slice_seed=None,
            lucavirus_tasks=None, gpu_id=-1, batch_size=16, device='auto'):
    """All input records are predicted using one selected model length.

    For a fixed-length model, queries must have at least the window length;
    deterministic windows use the same SHA-256 algorithm, keyed by QUERY ID.
    No undocumented auto-selection of models is made.
    """
    model_dir=Path(model_dir)
    manifest_path=model_dir/'model_manifest.json'
    model_path=model_dir/'model.joblib'
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError('Expected model.joblib + model_manifest.json in model directory')
    meta=json.loads(manifest_path.read_text(encoding='utf-8'))
    if meta['length']!=length:raise ValueError(f'Model length {meta["length"]} != selected {length}')
    feature=meta['feature'];algorithm=meta['algorithm']
    if feature not in FEATURE_DIMS:raise ValueError(f'Unsupported model feature {feature}')
    if meta.get('model_sha256') != sha256sum(model_path):
        raise ValueError('Model SHA256 mismatch; refusing to load altered model')
    if length!='full_length':
        saved_seed=meta.get('slice_seed')
        if saved_seed is not None:
            if slice_seed is not None and str(slice_seed)!=str(saved_seed):
                raise ValueError('Query --slice-seed does not match the model training seed')
            slice_seed=str(saved_seed)
        if slice_seed is None:
            raise ValueError('For fixed-length models provide --slice-seed (the training window seed)')
    if feature=='LucaVirus' and lucavirus_tasks is None:
        raise ValueError('--lucavirus-tasks is required for fresh LucaVirus inference')
    output=Path(output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite existing output {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Joblib uses pickle; user must only load trusted models from this project.
    clf=joblib.load(model_path)
    expected_classes=[0,1,2]
    model_classes=list(map(int,clf.classes_ if hasattr(clf,'classes_') else clf.named_steps['clf'].classes_))
    if sorted(model_classes)!=expected_classes:raise ValueError('Classifier has unexpected classes_')
    with tempfile.TemporaryDirectory(prefix='vh3_query_') as temp:
        root=Path(temp); seqdir=root/'sequences'/length;seqdir.mkdir(parents=True)
        fasta=seqdir/f'Other_{length}.fasta'
        items=[];known=set()
        with fasta.open('w',encoding='utf-8') as out:
            for header,seq in query_records(input_fasta):
                name=header.split()[0]
                if not name or name in known:raise ValueError(f'Duplicate/empty query ID: {name!r}')
                if not seq:raise ValueError(f'Empty sequence for query {name!r}')
                if re.search('[^ACGTRYSWKMBDHVN]',seq):
                    raise ValueError(f'Query {name!r} has unsupported nucleotide symbol')
                known.add(name)
                if length=='full_length':fragment=seq
                else:
                    result=choose_fragment(seq,int(length[:-2]),str(slice_seed),name)
                    if result is None:
                        raise ValueError(f'Query {name!r} has length {len(seq)} < required {length}; choose a shorter model')
                    fragment=result[1]
                surrogate=str(len(items)+1)
                write_fasta_record(out,surrogate,fragment)
                items.append((name,header,len(seq),surrogate))
        if not items:raise ValueError('No sequences in input FASTA')
        write_ids(seqdir/'Other_ids.tsv',[(s,'Other') for _,_,_,s in items])
        feats=root/'features'
        if feature.endswith('mer'):
            from .common import fasta_records
            k=int(feature[0])
            d=feats/length/feature/'Other';d.mkdir(parents=True,exist_ok=False)
            feature_rows=[kmer_vector(s,k) for _,_,s in fasta_records(fasta)]
            np.save(d/'features.npy',np.vstack(feature_rows).astype(np.float32))
            write_ids(d/'ids.tsv',[(s,'Other') for _,_,_,s in items])
        elif feature=='ESM2':
            infer_esm2(root/'sequences',feats,length,'Other',batch_size,device)
        elif feature=='LucaVirus':
            infer_luca(root/'sequences',feats,length,'Other',lucavirus_tasks,gpu_id,chunk_size=1000)
        else:raise ValueError(f'Feature not implemented for query inference: {feature}')
        X=np.load(feats/length/feature/'Other'/'features.npy')
        if X.shape!=(len(items),FEATURE_DIMS[feature]):raise ValueError('Query feature count/dimension mismatch')
        p0=clf.predict_proba(X)
        probs=np.zeros((len(items),3),dtype=float)
        for j,c in enumerate(model_classes):probs[:,c]=p0[:,j]
        if not np.isfinite(probs).all():raise ValueError('Non-finite prediction probabilities')
        preds=np.argmax(probs,axis=1)
        rows=[]
        for i,(name,header,nt_len,surrogate) in enumerate(items):
            rows.append({'query_id':name,'original_header':header,'input_length':nt_len,
                         'model_length':length,'model_feature':feature,'algorithm':algorithm,
                         'prediction':CLASSES[int(preds[i])],
                         'prob_Insect':f'{probs[i,0]:.6f}',
                         'prob_Plant':f'{probs[i,1]:.6f}',
                         'prob_Other':f'{probs[i,2]:.6f}'})
        with output.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    print(f'[PREDICT] {len(items)} queries -> {output}',flush=True)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='FASTA or FASTA.gz; arbitrary unique FASTA query names')
    p.add_argument('--model-dir',required=True,help='Fresh fitted model directory')
    p.add_argument('--output',required=True,help='New predictions.tsv path; never overwrites')
    p.add_argument('--length',required=True,choices=LENGTHS,help='Explicit length model; no unsupported auto-routing')
    p.add_argument('--slice-seed',default=None,help='Same fixed slicing seed as trained model for 1/3/5 kb')
    p.add_argument('--lucavirus-tasks',default=None,help='Official LucaVirusTasks git checkout')
    p.add_argument('--gpu-id',type=int,default=-1)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='auto')
    a=p.parse_args(argv)
    predict(a.input,a.model_dir,a.output,a.length,a.slice_seed,a.lucavirus_tasks,
            a.gpu_id,a.batch_size,a.device)
if __name__=='__main__':main()
