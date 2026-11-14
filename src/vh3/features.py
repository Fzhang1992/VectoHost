"""Extract 1/2/3/4-mer frequency features directly from freshly prepared FASTA."""
from pathlib import Path
import argparse
from itertools import product
from functools import lru_cache
import numpy as np
import shutil
from .common import CLASSES,LENGTHS,fasta_records,load_ids,write_ids
ALPHABET='ACGT'
@lru_cache(maxsize=4)
def _kmer_index(k):
    return {''.join(t):i for i,t in enumerate(product(ALPHABET,repeat=k))}

def kmer_vector(seq,k):
    idx=_kmer_index(k)
    vec=np.zeros(4**k,dtype=np.float32)
    for i in range(len(seq)-k+1):
        j=idx.get(seq[i:i+k]);
        if j is not None:vec[j]+=1
    total=float(vec.sum())
    if total:vec/=total
    return vec

def extract_kmer(sequence_root,feature_root,length,k):
    sequence_root=Path(sequence_root);feature_root=Path(feature_root)
    for cls in CLASSES:
        fasta=sequence_root/length/f'{cls}_{length}.fasta'
        ids=load_ids(sequence_root/length/f'{cls}_ids.tsv')
        if any(c!=cls for _,c in ids):raise ValueError('Class labels mismatch')
        target=feature_root/length/f'{k}mer'/cls
        if target.exists():
            existing=target/'features.npy'
            if existing.is_file() and (target/'ids.tsv').is_file():
                prev=load_ids(target/'ids.tsv')
                shape=np.load(existing,mmap_mode='r').shape
                if prev!=ids or shape!=(len(ids),4**k):
                    raise ValueError(f'Existing feature output does not match expected IDs/shape: {target}')
                print(f'[SKIP] Verified existing {length}/{k}mer/{cls}',flush=True)
                continue
            # A partly written *own* feature folder is safe to reconstruct.
            if any(x.name not in ('features.npy.part','ids.tsv') for x in target.iterdir()):
                raise RuntimeError(f'Unexpected unfinished files in {target}; inspect manually')
            shutil.rmtree(target)
        target.mkdir(parents=True)
        matrix=np.lib.format.open_memmap(target/'features.npy.part',mode='w+',dtype=np.float32,shape=(len(ids),4**k))
        n=0
        for i,(tid,header,seq) in enumerate(fasta_records(fasta)):
            if i>=len(ids) or tid!=ids[i][0]:raise ValueError(f'Incorrect TaxID order at {fasta} row={i}')
            matrix[i]=kmer_vector(seq,k);n+=1
        if n!=len(ids):raise ValueError(f'FASTA/IDs mismatch: {n} != {len(ids)}')
        matrix.flush();del matrix
        (target/'features.npy.part').replace(target/'features.npy')
        write_ids(target/'ids.tsv',ids)
        print(f'[DONE] {length}/{k}mer/{cls} shape=({n},{4**k})',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences',required=True);p.add_argument('--features-dir',required=True)
    p.add_argument('--length',choices=LENGTHS,required=True);p.add_argument('--k',type=int,choices=(1,2,3,4),required=True)
    a=p.parse_args(argv)
    extract_kmer(a.sequences,a.features_dir,a.length,a.k)
if __name__=='__main__':main()
