import argparse
from pathlib import Path
from collections import Counter
from .common import CLASSES,LENGTHS,fasta_records,load_ids,PUBLISHED_INPUT_COUNTS,CROSS_HOST

def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only pretraining consistency check')
    p.add_argument('--sequences',required=True);p.add_argument('--expect-published',action='store_true')
    a=p.parse_args(argv);root=Path(a.sequences)
    for length in LENGTHS:
        seen=set();counts={}
        for cl in CLASSES:
            ids=load_ids(root/length/f'{cl}_ids.tsv')
            fasta=root/length/f'{cl}_{length}.fasta'
            tids=[tid for tid,h,s in fasta_records(fasta)]
            if tids!=[tid for tid,_ in ids]:raise ValueError(f'{fasta}: order or IDs do not match ids.tsv')
            if len(set(tids))!=len(tids):raise ValueError(f'{fasta}: duplicated TaxID')
            if seen.intersection(tids):raise ValueError(f'{length}: duplicate TaxID between classes')
            seen.update(tids)
            if length!='full_length' and any(len(s)!=int(length[:-2]) for _,_,s in fasta_records(fasta)):
                raise ValueError(f'{fasta}: wrong length')
            counts[cl]=len(tids)
            if a.expect_published and len(tids)!=PUBLISHED_INPUT_COUNTS[length][cl]:
                raise ValueError(f'{length}/{cl}: count={len(tids)}, published={PUBLISHED_INPUT_COUNTS[length][cl]}')
        print(f'[PASS] {length}: {counts}, reserved_crosshost={len(seen&CROSS_HOST)}')
if __name__=='__main__':main()
