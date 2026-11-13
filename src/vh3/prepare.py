"""Prepare four length-specific classwise FASTAs with explicit provenance.

historical mode: copy exact previously used 1/3/5kbp sequences, never re-slice.
regenerate mode: deterministic SHA-256 windowing of all three full-length classes.
The latter is NOT asserted to recreate historical Insect/Plant slices.
"""
import argparse
import csv
from pathlib import Path
import shutil
from .common import (
    CLASSES,LENGTHS,PUBLISHED_INPUT_COUNTS,choose_fragment,fasta_records,
    write_fasta_record,sha256sum,write_ids
)

def find_historical(root,length,cls):
    for ext in ('.fasta','.fasta.gz','.fna','.fna.gz'):
        p=root/length/(cls+'_'+length+ext)
        if p.is_file():return p
    raise FileNotFoundError(f'Missing historical fragment: {root}/{length}/{cls}_{length}.fasta[.gz]')

def prepare(infiles, outroot, mode, seed=None, historical_dir=None, expect_published=False):
    outroot=Path(outroot)
    if outroot.exists() and any(outroot.iterdir()):
        raise FileExistsError(f'Destination must not contain previous output: {outroot}')
    if mode=='regenerate' and seed is None:
        raise ValueError('Specify an explicit --slice-seed for deterministic nucleotide windows')
    if mode=='historical' and historical_dir is None:
        raise ValueError('--historical-dir is required in historical mode')
    # Validate IDs and lengths without loading all full viral genomes into RAM.
    source_ids={}
    for cls in CLASSES:
        file=Path(infiles[cls])
        if not file.is_file(): raise FileNotFoundError(file)
        seen=set()
        for tid,header,seq in fasta_records(file):
            if tid in seen:raise ValueError(f'{cls}: duplicate TaxID {tid}')
            if not seq:raise ValueError(f'{cls}: empty sequence TaxID={tid}')
            seen.add(tid)
        if expect_published and len(seen)!=PUBLISHED_INPUT_COUNTS['full_length'][cls]:
            raise ValueError(f'{cls} full-length count {len(seen)}, expected {PUBLISHED_INPUT_COUNTS["full_length"][cls]}')
        source_ids[cls]=seen
    for i,cl in enumerate(CLASSES):
        for other in CLASSES[i+1:]:
            dup=source_ids[cl]&source_ids[other]
            if dup:raise ValueError(f'Input TaxIDs overlap across classes ({cl}/{other}): {sorted(dup)[:10]}')
    manifest=[]
    outroot.mkdir(parents=True,exist_ok=True)
    for length in LENGTHS:
        (outroot/length).mkdir(exist_ok=True)
    # Iterate one full-length FASTA per class only once for data production.
    for cls in CLASSES:
        src=Path(infiles[cls])
        direct_lengths = LENGTHS if mode=='regenerate' else ('full_length',)
        handles={length:(outroot/length/f'{cls}_{length}.fasta').open('w',encoding='utf-8') for length in direct_lengths}
        ids_by_length={length:[] for length in LENGTHS}
        try:
            for tid,header,seq in fasta_records(src):
                for length in direct_lengths:
                    if length=='full_length':piece=seq;out_header=header
                    else:
                        selected=choose_fragment(seq,int(length[:-2]),str(seed),tid)
                        if selected is None:continue
                        start,piece=selected
                        out_header=f'{tid}|class={cls}|length={length}|start={start+1}|source_length={len(seq)}'
                    write_fasta_record(handles[length],out_header,piece)
                    ids_by_length[length].append((tid,cls))
        finally:
            for h in handles.values():h.close()
        if mode=='historical':
            for length in LENGTHS[:-1]:
                source=find_historical(Path(historical_dir),length,cls)
                dest=outroot/length/f'{cls}_{length}.fasta'
                seen=set()
                with dest.open('w',encoding='utf-8') as dst:
                    for tid,header,seq in fasta_records(source):
                        if tid not in source_ids[cls]:raise ValueError(f'{source}: unknown TaxID {tid}')
                        if len(seq)!=int(length[:-2]):raise ValueError(f'{source}: {tid} has length {len(seq)}')
                        if tid in seen:raise ValueError(f'{source}: duplicate TaxID {tid}')
                        seen.add(tid)
                        write_fasta_record(dst,header,seq)
                        ids_by_length[length].append((tid,cls))
        for length in LENGTHS:
            dest=outroot/length/f'{cls}_{length}.fasta'
            ids=ids_by_length[length]
            if expect_published and len(ids)!=PUBLISHED_INPUT_COUNTS[length][cls]:
                raise ValueError(f'{length}/{cls}: got {len(ids)}, expected {PUBLISHED_INPUT_COUNTS[length][cls]}')
            write_ids(outroot/length/f'{cls}_ids.tsv',ids)
            manifest.append((length,cls,len(ids),str(dest.relative_to(outroot)),sha256sum(dest),mode,str(seed or 'historical_copy')))
            print(f'[OK] {length} {cls}: {len(ids)} sequences -> {dest}',flush=True)
    with (outroot/'sequence_manifest.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['length','class','count','filename','sha256','mode','seed']);w.writerows(manifest)
    print('Prepared all sequences and class-wise ids.tsv; originals unchanged.')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for cls in CLASSES:
        p.add_argument('--'+cls.lower(),required=True,help=f'{cls} full-length FASTA or .gz')
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['historical','regenerate'],required=True)
    p.add_argument('--slice-seed',default=None,help='Original string SEED value from the historical 07 script')
    p.add_argument('--historical-dir',default=None,help='Root with 1000bp/3000bp/5000bp historical FASTAs')
    p.add_argument('--expect-published',action='store_true',help='Require exact historical sample counts')
    a=p.parse_args(argv)
    prepare({c:getattr(a,c.lower()) for c in CLASSES},a.output,a.mode,a.slice_seed,a.historical_dir,a.expect_published)
if __name__=='__main__':main()
