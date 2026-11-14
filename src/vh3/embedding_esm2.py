"""NEW ESM-2 inference from class-specific nucleotide FASTA (never imports old embeddings).

Method transcribed from the project's original 11_extract_ESM2_final_binary.py:
  six reading frames -> longest stop-free translated segment -> 50-aa fallback
  -> first 1022 aa -> ESM-2 t6/8M/UR50D layer6 mean residue embedding.
The original 11.py is a *binary* workflow; this reimplementation extends it to
all three classes and four lengths. Numeric equivalence must be checked on server.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import hashlib
import json
import os
import numpy as np
from .common import CLASSES, LENGTHS, fasta_records, load_ids, write_ids

MIN_AA = 50
GENETIC_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L', 'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
COMP=str.maketrans('ACGTNacgtn','TGCANtgcan')

def clean_seq(seq):
    return re.sub(r'[^ACGTNacgtn]','N',str(seq)).upper()

def revcomp(seq):
    return clean_seq(seq).translate(COMP)[::-1].upper()

def translate_frame(seq, frame):
    seq=clean_seq(seq)
    return ''.join(GENETIC_CODE.get(seq[i:i+3],'X') for i in range(frame,len(seq)-2,3))

def get_longest_orf(seq):
    """Preserves original script's strict > tie breaking and fallback."""
    seq=clean_seq(seq)
    best=''; info=''
    for strand_name,s in (('+',seq),('-',revcomp(seq))):
        for frame in (0,1,2):
            aa=translate_frame(s,frame)
            orf=max(aa.split('*'),key=len)
            if len(orf)>len(best):
                best=orf;info=f'{strand_name}{frame}'
    if len(best)<MIN_AA:
        fallback='';fallback_info=''
        for strand_name,s in (('+',seq),('-',revcomp(seq))):
            for frame in (0,1,2):
                aa=translate_frame(s,frame).replace('*','')
                if len(aa)>len(fallback):
                    fallback=aa;fallback_info=f'{strand_name}{frame}_fallback_no_stop'
        best=fallback;info=fallback_info
    best=best.replace('*','')
    best=re.sub(r'[^ACDEFGHIKLMNPQRSTVWYX]','X',best)
    if len(best)>1022:
        best=best[:1022];info+='_truncated1022'
    return best,info

def infer_class(sequence_root,feature_root,length,cls,batch_size=16,device='auto'):
    """Extract fresh embeddings; checkpoint at a batch boundary for safe restart."""
    import torch
    source=Path(sequence_root)/length/f'{cls}_{length}.fasta'
    ids_file=Path(sequence_root)/length/f'{cls}_ids.tsv'
    ids=load_ids(ids_file)
    if any(label!=cls for tid,label in ids):raise ValueError('Class mismatch in ids')
    dest=Path(feature_root)/length/'ESM2'/cls
    from .common import sha256sum
    fingerprint={'sequence_sha256':sha256sum(source),'ids_sha256':sha256sum(ids_file),
                 'model':'esm2_t6_8M_UR50D','layer':6,'dimension':320}
    complete=dest/'features.npy'
    if complete.is_file() and (dest/'ids.tsv').is_file() and (dest/'inference_manifest.json').is_file():
        if (load_ids(dest/'ids.tsv')==ids and
            np.load(complete,mmap_mode='r').shape==(len(ids),320) and
            json.loads((dest/'inference_manifest.json').read_text())['input_fingerprint']==fingerprint):
            print(f'[SKIP] Verified ESM2 {length}/{cls}',flush=True)
            return
        raise RuntimeError(f'Existing ESM2 output does not match source data: {dest}')
    dest.mkdir(parents=True,exist_ok=True)
    state_path=dest/'progress.json'
    part=dest/'features.npy.part'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state.get('input_fingerprint')!=fingerprint or state.get('batch_size')!=batch_size:
            raise RuntimeError(f'ESM2 resume configuration/input changed: {dest}')
        completed=int(state['completed'])
        if not part.exists():raise RuntimeError(f'ESM2 progress exists but matrix missing: {dest}')
        output=np.lib.format.open_memmap(part,mode='r+')
        if output.shape!=(len(ids),320):raise RuntimeError('ESM2 unfinished matrix shape mismatch')
    else:
        if any(dest.iterdir()):
            raise RuntimeError(f'Unrecognized prior ESM2 output at {dest}; inspect before restarting')
        completed=0
        output=np.lib.format.open_memmap(part,mode='w+',dtype=np.float32,shape=(len(ids),320))
        state={'input_fingerprint':fingerprint,'batch_size':batch_size,'completed':0}
        state_path.write_text(json.dumps(state,indent=2)+'\n')
    try:
        import esm
    except ImportError as e:
        raise RuntimeError('Install fair-esm==2.0.0 for ESM-2 inference') from e
    try:
        model,alphabet=esm.pretrained.esm2_t6_8M_UR50D()
        converter=alphabet.get_batch_converter()
        torch_device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
        model=model.to(torch_device).eval()
        def save_progress(n):
            output.flush()
            state['completed']=n
            temp=state_path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(state,indent=2)+'\n')
            os.replace(temp,state_path)
        buffer=[];cursor=completed;seen=0
        def process(batch):
            nonlocal cursor
            _,strings,tokens=converter(batch)
            with torch.no_grad():
                reps=model(tokens.to(torch_device),repr_layers=[6],return_contacts=False)['representations'][6]
                for ix,aa in enumerate(strings):
                    n=len(aa)
                    vector=reps[ix,1:n+1].mean(dim=0).detach().cpu().numpy()
                    if vector.shape!=(320,) or not np.isfinite(vector).all():
                        raise ValueError(f'Invalid ESM2 embedding for {batch[ix][0]}')
                    output[cursor]=vector.astype(np.float32)
                    cursor+=1
            save_progress(cursor)
            print(f'[ESM2] {length}/{cls}: {cursor}/{len(ids)}',flush=True)
        for i,(tid,header,seq) in enumerate(fasta_records(source)):
            if i>=len(ids) or tid!=ids[i][0]:raise ValueError(f'TaxID order differs at {source} row={i}')
            seen+=1
            if i<completed:continue
            aa,_=get_longest_orf(seq)
            if not aa:aa='X'*MIN_AA
            buffer.append((tid,aa))
            if len(buffer)>=batch_size:
                process(buffer);buffer=[]
        if buffer:process(buffer)
        if cursor!=len(ids) or seen!=len(ids):raise ValueError('Incomplete ESM2 input processing')
        # Reconstruct the ORF audit from the same FASTA, separate from batch progress.
        info_file=dest/'orf_info.tsv.part'
        with info_file.open('w',encoding='utf-8',newline='') as fh:
            writer=csv.writer(fh,delimiter='\t');writer.writerow(['taxid','aa_length','orf_info'])
            for tid,_,seq in fasta_records(source):
                aa,info=get_longest_orf(seq)
                if not aa:aa='X'*MIN_AA;info='empty_orf_fallback'
                writer.writerow([tid,len(aa),info])
        output.flush();del output
        checkpoint=Path(torch.hub.get_dir())/'checkpoints'/'esm2_t6_8M_UR50D.pt'
        digest=None
        if checkpoint.is_file():
            h=hashlib.sha256()
            with checkpoint.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            digest=h.hexdigest()
        manifest={'model':'esm2_t6_8M_UR50D','layer':6,'dimension':320,
                  'ORF':'six_frame_longest_stop_free; fallback_no_stop if length<50; cap1022',
                  'batch_size':batch_size,'device':str(torch_device),'torch':torch.__version__,
                  'checkpoint_path':str(checkpoint) if checkpoint.is_file() else None,
                  'checkpoint_sha256':digest,'records':cursor,'input_fingerprint':fingerprint}
        (dest/'inference_manifest.json.tmp').write_text(json.dumps(manifest,indent=2)+'\n')
        os.replace(dest/'features.npy.part',dest/'features.npy')
        os.replace(dest/'orf_info.tsv.part',dest/'orf_info.tsv')
        write_ids(dest/'ids.tsv',ids)
        os.replace(dest/'inference_manifest.json.tmp',dest/'inference_manifest.json')
        state_path.unlink(missing_ok=True)
        (dest/'FAILED.txt').unlink(missing_ok=True)
        print(f'[DONE] ESM2 {length}/{cls} ({cursor},320)',flush=True)
    except BaseException:
        (dest/'FAILED.txt').write_text('Interrupted or failed: rerun with unchanged input/configuration to resume.\n')
        raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences',required=True);p.add_argument('--features-dir',required=True)
    p.add_argument('--length',choices=LENGTHS,required=True)
    p.add_argument('--class-name',choices=CLASSES,required=True)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='auto')
    a=p.parse_args(argv)
    if a.batch_size<1: p.error('--batch-size must be positive')
    infer_class(a.sequences,a.features_dir,a.length,a.class_name,a.batch_size,a.device)
if __name__=='__main__':main()
