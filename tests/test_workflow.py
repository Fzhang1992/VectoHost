import gzip
import random
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from vh3.common import choose_fragment, fasta_records, CLASSES, CROSS_HOST
from vh3.features import kmer_vector, extract_kmer
from vh3.prepare import prepare
from vh3.train import train
from vh3.evaluate import evaluate_table
from vh3.embedding_esm2 import get_longest_orf, translate_frame
from vh3.embedding_lucavirus import infer_class


def write_dummy_fasta(path, taxids, label, size=5100):
    rng=random.Random(12345)
    with (gzip.open(path,'wt',encoding='utf-8') if str(path).endswith('.gz') else open(path,'w',encoding='utf-8')) as f:
        for tid in taxids:
            seq=''.join(rng.choices('ACGT',k=size))
            f.write(f'>{tid}|class={label}\n{seq}\n')

def test_deterministic_fragment():
    x='ACGT'*100
    assert choose_fragment(x,100,'test','199')==choose_fragment(x,100,'test','199')
    assert choose_fragment(x,999,'test','199') is None

def test_kmer():
    x=kmer_vector('AAAACN',2)
    assert x.shape==(16,)
    assert np.isclose(x.sum(),1.0)
    assert np.isclose(x[0],3/4)

def test_esm2_historical_six_frame_orf():
    aa,info=get_longest_orf('ATG'+'GCT'*30+'TAA')
    assert isinstance(info,str) and info
    assert len(aa)>0
    assert all(x in 'ACDEFGHIKLMNPQRSTVWYX' for x in aa)
    assert len(get_longest_orf('ATG'*5000)[0])<=1022

def test_full_toy_kmer_train_evaluate(tmp_path):
    sources=tmp_path/'sources';sources.mkdir()
    all_ids={'Insect':[str(100+i) for i in range(6)]+['1277443'],
             'Plant':[str(200+i) for i in range(6)],
             'Other':[str(300+i) for i in range(6)]}
    infiles={}
    for cls in CLASSES:
        p=sources/(cls+'.fasta.gz' if cls=='Other' else cls+'.fasta')
        write_dummy_fasta(p,all_ids[cls],cls)
        infiles[cls]=p
    seqroot=tmp_path/'sequences'
    prepare(infiles,seqroot,'regenerate',seed='original-example-seed')
    assert infiles['Other'].exists()
    featroot=tmp_path/'features'
    extract_kmer(seqroot,featroot,'1000bp',3)
    result=tmp_path/'results'
    train(featroot,'1000bp',result,['3mer'],['GaussianNB'],expected=False)
    df=pd.read_csv(result/'oof_predictions.tsv',sep='\t',dtype={'taxid':str})
    assert len(df)==19
    assert df.loc[df.taxid=='1277443','true_class'].iat[0]=='Insect'
    summary,cm,audit=evaluate_table(result/'oof_predictions.tsv','1000bp')
    assert len(summary)==1
    assert audit.crosshost_removed.iat[0]==1
    assert audit.evaluated_n.iat[0]==18


def test_lucavirus_wrapper_requires_new_pt_by_taxid(tmp_path):
    # Fake producer exercises wrapper I/O and ID validation ONLY, not the real model.
    import torch
    source=tmp_path/'sequences'/'1000bp';source.mkdir(parents=True)
    with (source/'Insect_1000bp.fasta').open('w') as f:
        f.write('>100\n'+'ACGT'*250+'\n>200\n'+'GCTA'*250+'\n')
    from vh3.common import write_ids
    write_ids(source/'Insect_ids.tsv',[('100','Insect'),('200','Insect')])
    repo=tmp_path/'fake_lucavirus_tasks'
    script=repo/'src/llm/lucavirus/get_embedding.py'
    script.parent.mkdir(parents=True)
    script.write_text('''
import argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for a in ('llm_dir','llm_type','llm_version','llm_time_str','llm_step','seq_type','input_file','save_path','embedding_type','embedding_fixed_len_a_time','gpu_id'):
    p.add_argument('--'+a)
p.add_argument('--embedding_complete',action='store_true')
a=p.parse_args()
out=Path(a.save_path);out.mkdir(parents=True,exist_ok=True)
for line in Path(a.input_file).read_text().splitlines():
    if line.startswith('>'):
        tid=line[1:]
        torch.save(torch.full((2560,),float(int(tid)),dtype=torch.float32),out/f'vector_{tid}.pt')
''')
    dest=tmp_path/'features'
    infer_class(tmp_path/'sequences',dest,'1000bp','Insect',repo,-1,1,sys.executable)
    feature=np.load(dest/'1000bp'/'LucaVirus'/'Insect'/'features.npy')
    assert feature.shape==(2,2560)
    assert feature[0,0]==100 and feature[1,0]==200
    assert (dest/'1000bp'/'LucaVirus'/'Insect'/'inference_manifest.json').exists()
