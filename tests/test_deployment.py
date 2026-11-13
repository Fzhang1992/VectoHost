"""Deployment-path tests with synthetic sequences only (no pretrained weights)."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from vh3.common import write_ids
from vh3.final_model import fit_deploy_model
from vh3.predict import predict, query_records


def make_features(tmp_path):
    root=tmp_path/'features'/'full_length'/'3mer'
    for c,ids in {'Insect':['10001','10002','10003','10004','10005','10006','1277443'],
                  'Plant':['20001','20002','20003','20004','20005','20006'],
                  'Other':['30001','30002','30003','30004','30005','30006']}.items():
        d=root/c;d.mkdir(parents=True)
        vec=np.zeros((len(ids),64),dtype=np.float32)
        for i,v in enumerate(ids):
            vec[i,int(v)%64]=1.
        np.save(d/'features.npy',vec)
        write_ids(d/'ids.tsv',[(s,c) for s in ids])
    return tmp_path/'features'


def test_save_and_predict_fresh_kmer(tmp_path):
    root=make_features(tmp_path)
    model=fit_deploy_model(root,'full_length','3mer','GaussianNB',tmp_path/'models'/'full_length'/'3mer_GaussianNB')
    meta=json.loads((model/'model_manifest.json').read_text())
    assert meta['training_class_counts_before_undersampling']=={'Insect':6,'Plant':6,'Other':6}
    assert meta['training_rows_after_undersampling']==18
    query=tmp_path/'query.fasta'
    query.write_text('>seq_A extra_annotation\nAAAAAAAACCCCGGGGTTTT\n>seq_B\nCCCCCCTTTTTAAAAAGGG\n')
    result=tmp_path/'predictions.tsv'
    out=predict(query,model,result,'full_length')
    assert [x['query_id'] for x in out]==['seq_A','seq_B']
    assert {r['prediction'] for r in out} <= {'Insect','Plant','Other'}
    for r in out:
        assert abs(sum(float(r[k]) for k in ('prob_Insect','prob_Plant','prob_Other'))-1)<0.000003
    assert len(pd.read_csv(result,sep='\t'))==2
    with pytest.raises(FileExistsError):
        predict(query,model,result,'full_length')


def test_fixed_length_requires_original_seed(tmp_path):
    root=tmp_path/'features'/'1000bp'/'3mer'
    for cl,offset in [('Insect',10000),('Plant',20000),('Other',30000)]:
        d=root/cl;d.mkdir(parents=True)
        ids=[str(offset+j) for j in range(7)]
        vec=np.zeros((7,64),dtype=np.float32)
        for i in range(7):
            vec[i,(i+offset//10000)%64]=0.7
            vec[i,(i*3+offset//10000+5)%64]+=0.3
        np.save(d/'features.npy',vec)
        write_ids(d/'ids.tsv',[(s,cl) for s in ids])
    model=fit_deploy_model(tmp_path/'features','1000bp','3mer','GaussianNB',tmp_path/'model')
    query=tmp_path/'query.fasta';query.write_text('>query1\n'+'ACGT'*260+'\n')
    with pytest.raises(ValueError,match='--slice-seed'):
        predict(query,model,tmp_path/'p.tsv','1000bp')
    out=predict(query,model,tmp_path/'p.tsv','1000bp',slice_seed='original_test_seed')
    assert len(out)==1


def test_arbitrary_query_ids_and_gzip(tmp_path):
    import gzip
    f=tmp_path/'x.fasta.gz'
    with gzip.open(f,'wt') as handle:handle.write('>virus_A some context\nACGT\n>virus-B\nGGGG\n')
    assert list(query_records(f))==[('virus_A some context','ACGT'),('virus-B','GGGG')]


def test_predict_lucavirus_invokes_fresh_inference_not_old_features(tmp_path):
    """Tests wrapper plumbing with mock LucaVirus producer; NOT real foundation model."""
    import sys
    root=tmp_path/'features'/'full_length'/'LucaVirus'
    for index,(cl,offset) in enumerate((('Insect',10000),('Plant',20000),('Other',30000))):
        d=root/cl;d.mkdir(parents=True)
        ids=[str(offset+j) for j in range(7)]
        vec=np.zeros((7,2560),dtype=np.float32)
        for j in range(7):vec[j,:]=index*10+j/8
        np.save(d/'features.npy',vec)
        write_ids(d/'ids.tsv',[(s,cl) for s in ids])
    model=fit_deploy_model(tmp_path/'features','full_length','LucaVirus','KNN',tmp_path/'final')
    query=tmp_path/'q.fasta'
    query.write_text('>new_virus_A\n'+'ACTG'*300+'\n>new_virus_B\n'+'GCTA'*300+'\n')
    repo=tmp_path/'test_only_mock_upstream'
    producer=repo/'src/llm/lucavirus/get_embedding.py'
    producer.parent.mkdir(parents=True)
    producer.write_text('''
import argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for x in ('llm_dir','llm_type','llm_version','llm_time_str','llm_step','seq_type','input_file','save_path','embedding_type','embedding_fixed_len_a_time','gpu_id'):
    p.add_argument('--'+x)
p.add_argument('--embedding_complete',action='store_true')
a=p.parse_args()
out=Path(a.save_path);out.mkdir(parents=True,exist_ok=True)
for ln in Path(a.input_file).read_text().splitlines():
    if ln.startswith('>'):
        taxid=ln[1:].strip()
        torch.save(torch.full((2560,),float(int(taxid))),out/f'vector_{taxid}.pt')
''')
    out= predict(query,model,tmp_path/'result.tsv','full_length',lucavirus_tasks=repo)
    assert len(out)==2
    assert (tmp_path/'result.tsv').is_file()
