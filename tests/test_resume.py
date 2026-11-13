"""Restart and provenance tests use only artificial sequences and model stubs."""
import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest
import torch
from vh3.common import write_ids
from vh3.embedding_esm2 import infer_class as infer_esm
from vh3.embedding_lucavirus import infer_class as infer_luca
from vh3.features import extract_kmer
from vh3.run_identity import verify


def build_class(tmp_path, n=4):
    root=tmp_path/'sequences'
    d=root/'1000bp'
    d.mkdir(parents=True)
    ids=[str(10000+i) for i in range(n)]
    with (d/'Insect_1000bp.fasta').open('w') as fh:
        for tid in ids:
            fh.write('>'+tid+'\n'+'ACGT'*250+'\n')
    write_ids(d/'Insect_ids.tsv',[(x,'Insect') for x in ids])
    return root,ids


def test_kmer_reentry_verifies_completed_files(tmp_path):
    # Create the three class files that the combined kmer extractor requires.
    seq=tmp_path/'sequences'/'1000bp'
    seq.mkdir(parents=True)
    for cl,offset in [('Insect',10),('Plant',20),('Other',30)]:
        (seq/f'{cl}_1000bp.fasta').write_text(f'>{offset}\n'+'ACGT'*250+'\n')
        write_ids(seq/f'{cl}_ids.tsv',[(str(offset),cl)])
    extract_kmer(tmp_path/'sequences',tmp_path/'features','1000bp',3)
    m=tmp_path/'features'/'1000bp'/'3mer'/'Insect'/'features.npy'
    first=m.read_bytes()
    extract_kmer(tmp_path/'sequences',tmp_path/'features','1000bp',3)
    assert m.read_bytes()==first
    # An incorrect complete file must never be silently reused.
    np.save(m,np.zeros((2,64),dtype=np.float32))
    with pytest.raises(ValueError,match='does not match'):
        extract_kmer(tmp_path/'sequences',tmp_path/'features','1000bp',3)


def test_esm2_continues_from_completed_batch(tmp_path,monkeypatch):
    seq,ids=build_class(tmp_path,4)
    calls={'n':0,'fail':True}
    class FakeAlphabet:
        def get_batch_converter(self):
            def convert(batch):
                strings=[x[1] for x in batch]
                tokens=torch.zeros((len(strings),max(map(len,strings))+2),dtype=torch.long)
                return [x[0] for x in batch],strings,tokens
            return convert
    class FakeModel:
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,tokens,repr_layers,return_contacts):
            calls['n']+=1
            if calls['fail'] and calls['n']==2:raise RuntimeError('synthetic preemption')
            return {'representations':{6:torch.ones((tokens.shape[0],tokens.shape[1],320))}}
    fake=types.SimpleNamespace(pretrained=types.SimpleNamespace(
        esm2_t6_8M_UR50D=lambda:(FakeModel(),FakeAlphabet())))
    monkeypatch.setitem(sys.modules,'esm',fake)
    with pytest.raises(RuntimeError,match='synthetic preemption'):
        infer_esm(seq,tmp_path/'features','1000bp','Insect',batch_size=2,device='cpu')
    dest=tmp_path/'features'/'1000bp'/'ESM2'/'Insect'
    assert json.loads((dest/'progress.json').read_text())['completed']==2
    calls['fail']=False
    infer_esm(seq,tmp_path/'features','1000bp','Insect',batch_size=2,device='cpu')
    assert np.load(dest/'features.npy').shape==(4,320)
    assert not (dest/'progress.json').exists()
    assert not (dest/'FAILED.txt').exists()
    infer_esm(seq,tmp_path/'features','1000bp','Insect',batch_size=2,device='cpu')


def test_lucavirus_reuses_verified_raw_vectors(tmp_path):
    seq,ids=build_class(tmp_path,3)
    repo=tmp_path/'mock_upstream'
    script=repo/'src/llm/lucavirus/get_embedding.py'
    script.parent.mkdir(parents=True)
    (repo/'fail_once').write_text('1')
    script.write_text('''
import argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for name in ('llm_dir','llm_type','llm_version','llm_time_str','llm_step','seq_type','input_file','save_path','embedding_type','embedding_fixed_len_a_time','gpu_id'):
    p.add_argument('--'+name)
p.add_argument('--embedding_complete',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
out=Path(a.save_path);out.mkdir(parents=True,exist_ok=True)
for row in Path(a.input_file).read_text().splitlines():
    if not row.startswith('>'):continue
    tid=row[1:].strip()
    with (root/'calls.txt').open('a') as fh:fh.write(tid+'\\n')
    if tid=='10001' and (root/'fail_once').exists():
        (root/'fail_once').unlink()
        raise SystemExit(2)
    torch.save(torch.ones(2560)*int(tid),out/f'vector_{tid}.pt')
''')
    with pytest.raises(Exception):
        infer_luca(seq,tmp_path/'features','1000bp','Insect',repo,chunk_size=1)
    dest=tmp_path/'features'/'1000bp'/'LucaVirus'/'Insect'
    assert (dest/'progress.json').exists()
    assert (dest/'raw_vectors'/'vector_10000.pt').is_file()
    infer_luca(seq,tmp_path/'features','1000bp','Insect',repo,chunk_size=1)
    assert np.load(dest/'features.npy').shape==(3,2560)
    assert (repo/'calls.txt').read_text().splitlines().count('10000')==1
    assert not (dest/'progress.json').exists()
    infer_luca(seq,tmp_path/'features','1000bp','Insect',repo,chunk_size=1)


def test_run_identity_prevents_cross_input_mixing(tmp_path):
    inputs=[]
    for name in ('config.env','Insect.fasta','Plant.fasta','Other.fasta'):
        path=tmp_path/name;path.write_text(name)
        inputs.append(path)
    repo=tmp_path/'repo'
    script=repo/'src/llm/lucavirus/get_embedding.py'
    script.parent.mkdir(parents=True);script.write_text('pass\n')
    marker=tmp_path/'work/run_identity.json'
    verify(inputs[0],inputs[1:],repo,marker)
    verify(inputs[0],inputs[1:],repo,marker)
    inputs[1].write_text('different')
    with pytest.raises(SystemExit,match='changed'):
        verify(inputs[0],inputs[1:],repo,marker)
