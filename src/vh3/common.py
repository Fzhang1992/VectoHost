from pathlib import Path
import hashlib
import re
import gzip
import csv

CLASSES = ('Insect', 'Plant', 'Other')
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
LENGTHS = ('1000bp', '3000bp', '5000bp', 'full_length')
FEATURE_DIMS = {'1mer':4,'2mer':16,'3mer':64,'4mer':256,'ESM2':320,'LucaVirus':2560}
CROSS_HOST = frozenset({
    '1277443','228582','2170100','1985699','12287','223287','1795442',
    '3052585','1774200','187978','10991','10986','11307','12851'
})
# Counts in the historical input feature sets, BEFORE removing 14 cross-host TaxIDs.
PUBLISHED_INPUT_COUNTS = {
    '1000bp':{'Insect':1040,'Plant':2111,'Other':30314},
    '3000bp':{'Insect':927,'Plant':1511,'Other':29494},
    '5000bp':{'Insect':838,'Plant':1340,'Other':28313},
    'full_length':{'Insect':1040,'Plant':2111,'Other':30314},
}
PUBLISHED_EVAL_COUNTS = {
    '1000bp':{'Insect':1026,'Plant':2111,'Other':30314},
    '3000bp':{'Insect':914,'Plant':1511,'Other':29494},
    '5000bp':{'Insect':826,'Plant':1340,'Other':28313},
    'full_length':{'Insect':1026,'Plant':2111,'Other':30314},
}

def open_text(path):
    path = Path(path)
    return gzip.open(path,'rt',encoding='utf-8-sig') if path.suffix == '.gz' else open(path,'r',encoding='utf-8-sig')

def taxid_from_header(header):
    first = header.split('|',1)[0].split()[0].strip()
    if first.isdecimal():
        return first
    match = re.search(r'(?:^|[|\s])taxid\s*=\s*(\d+)(?:$|[|\s])',header,re.I)
    if match:
        return match.group(1)
    raise ValueError(f'FASTA header has no parseable TaxID: {header[:120]}')

def fasta_records(path):
    with open_text(path) as f:
        header = None
        blocks = []
        for lineno,line in enumerate(f,1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield taxid_from_header(header), header, ''.join(blocks).upper()
                header = line[1:]
                blocks = []
            else:
                if header is None:
                    raise ValueError(f'FASTA sequence before header at {path}:{lineno}')
                blocks.append(line)
        if header is not None:
            yield taxid_from_header(header),header,''.join(blocks).upper()

def write_fasta_record(out, header, seq):
    out.write('>'+header+'\n')
    for i in range(0,len(seq),80):
        out.write(seq[i:i+80]+'\n')

def sha256sum(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def choose_fragment(seq, n, seed, taxid):
    """Historical Other slicing rule: first 8 SHA-256 bytes, big endian, modulo valid starts."""
    if len(seq)<n:
        return None
    if len(seq)==n:
        start=0
    else:
        digest=hashlib.sha256(f'{seed}|{taxid}|{n}'.encode('utf-8')).digest()
        start=int.from_bytes(digest[:8],byteorder='big')%(len(seq)-n+1)
    return start,seq[start:start+n]

def load_ids(path):
    with open(path,'r',encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f,delimiter='\t')
        if not rd.fieldnames or not {'taxid','class'}.issubset(rd.fieldnames):
            raise ValueError(f'{path}: ids.tsv requires taxid and class columns')
        result=[(r['taxid'].strip(),r['class'].strip()) for r in rd]
    if not result or any(not tid.isdecimal() or cl not in CLASSES for tid,cl in result):
        raise ValueError(f'{path}: invalid TaxID/class')
    if len({tid for tid,_ in result})!=len(result):
        raise ValueError(f'{path}: duplicated TaxID')
    return result

def write_ids(path, rows):
    with open(path,'w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter='\t')
        wr.writerow(['taxid','class'])
        wr.writerows(rows)
