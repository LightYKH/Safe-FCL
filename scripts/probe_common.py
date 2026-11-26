"""Portable helpers used inside each newly prepared probe stage."""
import hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def verify(p):
    for name,digest in load(p)['files'].items():
        if sha(name)!=digest:raise ValueError('Frozen collection input changed: '+name)

def freeze(p,files):write(p,dict(files={str(Path(f).resolve()):sha(f) for f in files}))
