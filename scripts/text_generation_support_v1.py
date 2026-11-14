"""Pinned CPU contracts, counters and checkpoints for a NEW generation pilot."""
from __future__ import annotations
import hashlib,json,os,random,shutil,time
from pathlib import Path
import torch
import learning_io_v2 as io

SOURCE_SHA='5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675'
CORE_SHA='eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352'
SCHEMA='fdt_text_generation_pilot_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def state_sha(state):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(b'T'+str(x.dtype).encode()+str(tuple(x.shape)).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'D')
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))): visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(b'L' if isinstance(x,list) else b'Q')
            for v in x: visit(v)
        else: h.update((type(x).__name__+':'+repr(x)+';').encode())
    visit(state); return h.hexdigest()

def core_sha(root):
    h=hashlib.sha256()
    for p in sorted((Path(root)/'src/fdt_local').glob('*.py')): h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def validate(plan_path,output,worker=None):
    plan=io.load_json(plan_path); output=Path(output).resolve()
    if plan.get('schema')!=SCHEMA or type(plan.get('authorized')) is not bool: raise ValueError('schema/authorization')
    if Path(plan['output']).resolve()!=output: raise ValueError('output identity')
    limits={'attempts':256,'batch':8,'wall':900,'reserve':90,'max_calls':1300,
            'max_output_bytes':12*1024**3,'start_free_bytes':36*1024**3,'min_gpu_free_mib':12288}
    if plan['limits']!=limits: raise ValueError('prospective fixed limits mismatch')
    if plan['source_sha']!=SOURCE_SHA: raise ValueError('source checkpoint identity')
    if plan['optimizer_policy']!='fresh AdamW on new text branch; frozen copied language': raise ValueError('optimizer policy')
    root=Path(plan['root']).resolve()
    if core_sha(root)!=CORE_SHA: raise ValueError('core source changed')
    for file,expected in plan['files'].items():
        if sha(file)!=expected: raise ValueError('pinned file changed: '+file)
    if worker and str(Path(worker).resolve()) not in plan['files']: raise ValueError('worker unpinned')
    if plan.get('e2_final_accessed') is not False: raise ValueError('E2 final forbidden')
    if plan.get('deadline') is not None and plan['cutoff']!=plan['deadline']-90: raise ValueError('shutdown reserve')
    from research.text_generation_data_v1 import check
    data={name:io.load_json(Path(plan['data'])/(name+'.json')) for name in ('train','train_probe','dev_pairs','dev_wording')}
    review=check(data)
    if len(data['train'])<2048: raise ValueError('finite train pool too small')
    return plan,data,review

class Budget:
    def __init__(self,plan):
        self.plan=plan;self.output=Path(plan['output']);self.calls={'encode':0,'denoise':0};self.started=time.time()
    def check(self):
        if (self.output/'STOP').exists(): raise InterruptedError('cooperative STOP')
        if time.time()>=self.plan['cutoff']: raise TimeoutError('fixed worker cutoff')
        if shutil.disk_usage(self.output).free < 22*1024**3: raise OSError('20GiB + checkpoint temporary reserve')
        size=sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
        if size>=self.plan['limits']['max_output_bytes']: raise OSError('owned output cap')
    def charge(self,kind):
        self.check()
        if sum(self.calls.values())>=self.plan['limits']['max_calls']: raise RuntimeError('actual call cap')
        self.calls[kind]+=1

def save_full(output,payload):
    """No rotation/deletion. New preserved payload + SHA/safeload/digest marker."""
    output=Path(output); directory=output/'checkpoints'; directory.mkdir(exist_ok=True)
    if shutil.disk_usage(output).free<22*1024**3: raise OSError('checkpoint temporary reserve')
    stem=f"state_{payload['attempts']:04d}";path=directory/(stem+'.pt')
    if path.exists(): raise FileExistsError('checkpoint overwrite forbidden')
    temp=directory/(stem+'.pt.tmp')
    digest=state_sha(payload)
    with temp.open('xb') as f: torch.save(payload,f);f.flush();os.fsync(f.fileno())
    io.replace_with_retry(temp,path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if state_sha(restored)!=digest: raise ValueError('complete fullstate roundtrip mismatch')
    meta={'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'algorithm_sha256':digest,
          'attempts':payload['attempts'],'updates':payload['updates'],'sampler_cursor':payload['cursor'],
          'full_state_verified':True}
    io.atomic_json(directory/(stem+'.json'),meta);io.atomic_json(output/'latest.json',meta)
    return meta
