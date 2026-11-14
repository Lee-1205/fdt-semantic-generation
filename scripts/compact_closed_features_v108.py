"""Bounded reversible NTFS compression of three CLOSED TRAIN feature files.
No weight/content rewrite/delete. SHA and CPU safeload stay identical.
"""
import ctypes,gc,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
import learning_io_v2 as io
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def physical(p):
    high=ctypes.c_ulong();fn=ctypes.windll.kernel32.GetCompressedFileSizeW
    fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)];fn.restype=ctypes.c_ulong
    low=fn(str(p),ctypes.byref(high));return (high.value<<32)|low

def main():
    a=P/'research/artifacts/closed_feature_compression_20261114_v108';a.mkdir(exist_ok=False)
    start=time.monotonic();torch.set_num_threads(1);rows=[];candidates=[]
    for name in ('ordered_fuzzy_20261111_v55','commit_readout_20261111_v54','semantic_binding_20261111_v59'):
        run=P/'runs'/name;closure=run/'final_status.json';c=io.load_json(closure)
        assert c['status']=='COMPLETE' and c['exit']['all_owned_processes_gone']
        p=(run/'train_features.pt').resolve();assert p.is_relative_to((P/'runs').resolve())
        candidates.append(dict(path=str(p),bytes=p.stat().st_size,physical_before=physical(p),closure=str(closure),closure_SHA=sha(closure)))
    io.atomic_json(a/'scope.json',dict(candidates=candidates,max_files=3,max_seconds=180,target_free_bytes=int(108.5*1024**3),delete0=True,weight_changes0=True))
    for c in candidates:
        if shutil.disk_usage(P).free>=int(108.5*1024**3) or time.monotonic()-start>=170:break
        assert shutil.disk_usage(P).free>=100*1024**3
        p=Path(c['path']);before=sha(p);obj=torch.load(p,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        if c['physical_before']<c['bytes']*.98:continue
        r=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(p)],capture_output=True,timeout=min(80,175-(time.monotonic()-start)),creationflags=0x08000000)
        with (a/'compact.stdout.log').open('ab') as f:f.write(r.stdout+r.stderr)
        assert r.returncode==0 and sha(p)==before and p.stat().st_size==c['bytes']
        obj=torch.load(p,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        rows.append(dict(c,SHA=before,physical_after=physical(p),safeload_before_after=True,logical_content_unchanged=True))
        io.atomic_json(a/'progress.json',dict(rows=rows,Cfree=shutil.disk_usage(P).free,seconds=time.monotonic()-start))
    out=dict(status='CLOSED_FEATURES_LOSSLESS_COMPRESSION_COMPLETE',rows=rows,physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in rows),
        Cfree=shutil.disk_usage(P).free,seconds=time.monotonic()-start,deletions0=True,weight_changes0=True,NNcalls0=True,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(a/'result.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()
