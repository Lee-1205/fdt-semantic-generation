"""Bounded transparent compression of immutable JSON evidence/data; no rewrite."""
import ctypes,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];W=P.parent;sys.path.insert(0,str(P/'scripts'))
import learning_io_v2 as io

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def allocated(p):
    h=ctypes.c_ulong();fn=ctypes.windll.kernel32.GetCompressedFileSizeW
    fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)];fn.restype=ctypes.c_ulong
    l=fn(str(p),ctypes.byref(h));return (h.value<<32)|l

def main():
    a=Path('D:/Codex/FDT/research/closed_json_storage_20261114_v111');a.mkdir(exist_ok=False)
    began=time.monotonic();items=[];done=[]
    bases=[P/'runs',P/'research/artifacts']
    for base in bases:
        for f in base.rglob('*.json'):
            if f.stat().st_size<65536 or any(s in str(f).lower() for s in ('e2','holdout','sealed','registry','v108')):continue
            if base.name=='runs':
                run=P/'runs'/f.relative_to(base).parts[0];closure=run/'final_status.json'
                if not closure.exists():continue
                try:c=io.load_json(closure)
                except (ValueError,OSError):continue
                if not c.get('exit',{}).get('all_owned_processes_gone'):continue
            size=allocated(f)
            if size<f.stat().st_size*.98:continue
            items.append(dict(path=str(f.resolve()),bytes=f.stat().st_size,physical_before=size))
    for f in (W/'outputs').glob('FDT_*.json'):
        if f.stat().st_size<65536 or any(s in f.name.lower() for s in ('e2','holdout','sealed','registry')):continue
        size=allocated(f)
        if size>=f.stat().st_size*.98:items.append(dict(path=str(f.resolve()),bytes=f.stat().st_size,physical_before=size))
    items.sort(key=lambda x:-x['bytes']);items=items[:600]
    io.atomic_json(a/'scope.json',dict(files=items,max_files=600,max_logical_bytes=4*1024**3,max_seconds=180,target_Cfree=int(109.0*1024**3),deletion0=True))
    logical=0
    for c in items:
        if shutil.disk_usage(W).free>=int(109.0*1024**3) or time.monotonic()-began>165 or logical+c['bytes']>4*1024**3:break
        assert shutil.disk_usage(W).free>=100*1024**3
        f=Path(c['path']);before=sha(f)
        r=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(f)],capture_output=True,timeout=min(20,175-(time.monotonic()-began)),creationflags=0x08000000)
        with (a/'compact.stdout.log').open('ab') as log:log.write(r.stdout+r.stderr)
        assert r.returncode==0 and sha(f)==before and f.stat().st_size==c['bytes']
        done.append(dict(c,SHA=before,physical_after=allocated(f),logical_bytes_unchanged=True));logical+=c['bytes']
    result=dict(status='CLOSED_JSON_STORAGE_COMPRESSION_COMPLETE',files=done,physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in done),
        Cfree=shutil.disk_usage(W).free,seconds=time.monotonic()-began,NNcalls0=True,deletion0=True,logical_model_changes0=True)
    io.atomic_json(a/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':main()
