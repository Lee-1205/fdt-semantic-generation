"""Bounded reversible NTFS compression, closed noncurrent checkpoints only.
Logical SHA and safe-load remain unchanged. No deletion, model call or RNG use.
"""
import gc,json,shutil,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from compact_closed_features_v108 import physical,sha
import learning_io_v2 as io
import torch


def main():
    a=Path('D:/Codex/FDT/research/closed_runtime_storage_20261114_v111');a.mkdir(exist_ok=False)
    torch.set_num_threads(1);began=time.monotonic();candidates=[];done=[]
    for run in (P/'runs').iterdir():
        if not run.is_dir() or '202611' not in run.name or any(s in run.name for s in ('v74','v88','v102','v105','v111')):continue
        closure=run/'final_status.json'
        if not closure.exists():continue
        c=io.load_json(closure)
        if c.get('status') not in ('COMPLETE','FAILED','STOPPED') or not c.get('exit',{}).get('all_owned_processes_gone'):continue
        paths=[]
        for pattern in ('model/runtime.pt','runtime.pt','model/payload_*.pt','payload_*.pt','model/checkpoints/*.pt','checkpoints/*.pt','train_features.pt','raw/*.pt'):
            paths.extend(run.glob(pattern))
        for f in sorted(set(paths)):
            n=f.stat().st_size
            if n<32*1024**2:continue
            alloc=physical(f)
            if alloc<n*.98:continue
            candidates.append(dict(path=str(f.resolve()),bytes=n,physical_before=alloc,closure=str(closure),closure_SHA=sha(closure)))
    candidates.sort(key=lambda r:-r['bytes']);chosen=[];total=0
    for c in candidates:
        if len(chosen)>=80 or total+c['bytes']>32*1024**3:break
        chosen.append(c);total+=c['bytes']
    io.atomic_json(a/'scope.json',dict(files=chosen,max_files=80,max_logical_bytes=32*1024**3,max_seconds=240,target_Cfree=109*1024**3,
        current_V74_V88_V102_V105_excluded=True,logical_weights_Adam_RNG_unchanged=True,deletions=0,model_calls=0))
    for c in chosen:
        if shutil.disk_usage(P).free>=109*1024**3 or time.monotonic()-began>220:break
        f=Path(c['path']);assert f.is_relative_to((P/'runs').resolve()) and shutil.disk_usage(P).free>100*1024**3
        before=sha(f)
        obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        r=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(f)],capture_output=True,timeout=min(35,230-(time.monotonic()-began)),creationflags=0x08000000)
        with (a/'compact.stdout.log').open('ab') as log:log.write(r.stdout+r.stderr)
        assert r.returncode==0 and sha(f)==before and f.stat().st_size==c['bytes']
        obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        done.append(dict(c,SHA=before,physical_after=physical(f),CPU_safeload_before_after=True))
        io.atomic_json(a/'progress.json',dict(files=len(done),saved_bytes=sum(x['physical_before']-x['physical_after'] for x in done),Cfree=shutil.disk_usage(P).free))
    result=dict(status='CLOSED_STORAGE_COMPRESSION_COMPLETE',files=done,physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in done),
        Cfree=shutil.disk_usage(P).free,seconds=time.monotonic()-began,logical_file_changes=0,deletions=0,model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(a/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':main()
