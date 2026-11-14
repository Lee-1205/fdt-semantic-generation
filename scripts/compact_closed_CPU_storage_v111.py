"""Separate bounded NTFS storage pass on closed CPU-fixture fullstates only."""
import gc,json,shutil,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from compact_closed_features_v108 import physical,sha
import learning_io_v2 as io
import torch


def main():
    a=Path('D:/Codex/FDT/research/closed_CPU_storage_20261114_v111');a.mkdir(exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic();candidates=[];done=[]
    for artifact in (P/'research/artifacts').iterdir():
        if not artifact.is_dir() or any(s in artifact.name.lower() for s in ('e2','holdout','registry','sealed','v111')):continue
        closure=artifact/'cpu_preflight_owned/final_status.json'
        if not closure.exists():continue
        c=io.load_json(closure)
        if c.get('status')!='CPU_PREFLIGHT_PASS' or not c.get('exit',{}).get('all_owned_processes_gone'):continue
        for f in (artifact/'cpu_preflight_owned/runtime').glob('*.pt'):
            n=f.stat().st_size
            if n<32*1024**2:continue
            alloc=physical(f)
            if alloc<n*.98:continue
            candidates.append(dict(path=str(f.resolve()),bytes=n,physical_before=alloc,closure=str(closure),closure_SHA=sha(closure)))
    candidates.sort(key=lambda x:-x['bytes']);chosen=[];size=0
    for c in candidates:
        if len(chosen)>=100 or size+c['bytes']>48*1024**3:break
        chosen.append(c);size+=c['bytes']
    io.atomic_json(a/'scope.json',dict(files=chosen,max_files=100,max_logical_bytes=48*1024**3,max_seconds=240,target_Cfree=109*1024**3,CPUfixtures_only=True,deletions=0,model_calls=0))
    for c in chosen:
        if shutil.disk_usage(P).free>=109*1024**3 or time.monotonic()-start>220:break
        f=Path(c['path']);assert f.is_relative_to((P/'research/artifacts').resolve()) and shutil.disk_usage(P).free>100*1024**3
        before=sha(f);obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        r=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(f)],capture_output=True,timeout=min(35,230-(time.monotonic()-start)),creationflags=0x08000000)
        with (a/'compact.stdout.log').open('ab') as log:log.write(r.stdout+r.stderr)
        assert r.returncode==0 and sha(f)==before and f.stat().st_size==c['bytes']
        obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        done.append(dict(c,SHA=before,physical_after=physical(f),CPU_safeload_before_after=True))
        io.atomic_json(a/'progress.json',dict(files=len(done),physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in done),Cfree=shutil.disk_usage(P).free))
    report=dict(status='CLOSED_CPU_FIXTURE_COMPRESSION_COMPLETE',files=done,physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in done),
        seconds=time.monotonic()-start,Cfree=shutil.disk_usage(P).free,logical_weights_Adam_RNG_changes0=True,deletions=0,model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(a/'result.json',report);print(json.dumps({k:v for k,v in report.items() if k!='files'}))

if __name__=='__main__':main()
