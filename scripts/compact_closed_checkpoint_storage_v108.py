"""Lossless NTFS storage change ONLY, no logical checkpoint/weight/RNG change."""
import gc,json,shutil,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from compact_closed_features_v108 import physical,sha
import learning_io_v2 as io
import torch

def main():
    a=P/'research/artifacts/closed_checkpoint_storage_20261114_v108';a.mkdir(exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic();candidates=[];rows=[]
    for run in (P/'runs').iterdir():
        if not run.is_dir() or '202611' not in run.name or any(t in run.name for t in ('v74','v88','v102','v105')):continue
        closure=run/'final_status.json'
        if not closure.exists():continue
        state=io.load_json(closure)
        if state.get('status') not in ('COMPLETE','FAILED','STOPPED') or not state.get('exit',{}).get('all_owned_processes_gone'):continue
        for f in run.rglob('state_*.pt'):
            n=f.stat().st_size
            if n<256*1024**2:continue
            alloc=physical(f)
            if alloc<n*.98:continue
            candidates.append(dict(path=str(f.resolve()),bytes=n,physical_before=alloc,closure=str(closure),closure_SHA=sha(closure)))
    candidates.sort(key=lambda x:-x['bytes']);chosen=[];size=0
    for c in candidates:
        if len(chosen)>=24 or size+c['bytes']>16*1024**3:break
        chosen.append(c);size+=c['bytes']
    io.atomic_json(a/'scope.json',dict(files=chosen,max_files=24,max_logical_bytes=16*1024**3,max_seconds=180,
        target_Cfree=int(108.5*1024**3),logical_checkpoint_rewrite0=True,deletions0=True,excludes_current_codec_and_thought=True))
    for c in chosen:
        if shutil.disk_usage(P).free>=int(108.5*1024**3) or time.monotonic()-start>=165:break
        assert shutil.disk_usage(P).free>=100*1024**3
        f=Path(c['path']);assert f.is_relative_to((P/'runs').resolve()) and f.name.startswith('state_')
        before=sha(f);obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        result=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(f)],capture_output=True,
            timeout=min(25,175-(time.monotonic()-start)),creationflags=0x08000000)
        with (a/'compact.stdout.log').open('ab') as log:log.write(result.stdout+result.stderr)
        assert result.returncode==0 and sha(f)==before and f.stat().st_size==c['bytes']
        obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        rows.append(dict(c,SHA=before,physical_after=physical(f),CPU_safeload_before_after=True,logical_file_unchanged=True))
        io.atomic_json(a/'progress.json',dict(files=len(rows),physical_saved=sum(x['physical_before']-x['physical_after'] for x in rows),Cfree=shutil.disk_usage(P).free,seconds=time.monotonic()-start))
    result=dict(status='CLOSED_CHECKPOINT_STORAGE_COMPRESSION_COMPLETE',rows=rows,
        physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in rows),Cfree=shutil.disk_usage(P).free,
        seconds=time.monotonic()-start,logical_weight_Adam_RNG_changes0=True,deletions0=True,NNcalls0=True,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(a/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
