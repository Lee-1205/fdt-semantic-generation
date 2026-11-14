"""Bounded lossless NTFS compression of explicitly closed raw evidence only.

No deletion/move/payload rewrite/model forward. Stop once global C>=110GiB.
Skip already physically compressed files. Logical SHA and safe-load verified.
"""
import ctypes,gc,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support


def physical(path):
    high=ctypes.c_ulong();fn=ctypes.windll.kernel32.GetCompressedFileSizeW
    fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)];fn.restype=ctypes.c_ulong
    low=fn(str(path),ctypes.byref(high))
    if low==0xffffffff and ctypes.GetLastError():raise ctypes.WinError()
    return (high.value<<32)|low


def main():
    a=ROOT/'research/artifacts/closed_raw_compression_20261114_v4';a.mkdir(exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic();before=shutil.disk_usage(ROOT).free;rows=[];candidates=[];skips=[]
    for run in (ROOT/'runs').iterdir():
        if not run.is_dir():continue
        closure=None
        for f in (run/'final_status.json',run/'supervisor/final_status.json'):
            if not f.exists():continue
            state=io.load_json(f)
            if state.get('status') in ('COMPLETE','STOPPED','FAILED') and state.get('exit',{}).get('all_owned_processes_gone'):
                closure=dict(path=str(f),SHA=support.sha(f),status=state['status']);break
        if closure:
            for f in run.glob('raw*.pt'):
                size=f.stat().st_size
                if size<5*1024**2:continue
                allocated=physical(f)
                if allocated<size*.98:skips.append(dict(path=str(f),reason='already physically compressed'));continue
                candidates.append(dict(path=str(f),bytes=size,physical_before=allocated,closure=closure))
    candidates.sort(key=lambda r:r['bytes'],reverse=True)
    io.atomic_json(a/'scope.json',dict(start_Cfree=before,candidates=candidates,skip_already_compressed=skips,
        fixed_seconds=240,max_files=300,max_logical_bytes=20*1024**3,target_global_Cfree=110*1024**3,delete0=True,weights0=True,modelcalls0=True))
    logical=0
    for c in candidates[:300]:
        if shutil.disk_usage(ROOT).free>=110*1024**3 or time.monotonic()-start>=240 or logical+c['bytes']>20*1024**3:break
        path=Path(c['path']).resolve()
        if not path.is_relative_to((ROOT/'runs').resolve()) or not path.name.startswith('raw') or path.suffix!='.pt':raise ValueError('explicit closed raw scope')
        sha=support.sha(path)
        try:obj=torch.load(path,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        except (ValueError,RuntimeError,TypeError) as e:
            skips.append(dict(path=str(path),reason='safe-load format unsupported; left intact',error=repr(e)));continue
        left=240-(time.monotonic()-start)
        if left<5:break
        result=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(path)],capture_output=True,timeout=min(25,left),creationflags=0x08000000)
        with (a/'compact.stdout.log').open('ab') as f:f.write(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('compact failed '+str(path))
        assert support.sha(path)==sha and path.stat().st_size==c['bytes']
        obj=torch.load(path,map_location='cpu',weights_only=True,mmap=True);del obj;gc.collect()
        after=physical(path);rows.append(dict(c,SHA=sha,logical_SHA_unchanged=True,physical_after=after,CPU_safeload_before_after=True));logical+=c['bytes']
        io.atomic_json(a/'progress.json',dict(files=len(rows),saved_bytes=sum(x['physical_before']-x['physical_after'] for x in rows),Cfree=shutil.disk_usage(ROOT).free,elapsed=time.monotonic()-start))
    final=dict(status='CLOSED_RAW_TRANSPARENT_COMPRESSION_FINISHED',files=rows,skipped=skips,
        physical_saved_bytes=sum(x['physical_before']-x['physical_after'] for x in rows),logical_bytes=logical,seconds=time.monotonic()-start,
        Cfree_before=before,Cfree_after=shutil.disk_usage(ROOT).free,global_delta_not_assumed_equal_reclaimed=True,
        all_logical_SHA_and_CPU_safeload_unchanged=True,deletions=0,weights_changed=0,model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(a/'root_result.json',final);print(json.dumps({k:v for k,v in final.items() if k not in ('files','skipped')}))


if __name__=='__main__':main()
