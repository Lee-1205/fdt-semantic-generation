"""Bounded lossless filesystem compression of CLOSED, generated old language-branch checkpoints.

No deletion/moving/serialization change, no source documents/core/typed weights,
no V9/original typed/source documents/raw corpus or latest AE27-AE30 runs, no model forward, optimizer, CUDA or cloud.
Each exact file's logical SHA/length must remain identical after compact exits.
"""
import ctypes, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/artifacts/owt_checkpoint_lossless_space_20261029_v4'
MAX_FILES=40
MAX_LOGICAL=28*1024**3
TARGET_FREE=int(40.5*1024**3)
WALL=480
PER_FILE=45

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name; temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,p)

def allocated(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    fn=kernel.GetCompressedFileSizeW
    fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)]
    fn.restype=ctypes.c_ulong
    high=ctypes.c_ulong(0);ctypes.set_last_error(0)
    low=fn(str(path),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
    return (high.value<<32)|low

def main():
    if os.name!='nt':raise RuntimeError('Windows NTFS only')
    if OUT.exists():raise FileExistsError('new bounded storage operation only')
    candidates=[];total=0
    run_names=['text_generation_20261028_v1','text_context_decoder_20261028_v2','text_semantic_alignment_20261028_v3','text_span_alignment_20261028_v4','text_language_continuation_20261028_v5','text_language_continuation_20261028_v5_deterministic_v2','text_language_continuation_20261028_v6','text_language_continuation_20261028_v7','text_decoder_alignment_20261028_v8']
    allowed=set(run_names)
    for run_name in run_names:
        run=ROOT/'runs'/run_name
        terminal=json.loads((run/'supervisor/final_status.json').read_text(encoding='utf-8-sig'))
        if not terminal.get('exit',{}).get('all_owned_processes_gone') or terminal['status'] not in ('COMPLETE','FAILED','STOPPED'):raise RuntimeError('closed generated run only')
        for stage in ('checkpoints','model/checkpoints','import/checkpoints','initial/checkpoints','reader/checkpoints','realizer/checkpoints'):
            for path in sorted((run/stage).glob('*.pt')):
                resolved=path.resolve(strict=True)
                if path.is_symlink() or resolved!=path or resolved.relative_to(ROOT/'runs').parts[0] not in allowed:raise ValueError('exact older generated path only')
                if not path.stem.startswith(('state_','payload_')):raise ValueError('fullstate names only, no raw tensor cache')
                if len(candidates)>=MAX_FILES or total+path.stat().st_size>MAX_LOGICAL:break
                candidates.append(path);total+=path.stat().st_size
    OUT.mkdir();start=time.monotonic();before=shutil.disk_usage(ROOT).free
    plan={'root':str(ROOT),'method':'compact.exe /c /exe:lzx /q EXACT_FILE',
        'official_reference':'https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact',
        'max_files':MAX_FILES,'max_logical_bytes':MAX_LOGICAL,'wall_seconds':WALL,'per_file_seconds':PER_FILE,
        'target_C_free_bytes':TARGET_FREE,'paths':[str(p) for p in candidates],'planned_logical_bytes':total,
        'script_sha256':sha(Path(__file__)),'no_deletion_or_move':True,'logical_SHA_must_stay_identical':True,
        'all_candidates_generated_closed_language_branches_only':True,'protected_source_and_latest_runs':['typed','V9','AE27','AE28','AE29','AE30'],
        'C_free_before':before,'model_forwards':0,'optimizer_updates':0,'CUDA_initialized':False}
    save('plan.json',plan);records=[];status='BOUNDED_COMPLETE'
    for path in candidates:
        if shutil.disk_usage(ROOT).free>=TARGET_FREE:status='SPACE_TARGET_REACHED';break
        if time.monotonic()-start+PER_FILE+10>=WALL:status='FIXED_TIME_CAP';break
        record={'path':str(path),'logical_bytes_before':path.stat().st_size,'sha256_before':sha(path),
            'allocated_bytes_before':allocated(path),'C_free_before':shutil.disk_usage(ROOT).free}
        records.append(record);save('progress.json',{'status':'PREPARED_FILE','records':records})
        proc=None;error=None;began=time.monotonic()
        command=['compact.exe','/c','/exe:lzx','/q',str(path)]
        try:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,shell=False)
            record.update(owned_compact_pid=proc.pid,command=command)
            save('progress.json',{'status':'COMPACT_RUNNING','records':records})
            stdout,stderr=proc.communicate(timeout=PER_FILE)
            record.update(exit_code=proc.returncode,stdout_hex=stdout.hex(),stderr_hex=stderr.hex())
            if proc.returncode:raise RuntimeError('compact returned nonzero')
        except BaseException as exc:
            error=repr(exc)
        finally:
            if proc is not None:
                if proc.poll() is None:proc.kill()
                proc.wait(timeout=10)
                record.update(owned_compact_exit_code=proc.returncode,owned_compact_gone=True)
        record.update(sha256_after=sha(path),logical_bytes_after=path.stat().st_size,
            allocated_bytes_after=allocated(path),C_free_after=shutil.disk_usage(ROOT).free,seconds=time.monotonic()-began,error=error)
        identical=record['sha256_after']==record['sha256_before'] and record['logical_bytes_after']==record['logical_bytes_before']
        record['logical_content_identical']=identical
        save('progress.json',{'status':'FILE_VERIFIED','records':records})
        print(json.dumps({'file':str(path.relative_to(ROOT)),'identical':identical,
            'saved_allocated_bytes':record['allocated_bytes_before']-record['allocated_bytes_after'],
            'C_free_bytes':record['C_free_after'],'seconds':record['seconds']}),flush=True)
        if not identical or error:
            status='STOP_CONTENT_MISMATCH' if not identical else 'STOP_COMPACT_ERROR';break
    final={'status':status,'records':records,'files_processed':len(records),'all_logical_content_identical':all(r.get('logical_content_identical',False) for r in records),
        'all_owned_compact_processes_gone':all(r.get('owned_compact_gone',False) for r in records),
        'allocated_bytes_saved':sum(r.get('allocated_bytes_before',0)-r.get('allocated_bytes_after',r.get('allocated_bytes_before',0)) for r in records),
        'C_free_before':before,'C_free_after':shutil.disk_usage(ROOT).free,'target_C_free_bytes':TARGET_FREE,
        'seconds':time.monotonic()-start,'model_forwards':0,'optimizer_updates':0,'file_deletions':0,'file_moves':0}
    if final['C_free_after']>=TARGET_FREE and not status.startswith('STOP'):final['status']='SPACE_TARGET_REACHED'
    save('result.json',final)
    print(json.dumps({k:v for k,v in final.items() if k!='records'}),flush=True)
    return 0 if final['all_logical_content_identical'] and final['all_owned_compact_processes_gone'] and not final['status'].startswith('STOP') else 2

if __name__=='__main__':raise SystemExit(main())
