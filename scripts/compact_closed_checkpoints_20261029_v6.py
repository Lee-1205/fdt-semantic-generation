"""New bounded transparent NTFS compression; logical checkpoints never change.

Explicit generated CLOSED run whitelist only. Current V37 final source, typed
weights, code, documents, datasets, raw outputs and user originals are excluded.
Plan first; ONE execution; original marker/SHA/length verified before and after.
"""
import argparse,ctypes,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import supervise_three_memory_codec_20261029_v37 as os_owner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/artifacts/owt_checkpoint_lossless_space_20261029_v6'
LIMITS={'max_files':44,'max_logical_bytes':34*1024**3,'wall_seconds':480,'per_file_seconds':45,
    'target_C_free_bytes':int(35.25*1024**3),'required_training_start_bytes':35*1024**3}
RUNS=['text_language_continuation_20261028_v7','text_decoder_alignment_20261028_v8',
    'text_decoder_addressing_20261028_v9']+[f'owt_fuzzy_reconstruction_20261029_v{v}' for v in (27,28,29,30,31)]+[
    f'three_memory_codec_20261029_v{v}' for v in (33,34,35,36,37)]
STAGES=('checkpoints','model/checkpoints','import/checkpoints','initial/checkpoints','reader/checkpoints','realizer/checkpoints','emergency/checkpoints')
PROTECTED=ROOT/'runs/three_memory_codec_20261029_v37/model/checkpoints/state_4096.pt'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(name,value):
    p=OUT/name;temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def allocated(p):
    fn=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
    fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)];fn.restype=ctypes.c_ulong
    high=ctypes.c_ulong();ctypes.set_last_error(0);low=fn(str(p),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
    return high.value*2**32+low
def checked_path(value):
    p=Path(value)
    if not p.is_absolute() or p.is_symlink() or p.resolve(strict=True)!=p:raise ValueError('resolved absolute file only')
    rel=p.relative_to(ROOT/'runs')
    if rel.parts[0] not in RUNS or p.suffix!='.pt' or not p.stem.startswith(('state_','payload_')) or p==PROTECTED:
        raise ValueError('outside explicit generated checkpoint whitelist')
    if '/'.join(rel.parts[1:-1]) not in STAGES:raise ValueError('checkpoint directory only')
    end=read(ROOT/'runs'/rel.parts[0]/'supervisor/final_status.json')
    if end['status'] not in ('COMPLETE','FAILED','STOPPED') or not end.get('exit',{}).get('all_owned_processes_gone'):
        raise ValueError('closed and owned-tree gone only')
    return p
def prepare():
    if OUT.exists():raise FileExistsError('NEW storage purpose only')
    if os_owner.duplicates(None):raise RuntimeError('model worker active')
    rows=[];total=0;skipped=[]
    for run_name in RUNS:
        run=ROOT/'runs'/run_name
        # Every selected directory has its own terminal proof; no broad run scan.
        for stage in STAGES:
            for raw in sorted((run/stage).glob('*.pt')):
                if raw==PROTECTED:continue
                path=checked_path(str(raw));size=path.stat().st_size
                if allocated(path)<size:skipped.append(str(path));continue
                marker=read(path.with_suffix('.json'))
                if marker.get('bytes')!=size or not marker.get('sha256'):raise ValueError('original complete marker required')
                if len(rows)>=LIMITS['max_files'] or total+size>LIMITS['max_logical_bytes']:continue
                rows.append({'path':str(path),'logical_bytes':size,'original_sha256':marker['sha256']});total+=size
    OUT.mkdir(exist_ok=False)
    p={'schema':'closed_checkpoint_NTFS_lossless_v6','limits':LIMITS,'paths':rows,'planned_logical_bytes':total,
        'script_sha256':sha(Path(__file__)),'protected_current_source':str(PROTECTED),
        'official_reference':'https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact',
        'method':'compact.exe /c /exe:lzx /q EXACT_FILE','already_compressed_skipped':skipped,
        'file_deletions':0,'file_moves':0,'tensor_serialization_changes':0,'model_forwards':0,'CUDA_initialized':False,
        'C_free_before':shutil.disk_usage(ROOT).free}
    save('plan.json',p)
    print(json.dumps({'status':'NEW_STORAGE_PLAN_ONLY','files':len(rows),'logical_bytes':total,'limits':LIMITS,
        'run_counts':{name:sum(Path(x['path']).relative_to(ROOT/'runs').parts[0]==name for x in rows) for name in RUNS}}))
def execute():
    p=read(OUT/'plan.json')
    if p['limits']!=LIMITS or p['script_sha256']!=sha(Path(__file__)):raise ValueError('fixed storage code/caps')
    if (OUT/'execution_started.json').exists():raise FileExistsError('closed allocation never replayed')
    if os_owner.duplicates(None):raise RuntimeError('model worker active')
    for row in p['paths']:checked_path(row['path'])
    start=time.monotonic();before=shutil.disk_usage(ROOT).free
    save('execution_started.json',{'started_unix':time.time(),'fixed_wall_seconds':480,'plan_sha256':sha(OUT/'plan.json')})
    records=[];status='BOUNDED_COMPLETE'
    for row in p['paths']:
        if (OUT/'STOP').exists():status='COOPERATIVE_STOP';break
        if shutil.disk_usage(ROOT).free>=LIMITS['target_C_free_bytes']:status='SPACE_TARGET_REACHED';break
        if time.monotonic()-start+60>=480:status='FIXED_TIME_CAP';break
        if os_owner.duplicates(None):status='STOP_NEW_MODEL_WORKER';break
        path=checked_path(row['path']);rec={'path':str(path),'logical_bytes_before':path.stat().st_size,
            'sha256_before':sha(path),'allocated_bytes_before':allocated(path),'C_free_before':shutil.disk_usage(ROOT).free}
        if rec['logical_bytes_before']!=row['logical_bytes'] or rec['sha256_before']!=row['original_sha256']:
            raise ValueError('precompression original content mismatch')
        records.append(rec);save('progress.json',{'status':'PREPARED_FILE','records':records})
        process=None;error=None;began=time.monotonic()
        command=['compact.exe','/c','/exe:lzx','/q',str(path)]
        try:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,shell=False)
            rec.update(owned_compact_pid=process.pid,command=command);save('progress.json',{'status':'COMPACT_RUNNING','records':records})
            stdout,stderr=process.communicate(timeout=45)
            rec.update(exit_code=process.returncode,stdout_hex=stdout.hex(),stderr_hex=stderr.hex())
            if process.returncode:raise RuntimeError('compact returned nonzero')
        except BaseException as exc:error=repr(exc)
        finally:
            if process is not None:
                if process.poll() is None:process.kill()
                process.wait(timeout=10);rec.update(owned_compact_exit_code=process.returncode,owned_compact_gone=True)
        rec.update(sha256_after=sha(path),logical_bytes_after=path.stat().st_size,allocated_bytes_after=allocated(path),
            C_free_after=shutil.disk_usage(ROOT).free,seconds=time.monotonic()-began,error=error)
        identical=rec['sha256_before']==rec['sha256_after'] and rec['logical_bytes_before']==rec['logical_bytes_after']
        rec['logical_content_identical']=identical;save('progress.json',{'status':'FILE_VERIFIED','records':records})
        print(json.dumps({'file':str(path.relative_to(ROOT)),'identical':identical,
            'saved_allocated_bytes':rec['allocated_bytes_before']-rec['allocated_bytes_after'],'C_free_bytes':rec['C_free_after']}),flush=True)
        if error or not identical:status='STOP_CONTENT_MISMATCH' if not identical else 'STOP_COMPACT_ERROR';break
    free=shutil.disk_usage(ROOT).free
    if free>=LIMITS['target_C_free_bytes'] and not status.startswith('STOP'):status='SPACE_TARGET_REACHED'
    result={'status':status,'records':records,'files_processed':len(records),'seconds':time.monotonic()-start,
        'all_logical_content_identical':all(x.get('logical_content_identical',False) for x in records),
        'all_owned_compact_processes_gone':all(x.get('owned_compact_gone',False) for x in records),
        'allocated_bytes_saved':sum(x['allocated_bytes_before']-x['allocated_bytes_after'] for x in records),
        'C_free_before':before,'C_free_after':free,'training_start_floor_met':free>=35*1024**3,
        'target_C_free_bytes':LIMITS['target_C_free_bytes'],'file_deletions':0,'file_moves':0,'model_forwards':0,
        'tensor_serialization_changes':0,'current_V37_final_source_untouched':True}
    save('result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    return 0 if result['all_logical_content_identical'] and result['all_owned_compact_processes_gone'] and not status.startswith('STOP') else 2
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:raise SystemExit(execute())
    prepare()
