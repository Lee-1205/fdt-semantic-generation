"""New owner's real Windows natural exit, CPU fixture only; no model/GPU."""
import json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import supervise_three_memory_codec_20261029_v35 as owner

def main():
    a=ROOT/'research/artifacts/three_memory_codec_20261029_v35_worker_cpu';out=a/'real_owner_CPU_fixture';out.mkdir(exist_ok=False)
    fixture=a/'run_three_memory_owner_v35_CPU_fixture.py'
    fixture.write_text("""import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()
if a.preflight:
 print('CPU_PREFLIGHT_PASS');raise SystemExit(0)
o=Path(a.output)
(o/'progress.json').write_text(json.dumps({'phase':'CPU_FIXTURE_ONLY','attempts':0,'allocation_calls':{'encode':0,'denoise':0,'decoder':0},'native_pid':os.getpid(),'parent_pid':os.getppid()}),encoding='utf-8')
time.sleep(2)
(o/'result.json').write_text(json.dumps({'status':'THREE_MEMORY_CODEC_V35_COMPLETE','CPU_fixture_only':True,'model_calls':0}),encoding='utf-8')
""",encoding='utf-8')
    p=json.loads((a/'plan_unsealed.json').read_text(encoding='utf-8'))
    p.update(output=str(out),authorized=True,allocation_start=time.time(),CPU_fixture_only=True)
    p.update(deadline=p['allocation_start']+480,cutoff=p['allocation_start']+390)
    p['files'][str(fixture)]=owner.native.sha256(fixture)
    path=out/'plan.json';path.write_text(json.dumps(p,indent=2),encoding='utf-8')
    original=owner.resources
    owner.resources=lambda _: {'CPU_fixture_only':True,'C_free_bytes':shutil.disk_usage(ROOT).free,'GPU_access':False}
    try:code=owner.supervise(path,out,fixture)
    finally:owner.resources=original
    end=json.loads((out/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert code==0 and end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and end['actual']['identity_verified']
    assert 'torch' not in sys.modules
    result={'status':'REAL_WINDOWS_OWNER_CPU_FIXTURE_PASS','owner_sha256':owner.native.sha256(Path(owner.__file__)),
        'exit':end['exit'],'actual_native':end['actual'],'actual_model_worker':False,'neural_calls':0,'GPU_training':False,'owner_imported_Torch':False}
    with (a/'real_windows_owner_review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
