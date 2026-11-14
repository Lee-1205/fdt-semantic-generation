"""New readonly scope, race-fix regression and owned preflight/root review."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_missing_assessment_v92 import validate,SCHEMA,SOURCE_SHA,ALGORITHM,OUTPUT,START


def main(mode):
    a=ROOT/'research/artifacts/missing_assessment_20261113_v92_cpu_v1';path=a/'plan_unsealed.json'
    if mode=='prepare':
        a.mkdir(exist_ok=False);old=io.load_json(ROOT/'runs/feasible_repair_20261113_v91/plan.json')
        run=ROOT/'runs/feasible_repair_20261113_v91';incident=io.load_json(run/'incident.json')
        previous=(ROOT/'scripts/supervise_feasible_repair_v91.py').read_text(encoding='utf-8');prefix,suffix=previous.split("    state=dict(status='PREPARING'",1)
        prefix=prefix.replace('run_feasible_repair_20261113_v91.py','run_missing_assessment_v92.py')
        start=prefix.index("    if (p['schema']");end=prefix.index('    for name,digest',start)
        prefix=prefix[:start]+"    if (p['schema']!='"+SCHEMA+"' or p['wall_seconds']!=180 or p['max_calls']!=73 or p['expected_rows']!=73 or p['max_output_bytes']!=32*1024**2):\n        raise ValueError('fixed NEW missing73 readonly contract')\n"+prefix[end:]
        prefix=prefix.replace("p['deadline']-p['start']!=600","p['deadline']-p['start']!=180").replace("p['cutoff']!=p['deadline']-120","p['cutoff']!=p['deadline']-45")
        prefix=prefix.replace('108*1024**3+144*1024**2','108*1024**3+32*1024**2').replace('108.140625GiB start:100 protected+144MiB owned incl STOPtemp+8 headroom','108.03125GiB start:100 protected+32MiB owned+8 headroom')
        owner=ROOT/'scripts/supervise_missing_assessment_v92.py'
        with owner.open('x',encoding='utf-8',newline='\n') as f:f.write(prefix+"    state=dict(status='PREPARING'"+suffix)
        files=dict(old['files']);extra=[Path(__file__).resolve(),ROOT/'scripts/run_missing_assessment_v92.py',owner,Path(incident['emergency']['path']),run/'incident.json',run/'ROOT_terminal_native.json',run/'root_saved_review.json',run/'root_missing_inputs.json',ROOT/'scripts/owned_output_size_v52.py']
        for f in extra:files[str(f.resolve())]=support.sha(f)
        p=dict(schema=SCHEMA,authorized=False,root=str(ROOT),output=str(ROOT/'runs/missing_assessment_20261113_v92'),source=incident['emergency']['path'],source_SHA=SOURCE_SHA,source_algorithm_SHA=ALGORITHM,
            eval_inputs=str(run/'root_missing_inputs.json'),original_eval_inputs=old['eval_inputs'],saved_review=str(run/'root_saved_review.json'),expected_rows=73,
            max_calls=73,max_output_bytes=OUTPUT,start_free_bytes=START,wall_seconds=180,optimizer_steps=0,encoder_calls=0,denoise_calls=0,files=files,
            purpose='ONLY73never-produced V91 final outputs from exact emergency6update state; static121 not repeated; originalFAILED preserved')
        io.atomic_json(path,p);validate(path,Path(p['output']));print(json.dumps(dict(unsealed_plan=str(path),pins=len(files),newGPU=0)));return
    p=io.load_json(path);out=Path(p['output']);validate(path,out)
    if mode=='review':
        from unittest.mock import patch
        from owned_output_size_v52 import owned_output_bytes
        from run_critical_repair_20261113_v88 import size as unsafe_size
        stable=a/'stable_bytes.bin'
        with stable.open('xb') as f:f.write(b'abc')
        vanished=a/'state.json.replaced.tmp';original=Path.rglob;old_is_file=Path.is_file
        def racing(self,pattern):yield vanished;yield from original(self,pattern)
        def was_file(self):return True if self==vanished else old_is_file(self)
        with patch.object(Path,'rglob',racing),patch.object(Path,'is_file',was_file):
            try:unsafe_size(a);raise AssertionError('original race not reproduced')
            except FileNotFoundError:pass
            assert owned_output_bytes(a)>=3
        pre=io.load_json(a/'cpu_preflight_owned/final_status.json');assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['worker_exit_code']==0 and not pre['report']['CUDA_initialized']
        assert pre['report']['actual_encoder_forwards']==pre['report']['actual_readout_forwards']==0 and pre['report']['optimizer_steps']==0
        assert pre['report']['source_updates']==6 and pre['report']['worker_uses_existing_v52_race_fix']
        old=(ROOT/'scripts/supervise_feasible_repair_v91.py').read_text(encoding='utf-8');new=(ROOT/'scripts/supervise_missing_assessment_v92.py').read_text(encoding='utf-8')
        assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
        assert shutil.disk_usage(ROOT).free>=START
        io.atomic_json(a/'root_independent_validate.json',dict(status='ROOT_V92_READONLY_MISSING73_READY',actual_fullsource_CPU=pre['report'],
            old_worker_size_race_reproduced=True,existing_v52_fix_PASS=True,originalV91sealed_code_unchanged=True,unchanged_postPopen_cleanup=True,
            max_calls73=73,optimizer0=True,missing_only73=True,output32MiB=True,Cfree=shutil.disk_usage(ROOT).free));print('ROOT_V92_READONLY_MISSING73_READY');return
    pre=io.load_json(a/'cpu_preflight_owned/final_status.json');review=io.load_json(a/'root_independent_validate.json')
    assert mode=='seal' and pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and review['status']=='ROOT_V92_READONLY_MISSING73_READY' and shutil.disk_usage(ROOT).free>=START
    for f in (a/'cpu_preflight_owned/final_status.json',a/'root_independent_validate.json'):p['files'][str(f)]=support.sha(f)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+135,deadline=now+180);out.mkdir(exist_ok=False);io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
    snap=out/'code_snapshot';snap.mkdir()
    for file in (Path(__file__).resolve(),ROOT/'scripts/run_missing_assessment_v92.py',ROOT/'scripts/supervise_missing_assessment_v92.py',ROOT/'scripts/owned_output_size_v52.py'):(snap/file.name).write_bytes(file.read_bytes())
    print(json.dumps(dict(planSHA=support.sha(out/'plan.json'),cutoff=p['cutoff'],deadline=p['deadline'],readout73=True,optimizer0=True)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','review','seal']);main(a.parse_args().mode)
