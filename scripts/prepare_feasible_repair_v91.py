"""Prepare/review then seal ONE new fixed-clock V91 trial; no implicit launch."""
import argparse, ast, json, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_feasible_repair_20261113_v91 import validate, SCHEMA, SOURCE_SHA, ALGORITHM, STORAGE_SHA, ANCHOR_SHA, OUTPUT, START, CAPS, COST


def main(seal=False):
    a=ROOT/'research/artifacts/feasible_repair_20261113_v91_cpu_v1'; path=a/'plan_unsealed.json'
    if seal:
        p=io.load_json(path); out=Path(p['output']); validate(path,out)
        pre=io.load_json(a/'cpu_preflight_owned/final_status.json'); review=io.load_json(a/'root_independent_validate.json')
        if (pre['status']!='CPU_PREFLIGHT_PASS' or not pre['exit']['all_owned_processes_gone']
                or pre['report']['CUDA_initialized'] or not pre['report']['actual_only2_gradient']
                or review['status']!='ROOT_V91_ACTUAL_FEASIBLE_REPAIR_READY' or shutil.disk_usage(ROOT).free<START):
            raise ValueError('preflight/review/actual current start resources')
        for f in (a/'cpu_preflight_owned/final_status.json',a/'root_independent_validate.json',ROOT/'scripts/review_feasible_repair_v91_preseal_cpu.py'):
            p['files'][str(f)]=support.sha(f)
        now=time.time(); p.update(authorized=True,start=now,cutoff=now+480,deadline=now+600)
        out.mkdir(exist_ok=False); io.atomic_json(out/'plan.json',p); validate(out/'plan.json',out)
        snapshot=out/'code_snapshot'; snapshot.mkdir()
        for name in p['new_code_files']:
            (snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
        print(json.dumps(dict(planSHA=support.sha(out/'plan.json'),start=p['start'],cutoff=p['cutoff'],deadline=p['deadline'],max_calls=1400,expected_max=1348,owned_output_MiB=144)))
        return
    a.mkdir(exist_ok=False)
    old=io.load_json(ROOT/'runs/critical_repair_20261113_v88/plan.json')
    source=ROOT/'runs/critical_repair_20261113_v88/model/runtime.pt'
    source_env=torch.load(source,map_location='cpu',weights_only=True)
    assert support.sha(source)==SOURCE_SHA and source_env['reconstructed_fullstate_SHA']==ALGORITHM
    anchor=ROOT/'research/artifacts/feasible_readout_delta_20261113_v89_cpu_v1/original_V80_FP32_reference_anchor.json'
    operators=anchor.parent/'operator_files.json'
    baseline=ROOT/'runs/critical_repair_20261113_v88/assessment/final/reconstructions.json'
    records=io.load_json(baseline); assert len(records)==194
    # Reuse the already reviewed owner cleanup suffix BYTE-FOR-BYTE. Only its
    # pre-Popen purpose/schema/worker/clock/start-output constants change.
    previous=(ROOT/'scripts/supervise_critical_repair_v88.py').read_text(encoding='utf-8')
    prefix,suffix=previous.split("    state=dict(status='PREPARING'",1)
    prefix=prefix.replace('cause-directed FP32/critical coverage V88','feasible actual-Adam original-anchor V91')
    prefix=prefix.replace('run_critical_repair_20261113_v88.py','run_feasible_repair_20261113_v91.py')
    prefix=prefix.replace('fdt_cause_directed_critical_repair_v88',SCHEMA)
    prefix=prefix.replace("p['wall_seconds']!=900","p['wall_seconds']!=600").replace("p['max_calls']!=5100","p['max_calls']!=1400")
    prefix=prefix.replace("p['attempts']!=192 or p['teacher_epochs']!=1","p['attempts']!=96 or p['fresh_teacher_calls']!=0")
    prefix=prefix.replace('192*1024**2','144*1024**2').replace("p['deadline']-p['start']!=900","p['deadline']-p['start']!=600")
    prefix=prefix.replace('108.1875GiB start:100 protected+192MiB owned incl STOPtemp+8 headroom','108.140625GiB start:100 protected+144MiB owned incl STOPtemp+8 headroom')
    owner=ROOT/'scripts/supervise_feasible_repair_v91.py'
    with owner.open('x',encoding='utf-8',newline='\n') as f: f.write(prefix+"    state=dict(status='PREPARING'"+suffix)
    names=['scripts/run_feasible_repair_20261113_v91.py','scripts/prepare_feasible_repair_v91.py','scripts/supervise_feasible_repair_v91.py',
        'scripts/review_feasible_repair_v91_preseal_cpu.py','scripts/feasible_critical_engine_v90.py','research/feasible_adam_transaction_v90.py',
        'research/diagnostic_reference_v90.py','research/feasible_readout_delta_v89.py','research/immutable_reference_anchor_v89.py']
    extra=[ROOT/name for name in names]
    extra+=[source,anchor,operators,baseline,ROOT/'runs/critical_repair_20261113_v88/result.json',
        ROOT/'runs/critical_repair_20261113_v88/root_result.json',ROOT/'runs/critical_repair_20261113_v88/root_evidence_SHA.json',
        ROOT/'research/artifacts/feasible_critical_engine_20261113_v90_cpu_v1/root_engine_cpu_review.json',
        ROOT/'research/artifacts/feasible_critical_engine_20261113_v90_cpu_v1/root_independent_review.json',
        ROOT/'research/artifacts/closed_raw_compression_20261113_v2/root_result.json']
    extra+=[Path(r['trace']) for r in records]+[Path(r['targets']) for r in source_env['runtime']['cache_refs']]
    extra+=[Path(source_env['runtime']['bank_ref'][k]) for k in ('features','targets')]
    files=dict(old['files'])
    for f in extra:
        files[str(f.resolve())]=support.sha(f)
        if f.suffix=='.py': ast.parse(f.read_text(encoding='utf-8-sig'))
    p=dict(schema=SCHEMA,authorized=False,root=str(ROOT),output=str(ROOT/'runs/feasible_repair_20261113_v91'),data=old['data'],
        source=str(source),source_SHA=SOURCE_SHA,source_algorithm_SHA=ALGORITHM,storage_parent=old['storage_parent'],storage_parent_SHA=STORAGE_SHA,
        source_result=str(ROOT/'runs/critical_repair_20261113_v88/result.json'),baseline_records=str(baseline),
        anchor=str(anchor),anchor_SHA=ANCHOR_SHA,representation_operators=str(operators),anchor_selection=old['anchor_selection'],symbol_profile=old['symbol_profile'],
        eval_inputs=old['eval_inputs'],attempts=96,group_size=5,reused_static_teacher_rows=1024,fresh_teacher_calls=0,
        max_calls=1400,expected_max_calls=1348,max_output_bytes=OUTPUT,start_free_bytes=START,wall_seconds=600,
        precision_policy=old['precision_policy'],no_accepted_window=32,reference_calls_per_group_cap=2,
        reference_patch_work_cap=8192*194,component_caps=CAPS,cost=COST,
        correction=dict(passes=2,max_active_constraints=16,relative_radius=1.,padding=1e-6),
        trainable_names=['readout.readout.net.2.weight','readout.readout.net.2.bias'],trainable_parameters=2117664,storage_override_tensors=7,
        fresh_optimizer='AdamW1e-5 WD.01 enabledscaler1024 clip1; accepted correctedweights keep actualproposal moments; V88 oldAdamRNG preserved external',
        purpose='Correct actual Adam direction against ORIGINAL V80 FP32 constraints from warm V88; no scalar-only repeat/no floor-ratchet',
        limits='Previously observed192short+2cachedlong; static original teacher/baseline reuse explicitly not new NN exposure; fullbank finiteTRAIN only; no semantic-complete/generalization promotion.',
        Exact_entries=0,copy_bytes=0,denoise_calls=0,automatic_promotion=False,E2final_access=False,
        new_code_files=names,files=files)
    io.atomic_json(path,p); validate(path,Path(p['output']))
    io.atomic_json(a/'root_scope_cost_CPU_review.json',dict(schema_PASS=True,pins=len(files),max_calls=1400,expected_max_calls=1348,
        code_cost=COST,full_output_with_STOPtemp=OUTPUT,component_caps=CAPS,start_required=START,
        inherited_postPopen_cleanup_unchanged=True,new_actual_GPU_calls=0,actual_fullsize_preflight_still_required=True))
    print(json.dumps(dict(plan=str(path),pins=len(files),expected_max_calls=1348,max_calls=1400,output_MiB=144,new_GPU_calls=0)))


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--seal',action='store_true'); main(a.parse_args().seal)
