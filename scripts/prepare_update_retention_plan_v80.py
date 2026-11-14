"""Canonical NEW scope/cost/output plan; no unchecked dict(old) inheritance."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_update_retention_20261113_v80 import validate,OUTPUT_CAP,START_FREE,PARENT_SHA,ALGORITHM_SHA,STORAGE_PARENT_SHA
from research.fuzzy_factual_objective_v70 import training_weights
from research.update_retention_objective_v80 import OBJECTIVE,frozen_objective_report
from plan_scope_contract_v80_cpu import validate_scope


def main(seal):
    data=ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1';review=data/'execution_review_v3';path=review/'plan_unsealed.json'
    if seal:
        p=io.load_json(path);out=Path(p['output']);assert not out.exists();validate(path,out)
        cp=review/'cpu_preflight_owned/final_status.json';rootproof=review/'root_independent_validate.json'
        c=io.load_json(cp);q=io.load_json(rootproof)
        assert c['status']=='CPU_PREFLIGHT_PASS' and c['exit']['all_owned_processes_gone'] and not c['owner_imported_Torch']
        assert c['report']['actual_only2_gradient'] and c['report']['optimizer_steps']==0 and not c['report']['CUDA_initialized']
        assert q['status']=='ROOT_INDEPENDENT_V80_READY'
        for f in (cp,rootproof):p['files'][str(f)]=support.sha(f)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900);out.mkdir();io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
        snap=out/'code_snapshot';snap.mkdir()
        for name in p['new_code_files']:(snap/Path(name).name).write_bytes((ROOT/name).read_bytes())
        print(json.dumps(dict(plan_SHA=support.sha(out/'plan.json'),cost=p['cost'],output_cap=OUTPUT_CAP,source_rows=1920,group_attempts=384)));return
    assert not review.exists()
    old=io.load_json(ROOT/'runs/last_linear_20261113_v79/plan.json')
    # Copy only unchanged input/evaluation paths and read-only pins, never stale
    # scope, estimate, epoch, optimizer-count, or authorization descriptions.
    pathkeys=('data','eval_data','cycle_source','storage_parent','baseline_run','retention_OWT','parent_data','old_eval_data','old_expression_data','observed_V74_data','observed_V75_data','CPU_reference_cases')
    p={k:old[k] for k in pathkeys};p['data']=str(ROOT/'research/artifacts/numeric_coverage_20261113_v80_cpu_v1')
    rows=io.load_json(Path(p['data'])/'train_features.json');_,objective=training_weights(rows)
    profile=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1';objective=frozen_objective_report(objective,io.load_json(profile/'symbol_weight_source.json'))
    envelope=torch.load(p['cycle_source'],map_location='cpu',weights_only=True);stored=envelope['active_weights'];active={k:v for k,v in stored.items() if k in ('readout.readout.net.2.weight','readout.readout.net.2.bias')};assert len(active)==2 and sum(v.numel() for v in active.values())==2117664
    scope=io.load_json(ROOT/'research/artifacts/last_linear_20261113_v79_cpu_v1/manifest.json');policy=io.load_json(data/'prospective_policy.json');assert sorted(active)==sorted(scope['active_optimizer_scope']['names'])
    names=['research/update_retention_objective_v80.py','research/actual_update_retention_v80.py','research/lossless_raw_tensor_v80.py',
        'scripts/update_retention_engine_v80.py','scripts/update_retention_assess_v80.py','scripts/plan_scope_contract_v80_cpu.py',
        'scripts/test_update_retention_engine_v80_cpu.py','scripts/run_update_retention_20261113_v80.py','scripts/supervise_update_retention_v80.py',
        'scripts/prepare_update_retention_plan_v80.py','scripts/review_update_retention_v80_preseal_cpu.py',
        'tests/test_actual_update_retention_v80.py','tests/test_lossless_raw_tensor_v80.py']
    cost_detail=dict(encode_total=3076,readout_total=3418,stream_chunk_total=258,block_work=2322,denoise=0,reference_linear_total=1538,
        teacher1024=2048,head1920_micros_384_attempts=3840,new64_before_after=256,observed340=340,long_fresh_and_comparison=6,native_stream_and_chunks=262,reference_initial_final_and_max4x384=1538)
    components=policy['estimate_components'];assert sum(components.values())==1023415424
    p.update(schema='fdt_guarded_update_v80',authorized=False,root=str(ROOT),output=str(ROOT/'runs/update_retention_20261113_v80'),
        balance_data=p['data'],scope_data=str(data),symbol_profile_data=str(profile),phase='language_verbalizer_repair_ONLY_not_diffusion',
        cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,storage_parent_sha=STORAGE_PARENT_SHA,
        teacher_epochs=1,head_schedule='finite1920rows384groups; not one unique epoch',group_size=5,attempts=384,capacity=16384,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=8500,cost=8290,cost_detail=cost_detail,
        objective=OBJECTIVE,objective_report=objective,optimizer='fresh final2AdamW1e-5 WD.01 BF16 scaler1024; five .2 losses accumulated at same parameters, unscale/clip1/Adam proposal once + bounded TRAIN-reference damping/rollback',
        model_scope=dict(tensors=2,parameters=2117664,names=sorted(active)),storage_delta_scope=scope['storage_delta_scope'],resource_estimate=dict(components=components,total_bytes=sum(components.values())),
        max_output_bytes=OUTPUT_CAP,start_free_bytes=START_FREE,protected_free_bytes=100*1024**3,owner_stop_free_bytes=104*1024**3,
        GPU_peak_cap_bytes=12*1024**3,source_encoder_optimizer_updates=0,stream_chunk_cap=258,long_block_work_cap=2322,
        purpose='Broaden independent A/B sign and numeric-width TRAIN coverage; guard actual lastLinear2 Adam proposals using100 fixed TRAIN reference sources; no raw/Exact bypass',
        exposure=io.load_json(Path(p['data'])/'manifest.json'),prospective_CPU_policy_snapshot_NOT_execution_status=policy,reference_patch_work_cap=1538*7487,new_code_files=names,maximum_additional_STOP_checkpoint=1,
        checkpoints=['initial linked','teacher_complete linked','model linked','one emergency linked if stopped'],
        phase_reserves=dict(teacher_before_cutoff=180,head_before_cutoff=120,outer_shutdown=120),
        exact_policy='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST',Exact_content_entries=0,source_copy_bytes=0,denoise_calls=0,certified_prior_writer=False,automatic_promotion=False,
        initial_restore_contract='V75 complete warm NN; standalone V74 model-only storage reference plus stored7 deltas (frozen5 prefix, optimizer2); preserve all old Adam/scaler/RNG',
        recovery='allNN/2Adam/scaler/ALLRNG/teacher margins/orders/group-micro consumed exposure/history/inflightSTOP/call/evalrefs; no automatic replay',
        limitations='TRAIN data, quartet grouping and actual-delta guard change jointly. 384attempt cap; accepted updates can be fewer, rejects consume source/scaler/cursor. 100TRAIN references do not certify unseen meaning. Same observed evaluation; no blind. Full codec remains fixed except2lastLinear; no diffusion.',
        files=dict(old['files']))
    files=[ROOT/name for name in names]
    files.extend(data/name for name in ('root_result.json','root_input_review.json','root_primitive_CPU_tests.json','root_lossless_raw_review_v3.json','prospective_policy.json','anchor_selection.json','root_engine_CPU_contract_v80_v4.json','initial_engine_CPU_incident.json'))
    files.extend([ROOT/'scripts/plan_scope_contract_v80_cpu.py',ROOT/'configs/research_base_selection_v79_review_20261113.json',ROOT/'configs/reasoning_only_training_policy_20261113_v1.json'])
    for name in ('train_features.json','train_targets.json','orders.json','manifest.json','root_independent_CPU_review.json'):files.append(Path(p['data'])/name)
    for f in files:p['files'][str(f)]=support.sha(f)
    for file,sha in p['files'].items():
        assert support.sha(file)==sha,file
        if file.endswith('.py'):ast.parse(Path(file).read_text(encoding='utf-8-sig'))
    assert sum(cost_detail[k] for k in ('teacher1024','head1920_micros_384_attempts','new64_before_after','observed340','long_fresh_and_comparison','native_stream_and_chunks','reference_initial_final_and_max4x384'))==8290
    validate_scope(p,active);review.mkdir();io.atomic_json(path,p);validate(path,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),model_scope=p['model_scope'],max8290calls=True,attempts384=True,resource_estimate=p['resource_estimate'],hardcap=OUTPUT_CAP)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
