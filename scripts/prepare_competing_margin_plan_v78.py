"""Canonical NEW scope/cost/output plan; no unchecked dict(old) inheritance."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_competing_margin_20261113_v78 import validate,OUTPUT_CAP,START_FREE,PARENT_SHA,ALGORITHM_SHA,STORAGE_PARENT_SHA
from research.fuzzy_factual_objective_v70 import training_weights
from research.competing_margin_objective_v78 import OBJECTIVE,frozen_objective_report
from plan_scope_contract_v78_cpu import validate_scope


def main(seal):
    data=ROOT/'research/artifacts/competing_margin_20261113_v78_cpu_v1';review=data/'execution_review_v2';path=review/'plan_unsealed.json'
    if seal:
        p=io.load_json(path);out=Path(p['output']);assert not out.exists();validate(path,out)
        cp=review/'cpu_preflight_owned/final_status.json';rootproof=review/'root_independent_validate.json'
        c=io.load_json(cp);q=io.load_json(rootproof)
        assert c['status']=='CPU_PREFLIGHT_PASS' and c['exit']['all_owned_processes_gone'] and not c['owner_imported_Torch']
        assert c['report']['actual_only7_gradient'] and c['report']['optimizer_steps']==0 and not c['report']['CUDA_initialized']
        assert q['status']=='ROOT_INDEPENDENT_V78_READY'
        for f in (cp,rootproof):p['files'][str(f)]=support.sha(f)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900);out.mkdir();io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
        snap=out/'code_snapshot';snap.mkdir()
        for name in p['new_code_files']:(snap/Path(name).name).write_bytes((ROOT/name).read_bytes())
        print(json.dumps(dict(plan_SHA=support.sha(out/'plan.json'),cost=p['cost'],output_cap=OUTPUT_CAP,source_rows=1920,group_attempts=384)));return
    assert not review.exists()
    old=io.load_json(ROOT/'runs/update_balance_20261113_v77/plan.json')
    # Copy only unchanged input/evaluation paths and read-only pins, never stale
    # scope, estimate, epoch, optimizer-count, or authorization descriptions.
    pathkeys=('data','eval_data','cycle_source','storage_parent','baseline_run','retention_OWT','parent_data','old_eval_data','old_expression_data','observed_V74_data','observed_V75_data','CPU_reference_cases')
    p={k:old[k] for k in pathkeys}
    rows=io.load_json(Path(p['data'])/'train_features.json');_,objective=training_weights(rows)
    profile=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1';objective=frozen_objective_report(objective,io.load_json(profile/'symbol_weight_source.json'))
    envelope=torch.load(p['cycle_source'],map_location='cpu',weights_only=True);active=envelope['active_weights'];assert len(active)==7 and sum(v.numel() for v in active.values())==2908192
    names=['research/competing_margin_objective_v78.py','scripts/competing_margin_engine_v78.py','tests/test_competing_margin_v78.py',
        'scripts/test_competing_margin_engine_v78_cpu.py','scripts/prepare_competing_margin_v78_cpu.py','scripts/run_competing_margin_20261113_v78.py',
        'scripts/supervise_competing_margin_v78.py','scripts/prepare_competing_margin_plan_v78.py','scripts/review_competing_margin_v78_preseal_cpu.py']
    cost_detail=dict(encode_total=3076,readout_total=3418,stream_chunk_total=258,block_work=2322,denoise=0,
        teacher1024=2048,head1920_micros_same384_updates=3840,new64_before_after=256,observed340=340,long_fresh_and_comparison=6,native_stream_and_chunks=262)
    components=dict(raw_and_parity=921785600,loss_only_teacher_with_margin=32*1024**2,
        initial_teacher_states2=2*(2908192*4+8*1024**2),final_STOP_and_temp3=3*(2908192*12+8*1024**2),metadata_slack=32*1024**2)
    p.update(schema='fdt_competing_margin_v78',authorized=False,root=str(ROOT),output=str(ROOT/'runs/competing_margin_20261113_v78'),
        balance_data=str(data),symbol_profile_data=str(profile),phase='language_verbalizer_repair_ONLY_not_diffusion',
        cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,storage_parent_sha=STORAGE_PARENT_SHA,
        teacher_epochs=1,head_schedule='finite1920rows384groups; not one unique epoch',group_size=5,attempts=384,capacity=16384,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=7000,cost=6752,cost_detail=cost_detail,
        objective=OBJECTIVE,objective_report=objective,optimizer='fresh final7AdamW1e-5 WD.01 BF16 scaler1024; five .2 losses accumulated at same parameters, unscale/clip1/Adam once',
        model_scope=dict(tensors=7,parameters=2908192,names=sorted(active)),resource_estimate=dict(components=components,total_bytes=sum(components.values())),
        max_output_bytes=OUTPUT_CAP,start_free_bytes=START_FREE,protected_free_bytes=100*1024**3,owner_stop_free_bytes=104*1024**3,
        GPU_peak_cap_bytes=12*1024**3,source_encoder_optimizer_updates=0,stream_chunk_cap=258,long_block_work_cap=2322,
        purpose='Correct weak TRAIN competitor margins and signed/unsigned/OWT gradient interference in SAME update; preserve fuzzy encoder and Commit; no global plus-bias/filter/exact bypass',
        exposure=io.load_json(data/'manifest.json'),new_code_files=names,maximum_additional_STOP_checkpoint=1,
        checkpoints=['initial linked','teacher_complete linked','model linked','one emergency linked if stopped'],
        phase_reserves=dict(teacher_before_cutoff=180,head_before_cutoff=120,outer_shutdown=120),
        exact_policy='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST',Exact_content_entries=0,source_copy_bytes=0,denoise_calls=0,certified_prior_writer=False,automatic_promotion=False,
        initial_restore_contract='V75 complete warm NN; standalone V74 model-only storage reference plus active7; preserve all old Adam/scaler/RNG',
        recovery='allNN/7Adam/scaler/ALLRNG/teacher margins/orders/group-micro consumed exposure/history/inflightSTOP/call/evalrefs; no automatic replay',
        limitations='8lossCPU+tinyenginecontracts; same1024TRAIN and1920source exposures/384freshAdamupdates. Existing grammar-revised64 and340observed+2authored262KiB sources are NOT new blind data; margin and grouping jointly changed.',
        files=dict(old['files']))
    files=[ROOT/name for name in names]
    files.extend(data/name for name in ('orders.json','manifest.json','builder_SHA.json','root_engine_CPU_contract_v2.json','initial_CPU_incident.json'))
    files.extend([ROOT/'scripts/plan_scope_contract_v78_cpu.py',ROOT/'configs/research_base_selection_v77_review_20261113.json',ROOT/'configs/reasoning_only_training_policy_20261113_v1.json'])
    for f in files:p['files'][str(f)]=support.sha(f)
    for file,sha in p['files'].items():
        assert support.sha(file)==sha,file
        if file.endswith('.py'):ast.parse(Path(file).read_text(encoding='utf-8-sig'))
    assert sum(cost_detail[k] for k in ('teacher1024','head1920_micros_same384_updates','new64_before_after','observed340','long_fresh_and_comparison','native_stream_and_chunks'))==6752
    validate_scope(p,active);review.mkdir();io.atomic_json(path,p);validate(path,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),model_scope=p['model_scope'],expected6752calls=True,attempts384=True,resource_estimate=p['resource_estimate'],hardcap=OUTPUT_CAP)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
