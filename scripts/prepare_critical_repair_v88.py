"""New finite cause-directed actual plan, no launch and no reused deadline."""
import argparse,ast,gc,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import unpack
from run_critical_repair_20261113_v88 import validate,CAPS,OUTPUT,START,PARENT_SHA,ALGORITHM,STORAGE_SHA

def main(seal=False):
    a=ROOT/'research/artifacts/critical_repair_20261113_v88_cpu_v1';path=a/'plan_unsealed.json'
    if seal:
        p=io.load_json(path);out=Path(p['output']);validate(path,out)
        pre=io.load_json(a/'cpu_preflight_owned/final_status.json');review=io.load_json(a/'root_independent_validate.json')
        assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and not pre['report']['CUDA_initialized'] and pre['report']['actual_only2_gradient']
        assert review['status']=='ROOT_V88_ACTUAL_REPAIR_READY' and review['Cfree']>=START
        for f in (a/'cpu_preflight_owned/final_status.json',a/'root_independent_validate.json',ROOT/'scripts/review_critical_repair_v88_preseal_cpu.py'):p['files'][str(f)]=support.sha(f)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900);out.mkdir(exist_ok=False);io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
        snapshot=out/'code_snapshot';snapshot.mkdir()
        for name in p['new_code_files']:(snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
        print(json.dumps(dict(planSHA=support.sha(out/'plan.json'),cutoff=p['cutoff'],deadline=p['deadline'],output192MiB=True,max_calls=5100,expected_max5062=True)));return
    a.mkdir(exist_ok=False);old=io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json');items=[];extra=[]
    specs=[('new',Path(old['eval_data'])/'features.json','new_English'),('observed75',Path(old['observed_V75_data'])/'dev_features.json','observed_V75_DEV'),
        ('observed74',Path(old['observed_V74_data'])/'dev_features.json','observed_V74_DEV'),('long',Path(old['old_eval_data'])/'long_features.json','long')]
    for panel,file,tag in specs:
        rows=io.load_json(file);rows=rows[:2] if panel=='long' else rows;extra.append(file)
        for i,row in enumerate(rows):
            raw=ROOT/f'runs/update_retention_20261113_v80/raw_{tag}_{i:02d}.pt';packet=unpack(torch.load(raw,map_location='cpu',weights_only=True));extra.append(raw)
            if 'fuzzy_commit_reference' in packet:extra.append(Path(packet['fuzzy_commit_reference']['path']))
            patches=packet['targets'].shape[0]
            if panel=='new':idx=io.load_json(Path(old['eval_data'])/'targets.json')[i]['B_span']['start']//16;sample=[idx]
            elif panel=='long':
                points=[0,patches//4,patches//2,3*patches//4,patches-1]+([226540//16] if i==0 else [249668//16,256831//16])+[patches//3,2*patches//3]
                sample=list(dict.fromkeys(points))[:8]
            else:sample=[(len(row['prompt'].encode())//2)//16]
            items.append(dict(panel=panel,index=i,row=row,raw=str(raw),sample_patches=sample));del packet;gc.collect()
    assert len(items)==194;io.atomic_json(a/'eval_inputs.json',items)
    owner=(ROOT/'scripts/supervise_update_retention_v80.py').read_text(encoding='utf-8')
    owner=owner.replace('one actual-update V80 readout trial','one cause-directed FP32/critical coverage V88 trial').replace('run_update_retention_20261113_v80.py','run_critical_repair_20261113_v88.py')
    start=owner.index("    if (p['schema']");end=owner.index('    for name,digest',start)
    owner=owner[:start]+"    if (p['schema']!='fdt_cause_directed_critical_repair_v88' or p['wall_seconds']!=900 or p['max_calls']!=5100\n            or p['attempts']!=192 or p['teacher_epochs']!=1 or p['group_size']!=5 or p['no_accepted_window']!=32\n            or p['max_output_bytes']!=192*1024**2):\n        raise ValueError('fixed cause-directed training contract')\n"+owner[end:]
    owner=owner.replace('108*1024**3+31*1024**3//32','108*1024**3+192*1024**2').replace('108.96875GiB start:100 protected+0.96875 owned incl temp+8 headroom','108.1875GiB start:100 protected+192MiB owned incl STOPtemp+8 headroom')
    f=ROOT/'scripts/supervise_critical_repair_v88.py'
    with f.open('x',encoding='utf-8',newline='\n') as stream:stream.write(owner)
    original=(ROOT/'scripts/supervise_update_retention_v80.py').read_text(encoding='utf-8');assert owner.split("    state=dict(status='PREPARING'",1)[1]==original.split("    state=dict(status='PREPARING'",1)[1]
    names=['research/readout_decision_trace_v88.py','tests/test_readout_decision_trace_v88.py','scripts/run_critical_repair_20261113_v88.py','scripts/prepare_critical_repair_v88.py','scripts/supervise_critical_repair_v88.py',
        'scripts/critical_bank_engine_v86.py','scripts/compact_critical_engine_v87.py','research/critical_bank_storage_v87.py','research/critical_reference_device_v86.py','research/fp32_ordered_readout_v86.py',
        'research/critical_reference_bank_v85.py','research/train_critical_frontier_v84.py','research/readout_precision_policy_v81.py']
    extra += list((ROOT/'research').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src/fdt_local').glob('*.py'))
    extra += [Path(old['data'])/name for name in ('train_features.json','train_targets.json','orders.json','manifest.json')]
    source=ROOT/'runs/update_retention_20261113_v80/model/runtime.pt';storage=Path(old['storage_parent']);extra.extend([source,storage,a/'eval_inputs.json',Path(old['eval_data'])/'targets.json',
        ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1/anchor_selection.json',ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1/symbol_weight_source.json',
        ROOT/'research/artifacts/compact_critical_engine_20261113_v87_cpu_v1/root_engine_cpu_review.json',ROOT/'research/artifacts/critical_bank_engine_20261113_v86_cpu_v2/root_engine_cpu_review.json',
        ROOT/'research/artifacts/closed_raw_compression_20261113_v1/root_result.json',ROOT/'runs/precision_fix_check_20261113_v82/reconstructions.json'])
    extra+=list((ROOT/'runs/precision_fix_check_20261113_v82').glob('raw_*.pt'))
    files={str(f.resolve()):support.sha(f) for f in sorted(set(extra))}
    for file in files:
        if file.endswith('.py'):ast.parse(Path(file).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_cause_directed_critical_repair_v88',authorized=False,root=str(ROOT),output=str(ROOT/'runs/critical_repair_20261113_v88'),data=old['data'],
        source=str(source),source_SHA=PARENT_SHA,source_algorithm_SHA=ALGORITHM,storage_parent=str(storage),storage_parent_SHA=STORAGE_SHA,
        anchor_selection=str(ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1/anchor_selection.json'),symbol_profile=str(ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1/symbol_weight_source.json'),
        eval_inputs=str(a/'eval_inputs.json'),baseline_new_run=str(ROOT/'runs/precision_fix_check_20261113_v82'),attempts=192,group_size=5,teacher_epochs=1,
        precision_policy='fixed_full_FP32_Commit_readout_no_Gold_no_Exact_v81',no_accepted_window=32,wall_seconds=900,max_calls=5100,expected_max_calls=5062,
        cost=dict(teacher_encode=1024,teacher_readout=1024,head_encode_max=960,head_readout_max=960,reference_linear_max=770,baseline_cached_readout=130,final_cached_readout=194),
        component_caps=CAPS,max_output_bytes=OUTPUT,start_free_bytes=START,reference_patch_work_cap=8192*(4*192+2),max_bank_patches=8192,
        trainable_names=['readout.readout.net.2.weight','readout.readout.net.2.bias'],trainable_parameters=2117664,storage_override_tensors=7,
        max_additional_emergency_checkpoint=1,checkpoint_temp_included=True,source_encoder_updates=0,denoise_calls=0,Exact_entries=0,copy_bytes=0,
        fresh_optimizer='2AdamW1e-5 WD.01 enabledGradScaler1024, clip1; source V80 Adam+ALLRNG preserved external, not resumed trajectory',
        purpose='Repair confirmed output numerical path + TRAIN weak-critical reference coverage; unchanged data, finite192groups, no blind repeat/no DEV training',
        limitations='Same already observed correlated64+64+64+2long. Long assessment uses pinned cached Commit, no fresh long encoder claim. Trace stores all statewise UTF8 choices/goldLP/rawargmax and sampled fullscores, NOT full258 raw.',
        new_code_files=names,files=files,e2_final_accessed=False,automatic_promotion=False)
    assert sum(p['cost'].values())==5062 and files[str(source)]==PARENT_SHA and files[str(storage)]==STORAGE_SHA
    io.atomic_json(path,p);validate(path,Path(p['output']))
    n=sum(((len(x['row']['prompt'].encode())+15)//16+1)*16 for x in items);baseline_n=sum(((len(x['row']['prompt'].encode())+15)//16+1)*16 for x in items if x['panel'] in ('observed75','observed74'))
    estimate=(n+baseline_n)*26+sum(len(x['sample_patches'])*16*258*4 for x in items)+2*16*258*4
    io.atomic_json(a/'root_scope_cost_CPU_review.json',dict(schema_PASS=True,pins=len(files),max_calls=5100,expected_max5062=5062,head_rows960=True,not_unique_epoch=True,
        trace_tensor_value_estimate_bytes=estimate,assessment_cap=CAPS['assessment'],total_caps=OUTPUT,start_required=START,
        CPU9component_and_full_state_engine_contracts_previouslyPASS=True,actual_new_GPU_calls=0,unchanged_postPopen_cleanup=True,actual_fullsize_preflight_still_required=True))
    print(json.dumps(dict(plan=str(path),pins=len(files),trace_tensor_value_estimate=estimate,assessment_cap=CAPS['assessment'],max5062=True,newGPU=0)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');main(a.parse_args().seal)
