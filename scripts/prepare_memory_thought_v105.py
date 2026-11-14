"""New finite thought-only scope, actual worker preflight and independent review."""
import argparse,collections,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_memory_thought_20261114_v105 import validate,Data,SCHEMA,OUTPUT,START

A=ROOT/'research/artifacts/grid_tail_thought_20261114_v105_cpu_v1';PLAN=A/'plan_unsealed.json'
WORKER=ROOT/'scripts/run_memory_thought_20261114_v105.py';OWNER=ROOT/'scripts/supervise_memory_thought_v105.py'


def main(mode):
    if mode=='prepare':
        A.mkdir(exist_ok=False)
        old=(ROOT/'scripts/supervise_thought_cache_v95.py').read_text(encoding='utf-8');prefix,suffix=old.split("    state=dict(status='PREPARING'",1)
        prefix=prefix.replace('run_thought_cache_v95.py','run_memory_thought_20261114_v105.py')
        start=prefix.index("    if (p['schema']");end=prefix.index('    for name,digest',start)
        prefix=prefix[:start]+"    if (p['schema']!='"+SCHEMA+"' or p['wall_seconds']!=600 or p['max_calls']!=14400 or p['max_attempts']!=768 or p['train_rows']!=6144 or p['max_output_bytes']!=256*1024**2):\n        raise ValueError('fixed NEW grid-tail thought-only trial')\n"+prefix[end:]
        prefix=prefix.replace("p['deadline']-p['start']!=300","p['deadline']-p['start']!=600").replace("p['cutoff']!=p['deadline']-60","p['cutoff']!=p['deadline']-120")
        prefix=prefix.replace('108*1024**3+128*1024**2','108*1024**3+256*1024**2').replace('108.125GiB start:100 protected+128MiB owned+8 headroom','108.25GiB start:100 protected+256MiB incl STOPtemp+8 headroom')
        # Explicit NEW CPU-only execution, not a GPU launch below its12GiB
        # start floor. Device is schema-fixed; no CUDA tensor/context is used.
        gpu_start=prefix.index("    mem=native.run_native(['nvidia-smi'")
        prefix=prefix[:gpu_start]+"    if p.get('device')!='cpu':raise ValueError('CPU-only profile; GPU execution forbidden')\n"
        with OWNER.open('x',encoding='utf-8',newline='\n') as f:f.write(prefix+"    state=dict(status='PREPARING'"+suffix)
        selection=ROOT/'configs/frozen_thought_codec_20261113_v94.json';s=io.load_json(selection)
        cache=ROOT/'runs/thought_cache_20261113_v95';data=ROOT/'research/artifacts/memory_thought_data_20261113_v95_cpu_v1'
        files=[f for folder in ('research','scripts','src/fdt_local') for f in (ROOT/folder).glob('*.py')]+list(data.glob('*.json'))
        files += [selection,Path(s['selected_runtime']),Path(s['mandatory_standalone_base']),Path(s['selected_runtime']).with_suffix('.json')]
        files += [cache/name for name in ('result.json','input_index.json','target_index_loss_only.json','static_memories.pt','root_cache_and_termination_review.json','ROOT_terminal_native.json','final_status.json')]
        files += [Path(r['path']) for rows in io.load_json(cache/'input_index.json').values() for r in rows]
        files += [Path(r['target']['path']) for r in io.load_json(cache/'target_index_loss_only.json')]
        parent=ROOT/'runs/memory_thought_20261114_v102';warm=io.load_json(parent/'result.json')['checkpoints'][-1]
        curriculum=ROOT/'research/artifacts/thought_grid_tail_20261114_v104_cpu_v1/sixteen_finite_pair_orders_times.json'
        warm_full=torch.load(warm['path'],map_location='cpu',weights_only=True)
        assert support.sha(warm['path'])==warm['SHA'] and support.state_sha(warm_full)==warm['algorithm_SHA']
        files += [Path(warm['path']),curriculum,parent/'assessment.json',parent/'result.json',parent/'root_saved_review.json',parent/'ROOT_terminal_native.json',
            ROOT/'research/artifacts/thought_grid_tail_20261114_v104_cpu_v1/root_CPU_review.json']
        files += [Path(x['raw']['path']) for x in io.load_json(parent/'assessment.json') if x['phase']=='final']
        pins={str(f.resolve()):support.sha(f) for f in files}
        p=dict(schema=SCHEMA,device='cpu',authorized=False,root=str(ROOT),output=str(ROOT/'runs/grid_tail_thought_20261114_v105'),selection=str(selection),codec_SHA=s['selected_SHA'],
            data=str(data),cache=str(cache),files=pins,max_calls=14400,expected_calls=14192,max_attempts=768,train_rows=6144,batch=8,epochs=16,
            eval_rows=96,control_rows=16,steps=16,wall_seconds=600,max_output_bytes=OUTPUT,start_free_bytes=START,seed=1111405,LR=1e-4,tail_weight=.05,
            warm=warm,warm_model_SHA=support.state_sha(warm_full['engine']['thought']),curriculum=str(curriculum),baseline_assessment=str(parent/'assessment.json'),
            purpose='Correct high-noise time coverage and direct tail loss. EachTRAINrow all16 sampler times. Warm ONLY V102thought; FRESHAdam/scaler. Allcodecweightsbuffersfrozen. No target/length during inference.',
            cost=dict(training=dict(rows=6144,denoise=6144,readout=6144),baseline_final=dict(rows=96,denoise=1536,readout=96),
                final_controls=dict(rows=16,denoise=256,readout=16),baseline_inherited=dict(rows=96,denoise=0,readout=0),total=14192),
            fullstate=dict(checkpoints=[0,384,768],STOP_current=True,scaled_pending_gradients=True,populated_Adam=True,enabled_scaler=True,ALL_RNG=True,no_old_budget_resume=True),
            success_scope='Same source/forward/sampler and warm weights→reuse prior saved96 baseline, one actual parity check. Report actual factual output vs earlyEOS, not only NLL. No automatic extraSGD/promotion.')
        io.atomic_json(PLAN,p);validate(PLAN,Path(p['output']));d=Data(p)
        tt={i:[] for i in d.train}
        for i,t in zip(d.orders,d.times):tt[i].append(t)
        io.atomic_json(A/'root_finite_scope.json',dict(TRAIN_unique384=len(set(d.orders))==384,sixteen_finite_6144=len(d.orders)==6144,
            each_TRAIN_exactly16=all(n==16 for n in collections.Counter(d.orders).values()),each_TRAIN_each16_times=all(sorted(v)==sorted(1-j/16 for j in range(16)) for v in tt.values()),
            DEV_literal_overlap0=not(set(d.train)&{x['id'] for x in d.evals}),
            TRAINtargetbytes108544=sum(len(d.train[i]['answer'].encode()) for i in d.orders),new_context_encoder_calls=0,source_only_cache=True,
            warm_model_SHA=p['warm_model_SHA'],old_Adam_and_RNG_preserved_not_resumed=True,
            codec_SHA=p['codec_SHA'],pure_readonly_cache_inputs=True,proposed_live_calls=14192,baseline_reuse_calls0=True,output256MiB_includes_STOP_gradients_and_temp=True))
        print(json.dumps(dict(unsealed=str(PLAN),pins=len(pins),actual_GPU_training=0)));return
    p=validate(PLAN,Path(io.load_json(PLAN)['output']));out=Path(p['output']);pre=io.load_json(A/'cpu_preflight_owned/final_status.json')
    assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['worker_exit_code']==0 and pre['exit']['all_owned_processes_gone']
    r=pre['report'];assert not r['CUDA_initialized'] and r['all_old_codec_parameters_buffers_unchanged'] and r['only_new_thought_optimizer']
    assert r['actual_populated_Adam_scaler_RNG_READY_exact'] and r['actual_thought_parameters']==3224322 and r['actual_encoder_forwards']==0
    assert r['actual_fullsize_warm_cached_row_bitwise_parity'] and r['old_Adam_RNG_not_loaded_into_new_optimizer'] and r['exact_T1_mid_low_grid_training']
    if mode=='review':
        old=(ROOT/'scripts/supervise_thought_cache_v95.py').read_text(encoding='utf-8');new=OWNER.read_text(encoding='utf-8')
        assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
        saved=torch.load(A/'cpu_preflight_owned/runtime/populated_READY_fullstate.pt',map_location='cpu',weights_only=True)
        assert saved['runtime']['phase']=='READY' and saved['runtime']['cursor']==3 and saved['runtime']['updates']==1
        assert saved['optimizer']['state'] and saved['scaler']['scale']==1024 and saved['ALL_RNG']['CUDA']==[]
        assert any(g is not None for g in saved['gradients'].values()) and support.state_sha(saved)==r['saved_READY_algorithm_SHA']
        assert saved['schema']=='grid_tail_thought_fullstate_v104' and saved['fixed_times']==[1.,.5,.0625] and saved['tail_weight']==.05
        scope=io.load_json(A/'root_finite_scope.json');assert scope['TRAINtargetbytes108544']==108544 and scope['DEV_literal_overlap0'] and scope['each_TRAIN_exactly16'] and scope['each_TRAIN_each16_times']
        assert sum(p['cost'][phase][kind] for phase in ('training','baseline_final','final_controls') for kind in ('denoise','readout'))==14192
        assert shutil.disk_usage(ROOT).free>=START
        io.atomic_json(A/'root_independent_validate.json',dict(status='ROOT_MEMORY_THOUGHT_FIRST_TRIAL_READY',actual_fullsize_CPU=r,actual_Adam_scaler_pendinggrads_RNG_safeload=True,
            frozen_all_codec_and_buffers=True,unchanged_all_postPopen_owned_OS_cleanup=True,new_only_thought_optimizer=True,TRAINonly_finite_scope=scope,
            newcaps=dict(calls=14400,expected=14192,attempts=768,rows=6144,wall=600,model_cutoff=480,reserve=120,output=OUTPUT),
            exact_CPU_restore_not_CUDA_resume=True,Cfree=shutil.disk_usage(ROOT).free))
        print('ROOT_MEMORY_THOUGHT_FIRST_TRIAL_READY');return
    assert mode=='seal' and io.load_json(A/'root_independent_validate.json')['status']=='ROOT_MEMORY_THOUGHT_FIRST_TRIAL_READY' and shutil.disk_usage(ROOT).free>=START
    for f in (A/'cpu_preflight_owned/final_status.json',A/'root_independent_validate.json',A/'root_finite_scope.json'):p['files'][str(f)]=support.sha(f)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+480,deadline=now+600);out.mkdir(exist_ok=False);io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
    snap=out/'code_snapshot';snap.mkdir()
    for f in (Path(__file__).resolve(),WORKER,OWNER,ROOT/'research/thought_train_engine_v98.py',ROOT/'research/thought_language_objective_v96.py',ROOT/'research/memory_thought_diffusion_v94.py',ROOT/'research/thought_grid_tail_v104.py'):(snap/f.name).write_bytes(f.read_bytes())
    print(json.dumps(dict(planSHA=support.sha(out/'plan.json'),output=str(out),deadline=p['deadline'],calls=14400,expected14192=True)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','review','seal']);main(a.parse_args().mode)
