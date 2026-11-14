"""New finite thought-only scope, actual worker preflight and independent review."""
import argparse,collections,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_memory_thought_20261114_v100 import validate,Data,SCHEMA,OUTPUT,START

A=ROOT/'research/artifacts/first_memory_thought_20261114_v100_cpu_v1';PLAN=A/'plan_unsealed.json'
WORKER=ROOT/'scripts/run_memory_thought_20261114_v100.py';OWNER=ROOT/'scripts/supervise_memory_thought_v100.py'


def main(mode):
    if mode=='prepare':
        A.mkdir(exist_ok=False)
        old=(ROOT/'scripts/supervise_thought_cache_v95.py').read_text(encoding='utf-8');prefix,suffix=old.split("    state=dict(status='PREPARING'",1)
        prefix=prefix.replace('run_thought_cache_v95.py','run_memory_thought_20261114_v100.py')
        start=prefix.index("    if (p['schema']");end=prefix.index('    for name,digest',start)
        prefix=prefix[:start]+"    if (p['schema']!='"+SCHEMA+"' or p['wall_seconds']!=600 or p['max_calls']!=6800 or p['max_attempts']!=192 or p['train_rows']!=1536 or p['max_output_bytes']!=256*1024**2):\n        raise ValueError('fixed NEW finite thought-only trial')\n"+prefix[end:]
        prefix=prefix.replace("p['deadline']-p['start']!=300","p['deadline']-p['start']!=600").replace("p['cutoff']!=p['deadline']-60","p['cutoff']!=p['deadline']-120")
        prefix=prefix.replace('108*1024**3+128*1024**2','108*1024**3+256*1024**2').replace('108.125GiB start:100 protected+128MiB owned+8 headroom','108.25GiB start:100 protected+256MiB incl STOPtemp+8 headroom')
        with OWNER.open('x',encoding='utf-8',newline='\n') as f:f.write(prefix+"    state=dict(status='PREPARING'"+suffix)
        selection=ROOT/'configs/frozen_thought_codec_20261113_v94.json';s=io.load_json(selection)
        cache=ROOT/'runs/thought_cache_20261113_v95';data=ROOT/'research/artifacts/memory_thought_data_20261113_v95_cpu_v1'
        files=[f for folder in ('research','scripts','src/fdt_local') for f in (ROOT/folder).glob('*.py')]+list(data.glob('*.json'))
        files += [selection,Path(s['selected_runtime']),Path(s['mandatory_standalone_base']),Path(s['selected_runtime']).with_suffix('.json')]
        files += [cache/name for name in ('result.json','input_index.json','target_index_loss_only.json','static_memories.pt','root_cache_and_termination_review.json','ROOT_terminal_native.json','final_status.json')]
        files += [Path(r['path']) for rows in io.load_json(cache/'input_index.json').values() for r in rows]
        files += [Path(r['target']['path']) for r in io.load_json(cache/'target_index_loss_only.json')]
        pins={str(f.resolve()):support.sha(f) for f in files}
        p=dict(schema=SCHEMA,authorized=False,root=str(ROOT),output=str(ROOT/'runs/memory_thought_20261114_v100'),selection=str(selection),codec_SHA=s['selected_SHA'],
            data=str(data),cache=str(cache),files=pins,max_calls=6800,expected_calls=6608,max_attempts=192,train_rows=1536,batch=8,epochs=4,
            eval_rows=96,control_rows=16,steps=16,wall_seconds=600,max_output_bytes=OUTPUT,start_free_bytes=START,seed=11114,LR=1e-4,
            purpose='FIRST new memory thought ONLY3.2M weights learn: sourceZ+K_P/K_C/NULLExact→conditionednoise→16stepFINALmeaning→frozenD. VP-v+.25clean+.5byteCE+.5EOSCE. Allcodecparametersbuffersfrozen.',
            cost=dict(training=dict(rows=1536,denoise=1536,readout=1536),baseline_final=dict(rows=192,denoise=3072,readout=192),
                final_controls=dict(rows=16,denoise=256,readout=16),total=6608),
            fullstate=dict(checkpoints=[0,96,192],STOP_current=True,scaled_pending_gradients=True,populated_Adam=True,enabled_scaler=True,ALL_RNG=True,no_old_budget_resume=True),
            success_scope='First bounded learning evidence, NOT language/meaning/generalreasoning completion. Compare same-noise baseline/final positive relations vs unknown baseline and tail/EOS. No automatic extraSGD/promotion.')
        io.atomic_json(PLAN,p);validate(PLAN,Path(p['output']));d=Data(p)
        io.atomic_json(A/'root_finite_scope.json',dict(TRAIN_unique384=len(set(d.orders))==384,four_finite_1536=len(d.orders)==1536,
            each_TRAIN_exactly4=all(n==4 for n in collections.Counter(d.orders).values()),DEV_literal_overlap0=not(set(d.train)&{x['id'] for x in d.evals}),
            TRAINtargetbytes27136=sum(len(d.train[i]['answer'].encode()) for i in d.orders),new_context_encoder_calls=0,source_only_cache=True,
            codec_SHA=p['codec_SHA'],pure_readonly_cache_inputs=True,proposed_live_calls=6608,output256MiB_includes_STOP_gradients_and_temp=True))
        print(json.dumps(dict(unsealed=str(PLAN),pins=len(pins),actual_GPU_training=0)));return
    p=validate(PLAN,Path(io.load_json(PLAN)['output']));out=Path(p['output']);pre=io.load_json(A/'cpu_preflight_owned/final_status.json')
    assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['worker_exit_code']==0 and pre['exit']['all_owned_processes_gone']
    r=pre['report'];assert not r['CUDA_initialized'] and r['all_old_codec_parameters_buffers_unchanged'] and r['only_new_thought_optimizer']
    assert r['actual_populated_Adam_scaler_RNG_READY_exact'] and r['actual_thought_parameters']==3224322 and r['actual_encoder_forwards']==0
    if mode=='review':
        old=(ROOT/'scripts/supervise_thought_cache_v95.py').read_text(encoding='utf-8');new=OWNER.read_text(encoding='utf-8')
        assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
        saved=torch.load(A/'cpu_preflight_owned/runtime/populated_READY_fullstate.pt',map_location='cpu',weights_only=True)
        assert saved['runtime']['phase']=='READY' and saved['runtime']['cursor']==3 and saved['runtime']['updates']==1
        assert saved['optimizer']['state'] and saved['scaler']['scale']==1024 and saved['ALL_RNG']['CUDA']==[]
        assert any(g is not None for g in saved['gradients'].values()) and support.state_sha(saved)==r['saved_READY_algorithm_SHA']
        scope=io.load_json(A/'root_finite_scope.json');assert scope['TRAINtargetbytes27136']==27136 and scope['DEV_literal_overlap0'] and scope['each_TRAIN_exactly4']
        assert sum(p['cost'][phase][kind] for phase in ('training','baseline_final','final_controls') for kind in ('denoise','readout'))==6608
        assert shutil.disk_usage(ROOT).free>=START
        io.atomic_json(A/'root_independent_validate.json',dict(status='ROOT_MEMORY_THOUGHT_FIRST_TRIAL_READY',actual_fullsize_CPU=r,actual_Adam_scaler_pendinggrads_RNG_safeload=True,
            frozen_all_codec_and_buffers=True,unchanged_all_postPopen_owned_OS_cleanup=True,new_only_thought_optimizer=True,TRAINonly_finite_scope=scope,
            newcaps=dict(calls=6800,expected=6608,attempts=192,rows=1536,wall=600,model_cutoff=480,reserve=120,output=OUTPUT),
            exact_CPU_restore_not_CUDA_resume=True,Cfree=shutil.disk_usage(ROOT).free))
        print('ROOT_MEMORY_THOUGHT_FIRST_TRIAL_READY');return
    assert mode=='seal' and io.load_json(A/'root_independent_validate.json')['status']=='ROOT_MEMORY_THOUGHT_FIRST_TRIAL_READY' and shutil.disk_usage(ROOT).free>=START
    for f in (A/'cpu_preflight_owned/final_status.json',A/'root_independent_validate.json',A/'root_finite_scope.json'):p['files'][str(f)]=support.sha(f)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+480,deadline=now+600);out.mkdir(exist_ok=False);io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
    snap=out/'code_snapshot';snap.mkdir()
    for f in (Path(__file__).resolve(),WORKER,OWNER,ROOT/'research/thought_train_engine_v98.py',ROOT/'research/thought_language_objective_v96.py',ROOT/'research/memory_thought_diffusion_v94.py'):(snap/f.name).write_bytes(f.read_bytes())
    print(json.dumps(dict(planSHA=support.sha(out/'plan.json'),output=str(out),deadline=p['deadline'],calls=6800,expected6608=True)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','review','seal']);main(a.parse_args().mode)
