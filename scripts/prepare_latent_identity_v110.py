"""Selected134 verified meanings, finite identity thought-only preparation."""
import argparse,collections,hashlib,json,random,shutil,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import learning_io_v2 as io

A=Path('D:/Codex/FDT/research/latent_identity_20261114_v110_cpu_v1')
PLAN=A/'plan_unsealed.json'
WORKER=P/'scripts/run_latent_identity_v110_cpu.py';OWNER=P/'scripts/supervise_latent_identity_v110.py'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main(mode):
    if mode=='prepare':
        A.mkdir(parents=True,exist_ok=False)
        old=io.load_json(P/'runs/grid_tail_thought_20261114_v105/plan.json')
        screen=P/'research/artifacts/roundtrip_identity_screen_20261114_v108_cpu_v2'
        terminal=io.load_json(screen/'ROOT_terminal_native.json');assert terminal['PythonCount']==0 and terminal['All4OwnedProxyGone']
        runtime=screen/'cpu_preflight_owned/runtime';selected=io.load_json(runtime/'selected_TRAIN_only.json');assert len(selected)==134
        orders=[]
        for epoch in range(4):
            ids=sorted(r['id'] for r in selected);random.Random(110000+epoch).shuffle(ids)
            orders.append([dict(id=i,t=1-((epoch+int(hashlib.sha256(i.encode()).hexdigest()[:8],16))%16)/16) for i in ids])
        io.atomic_json(A/'four_finite_orders.json',orders)
        pins=dict(old['files'])
        additions=[WORKER,OWNER,Path(__file__).resolve(),P/'research/runtime_aligned_thought_v107.py',P/'research/latent_identity_thought_v110.py',
            P/'tests/test_runtime_aligned_thought_v107.py',screen/'plan.json',screen/'ROOT_terminal_native.json',screen/'cpu_preflight_owned/final_status.json',
            runtime/'result.json',runtime/'all480.json',runtime/'selected_TRAIN_only.json',A/'four_finite_orders.json']
        additions += [Path(r['verified_roundtrip_raw']['path']) for r in selected]
        for f in additions:pins[str(f.resolve())]=sha(f)
        p={k:old[k] for k in ('root','selection','codec_SHA','data','cache')}
        p.update(schema='fdt_fullrank_LATENT_ONLY_identity_thought_v110',device='cpu',authorized=False,
            output='D:/Codex/FDT/runs/latent_identity_20261114_v110',files=pins,screen=str(runtime),curriculum=str(A/'four_finite_orders.json'),
            max_calls=13200,expected_calls=12920,max_attempts=67,train_rows=536,unique_TRAIN=134,batch=8,epochs=4,eval_rows=112,steps=16,
            wall_seconds=600,max_output_bytes=640*1024**2,seed=1111410,LR=1e-4,
            task='LATENT_ONLY identity denoising of verified E(X); no D forward/loss during training; no relation reasoning',
            source_and_realizer_frozen='ENTIRE123497080parameters+ALLbuffers+K_P/K_C/K_E/binding/output',
            cost=dict(train_rows536_thought17_D0=9112,baseline112_times17=1904,final112_times17=1904,total=12920),
            fresh_random_thought_and_Adam=True,old_weights_AdamRNG_preserved=True,
            fullstate=dict(checkpoints=[0,34,67],STOP_current=True,ALL_RNG=True,scaler=True,all_gradients=True,exact_cursor=True),
            resources=dict(C_start=108*1024**3+1024**2,C_protected=100*1024**3,D_start=8*1024**3+640*1024**2,D_protected=1024**3,output_on_D=True),
            no_automatic_promotion_or_repeat=True)
        io.atomic_json(PLAN,p)
        from run_latent_identity_v110_cpu import validate
        validate(PLAN,Path(p['output']).resolve())
        io.atomic_json(A/'root_finite_scope.json',dict(selected134_exact_TRAIN=True,DEV_literal_overlap0=True,each_selected_once_per4epochs=True,
            total_rows536=True,orders_SHA=sha(A/'four_finite_orders.json'),raw_THETA_input_has_no_labels=True,main_trajectory16_per_row=True,
            auxiliary_times4_per_row_not_all16=True,selected_pairs_not_necessarily_complete=True,fullstate_on_D_not_C=True,D_during_training0=True,language_loss_NOT_used=True,hidden520=True))
        link=P/'runs/latent_identity_20261114_v110_LINK.json'
        io.atomic_json(link,dict(status='PREPARED_NOT_LAUNCHED',actual_artifact=str(A),actual_output=p['output'],plan_unsealed=str(PLAN)))
        print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',plan=str(PLAN),plan_SHA=sha(PLAN),rows=536,calls=12920)));return
    p=io.load_json(PLAN)
    from run_latent_identity_v110_cpu import validate
    validate(PLAN,Path(p['output']).resolve())
    pre=io.load_json(A/'cpu_preflight_owned/final_status.json');assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['worker_exit_code']==0
    r=pre['report'];assert r['thought_parameters']==9576426 and r['effective_velocity_rank']==512 and r['D_calls_during_training']==0 and r['only_thought_optimizer'] and r['allcodec_parameters_buffers_frozen']
    assert r['actual_fullsize_READY_Adam_scaler_RNG_nextupdate_exact'] and r['actual_padded_selected_target_roundtrip_exact'] and r['actual16step_training_vs_eval_maxgap']<1e-4
    assert r['training_micros3_seconds']/3*536<400 and not r['CUDA_initialized']
    if mode=='review':
        import torch
        import text_generation_support_v1 as support
        saved=torch.load(A/'cpu_preflight_owned/runtime/populated_READY_fullstate.pt',map_location='cpu',weights_only=True)
        assert support.state_sha(saved)==r['saved_READY_algorithm_SHA'] and saved['schema']=='latent_identity_fullstate_v110' and not saved['language_losses_used']
        assert saved['runtime']['phase']=='READY' and saved['runtime']['cursor']==3 and saved['runtime']['updates']==1 and saved['ALL_RNG']['CUDA']==[]
        assert saved['optimizer']['state'] and saved['scaler']['scale']==1024 and any(g is not None for g in saved['gradients'].values())
        old=(P/'scripts/supervise_memory_thought_v105.py').read_text(encoding='utf-8');new=OWNER.read_text(encoding='utf-8')
        assert old.split('        finally:',1)[1]==new.split('        finally:',1)[1]
        io.atomic_json(A/'root_independent_validate.json',dict(status='ROOT_PURE_LATENT_FULLRANK_V110_READY',actual_fullsize_CPU=r,
            exact_READY_fullstate_Adam_scaler_RNG=True,all_postPopen_cleanup_unchanged=True,selected134_TRAIN_only=True,
            current_identity_task_not_relation=True,finite536rows67attempts=True,expected_calls12920=True,
            C100GiB_protected_and8GiB_startheadroom=True,D640MiB_all_checkpoints_raw_STOPtemp=True))
        print('ROOT_PURE_LATENT_FULLRANK_V110_READY');return
    assert mode=='seal' and io.load_json(A/'root_independent_validate.json')['status']=='ROOT_PURE_LATENT_FULLRANK_V110_READY'
    out=Path(p['output']);out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(P).free>=p['resources']['C_start'] and shutil.disk_usage(out.parent).free>=p['resources']['D_start']
    for f in (A/'cpu_preflight_owned/final_status.json',A/'root_independent_validate.json',A/'root_finite_scope.json'):p['files'][str(f)]=sha(f)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+480,deadline=now+600);out.mkdir(exist_ok=False);io.atomic_json(out/'plan.json',p)
    validate(out/'plan.json',out.resolve());snap=out/'code_snapshot';snap.mkdir()
    for f in (WORKER,OWNER,Path(__file__).resolve(),P/'research/runtime_aligned_thought_v107.py',P/'research/latent_identity_thought_v110.py',P/'research/thought_grid_tail_v104.py',P/'research/thought_train_engine_v98.py'):(snap/f.name).write_bytes(f.read_bytes())
    io.atomic_json(P/'runs/latent_identity_20261114_v110_LINK.json',dict(status='SEALED_NOT_YET_LAUNCHED',actual_artifact=str(A),actual_output=str(out),plan_SHA=sha(out/'plan.json')))
    print(json.dumps(dict(output=str(out),plan_SHA=sha(out/'plan.json'),new_start=now,new_cutoff=p['cutoff'],new_deadline=p['deadline'])) )

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','review','seal']);main(q.parse_args().mode)
