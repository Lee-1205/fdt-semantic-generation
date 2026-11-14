"""Root preparation/review/seal of ONE conditional E(Y) meaning trial."""
import argparse,collections,hashlib,json,shutil,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import learning_io_v2 as io
A=Path('D:/Codex/FDT/research/answer_meaning_20261114_v111_cpu_v1');PLAN=A/'plan_unsealed.json'
OUT=Path('D:/Codex/FDT/runs/answer_meaning_20261114_v111')
WORKER=P/'scripts/run_answer_meaning_v111_cpu.py';OWNER=P/'scripts/supervise_answer_meaning_v111.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main(mode):
    if mode=='prepare':
        A.mkdir(parents=True,exist_ok=False)
        old=io.load_json(Path('D:/Codex/FDT/runs/latent_identity_20261114_v110/plan.json'))
        parent=io.load_json(Path('D:/Codex/FDT/runs/latent_identity_20261114_v110/result.json'))
        terminal=io.load_json(Path('D:/Codex/FDT/runs/latent_identity_20261114_v110/ROOT_terminal_native.json'))
        assert terminal['PythonCount']==0
        contract=P/'research/artifacts/answer_meaning_objective_20261114_v1'
        selected={r['id'] for r in io.load_json(contract/'train_conditions.json')}
        allrows=io.load_json(Path(old['data'])/'train_targets.json');rows=[r for r in allrows if r['id'] in selected]
        from research.answer_meaning_thought_v111 import answer_curriculum
        orders=answer_curriculum(rows);io.atomic_json(A/'sixteen_finite_orders.json',orders)
        pairs=collections.defaultdict(list)
        for r in rows:pairs[r['pair_key']].append(r)
        positive=[v for k,v in sorted(pairs.items()) if len(v)==2 and all(r['label']!='unknown' for r in v)]
        unknown=[v for k,v in sorted(pairs.items()) if len(v)==2 and all(r['label']=='unknown' for r in v)]
        probe=[r['id'] for pair in positive[:4]+unknown[:4] for r in sorted(pair,key=lambda r:r['id'])]
        assert len(probe)==16
        pins=dict(old['files'])
        additions=[WORKER,OWNER,Path(__file__).resolve(),P/'research/answer_meaning_thought_v111.py',P/'tests/test_answer_meaning_v111.py',
            A/'sixteen_finite_orders.json',contract/'contract.json',contract/'train_conditions.json',contract/'train_targets_LOSS_ONLY.json',
            Path('D:/Codex/FDT/runs/latent_identity_20261114_v110/result.json'),Path('D:/Codex/FDT/runs/latent_identity_20261114_v110/ROOT_terminal_native.json')]
        warm=parent['checkpoints'][-1];additions.append(Path(warm['path']))
        for f in additions:pins[str(f.resolve())]=sha(f)
        import torch
        import text_generation_support_v1 as support
        saved=torch.load(warm['path'],map_location='cpu',weights_only=True)
        assert support.state_sha(saved)==warm['algorithm_SHA'];modelsha=support.state_sha(saved['engine']['thought']);del saved
        p={k:old[k] for k in ('root','selection','codec_SHA','data','cache')}
        p.update(schema='fdt_conditional_ANSWER_meaning_v111',device='cpu',authorized=False,output=str(OUT),files=pins,
            answer_contract=str(contract),curriculum=str(A/'sixteen_finite_orders.json'),probe_ids=probe,warm=warm,warm_thought_model_SHA=modelsha,
            max_calls=41000,expected_calls=40256,max_attempts=268,train_rows=2144,unique_TRAIN=134,batch=8,epochs=16,eval_rows=112,steps=16,
            wall_seconds=1200,max_output_bytes=1024**3,seed=1111411,LR=1e-4,
            task='Transformation: conditional NEW answer meaning E(Y), NOT source reconstruction E(X)',
            loss=dict(body_MSE=1.,tail_MSE=.05,VP_aux=.1,D_during_training=0),
            batch_policy='two-row scheduling blocks preserve actual58reversepairs and18singleton; no falsepairloss; flattened268attempts',
            fullstate=dict(checkpoints=[0,134,268],STOP_current=True,ALL_RNG=True,scaler=True,gradients=True,exact_cursor=True),
            CPU_preflight_scope=dict(max_calls=400,worker_seconds=90,reserve_seconds=30,max_output_bytes=512*1024**2,temporary_updates_including_replay=3),
            resources=dict(C_start=104*1024**3+64*1024**2,C_protected=100*1024**3,D_start=9*1024**3,D_protected=1024**3,
                RAM_start_available_GiB=8,output_on_D=True,C_write_cap_bytes=16*1024**2,
                rationale='New preseal resource review: all checkpoint/raw/preflight onD; Cwrites only tiny metadata. Maintain100GiBfloor+4GiB OS cleanupreserve+64MiBmetadata. Earlier108GiBwasprospective, notasealedexecution; nooriginalassetschangedormoved.'),
            no_automatic_promotion_or_repeat=True,oldbudgets_closed=True)
        io.atomic_json(PLAN,p)
        from run_answer_meaning_v111_cpu import validate
        validate(PLAN,OUT.resolve())
        io.atomic_json(A/'root_scope_review.json',dict(task_is_EY_NOT_EX=True,selected134_TRAIN_only=True,DEV_training0=True,
            actual58reversepairs18singleton=True,eachrow_every16grid_once=True,total2144rows268attempts=True,
            answer_bytes37088=True,calls40256=True,architecture_hidden520=True,warm_ONLYtheta_newAdam=True,
            all_old_assets_preserved=True,latest_asset_no_compression_no_deletion_after_instruction=True,new_prospective_resource_contract=p['resources']))
        io.atomic_json(P/'runs/answer_meaning_20261114_v111_LINK.json',dict(status='PREPARED_NOT_LAUNCHED',actual_artifact=str(A),actual_output=str(OUT)))
        print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',plan=str(PLAN),callcap=41000)));return
    p=io.load_json(PLAN)
    from run_answer_meaning_v111_cpu import validate
    validate(PLAN,OUT.resolve())
    pre=io.load_json(A/'cpu_preflight_owned/final_status.json')
    assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['worker_exit_code']==0 and not pre['owner_imported_Torch']
    r=pre['report'];assert r['thought_parameters']==9576426 and r['effective_velocity_rank']==512 and r['D_calls_during_training']==0
    assert r['warm_ONLY_thought_exact'] and r['fresh_Adam_initially_empty'] and r['actual27_target_meaning_prefixes_correct']
    assert r['actual_fullsize_READY_Adam_scaler_RNG_nextupdate_exact'] and r['last_partial6micros_exact']
    assert sum(r['NN_calls'].values())<=400 and r['training_micros14_seconds']/14*2144<900
    if mode=='review':
        import torch
        import text_generation_support_v1 as support
        ready=torch.load(A/'cpu_preflight_owned/runtime/populated_READY_fullstate.pt',map_location='cpu',weights_only=True)
        assert support.state_sha(ready)==r['saved_READY_algorithm_SHA'] and ready['schema']=='conditional_answer_meaning_fullstate_v111'
        assert ready['runtime']['cursor']==14 and ready['runtime']['updates']==1 and ready['runtime']['phase']=='READY'
        assert ready['optimizer']['state'] and ready['scaler']['scale']==1024 and any(g is not None for g in ready['gradients'].values()) and ready['ALL_RNG']['CUDA']==[]
        oldowner=(P/'scripts/supervise_latent_identity_v110.py').read_text();newowner=OWNER.read_text()
        assert oldowner.split('        finally:',1)[1]==newowner.split('        finally:',1)[1]
        io.atomic_json(A/'root_independent_validate.json',dict(status='ROOT_CONDITIONAL_ANSWER_V111_READY_NOT_LAUNCHED',actual_fullsize_CPU=r,
            READY_Adam_scaler_ALLRNG_gradients_exact=True,all_postPopen_owned_cleanup_unchanged=True,
            task_target_EY_training_D0=True,all27oracle_prefixes_correct_with_tail_errors_separate=True,C_floor100GiB_unchanged=True,
            actual_CPU_output_bytes=sum(f.stat().st_size for f in (A/'cpu_preflight_owned/runtime').iterdir() if f.is_file())))
        print('ROOT_CONDITIONAL_ANSWER_V111_READY');return
    assert mode=='seal' and io.load_json(A/'root_independent_validate.json')['status']=='ROOT_CONDITIONAL_ANSWER_V111_READY_NOT_LAUNCHED'
    assert shutil.disk_usage(P).free>=p['resources']['C_start'] and shutil.disk_usage(OUT.parent).free>=p['resources']['D_start']
    for f in (A/'cpu_preflight_owned/final_status.json',A/'root_independent_validate.json',A/'root_scope_review.json'):
        p['files'][str(f)]=sha(f)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+1080,deadline=now+1200)
    OUT.mkdir(parents=True,exist_ok=False);io.atomic_json(OUT/'plan.json',p);validate(OUT/'plan.json',OUT.resolve())
    snap=OUT/'code_snapshot';snap.mkdir()
    for f in (WORKER,OWNER,Path(__file__).resolve(),P/'research/answer_meaning_thought_v111.py',P/'research/latent_identity_thought_v110.py',
              P/'research/runtime_aligned_thought_v107.py',P/'research/thought_grid_tail_v104.py',P/'research/thought_train_engine_v98.py'):
        (snap/f.name).write_bytes(f.read_bytes())
    io.atomic_json(P/'runs/answer_meaning_20261114_v111_LINK.json',dict(status='SEALED_NOT_YET_LAUNCHED',actual_artifact=str(A),actual_output=str(OUT),plan_SHA=sha(OUT/'plan.json')))
    print(json.dumps(dict(status='SEALED_NOT_YET_LAUNCHED',output=str(OUT),plan_SHA=sha(OUT/'plan.json'),start=now,cutoff=p['cutoff'],deadline=p['deadline'])))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','review','seal']);main(q.parse_args().mode)
