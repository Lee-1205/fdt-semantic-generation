"""Root-only preparation, actual preflight review and one new clock seal."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import learning_io_v2 as io
A=P/'research/artifacts/answer_pair_contrast_20261114_v112_C_cpu_v1'
OUT=P/'runs/answer_pair_contrast_20261114_v112';PLAN=A/'plan_unsealed.json'
WORKER=P/'scripts/run_answer_pair_contrast_v112_cpu.py';OWNER=P/'scripts/supervise_answer_pair_contrast_v112_cpu.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main(mode):
    from run_answer_pair_contrast_v112_cpu import validate
    if mode=='prepare':
        A.mkdir(parents=True,exist_ok=False)
        oldrun=P/'runs/answer_meaning_20261114_v111';pairrun=P/'research/artifacts/answer_meaning_v111_fullTRAIN_pair_readonly_20261114_v1'
        p=io.load_json(oldrun/'plan.json');parent=io.load_json(oldrun/'result.json')
        for r in (oldrun,pairrun):
            t=io.load_json(r/'ROOT_terminal_native.json');assert t['PythonCount']==0
            s=io.load_json(r/'final_status.json');assert s['status']=='COMPLETE' and s['exit']['all_owned_processes_gone']
        oldowner=P/'scripts/supervise_answer_meaning_v111_c.py'
        text=oldowner.read_text(encoding='utf-8').replace('supervise_answer_meaning_v111_c','supervise_answer_pair_contrast_v112_cpu')
        text=text.replace('run_answer_meaning_v111_c_cpu.py','run_answer_pair_contrast_v112_cpu.py')
        text=text.replace('fdt_conditional_ANSWER_meaning_v111','fdt_answer_pair_contrast_v112')
        text=text.replace('conditional answer-meaning V111','actual-pair contrast V112')
        text=text.replace('conditional answer E(Y)','paired contrast answer E(Y)')
        with OWNER.open('x',encoding='utf-8') as f:f.write(text)
        warm=parent['checkpoints'][-1]
        import torch
        import text_generation_support_v1 as support
        saved=torch.load(warm['path'],map_location='cpu',weights_only=True)
        assert support.state_sha(saved)==warm['algorithm_SHA'];modelsha=support.state_sha(saved['engine']['thought'])
        for k in ('start','cutoff','deadline'):p.pop(k,None)
        p.update(schema='fdt_answer_pair_contrast_v112',authorized=False,output=str(OUT),seed=1111412,warm=warm,
            warm_thought_model_SHA=modelsha,expected_calls=40358,eval_rows=230,pair_coefficient=.25,
            task='conditional E(Y), same-noise actual-pair endpoint-difference repair',
            baseline_TRAIN_assessment=str(pairrun/'assessment.json'),baseline_DEV_assessment=str(oldrun/'assessment.json'),
            baseline_forward_calls=0,loss=dict(body_MSE=1.,tail_MSE=.05,VP_aux=.1,pair_difference=.25,training_D0=True),
            fullstate=dict(checkpoints=[0,134,268],STOP_current=True,atomic_rows=2,ALL_RNG=True,scaled_gradients=True),
            CPU_preflight_scope=dict(max_calls=450,worker_seconds=90,reserve_seconds=30,max_output_bytes=512*1024**2,temporary_updates_including_replays=4))
        additions=[WORKER,OWNER,Path(__file__).resolve(),P/'research/answer_pair_contrast_v112.py',P/'tests/test_answer_pair_contrast_v112.py',
            oldrun/'plan.json',oldrun/'result.json',oldrun/'assessment.json',oldrun/'root_saved_review.json',oldrun/'root_result.json',
            oldrun/'ROOT_terminal_native.json',oldrun/'root_final_delivery.json',pairrun/'assessment.json',pairrun/'root_saved_review.json',
            pairrun/'ROOT_terminal_native.json',Path(warm['path'])]
        for f in additions:p['files'][str(f.resolve())]=sha(f)
        # Pin inherited trace payloads too; they are unchanged baseline evidence.
        baseline=io.load_json(pairrun/'assessment.json')+[r for r in io.load_json(oldrun/'assessment.json') if r['phase']=='final' and r['split'].startswith('dev_')]
        assert len(baseline)==230
        for row in baseline:p['files'][row['raw']['path']]=row['raw']['SHA']
        io.atomic_json(PLAN,p);validate(PLAN,OUT.resolve())
        io.atomic_json(A/'root_math_and_scope_review.json',dict(actual_pairs58_singletons18=True,
            each_epoch134_once=True,epochs16_rows2144=True,RNG_draws1216=True,expected_calls40358=True,
            equal_body_mask_pair_identity='endpoint mean = MSE(mean error) + .25 MSE(difference error)',
            added_coefficient=.25,antisymmetric_coefficient_doubles=True,
            shared_noise_preserves_per_row_marginal=True,coupled_intervention_NOT_separate_causal_estimates=True,
            optimizer_NEW_NOT_V111trajectory=True,all_E_D_and_memory_extractors_frozen=True,Exact_NULL=True,
            source_only_inference_unchanged=True,old_closed_budget_reuse=False,
            initial_CPU_test_cwd_import_error_then_wrong_exception_guard_fixed=True,focused_tests5_PASS=True,
            literature=[dict(title='Conditioning Matters (NeurIPS2025)',url='https://proceedings.nips.cc/paper_files/paper/2025/hash/ad6363efb7af02f7db13d087c7e649bd-Abstract-Conference.html',
                relevance='conditional loss collapse in robot action diffusion; NOT proof of this FDT cause or remedy'),
                dict(title='Your Diffusion Model is Secretly a Noise Classifier (NeurIPS2024)',url='https://papers.nips.cc/paper_files/paper/2024/hash/38d67d1df644cf2efe9ebd5521741dc5-Abstract-Conference.html',
                relevance='different contrastive noise-level objective; NOT this pair endpoint objective')]))
        print('PREPARED_NOT_LAUNCHED');return
    p=io.load_json(PLAN);validate(PLAN,OUT.resolve())
    pre=io.load_json(A/'cpu_preflight_owned/final_status.json')
    assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['worker_exit_code']==0
    r=pre['report'];assert r['atomic_pair_MID_and_partial6_READY_exact'] and r['old_baseline2actual_samples_bitwise'] and r['training_D0']
    assert r['thought_parameters']==9576426 and sum(r['NN_calls'].values())==388
    if mode=='review':
        import torch
        import text_generation_support_v1 as support
        for name in ('MID_after_complete_pair','READY_partial6'):
            file=A/'cpu_preflight_owned/runtime'/(name+'.pt');v=torch.load(file,map_location='cpu',weights_only=True)
            assert v['schema']=='answer_pair_contrast_fullstate_v112' and v['pair_coefficient']==.25 and v['optimizer']['state']
            assert v['ALL_RNG']['CUDA']==[] and any(x is not None for x in v['gradients'].values())
            assert v['runtime']['cursor'] in (10,14) and v['runtime']['updates']==1
        assert OWNER.read_text().split('        finally:',1)[1]==(P/'scripts/supervise_answer_meaning_v111_c.py').read_text().split('        finally:',1)[1]
        io.atomic_json(A/'root_independent_validate.json',dict(status='ROOT_ACTUAL_PAIR_V112_READY_NOT_LAUNCHED',actual_report=r,
            fullsize_atomic_MID_READY_populatedAdam_scaler_ALLRNG_gradients_verified=True,
            owner_all_postPopen_cleanup_identical=True,old_baseline_identity_verified=True,C_only_end100GiB=True,
            model_forwards_by_independent_review=0))
        print('ROOT_ACTUAL_PAIR_V112_READY');return
    assert mode=='seal' and io.load_json(A/'root_independent_validate.json')['status']=='ROOT_ACTUAL_PAIR_V112_READY_NOT_LAUNCHED'
    if shutil.disk_usage(P).free<p['resources']['C_start']:raise OSError('end100GiB plus output/temp headroom')
    for file in (A/'cpu_preflight_owned/final_status.json',A/'root_independent_validate.json',A/'root_math_and_scope_review.json'):
        p['files'][str(file)]=sha(file)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+1080,deadline=now+1200)
    OUT.mkdir(parents=True,exist_ok=False);io.atomic_json(OUT/'plan.json',p);validate(OUT/'plan.json',OUT.resolve())
    snap=OUT/'code_snapshot';snap.mkdir()
    for file in (WORKER,OWNER,Path(__file__).resolve(),P/'research/answer_pair_contrast_v112.py'):
        (snap/file.name).write_bytes(file.read_bytes())
    print(json.dumps(dict(status='SEALED_NOT_LAUNCHED',output=str(OUT),plan_SHA=sha(OUT/'plan.json'),start=now,cutoff=p['cutoff'],deadline=p['deadline'])))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','review','seal']);main(q.parse_args().mode)
