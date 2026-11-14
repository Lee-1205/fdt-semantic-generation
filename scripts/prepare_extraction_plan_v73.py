"""Fixed new trial budget, enlarged held-out assessment before ONE seal."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_fuzzy_extraction_20261112_v73 import validate,PARENT_SHA,ALGORITHM_SHA,EXACT_POLICY
from fuzzy_extraction_engine_v73 import OBJECTIVE
from research.fuzzy_factual_objective_v70 import training_weights
def main(seal):
    d=ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2';e=ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text());output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';a=json.loads(proof.read_text());r=a['report']
        assert a['status']=='CPU_PREFLIGHT_PASS' and a['exit']['all_owned_processes_gone'] and not a['owner_imported_Torch']
        assert all(r[k] for k in ('fullsize_engine_restore','actual_teacher_partial_restore','actual_loss_gradient_upstream23','all_parent_tensors_exact','all_old_Adam_exact','fresh23Adam','long_wrapper_all_tensors_exact','fixed_readout7','Exact_entries0'))
        assert not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['actual_upstream_parameters']==14769664 and r['active_tensors']==23
        assert r['engine_schema']=='fdt_live_fuzzy_extraction_v73_fullstate' and r['objective_report']==p['objective_report']
        independent=review/'root_independent_validate.json';assert json.loads(independent.read_text(encoding='utf-8-sig'))['status']=='ROOT_INDEPENDENT_V73_READY'
        for f in (proof,independent):p['files'][str(f)]=support.sha(f)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n');validate(path,output);snap=output/'code_snapshot';snap.mkdir()
        for name in p['new_code_files']:
            file=ROOT/name;(snap/file.name).write_bytes(file.read_bytes())
        print(json.dumps(dict(plan_SHA=support.sha(path),cost=p['cost'],max_calls=p['max_calls'],wall=900,cutoff=780,reserve=120)));return
    assert not unsealed.exists();oldrun=ROOT/'runs/frozen_verbalizer_20261112_v72';old=json.loads((oldrun/'plan.json').read_text());p=dict(old);parent=oldrun/'model/checkpoints/state_0640.pt'
    names=['research/fuzzy_extraction_v73.py','scripts/fuzzy_extraction_engine_v73.py','scripts/run_fuzzy_extraction_20261112_v73.py','scripts/extraction_assess_v73.py',
        'scripts/supervise_fuzzy_extraction_v73.py','scripts/prepare_extraction_plan_v73.py','scripts/test_extraction_engine_v73_cpu.py','scripts/prepare_fuzzy_extraction_v73_cpu.py',
        'scripts/review_fuzzy_extraction_v73_cpu.py','scripts/prepare_extraction_eval_v73_cpu.py','scripts/review_extraction_eval_v73_cpu.py']
    rows=json.loads((d/'train_features.json').read_text());_,obj=training_weights(rows);obj.update(objective=OBJECTIVE,anchor_coefficient=.05,LR_groups=[1e-6,3e-6,3e-6])
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    p.update(schema='fdt_live_extraction_v73',authorized=False,output=str(ROOT/'runs/fuzzy_extraction_20261112_v73'),data=str(d),eval_data=str(e),
        cycle_source=str(parent),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,baseline_run=str(oldrun),old_expression_data=str(ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu'),
        start=None,cutoff=None,deadline=None,attempts=960,epochs=1,batch=1,wall_seconds=900,max_calls=5600,cost=5346,objective=OBJECTIVE,objective_report=obj,exact_policy=EXACT_POLICY,
        cost_detail=dict(teacher960_encode_readout=1920,live960_encode_readout=1920,baseline128_encode_readout=256,final244_short_encode_readout=488,final10_long_encode_readout=20,
            final_long_chunks736=736,execution_parity_encode_readout4_plus_chunks2=6,encode_total=2304,readout_total=2304,stream_chunks_total=738,block_work=6642,denoise=0),
        phase_model_reserves_seconds=dict(teacher_before_cutoff=360,live_before_cutoff=180,outer_shutdown=120),
        checkpoints=['initial/0','teacher_complete/0','model/1','model/480','model/960'],maximum_additional_STOP_checkpoint=1,new_code_files=names,
        source_encoder_optimizer_updates='bounded<=960; actual ledger authoritative',source_encoder_optimizer_policy='fresh upstream23Adam; final readout7 fixed; live source encoding every step',
        optimizer='fresh upstream23AdamW last2language1e-6/valueconditionrole3e-6 WD.01 clip1 BF16 enabledscaler1024; source7Adam+allRNG preserved',
        exposure=dict(teacher960_rows=True,live960_rows=True,one_finite_epoch=True,TRAIN_source_UTF8_bytes=1392317,critical_bytes=39730,EOS_targets=960),
        max_output_bytes=6*1024**3,stream_chunk_cap=738,long_block_work_cap=6642,
        purpose='fix language readout, improve actual fuzzy extraction/binding, expand never-TRAIN English and beyondTRAIN-length assessment per latest human request',
        baseline_contract='reuse V72 existing observed panels; fresh pretraining source128 baseline; final244short+10long all live source; no cached Commit substitutes changing encoder',
        cache_contract='ONLY initial TRAIN loss targets/probabilities/flags cached; no Commit or source-H training features; every live step encodes actual raw source',
        limitations='26TRAIN names/newheld names and authored minimal pairs share families; new96short/new8long are untrained created inputs not blind natural benchmarks; oldprepared32/oldlong2 and retained panels already observed; 7835 long events correlated; no abstract-meaning/losslessness or diffusion-completion claim')
    files=[ROOT/n for n in names]+[parent,oldrun/'result.json',oldrun/'root_result.json',oldrun/'root_saved_review.json',Path(p['retention_OWT']),ROOT/'scripts/owned_cpu_preflight_v50.py']
    files.extend(oldrun/n for n in ('OWT_reconstructions.json','retained_English_reconstructions.json','prior_English_reconstructions.json','new_expression_reconstructions.json','long_reconstructions.json'))
    files.extend(d/n for n in ('train_features.json','train_targets.json','dev_features.json','dev_targets.json','orders.json','coverage.json','manifest.json','root_data_review.json','root_candidate_review.json','root_CPU_summary.json','root_engine_contract.json'))
    files.extend(e/n for n in ('short_features.json','short_targets.json','long_features.json','long_targets.json','manifest.json','root_data_review.json'))
    files.extend(Path(p['parent_data'])/n for n in ('retained_English.json','dev_features.json','dev_targets.json','manifest.json'))
    files.append(Path(p['old_expression_data'])/'dev_features.json')
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(parent)==PARENT_SHA
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    # Bound storage from actual row lengths, all states plus one STOP and64MiB.
    def rawbytes(r):
        slots=(len(r['prompt'].encode())+15)//16;return (slots+1)*16*(258*4+8)+(slots+2)*512*4+32768
    held=json.loads((e/'short_features.json').read_text());long=json.loads((e/'long_features.json').read_text())
    olds=json.loads(Path(p['retention_OWT']).read_text())+json.loads((Path(p['parent_data'])/'retained_English.json').read_text())+json.loads((Path(p['parent_data'])/'dev_features.json').read_text())+json.loads((Path(p['old_expression_data'])/'dev_features.json').read_text())
    raw=sum(rawbytes(r) for r in held+olds+held+long);model=123497080*4;adam=14769664*8
    teacher=sum(((len(r['prompt'].encode())+15)//16+1)*16*14+16384 for r in rows)
    estimate=raw+5*model+3*adam+(model+adam)+teacher+5*4*1024**2+16*1024**2+64*1024**2
    assert estimate<p['max_output_bytes']
    p['output_estimate_including_additional_STOP_and64MiB']=estimate
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n');validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),cost=5346,max_calls=5600,output_estimate=estimate,output_cap=p['max_output_bytes'],held128_long10=True,objective=obj)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');main(a.parse_args().seal)
