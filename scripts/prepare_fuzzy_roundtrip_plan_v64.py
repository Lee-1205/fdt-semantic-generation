"""Actual cycle schema; strict dormant Exact; fresh bounded execution only."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_fuzzy_roundtrip_20261112_v64 import validate,SOURCE_SHA,HEAD_SHA,EXACT_POLICY

def main(seal):
    data=ROOT/'research/artifacts/fuzzy_roundtrip_20261112_v64_cpu';review=data/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text(encoding='utf8'));output=Path(p['output']);validate(unsealed,output)
        if output.exists():raise FileExistsError('ONE fresh output')
        proof=review/'cpu_preflight_owned/final_status.json';r=json.loads(proof.read_text(encoding='utf8'));report=r['report']
        if (r['status']!='CPU_PREFLIGHT_PASS' or not r['exit']['all_owned_processes_gone'] or r['owner_imported_Torch']
            or not report['fullsize_engine_restore'] or not report['actual_warm_size_roundtrip_gradient']
            or report['parent_Adam_used_for_training'] or not report['fresh_joint_optimizer']
            or not report['Exact_entries0'] or report['CUDA_initialized'] or report['optimizer_steps']!=0
            or report['exact_policy']!=EXACT_POLICY or report['engine_schema']!='fdt_live_fuzzy_roundtrip_v64_fullstate'):
            raise ValueError('actual cycle worker CPU proof')
        p['files'][str(proof)]=support.sha(proof);p['actual_trainable_parameters']=report['actual_trainable_parameters']
        now=time.time();p.update(authorized=True,start=now,cutoff=now+1020,deadline=now+1200)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in ('research/fuzzy_roundtrip_v64.py','scripts/fuzzy_roundtrip_engine_v64.py','scripts/run_fuzzy_roundtrip_20261112_v64.py',
            'scripts/supervise_fuzzy_roundtrip_v64.py','scripts/prepare_fuzzy_roundtrip_plan_v64.py','scripts/review_fuzzy_roundtrip_data_v64_cpu.py'):
            source=ROOT/name;(snapshots/source.name).write_bytes(source.read_bytes())
        result=dict(plan_sha=support.sha(path),actual_trainable_parameters=p['actual_trainable_parameters'],cost=580,cap=640,
            wall_seconds=1200,reserve=180,Exact_policy=EXACT_POLICY,live_encoder_and_decoder=True,fresh_baseline_required=True)
        (output/'root_prelaunch_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result));return
    if unsealed.exists():raise FileExistsError('immutable prospective plan')
    source=ROOT/'runs/prefix_independent_codec_20261111_v53/model/checkpoints/state_8192.pt'
    head=ROOT/'runs/broad_ordered_20261111_v57/model/checkpoints/state_8192.pt'
    output=ROOT/'runs/fuzzy_roundtrip_20261112_v64';pins={}
    for oldrun in ('broad_ordered_20261111_v57','semantic_binding_20261112_v62'):
        old=json.loads((ROOT/'runs'/oldrun/'plan.json').read_text(encoding='utf8'))
        for name,digest in old['files'].items():
            if name.endswith('.py'):pins[name]=digest
    names=('research/fuzzy_roundtrip_v64.py','scripts/fuzzy_roundtrip_engine_v64.py','scripts/run_fuzzy_roundtrip_20261112_v64.py',
        'scripts/supervise_fuzzy_roundtrip_v64.py','scripts/test_fuzzy_roundtrip_v64_cpu.py','scripts/test_fuzzy_roundtrip_engine_v64_cpu.py',
        'scripts/prepare_fuzzy_roundtrip_data_v64_cpu.py','scripts/review_fuzzy_roundtrip_data_v64_cpu.py')
    files=[ROOT/name for name in names]+[Path(__file__),source,head]
    files.extend(data/name for name in ('train.json','dev.json','orders.json','manifest.json','root_data_review.json','root_engine_contract.json','root_boundary_gradient_contract.json'))
    for file in files:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_fuzzy_roundtrip_v64',authorized=False,root=str(ROOT),output=str(output),data=str(data),files=pins,
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=HEAD_SHA,exact_policy=EXACT_POLICY,
        capacity=16384,batch=1,attempts=256,epochs=2,amp_dtype='bfloat16',wall_seconds=1200,start=None,cutoff=None,deadline=None,
        max_calls=640,cost=580,cost_detail=dict(train_encode_readout=512,baseline_encode_readout=32,final_encode_readout=32,zero_fuzzy_readout=4),
        max_output_bytes=4*1024**3,checkpoints=[0,1,128,256],maximum_additional_STOP_checkpoint=1,
        purpose='actual same-input language -> fuzzy Commit -> language; remove automatic Exact content and jointly train extraction and realization',
        objective='uniform active byte+EOS CE + .1 EOS CE; target only loss, no target prefix/length in language-reader features',
        optimizer='FRESH AdamW: last2language1e-6,value+condition3e-6,readout1e-4; WD.01 clip1 BF16 GradScaler1024; parent96+8Adam unchanged and unused',
        exposure=dict(TRAIN_rows=128,unique_source_documents=124,row_selections=256,target_UTF8_bytes=1117008,EOS_targets=256),
        baseline_contract='fresh policy-compliant baseline mandatory; old automatic-Exact results pre-policy and not current baseline',
        recovery='full encoder+readout/frozenhash/activeAdam+ownership/scaler/ALLRNG/order/cursor/history/phase/exposure/calls/evalrefs/parentrefs/Exactpolicy',
        input_window_EOF_not_original_document_EOF=True,denoise_calls=0,Exact_content_entries=0,source_copy_bytes=0,
        certified_prior_writer=False,automatic_promotion=False,
        limitations='already observed OWT DEV; fuzzy structural codec, not unique abstract meaning proof; no reasoner training; no explicit copy-request API yet',
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,owner_stop_free_bytes=104*1024**3,protected_free_bytes=100*1024**3,
        GPU_peak_cap_bytes=12*1024**3)
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8');validate(unsealed,output)
    print(json.dumps(dict(actual_worker_schema_PASS=True,cost=580,max_calls=640,Exact_policy=EXACT_POLICY,pins=len(pins))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
