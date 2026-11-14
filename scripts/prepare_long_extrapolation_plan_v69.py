"""ROOT new frozen length-extrapolation contract, actual worker validation."""
import argparse,ast,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_long_extrapolation_20261112_v69 import validate,PARENT_SHA,EXACT_POLICY

def main(seal):
    data=ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu';review=data/'execution_review';path=review/'plan_unsealed.json'
    if seal:
        p=json.loads(path.read_text());output=Path(p['output']);validate(path,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';state=json.loads(proof.read_text());r=state['report']
        assert state['status']=='CPU_PREFLIGHT_PASS' and state['exit']['all_owned_processes_gone'] and not state['owner_imported_Torch']
        assert r['fullsize_model_restore'] and r['all_parent_tensors_exact'] and r['all_old_Adam_exact'] and r['parent_scaler_ALLRNG_preserved']
        assert not r['CUDA_initialized'] and r['optimizer_steps']==r['trainable_parameters']==0 and r['actual_parent_forward']==2
        assert r['fullsize_CPU_native_stream_small_parity'] and r['linear_UTF8_FSM_reference_parity']
        independent=review/'root_independent_validate.json';v=json.loads(independent.read_text());assert v['status']=='ROOT_INDEPENDENT_V69_READY'
        p['files'][str(proof)]=support.sha(proof);p['files'][str(independent)]=support.sha(independent)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();sealed=output/'plan.json';sealed.write_text(json.dumps(p,indent=2)+'\n');validate(sealed,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        print(json.dumps(dict(plan_sha=support.sha(sealed),cost=528,maxcalls=560,streamchunks500cap512=True,blockwork4518cap4700=True,optimizer0=True,wall900reserve120=True)));return
    assert not path.exists();oldrun=ROOT/'runs/hard_fuzzy_cycle_20261112_v67';old=json.loads((oldrun/'plan.json').read_text());p=dict(old)
    features=json.loads((data/'features.json').read_text());targets=json.loads((data/'targets.json').read_text());manifest=json.loads((data/'manifest.json').read_text())
    assert len(features)==len(targets)==10 and sum(len(r['prompt'].encode()) for r in features)==1014489
    count=0
    pattern=re.compile(r'in episode (\d+), (\w+) moved (\d+) (.*?) from the (\w+) shelf to the (\w+) drawer (before|after) (\w+) closed the (\w+) gate, and the (\w+) box contained (\d+) (.*?) that (\w+) did not remove',re.I)
    for f,t in zip(features,targets):
        assert set(f)=={'id','prompt'} and f['id']==t['id'] and len(f['prompt'].encode())==t['actual_bytes']
        assert abs(t['actual_bytes']-t['requested_max_bytes'])<300
        if t['style']=='single_sentence':assert f['prompt'].count('.')==1 and '\n' not in f['prompt']
        matches=list(pattern.finditer(f['prompt']));assert len(matches)==len(t['events'])
        for m,e in zip(matches,t['events']):
            a=m.groups();assert (int(a[0]),a[1],int(a[2]),a[3],a[4],a[5],a[6],a[7],int(a[10]))==(e['episode'],e['actor'],e['quantity'],e['item'],e['source_color'],e['destination_color'],e['order'],e['checker'],e['reserve'])
            assert a[11]==e['item'] and a[12]==e['actor'] and e['negative_removal'] and m.start()==e['byte_start'] and m.end()==e['byte_end']
            count+=1
    assert count==5265 and manifest['lengths']==[len(r['prompt'].encode()) for r in features]
    # New authored literals, not a claim of graph or pretrained semantic novelty.
    checked=[]
    for name in ('research/artifacts/hard_fuzzy_cycle_20261112_v67_cpu_v2/train.json','research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json'):
        f=ROOT/name
        if f.exists():
            corpus=json.loads(f.read_text(encoding='utf-8-sig'));assert not any(r['prompt']==s['prompt'] for r in features for s in corpus);checked.append(str(f))
    proof=dict(status='ROOT_INDEPENDENT_V69_DATA_PASS',rows=10,events=count,source_UTF8_bytes=1014489,
        training_exposure_rows=0,all_event_positions_and_actor_quantity_direction_order_negation_checked=True,
        feature_target_separation=True,single_sentence_each_scale=True,checked_literal_sources=checked,
        UTF8_byte_symbols_NOT_BPE=True,new_data_correlated_NOT_independent_graphs=True,E2_registry_access=0,model_calls=0)
    data_proof=data/'root_data_review.json';assert not data_proof.exists();data_proof.write_text(json.dumps(proof,indent=2)+'\n')
    names=['research/streaming_fuzzy_codec_v69.py','research/fixed_utf8_readout_v69.py','scripts/test_streaming_fuzzy_codec_v69_cpu.py',
        'scripts/prepare_long_extrapolation_v69_cpu.py','scripts/prepare_long_extrapolation_plan_v69.py',
        'scripts/run_long_extrapolation_20261112_v69.py','scripts/supervise_long_extrapolation_v69.py','scripts/owned_cpu_preflight_v66.py']
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    parent=oldrun/'model/checkpoints/state_1024.pt';assert support.sha(parent)==PARENT_SHA
    p.update(schema='fdt_long_extrapolation_v69',root=str(ROOT),authorized=False,output=str(ROOT/'runs/long_extrapolation_20261112_v69'),data=str(data),
        cycle_source=str(parent),cycle_sha=PARENT_SHA,new_code_files=names,attempts=0,epochs=0,wall_seconds=900,start=None,cutoff=None,deadline=None,
        capacity=16384,test_capacity=262144,compute_chunk=2048,amp_dtype='float32',exact_policy=EXACT_POLICY,
        max_calls=560,cost=528,cost_detail=dict(encode=14,readout=14,stream_chunk=500,block_work=4518,denoise=0),stream_chunk_cap=512,block_work_cap=4700,
        parity_thresholds=dict(commit_relative_L2=.001,logits_relative_L2=.002,bank_relative_L2=.001),max_output_bytes=2*1024**3,
        purpose='human-requested beyond-trained-length English meaning reconstruction; same frozen V67 weights, continuous local-attention KV carry and entire-document banks',
        exposure=dict(eval_rows=10,parity_rows=2,eval_source_UTF8_bytes=1014489,training_rows=0),checkpoints=[],maximum_additional_STOP_checkpoint=0,
        limitations=manifest['limitations'],progress_stall_seconds=120,stop_exact_nonNULL=True)
    for obsolete in ('objective_report','objective','optimizer','baseline_run','cycle_algorithm_sha'):p.pop(obsolete,None)
    files=[ROOT/n for n in names]+[parent,data/'features.json',data/'targets.json',data/'parity_features.json',data/'manifest.json',data_proof,data/'root_CPU_contract.json']
    for f in files:p['files'][str(f)]=support.sha(f)
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();path.write_text(json.dumps(p,indent=2)+'\n');validate(path,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,data_review=proof,cost=528,max_calls=560,pins=len(p['files']))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
