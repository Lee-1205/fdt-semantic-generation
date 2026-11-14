"""Independent saved-state review of V34's bounded partial broad epoch."""
import json,statistics,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def total(s,key):
    n=0
    while key in s:n+=s[key];s=s.get('ancestry',{})
    return n

def main():
    torch.set_num_threads(2);run=R/'runs/three_memory_codec_20261029_v34';p=read(run/'plan.json')
    end,native,incident=[read(run/name) for name in ('supervisor/final_status.json','code_snapshot/root_terminal_check_20261029_1838.json','incident.json')]
    assert end['status']=='FAILED' and end['exit_code']==2 and end['exit']['all_owned_processes_gone'] and native['native_count']==0
    assert incident['error']=="TimeoutError('final assessment reserve')" and not (run/'result.json').exists()
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    assert support.core_sha(R)==support.CORE_SHA and all(support.sha(path)==sha for path,sha in p['files'].items())
    parent=torch.load(p['source'],weights_only=True,map_location='cpu');initial=torch.load(run/'import/checkpoints/state_0000.pt',weights_only=True,map_location='cpu')
    assert support.state_sha(parent)==p['source_algorithm']
    parts=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    digest=support.state_sha({k:parent[k] for k in parts})
    assert digest==support.state_sha({k:initial[k] for k in parts})==initial['continuation']['imported_component_sha256']
    assert initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0} and not initial['evaluations']
    assert len(initial['optimizer']['state'])==91 and initial['optimizer_parameter_names']==parent['optimizer_parameter_names']
    for k in ('history','orders','exposure','evaluations','actual_calls','ancestry'):
        assert support.state_sha(initial['ancestry'][k])==support.state_sha(parent[k])
    steps={i:int(v['step']) for i,v in parent['optimizer']['state'].items()};scaler=dict(parent['scaler']);source_calls=dict(parent['actual_calls']);del parent
    trained={n for g in initial['optimizer_parameter_names'] for n in g};full=[]
    for attempt,path in ((0,run/'import/checkpoints/state_0000.pt'),(2048,run/'model/checkpoints/state_2048.pt'),(3407,run/'emergency/checkpoints/state_3407.pt')):
        s=torch.load(path,weights_only=True,map_location='cpu');mark=read(path.with_suffix('.json'));sha=support.sha(path);alg=support.state_sha(s)
        assert sha==mark['sha256'] and alg==mark['algorithm_sha256'] and s['schema']=='fdt_three_memory_language_codec_v34_fullstate'
        assert s['attempts']==attempt and s['cursor']==8*attempt and s['updates']==attempt-s['skips'] and not s['gradients'] and not s['evaluations']
        assert s['frozen_sha']==initial['frozen_sha'] and s['continuation']==initial['continuation'] and s['migration']==initial['migration'] and s['ancestry']==initial['ancestry']
        assert [r['rows'] for r in s['history']]==s['orders'][:attempt] and len(s['history'])==attempt
        assert s['memory_epoch']==end['plan_sha256']+f":attempt{attempt}:updates{s['updates']}"
        assert all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in trained)
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        for i,v in s['optimizer']['state'].items():assert int(v['step'])==steps[i]+s['updates']
        scale,tracker=scaler['scale'],scaler['_growth_tracker'];skips=[]
        for i,rec in enumerate(s['history']):
            assert rec['attempt']==i+1 and rec['mask_probability']==.5 and rec['denoise_calls']==rec['copy_bytes']==0
            assert abs(rec['loss']-.5*(rec['clean_CE']+rec['corrupted_prefix_CE']))<1e-5
            if rec['AMP_skip']:scale*=scaler['backoff_factor'];tracker=0;skips.append(i+1)
            else:
                tracker+=1
                if tracker==scaler['growth_interval']:scale*=scaler['growth_factor'];tracker=0
            assert rec['scale']==scale
        assert s['scaler']==dict(scaler,scale=scale,_growth_tracker=tracker)
        expected={'encode':attempt,'denoise':0,'decoder':2*attempt}
        assert s['allocation_calls']==expected and s['actual_calls']=={k:source_calls[k]+expected[k] for k in expected}
        full.append({'attempt':attempt,'updates':s['updates'],'AMP_skips':skips,'bytes':path.stat().st_size,'sha256':sha,
            'algorithm_sha256':alg,'stage':s['worker_stage'],'cursor':s['cursor'],'scaler':s['scaler']})
        if attempt==3407:state=s
        else:del s
    assert state['attempts']==3407 and state['updates']==3406 and state['skips']==1 and state['cursor']==27256
    flattened=[i for batch in state['orders'] for i in batch]
    assert len(state['orders'])==4096 and sorted(flattened)==list(range(32768))
    seen=flattened[:27256];remaining=flattened[27256:]
    assert len(seen)==len(set(seen))==27256 and len(remaining)==5512 and set(seen).isdisjoint(remaining)
    target=read(Path(p['artifact'])/'targets.json')['train']
    exposed=sum(len(target[i]['target'].encode('utf-8')) for i in seen)
    assert state['exposure']['selected_rows']==state['exposure']['processed_rows']==27256
    assert state['exposure']['target_UTF8_bytes']==state['exposure']['selected_target_UTF8_bytes']==exposed
    for key in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):
        assert state['exposure'][key]==sum(h[key] for h in state['history'])
    remaining_bytes=sum(len(target[i]['target'].encode('utf-8')) for i in remaining)
    assert exposed+remaining_bytes==2926478
    assert total(state,'attempts')==107827 and total(state,'updates')==107789
    assert state['external_closed_assessment_calls']==p['external_closed_assessment_calls'] and state['external_closed_assessment_calls']['total']==1936
    online=[]
    for start in (0,1024,2048,3072):
        rows=state['history'][start:min(start+1024,3407)]
        online.append({'attempts':[start+1,start+len(rows)],**{k:statistics.mean(r[k] for r in rows) for k in ('loss','clean_CE','corrupted_prefix_CE')}})
    out={'status':'V34_BOUNDED_PARTIAL_EPOCH_ROOT_REVIEWED','supervisor_status_preserved':'FAILED','incident':incident['error'],
        'stop_cause':'Fixed final-assessment time reserve; no numerical training error/owner failure is recorded.',
        'new_attempts':3407,'new_updates':3406,'AMP_skips':[h['attempt'] for h in state['history'] if h['AMP_skip']],
        'global_attempts':107827,'global_updates':107789,'cursor':27256,'training_rows_seen_once':27256,
        'training_docs_seen':len({target[i]['document_id'] for i in seen}),'target_UTF8_bytes_exposed':exposed,
        'remaining_finite_batches':689,'remaining_training_rows':5512,'remaining_target_UTF8_bytes':remaining_bytes,
        'one_epoch_total_rows':32768,'one_epoch_total_target_UTF8_bytes':2926478,'source_component_sha256':digest,
        'all91_populated_Adam_model_scaler_ALLstoredRNG_import_exact':True,'all_frozen_original_parameters_unchanged':True,
        'all_model_and_Adam_finite':True,'fullstates':full,'online_loss_blocks':online,'exposure':state['exposure'],
        'actual_calls':state['allocation_calls'],'total_new_calls':10221,'cumulative_calls':state['actual_calls'],
        'external_closed_assessment_calls':state['external_closed_assessment_calls'],
        'new_assessment_or_reconstruction_outputs':False,'performance_improvement_claimed':False,
        'plan_sha256':end['plan_sha256'],'all_owned_proxy_gone_before_analysis':True,'ROOT_worker_live_observation':True,
        'additional_model_construction_forward':0,'CUDA_replay_performed':False,'denoise_calls':0,'literal_copy_bytes':0,'E2_final_accessed':False,
        'next_purpose':'Separate new bounded completion of remaining689 finite batches and final SAME-input assessment; preserve exact full order and schedule position. Never reopen/extend old allocation or repeat consumed rows.'}
    write(run/'root_result.json',out)
    print(json.dumps(out,ensure_ascii=True))
if __name__=='__main__':main()
