"""Independent saved-state review after the fixed final-reserve guard; forward0."""
import json,sys,statistics
from pathlib import Path
from collections import Counter
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from owt_fuzzy_engine_v23 import COMPONENTS,cumulative_count

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    torch.set_num_threads(2);run=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v23_final'
    p,end,latest=read(run/'plan.json'),read(run/'supervisor/final_status.json'),read(run/'latest.json')
    incident=read(run/'incident.json');native=read(run/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='FAILED' and end['exit_code']==2 and end['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert incident['error']=="TimeoutError('final evaluation reserve')" and not (run/'result.json').exists()
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    parent=torch.load(p['source'],map_location='cpu',weights_only=True)
    initial=torch.load(run/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    digest=support.state_sha({k:parent[k] for k in COMPONENTS})
    assert digest==support.state_sha({k:initial[k] for k in COMPONENTS}) and initial['ancestry']['history']==parent['history']
    source_scaler=dict(parent['scaler']);del parent
    states=[]
    for count,path in ((0,run/'import/checkpoints/state_0000.pt'),(4096,run/'model/checkpoints/state_4096.pt'),(7440,run/'emergency/checkpoints/state_7440.pt')):
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['schema']=='fdt_owt_fuzzy_AE_v23_fullstate' and s['attempts']==count and s['cursor']==8*count and not s['gradients']
        assert s['ancestry']==initial['ancestry'] and [h['rows'] for h in s['history']]==s['orders'][:count]
        trained={n for group in s['optimizer_parameter_names'] for n in group}
        assert all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in trained)
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        steps=Counter(int(item['step']) for item in s['optimizer']['state'].values());assert steps=={39624+s['updates']:63}
        scale,tracker=source_scaler['scale'],source_scaler['_growth_tracker']
        for h in s['history']:
            if h['AMP_skip']:scale*=source_scaler['backoff_factor'];tracker=0
            else:
                tracker+=1
                if tracker==source_scaler['growth_interval']:scale*=source_scaler['growth_factor'];tracker=0
            assert h['scale']==scale
        assert s['scaler']==dict(source_scaler,scale=scale,_growth_tracker=tracker)
        states.append({'attempt':count,'updates':s['updates'],'path':str(path),'sha256':support.sha(path),'algorithm_sha256':support.state_sha(s),'Adam_steps':dict(steps),'scaler':s['scaler']})
        if count==7440:state=s
        else:del s
    assert states[-1]['sha256']==latest['sha256'] and states[-1]['algorithm_sha256']==latest['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['skips'],state['cursor'])==(7440,7437,3,59520)
    assert cumulative_count(state,'attempts')==47076 and cumulative_count(state,'updates')==47061
    for epoch in range(2):assert sorted(i for b in state['orders'][epoch*4096:(epoch+1)*4096] for i in b)==list(range(32768))
    history=state['history'];features=read(Path(p['artifact'])/'features.json')['train']
    exposure=sum(len(features[i]['prompt'].encode('utf-8')) for h in history for i in h['rows'])
    old_rows=sum(i<16384 for h in history for i in h['rows'])
    assert state['exposure']['selected_rows']==state['exposure']['processed_rows']==59520
    assert exposure==state['exposure']['target_UTF8_bytes']==state['exposure']['eligible_prefix_bytes']
    for key in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):
        assert sum(h[key] for h in history)==state['exposure'][key]
    assert state['exposure']['decoder_supervised_symbol_occurrences']==2*(exposure+59520)
    assert all(abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5 for h in history)
    calls={'encode':7440,'denoise':0,'decoder':14880}
    assert calls==state['allocation_calls']==incident['new_calls'] and state['actual_calls']==incident['cumulative_calls']
    out={'status':'CLOSED_PARTIAL_TIME_RESERVE_ROOT_VERIFIED','actual_supervisor_status_preserved':'FAILED',
        'error':incident['error'],'model_numeric_failure_observed':False,'new_attempts':7440,'new_updates':7437,'AMP_skip_attempts':[h['attempt'] for h in history if h['AMP_skip']],
        'global_attempts':47076,'global_updates':47061,'unattempted_old_target_not_reused':752,'source_all_components_exact':True,'source_component_sha256':digest,
        'all_source_ancestry_and_RNG_retained':True,'frozen_reasoning_prefix_basis_unchanged':True,'all_model_and_Adam_finite':True,'fullstates':states,
        'exposure':state['exposure'],'old_document_rows':old_rows,'new_document_rows':59520-old_rows,
        'online_epoch_segments':[{k:statistics.mean(h[k] for h in history[i:i+4096]) for k in ('loss','clean_CE','corrupted_prefix_CE')} for i in (0,4096)],
        'new_calls':calls,'total_new_calls':22320,'closed_cap':28000,'unused_calls_not_reused':5680,
        'quality_assessment_performed':False,'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False}
    with (run/'root_partial_state_review.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=True))

if __name__=='__main__':main()
