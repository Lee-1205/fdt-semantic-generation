"""One independent saved-state/raw review; no extra model forward."""
import json,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from owt_fuzzy_engine_v17 import COMPONENTS,cumulative_count
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_text_language_v6_20261028_cpu import gram_stats

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+int(x!=y)))
        row=nxt
    return row[-1]

def main():
    torch.set_num_threads(2);r=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v17';p=read(r/'plan.json');result=read(r/'result.json')
    end=read(r/'supervisor/final_status.json');native=read(r/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert support.sha(r/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    path=r/'model/checkpoints'/result['latest']['file'];assert support.sha(path)==result['latest']['sha256']
    state=torch.load(path,map_location='cpu',weights_only=True);assert support.state_sha(state)==result['latest']['algorithm_sha256']
    initial=torch.load(r/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    parent=torch.load(p['source'],map_location='cpu',weights_only=True)
    before=support.state_sha({k:parent[k] for k in COMPONENTS});after=support.state_sha({k:initial[k] for k in COMPONENTS})
    assert before==after==read(r/'import_review.json')['all_algorithm_components_sha256']
    assert (initial['attempts'],initial['updates'],initial['cursor'])==(0,0,0) and initial['ancestry']['history']==parent['history']
    assert (state['attempts'],state['updates'],state['cursor'])==(2048,2048,16384)
    assert [h['rows'] for h in state['history']]==state['orders'] and all(h['outcome']=='UPDATED' for h in state['history'])
    for epoch in range(2):assert sorted(i for b in state['orders'][epoch*1024:(epoch+1)*1024] for i in b)==list(range(8192))
    trained={n for group in state['optimizer_parameter_names'] for n in group}
    assert all(torch.equal(v,initial['model'][n]) for n,v in state['model'].items() if n not in trained)
    assert not any(n.startswith(('core.refiner.','core.velocity.','core.time.')) for n in trained)
    steps=Counter(int(v['step']) for v in state['optimizer']['state'].values());assert steps=={21200:63}
    assert state['scaler']['scale']==131072 and state['scaler']['_growth_tracker']==48 and not state['gradients']
    assert state['exposure']==result['exposure'] and state['exposure']['target_UTF8_bytes']==p['training_exposure']['target_UTF8_bytes']
    assert state['allocation_calls']==result['actual_calls'] and state['actual_calls']==result['cumulative_calls']
    assert cumulative_count(state,'updates')==21200 and cumulative_count(parent,'updates')==19152
    training_loss=[sum(h['loss'] for h in state['history'][i:i+1024])/1024 for i in range(0,2048,1024)]
    parent_last_epoch_mean=sum(h['loss'] for h in parent['history'][-100:])/100
    preservation={'source_all_algorithm_components_preserved':True,'imported_component_sha256':before,
        'frozen_reasoning_and_prefix_unchanged':True,'Adam_steps':dict(steps),'scaler':state['scaler'],
        'source_history_orders_exposure_retained':True}
    del state,initial,parent
    targets=read(Path(p['artifact'])/'targets.json')['dev'];data=torch.load(r/'final_logits.pt',map_location='cpu',weights_only=True)
    ids=data['target_ids'];active=ids!=257;metrics={}
    for i,y in enumerate(targets):assert ids[i].tolist()==y['target_ids']+[257]*(129-len(y['target_ids']))
    for mode in ('normal','swap','zero'):
        x=data[mode].double();nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1)
        value=float(nll[active].mean());correct=int((x.argmax(-1)[active]==ids[active]).sum())
        assert abs(value-result['final'][mode]['NLL'])<.001 and abs(correct-result['final'][mode]['correct'])<30
        metrics[mode]={'NLL_from_saved_logits':value,'byte_accuracy':correct/int(active.sum()),'correct':correct,'symbols':int(active.sum()),
            'FP16_argmax_rounding_difference':correct-result['final'][mode]['correct']}
    mu=data['memberships'].double();assert (mu>=0).all() and (mu<=1).all() and ((mu>0).sum(-1)<=8).all()
    fuzzy={'coverage_anchors':int((mu.sum((0,1))>0).sum()),'mean_membership_mass':float(mu.sum(-1).mean()),
        'mean_active_anchors_per_view':float((mu>0).double().sum(-1).mean()),'state_view_coordinate_RMS':float(data['views'].double().square().mean().sqrt())}
    del data
    raw=read(r/'final_generation.json');old=read(r/'parent_reference_generation.json');gen={}
    for mode,rows in raw.items():
        stats=[];errors=[];old_errors=[]
        for row,previous in zip(rows,old[mode]):
            assert row['id']==previous['id'] and row['source']==previous['source']
            g=row['generated'];b=bytes.fromhex(g['raw_hex']);source=row['source'].encode('utf-8')
            if g['utf8_valid']:assert b.decode('utf-8')==g['text']
            assert row['reconstruction_exact']==(g['well_formed'] and g['text']==row['source'])
            stats.append(gram_stats(g['text']));errors.append(edit_distance(source,b)/len(source))
            old_errors.append(edit_distance(source,bytes.fromhex(previous['generated']['raw_hex']))/len(source))
        gen[mode]={'rows':len(rows),'exact':sum(x['reconstruction_exact'] for x in rows),'UTF8':sum(x['generated']['utf8_valid'] for x in rows),
            'EOS':sum(x['generated']['eos'] for x in rows),'unique_outputs':len({x['generated']['text'] for x in rows}),
            'repeated_4word_fraction_mean':sum(x['repeated_4gram_fraction'] for x in stats)/len(stats),
            'posthoc_byte_edit_error_mean':sum(errors)/len(errors),'parent_posthoc_byte_edit_error_mean':sum(old_errors)/len(old_errors)}
    calls={'encode':2048+32+4,'denoise':0,'decoder':2048+96+sum(stop_count(rows) for rows in raw.values())}
    assert calls==result['actual_calls']
    out={'status':'OWT_FUZZY_AE_PERSISTENT_AMP_GUARD_FIX_CONTINUATION_COMPLETE','new_updates':2048,'global_AE_updates':21200,
        'training_documents':2048,'training_sentences':8192,'new_training_documents':0,'observed_held_documents_unchanged':64,
        'new_target_UTF8_exposure':1457826,'old1024doc_target_exposure':728356,'other1024doc_target_exposure':729470,
        'recorded_online_training_batch_CE_mean_by_epoch':training_loss,'source_last100_online_batch_CE_mean':parent_last_epoch_mean,
        'plan_sha256':end['plan_sha256'],'final_fullstate_sha256':result['latest']['sha256'],'final_algorithm_sha256':result['latest']['algorithm_sha256'],
        'preservation':preservation,'saved_raw_metrics':metrics,'fuzzy_state_statistics':fuzzy,'free_reconstruction':gen,
        'effects':{'byte_accuracy_gain_pp':100*(result['final']['normal']['byte_accuracy']-result['parent_reference']['normal']['byte_accuracy']),
            'normal_NLL_reduction':result['parent_reference']['normal']['NLL']-result['final']['normal']['NLL'],
            'swap_state_NLL_cost':result['final']['swap']['NLL']-result['final']['normal']['NLL'],
            'zero_state_NLL_cost':result['final']['zero']['NLL']-result['final']['normal']['NLL']},
        'reconstructed_new_calls':calls,'total_new_calls':sum(calls.values()),'seconds':result['seconds'],
        'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'reasoning_diffusion_training':False,'E2_final_accessed':False,
        'interpretation':'Same fuzzy-anchor AE; exact V16emergency import with new persistent AMP guard and two finite epochs. V16 partial6864successful updates and V17new2048 are both reflected versus V15reference; not isolated guard effect. Teacher accuracy roughlyunchanged and NLLregresses; free exact0/16/UTF8EOS16/16. All previous failures and useful states preserved, reasoning remains frozen.'}
    with (r/'root_result.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=True))
if __name__=='__main__':main()
