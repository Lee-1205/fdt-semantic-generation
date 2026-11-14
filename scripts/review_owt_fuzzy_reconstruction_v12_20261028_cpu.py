"""Saved-result review only, no extra model forward or training gate."""
import json,math,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import text_generation_support_v1 as support
from review_text_language_v6_20261028_cpu import gram_stats

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def stop_count(rows):
    total=0
    for start in range(0,len(rows),4):
        total+=max((r['generated']['token_ids'].index(256)+1 if 256 in r['generated']['token_ids'] else 129) for r in rows[start:start+4])
    return total

def main():
    torch.set_num_threads(2);r=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v12'
    result=read(r/'result.json');end=read(r/'supervisor/final_status.json');native=read(r/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert support.sha(r/'plan.json')==end['plan_sha256']
    path=r/'model/checkpoints'/result['latest']['file'];assert support.sha(path)==result['latest']['sha256']
    state=torch.load(path,map_location='cpu',weights_only=True);assert support.state_sha(state)==result['latest']['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'])==(2048,2048,16384) and len(state['history'])==2048
    assert all(h['outcome']=='UPDATED' for h in state['history']) and not state['gradients']
    assert [h['rows'] for h in state['history']]==state['orders']
    for epoch in range(8):assert sorted(i for b in state['orders'][epoch*256:(epoch+1)*256] for i in b)==list(range(2048))
    initial=torch.load(r/'initial/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    trained=set(n for group in state['optimizer_parameter_names'] for n in group)
    frozen={n:v for n,v in state['model'].items() if n not in trained}
    assert all(torch.equal(v,initial['model'][n]) for n,v in frozen.items())
    assert not any(n.startswith(('core.time.','core.refiner.','core.velocity.','core.initializer.','core.slot_router.')) for n in trained)
    steps=Counter(int(v['step']) for v in state['optimizer']['state'].values());assert steps=={2048:len(trained)}
    assert state['scaler']['scale']==2048 and state['scaler']['_growth_tracker']==48
    assert state['exposure']==result['exposure']
    preserved={'frozen_reasoning_prefix_unchanged':True,'trained_tensors':len(trained),'Adam_steps':dict(steps),'scaler':state['scaler'],
        'anchor_RMS_change':float((state['model']['core.anchors']-initial['model']['core.anchors']).double().square().mean().sqrt()),
        'prior_RMS_change':float((state['model']['core.prior']-initial['model']['core.prior']).double().square().mean().sqrt())}
    del state,initial,frozen
    targets=read(ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12/targets.json')['dev']
    reconstructed={};fuzzy={}
    for phase in ('initial','final'):
        data=torch.load(r/(phase+'_logits.pt'),map_location='cpu',weights_only=True);ids=data['target_ids'];active=ids!=257
        for i,row in enumerate(targets):assert ids[i].tolist()==row['target_ids']+[257]*(129-len(row['target_ids']))
        reconstructed[phase]={}
        for mode in ('normal','swap','zero'):
            logits=data[mode].double();loss=logits.logsumexp(-1)-logits.gather(-1,ids[...,None]).squeeze(-1)
            nll=float(loss[active].mean());correct=int((logits.argmax(-1)[active]==ids[active]).sum())
            stored=read(r/(phase+'_metrics.json'))[mode]
            assert abs(nll-stored['NLL'])<.001 and abs(correct-stored['correct'])<30
            reconstructed[phase][mode]={'NLL':nll,'correct_from_saved_FP16_logits':correct,'symbols':int(active.sum()),
                'worker_FP32_correct':stored['correct'],'FP16_argmax_rounding_difference':correct-stored['correct']}
        mu=data['memberships'].double();assert (mu>=0).all() and (mu<=1).all() and ((mu>0).sum(-1)<=8).all()
        fuzzy[phase]={'coverage_anchors':int((mu.sum((0,1))>0).sum()),'mean_membership_mass':float(mu.sum(-1).mean()),
            'mean_active_anchors_per_view':float((mu>0).double().sum(-1).mean()),'state_view_coordinate_RMS':float(data['views'].double().square().mean().sqrt())}
        del data
    raw=read(r/'final_generation.json');generation={}
    for mode,rows in raw.items():
        stats=[]
        for x in rows:
            g=x['generated'];b=bytes.fromhex(g['raw_hex'])
            if g['utf8_valid']:assert b.decode('utf-8')==g['text']
            exact=g['well_formed'] and g['text']==x['source'];assert exact==x['reconstruction_exact']
            stats.append(gram_stats(g['text']))
        generation[mode]={'rows':len(rows),'exact':sum(r['reconstruction_exact'] for r in rows),'UTF8':sum(r['generated']['utf8_valid'] for r in rows),
            'EOS':sum(r['generated']['eos'] for r in rows),'unique_outputs':len({r['generated']['text'] for r in rows}),
            'repeated_4word_fraction_mean':sum(x['repeated_4gram_fraction'] for x in stats)/len(stats)}
    calls={'encode':2048+32+32+4,'denoise':0,'decoder':2048+96+96+sum(stop_count(rows) for rows in raw.values())}
    assert calls==result['actual_calls']
    final=result['final'];before=result['initial']
    out={'status':'OWT_FUZZY_AE_FIRST_CYCLE_COMPLETE','updates':2048,'epochs':8,'train_documents':512,'train_sentences':2048,
        'dev_documents':64,'dev_sentences':256,'positive_target_UTF8_exposure':1449624,'exposure_rows':16384,
        'plan_sha256':end['plan_sha256'],'final_fullstate_sha256':result['latest']['sha256'],'final_algorithm_sha256':result['latest']['algorithm_sha256'],
        'preservation':preserved,'saved_logits_reconstruction':reconstructed,'fuzzy_state_statistics':fuzzy,'generation':generation,
        'reported_effects':{'normal_NLL_reduction':before['normal']['NLL']-final['normal']['NLL'],
            'normal_byte_accuracy_gain_pp':100*(final['normal']['byte_accuracy']-before['normal']['byte_accuracy']),
            'final_swapped_state_NLL_cost':final['swap']['NLL']-final['normal']['NLL'],
            'final_zero_state_NLL_cost':final['zero']['NLL']-final['normal']['NLL']},
        'reconstructed_calls':calls,'total_actual_calls':sum(calls.values()),'seconds':result['seconds'],
        'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'reasoning_diffusion_training':False,'E2_final_accessed':False,
        'interpretation':'Information preservation and use of extracted fuzzy state improved under fixed corpus/decoder controls; free exact reconstruction remains0/16 and outputs are not yet faithful sentences. Early training, not a whole-model failure or proof of universal semantic extraction. Continue this separate AE stage before training output-meaning diffusion.'}
    with (r/'root_result.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=True))
if __name__=='__main__':main()
