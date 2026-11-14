"""One saved-result milestone review, not another model/quality gate."""
import json,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import text_generation_support_v1 as support
from review_text_language_v6_20261028_cpu import gram_stats

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def stop_count(rows):
    count=0
    for start in range(0,len(rows),4):
        ends=[]
        for r in rows[start:start+4]:
            ids=r['generated']['token_ids'];ends.append(ids.index(256)+1 if 256 in ids else 129)
        count+=max(ends)
    return count

def main():
    torch.set_num_threads(2);r=ROOT/'runs/text_decoder_alignment_20261028_v8';result=read(r/'result.json')
    osend=read(r/'supervisor/final_status.json');native=read(r/'code_snapshot/root_native_terminal_check.json')
    assert osend['status']=='COMPLETE' and osend['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    path=r/'model/checkpoints'/result['latest']['file'];assert support.sha(path)==result['latest']['sha256']
    state=torch.load(path,map_location='cpu',weights_only=True);assert support.state_sha(state)==result['latest']['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'])==(1636,1636,13084)
    assert all(h['phase']=='reconstruction' and h['span_UL']==0 and h['velocity_mse']==0 and h['outcome']=='UPDATED' for h in state['history'])
    assert len(state['history'])==1636 and [h['rows'] for h in state['history']]==state['curriculum']['batches']
    for epoch in range(4):assert sorted(i for b in state['curriculum']['batches'][epoch*409:(epoch+1)*409] for i in b)==list(range(3271))
    vp={k[3:]:v for k,v in state['model'].items() if k.startswith('vp.')}
    assert support.state_sha(vp)==state['phase_vp_sha']
    initial=torch.load(r/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,initial['model']['vp.'+k]) for k,v in vp.items())
    old_moments={};new_moments={}
    for gi,key in enumerate(('language','other')):
        for name,index in zip(state['optimizer_parameter_names'][key],state['optimizer']['param_groups'][gi]['params']):
            if name.startswith('vp.'):
                old_moments[name]=initial['optimizer']['state'][index];new_moments[name]=state['optimizer']['state'][index]
    assert support.state_sha(old_moments)==support.state_sha(new_moments)
    steps=Counter(int(s['step']) for s in state['optimizer']['state'].values());assert steps=={2862:34,1636:1,1226:40,1078:70}
    interval=initial['scaler']['growth_interval'];consumed=initial['scaler']['_growth_tracker']+1636
    assert state['scaler']['_growth_tracker']==consumed%interval
    assert state['scaler']['scale']==initial['scaler']['scale']*initial['scaler']['growth_factor']**(consumed//interval)
    assert not torch.count_nonzero(initial['model']['decoder.memory_positions.weight'])
    positions=state['model']['decoder.memory_positions.weight'].double()
    counts={'model_parameters':sum(v.numel() for n,v in state['model'].items() if n not in {'vp.codec','vp.frequencies'}),
        'decoder_trainable_parameters':sum(v.numel() for n,v in state['model'].items() if n.startswith('decoder.')),
        'learned_memory_position_RMS':float(positions.square().mean().sqrt()),'learned_memory_position_norm_mean':float(positions.norm(dim=-1).mean()),
        'frozen_VP_unchanged':True,'frozen_VP_Adam_moments_unchanged':True,'Adam_step_histogram':dict(steps),'scaler':state['scaler']}
    del state,initial,vp,old_moments,new_moments
    reference=read(r/'parent_reference_NLL.json');actual=read(r/'final_NLL.json')
    generations=read(r/'final_generation.json');gen={}
    for mode,rows in generations.items():
        stats=[]
        for row in rows:
            g=row['generated'];data=bytes.fromhex(g['raw_hex']);assert data.decode('utf-8')==g['text']
            stats.append(gram_stats(g['text']))
        gen[mode]={'rows':len(rows),'UTF8':sum(x['generated']['utf8_valid'] for x in rows),'EOS':sum(x['generated']['eos'] for x in rows),
            'unique_strings':len({x['generated']['text'] for x in rows}),
            'repeated_4gram_fraction_mean':sum(x['repeated_4gram_fraction'] for x in stats)/len(stats),
            'the_word_fraction_mean':sum(x['the_word_fraction'] for x in stats)/len(stats)}
    probes={}
    for phase in ('initial','final'):
        stored=torch.load(r/(phase+'_interface_logits.pt'),map_location='cpu',weights_only=True)
        aggregate={s:[0.,0,0] for s in ('0.0','0.15')}
        for t in stored:
            x=t['logits'].double();ids=t['target_ids'];active=ids!=257
            nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1);a=aggregate[str(t['sigma'])]
            a[0]+=float(nll[active].sum());a[1]+=int((x.argmax(-1)[active]==ids[active]).sum());a[2]+=int(active.sum())
        probes[phase]={'teacher_reconstruction_ORACLE_only':{s:{'NLL':a[0]/a[2],'accuracy':a[1]/a[2],'symbols':a[2]} for s,a in aggregate.items()},
            'authored_oracle_causal_exact':read(r/(phase+'_oracle_causal.json'))['exact']}
    inferred_calls={'encode':3272+8+111+4,'denoise':111*16+4*16,
        'decoder':3272+16+111+sum(stop_count(rows) for rows in generations.values())+sum(stop_count(read(r/(p+'_oracle_causal.json'))['rows']) for p in ('initial','final'))}
    assert inferred_calls==result['allocation_calls']
    output={'status':'V8_FIRST_INTERFACE_ARM_COMPLETE','span_UL_beta':0,'new_updates':1636,'global_updates':2862,'new_finite_epochs':4,
        'same_train_documents':512,'positive_target_byte_exposure':1538700,'negative_training_forward_or_gradient':False,
        'final_fullstate_sha256':result['latest']['sha256'],'final_algorithm_sha256':result['latest']['algorithm_sha256'],
        'old_component_import_digest':read(r/'import_review.json')['old_model_Adam_scaler_RNG_component_sha256'],
        'counts_and_VP_preservation':counts,'diagnostic_interface_ONLY':probes,
        'actual_predicted_VP_metrics':{'parent_NLL':reference['NLL'],'final_NLL':actual['NLL'],'parent_accuracy':reference['byte_accuracy'],'final_accuracy':actual['byte_accuracy'],
            'true_EOF_argmax':actual['true_EOF_correct'],'true_EOF_total':actual['true_EOF'],'generation':gen},
        'actual_calls_reconstructed':inferred_calls,'total_actual_calls':sum(inferred_calls.values()),'worker_seconds':result['seconds'],
        'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False,
        'interpretation':'Ordered-memory paired decoder learning improved held oracle teacher reconstruction and predicted-VP NLL. Oracle opposite-order exact remains0/8; faithful state realization not demonstrated. Actual free generation remains repetitive. No VP semantic completion/whole-model failure conclusion; beta0 isolates this first interface arm from penalty effects.'}
    with (r/'root_result.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=True))
if __name__=='__main__':main()
