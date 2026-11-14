"""Reconstruct the completed V9 evidence from saved bytes, with no model forward."""
import json, sys
from collections import Counter
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
import text_generation_support_v1 as support
from review_text_language_v6_20261028_cpu import gram_stats
from review_text_decoder_alignment_v8_20261028_cpu import stop_count

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    torch.set_num_threads(2)
    r=ROOT/'runs/text_decoder_addressing_20261028_v9'
    result=read(r/'result.json'); end=read(r/'supervisor/final_status.json')
    native=read(r/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and not native['native']
    assert support.sha(r/'plan.json')==end['plan_sha256']
    final=r/'model/checkpoints'/result['latest']['file']
    assert support.sha(final)==result['latest']['sha256']
    state=torch.load(final,map_location='cpu',weights_only=True)
    assert support.state_sha(state)==result['latest']['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'])==(1636,1636,13084)
    assert len(state['history'])==1636
    assert all(h['phase']=='reconstruction' and h['span_UL']==0 and h['velocity_mse']==0 and h['outcome']=='UPDATED' for h in state['history'])
    assert [h['rows'] for h in state['history']]==state['curriculum']['batches']
    for epoch in range(4):
        assert sorted(i for b in state['curriculum']['batches'][409*epoch:409*(epoch+1)] for i in b)==list(range(3271))
    initial=torch.load(r/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    vp={k:v for k,v in state['model'].items() if k.startswith('vp.')}
    assert all(torch.equal(v,initial['model'][k]) for k,v in vp.items())
    old_moments={}; new_moments={}
    for gi,key in enumerate(('language','other')):
        for name,index in zip(state['optimizer_parameter_names'][key],state['optimizer']['param_groups'][gi]['params']):
            if name.startswith('vp.'):
                old_moments[name]=initial['optimizer']['state'][index]
                new_moments[name]=state['optimizer']['state'][index]
    assert support.state_sha(old_moments)==support.state_sha(new_moments)
    steps=Counter(int(s['step']) for s in state['optimizer']['state'].values())
    assert steps=={4498:34,3272:1,1226:40,1078:70}
    consumed=initial['scaler']['_growth_tracker']+1636; interval=initial['scaler']['growth_interval']
    assert state['scaler']['_growth_tracker']==consumed%interval
    assert state['scaler']['scale']==initial['scaler']['scale']*initial['scaler']['growth_factor']**(consumed//interval)
    preserved={'VP_model_unchanged':True,'VP_Adam_unchanged':True,'populated_Adam_states':len(state['optimizer']['state']),
        'Adam_steps':dict(steps),'scaler':state['scaler'],'frozen_VP_digest':support.state_sha(vp)}
    del state,initial,vp,old_moments,new_moments
    probes={}
    for phase in ('initial','final'):
        rows=torch.load(r/(phase+'_interface_logits.pt'),map_location='cpu',weights_only=True)
        aggregates={s:[0.,0,0] for s in ('0.0','0.15')}
        for row in rows:
            logits=row['logits'].double(); ids=row['target_ids']; active=ids!=257
            nll=logits.logsumexp(-1)-logits.gather(-1,ids[...,None]).squeeze(-1)
            a=aggregates[str(row['sigma'])]
            a[0]+=float(nll[active].sum());a[1]+=int((logits.argmax(-1)[active]==ids[active]).sum());a[2]+=int(active.sum())
        oracle=read(r/(phase+'_oracle_causal.json'))
        exact=sum(x['generated']['text']==x['expected'] and x['generated']['eos'] for x in oracle['rows'])
        assert exact==oracle['exact']
        probes[phase]={'ORACLE_teacher_reconstruction':{s:{'NLL':a[0]/a[2],'accuracy':a[1]/a[2],'correct':a[1],'symbols':a[2]} for s,a in aggregates.items()},
            'ORACLE_opposite_order_exact':exact,'ORACLE_rows':len(oracle['rows'])}
    raw=torch.load(r/'final_NLL_logits.pt',map_location='cpu',weights_only=True)
    logits=raw['logits'].double(); ids=raw['target_ids'];active=ids!=257
    nll=logits.logsumexp(-1)-logits.gather(-1,ids[...,None]).squeeze(-1)
    metrics={'symbols':int(active.sum()),'NLL':float(nll[active].mean()),
        'byte_accuracy':float((logits.argmax(-1)[active]==ids[active]).double().mean()),
        'true_EOF':int((ids==256).sum()),'true_EOF_correct':int(((ids==256)&(logits.argmax(-1)==256)).sum())}
    actual=read(r/'final_NLL.json')
    assert metrics['symbols']==actual['symbols'] and abs(metrics['NLL']-actual['NLL'])<1e-6
    assert abs(metrics['byte_accuracy']-actual['byte_accuracy'])<1e-12
    corpus=read(ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu/targets.json')['dev']
    # Raw ID tensors are also checked against the original held target bytes.
    for index,row in enumerate(corpus):
        target=row['target_ids']
        assert target==list(row['target'].encode('utf-8'))+([256] if row['source_eof'] else [])
        assert ids[index].tolist()==target+[257]*(129-len(target))
    generations=read(r/'final_generation.json'); gen={}
    for mode,rows in generations.items():
        stats=[]
        for row in rows:
            g=row['generated']; assert bytes.fromhex(g['raw_hex']).decode('utf-8')==g['text']
            stats.append(gram_stats(g['text']))
        gen[mode]={'rows':len(rows),'UTF8':sum(x['generated']['utf8_valid'] for x in rows),
            'EOS':sum(x['generated']['eos'] for x in rows),'unique_strings':len({x['generated']['text'] for x in rows}),
            'repeated_4gram_fraction_mean':sum(x['repeated_4gram_fraction'] for x in stats)/len(stats),
            'the_word_fraction_mean':sum(x['the_word_fraction'] for x in stats)/len(stats),
            'examples':[x['generated']['text'] for x in rows[:2]]}
    inferred={'encode':3272+8+111+4,'denoise':111*16+4*16,
        'decoder':3272+16+111+sum(stop_count(x) for x in generations.values())+
        sum(stop_count(read(r/(p+'_oracle_causal.json'))['rows']) for p in ('initial','final'))}
    assert inferred==result['allocation_calls']
    parent=read(r/'parent_reference_NLL.json')
    output={'status':'V9_ADDRESS_INTERFACE_LEARNED_ACTUAL_VP_OUTPUT_REGRESSED','new_updates':1636,'global_updates':4498,
        'plan_sha256':end['plan_sha256'],'final_fullstate_sha256':result['latest']['sha256'],
        'final_algorithm_sha256':result['latest']['algorithm_sha256'],'preservation':preserved,'interface_diagnostics_only':probes,
        'actual_predicted_VP':{'parent_architecture_reference_NLL':parent['NLL'],'parent_reference_byte_accuracy':parent['byte_accuracy'],
            **metrics,'generation':gen,'NLL_is_conditional_on_one_sample_not_marginal_likelihood':True},
        'reconstructed_calls':inferred,'total_calls':sum(inferred.values()),'seconds':result['seconds'],
        'train_docs':512,'new_row_exposures':13084,'positive_target_UTF8_bytes':1538700,
        'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False,
        'interpretation':'Same-index addressing + learning yields990/990 held oracle byte reconstruction and8/8 authored opposite-order oracle realization. Actual predicted-VP continuation is incoherent and conditional NLL regresses. Repetition reduction alone is not generation success. VP and language encoder were unchanged; no claim independent semantic completion or whole-model failure. Parent architecture reference differs from new initial function.'}
    with (r/'root_result.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=True))

if __name__=='__main__':main()
