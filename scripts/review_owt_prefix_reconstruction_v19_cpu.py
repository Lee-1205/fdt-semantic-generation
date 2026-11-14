"""Independent saved fullstate/raw review of deterministic prefix-reconstruction run."""
import json,sys,statistics
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from owt_fuzzy_engine_v19 import COMPONENTS,cumulative_count
from research.owt_prefix_reconstruction_v19 import OBJECTIVE
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_owt_free_similarity_20261028_cpu import metrics,chrf,REVIEWS

COMMENTS=[
 'He survives, but continued and lawyer are not acknowledging culpability/deceit; the core claim is absent.',
 'Fanstine resembles Fans; waving flowers, anticipation and standing absent.',
 'Mixed uppercase fragments; Disney/donut/sundae and surprised question absent.',
 'money loosely relates to ransom; victims restoring backups and causal payment claim absent.',
 'No panel/team/Slade Wilson/Moscow relationship.',
 'Syria now exactly survives; Israel/Assad/stable ceasefire relation is still absent.',
 'She and legal topic survive; sobriety test failure and DUI charge absent.',
 'It was replaces I then; stall/taking items/without-paying relation absent.',
 'But and more survive; invented2015percent, speech restriction/negation/cause absent.',
 'Robby remains but Fabbri/radar/two named scouts/three-month relation absent.',
 'We can see does not express asking a second academic to review anonymous article.',
 'Lawyer, named people, inaction and primary-concern relationship absent.',
 'Name-like list only; gunmen/held vessels and actual countries absent.',
 'No Jamaican team/bobsled qualification/Olympics/Sochi relation.',
 'Japan is not Janissa; prevention/assassination/wall absent.',
 'Initial The and function-like spelling fragments; exact function names/temporary string/pointer relationship absent.'
]

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def main():
    torch.set_num_threads(2);run=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v19';p=read(run/'plan.json');result=read(run/'result.json')
    end=read(run/'supervisor/final_status.json');native=read(run/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    final=run/'model/checkpoints'/result['latest']['file'];assert support.sha(final)==result['latest']['sha256']
    state=torch.load(final,map_location='cpu',weights_only=True);assert support.state_sha(state)==result['latest']['algorithm_sha256']
    initial=torch.load(run/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True);parent=torch.load(p['source'],map_location='cpu',weights_only=True)
    digest=support.state_sha({k:parent[k] for k in COMPONENTS});assert digest==support.state_sha({k:initial[k] for k in COMPONENTS})
    assert state['schema']=='fdt_owt_fuzzy_AE_v19_fullstate' and state['training_objective']==OBJECTIVE and state['loss_reduction']=='elementwise_none_then_sum'
    assert (state['attempts'],state['updates'],state['cursor'],state['skips'])==(2048,2046,16384,2)
    assert cumulative_count(state,'attempts')==23252 and cumulative_count(state,'updates')==23246
    assert initial['ancestry']['history']==parent['history'] and state['ancestry']==initial['ancestry']
    assert [h['rows'] for h in state['history']]==state['orders'] and not state['gradients']
    for epoch in range(2):assert sorted(i for b in state['orders'][epoch*1024:(epoch+1)*1024] for i in b)==list(range(8192))
    trained={n for group in state['optimizer_parameter_names'] for n in group}
    assert all(torch.equal(v,initial['model'][n]) for n,v in state['model'].items() if n not in trained)
    assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
    assert all(torch.isfinite(v).all() for item in state['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
    steps=Counter(int(v['step']) for v in state['optimizer']['state'].values());assert steps=={23246:63}
    history=state['history'];skips=[h['attempt'] for h in history if h['AMP_skip']];assert len(skips)==2
    assert all(h['outcome']==('AMP_SKIP' if h['AMP_skip'] else 'UPDATED') for h in history)
    assert all(abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5 for h in history)
    assert state['exposure']==result['exposure'] and state['exposure']['target_UTF8_bytes']==1457826
    assert sum(h['eligible_prefix_bytes'] for h in history)==state['exposure']['eligible_prefix_bytes']==1457826
    assert sum(h['masked_prefix_bytes'] for h in history)==state['exposure']['masked_prefix_bytes']==218800
    assert sum(h['decoder_supervised_symbol_occurrences'] for h in history)==state['exposure']['decoder_supervised_symbol_occurrences']==2948420
    losses=[{name:statistics.mean(h[name] for h in history[i:i+1024]) for name in ('loss','clean_CE','corrupted_prefix_CE')} for i in (0,1024)]
    preservation={'source_model_Adam_scaler_all_RNG_components_exact':True,'component_sha256':digest,'failed_V18_history_and_nested_ancestry_retained':True,
        'frozen_reasoning_prefix_unchanged':True,'all_model_and_Adam_tensors_finite':True,'Adam_steps':dict(steps),'scaler':state['scaler']}
    assert state['actual_calls']==result['cumulative_calls'] and state['allocation_calls']==result['actual_calls']
    del state,initial,parent,history
    raw=read(run/'final_generation.json');old=read(run/'parent_reference_generation.json');scores={};rows_out={}
    for mode,rows in raw.items():
        scored=[]
        for i,(row,previous) in enumerate(zip(rows,old[mode])):
            assert row['id']==previous['id'] and row['source']==previous['source']
            g=row['generated'];assert bytes.fromhex(g['raw_hex']).decode('utf-8')==g['text']
            assert row['reconstruction_exact']==(g['well_formed'] and row['source']==g['text'])
            x={'index':i,'id':row['id'],'source':row['source'],'output':g['text'],**metrics(row['source'],g['text'])}
            if mode=='normal':x.update(source_proposition=REVIEWS[i][0],ROOT_reading=COMMENTS[i],faithful_paraphrase=False)
            scored.append(x)
        summary={k:statistics.mean(x[k] for x in scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
        top=0;margin=[]
        for i,row in enumerate(scored):
            values=[chrf(other['source'],row['output']) for other in scored];maximum=max(values)
            top+=values[i]==maximum and values.count(maximum)==1;margin.append(values[i]-statistics.mean(v for j,v in enumerate(values) if j!=i))
        summary.update(own_source_unique_top1_of16=top,own_vs_other15_chrF_margin=statistics.mean(margin),exact=sum(x['reconstruction_exact'] for x in rows),UTF8=sum(x['generated']['utf8_valid'] for x in rows),EOS=sum(x['generated']['eos'] for x in rows),unique=len({x['output'] for x in scored}),content_word_overlap_rows=sum(bool(x['shared_content_words']) for x in scored))
        scores[mode]=summary;rows_out[mode]=scored
    calls={'encode':2048+32+4,'denoise':0,'decoder':4096+96+sum(stop_count(rows) for rows in raw.values())};assert calls==result['actual_calls']
    data=torch.load(run/'final_logits.pt',map_location='cpu',weights_only=True);targets=read(Path(p['artifact'])/'targets.json')['dev'];ids=data['target_ids'];active=ids!=257
    for i,target in enumerate(targets):assert ids[i].tolist()==target['target_ids']+[257]*(129-len(target['target_ids']))
    saved_metrics={}
    for mode in raw:
        x=data[mode].double();nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1);correct=int((x.argmax(-1)[active]==ids[active]).sum())
        assert correct==result['final'][mode]['correct'] and abs(float(nll[active].mean())-result['final'][mode]['NLL'])<.001
        saved_metrics[mode]={'NLL':float(nll[active].mean()),'correct':correct,'symbols':int(active.sum()),'byte_accuracy':correct/int(active.sum())}
    mu=data['memberships'].double();fuzzy={'coverage_anchors':int((mu.sum((0,1))>0).sum()),'mean_membership_mass':float(mu.sum(-1).mean()),'per_view_active':float((mu>0).double().sum(-1).mean())}
    confidence={}
    for label,d in (('V19',data),('V17',torch.load(run/'parent_reference_logits.pt',map_location='cpu',weights_only=True))):
        x=d['normal'].double()[active];gold=ids[active];prob=x.softmax(-1);conf,pred=prob.max(-1);ok=pred==gold;ece=0.
        for i in range(10):
            mask=(conf>=i/10)&(conf<(i+1)/10 if i<9 else conf<=1)
            if mask.any():ece+=int(mask.sum())/len(gold)*abs(float(ok[mask].double().mean())-float(conf[mask].mean()))
        confidence[label]={'mean_max_probability':float(conf.mean()),'mean_confidence_when_wrong':float(conf[~ok].mean()),'ECE10_posthoc':ece}
    out={'status':'V19_PREFIX_RECONSTRUCTION_COMPLETE_ROOT_REVIEWED','new_attempts':2048,'new_updates':2046,'AMP_skips':2,'AMP_skip_attempts':skips,'global_attempts':23252,'global_updates':23246,
        'objective':OBJECTIVE,'loss_reduction':'elementwise_none_then_sum','online_training_epoch_means':losses,'preservation':preservation,'supervision':result['exposure'],
        'masked_fraction':218800/1457826,'decoder_supervised_UTF8_occurrences':2915652,'saved_raw_metrics':saved_metrics,'free_similarity':scores,'confidence':confidence,'fuzzy_statistics':fuzzy,
        'plan_sha256':end['plan_sha256'],'final_sha256':result['latest']['sha256'],'final_algorithm_sha256':result['latest']['algorithm_sha256'],
        'reconstructed_new_calls':calls,'total_new_calls':sum(calls.values()),'seconds':result['seconds'],'all_owned_gone_before_analysis':True,'additional_model_forwards':0,
        'ROOT_faithful_paraphrase':0,'ROOT_reading_nonblind':True,'same_observed_dev_multiple_use':True,'E2_final_accessed':False,
        'interpretation':'Deterministic loss-kernel issue fixed without disabling determinism; two finite epochs of unchanged fuzzy AE with train-only prefix corruption complete. Teacher accuracy slightly decreases, NLL slightly recovers versus V17 but remains worse than V15. Free scores/ROOTreading retained; no complete semantic recovery or generic LM claim.'}
    with (run/'root_result.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    with (run/'root_free_similarity.json').open('x',encoding='utf-8') as f:json.dump({'summary':scores,'rows':rows_out,'additional_model_forwards':0,'limitations':'Posthoc lexical measures and unblinded ROOT reading on same16devdocs; not calibrated semantic probability.'},f,ensure_ascii=False,indent=2)
    lines=['# V19 all16 source-output pairs and ROOT reading','', 'No additional model forward. Posthoc same observed16devdocuments.']
    for row in rows_out['normal']:lines.extend(['',f"## {row['index']}",f"Source: {row['source']}",f"Output: {row['output']}",f"ROOT: {row['ROOT_reading']}",f"chrF={row['chrF2_1to6']:.3f}; sharedcontent={row['shared_content_words']}"])
    with (run/'root_all16_free_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(out,ensure_ascii=True))
if __name__=='__main__':main()
