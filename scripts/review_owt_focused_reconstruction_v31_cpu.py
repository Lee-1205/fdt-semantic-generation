"""Saved-only independent ROOT review of64 training reconstructions and16 held rows."""
import json,statistics,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support
from review_owt_free_similarity_20261028_cpu import metrics
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count

HELD_COMMENTS=[
"He remains but acknowledgement, culpability and deception of the American public absent; repeated confirmed is unrelated.",
"Fendangs is not Fans; waving light-up flower accessories, anticipation and standing absent.",
"Uppercase fragments do not express Disney's donut sundae and surprised lack-of-knowledge question.",
"Globally lead lost; ransom payments, victims and backup restoration absent.",
"WordPales and Washington replace actual panel/team/Slade Wilson/Moscow; no assault-team relation preserved.",
"The chancially is not Technically; Israel/Syria/Assad/ceasefire-governance relation absent.",
"The geal Starce replaces she; sobriety-test failure and DUI charge absent.",
"I remains; taking goods from Yanick's stall and leaving without paying absent.",
"But more and incidents/Workers words do not preserve no explosion or effectiveness of workplace speech restriction.",
"Name-like strings replace Fabbri/Chiarelli/Bowman; barely-on-radar and three-month timing absent.",
"We remains; asking second academic to review anonymous article absent.",
"Report/behave fragments do not preserve Wubbels, lawyer, inaction and primary concern.",
"Country-like words do not express gunmen holding vessels from the original six countries.",
"Baroness Royand resembles focused training text; Jamaican bobsled/Sochi Olympics qualification absent.",
"Launa's is not Janissa; preventing assassination with wall absent.",
"The functions is a lexical lead only; named functions, temporary-string/pointer-array relation absent."
]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def count(s,k):
    n=0
    while k in s:n+=s[k];s=s.get('ancestry',{})
    return n
def verify_rows(rows,features,comments=None):
    scored=[]
    for i,(row,feature) in enumerate(zip(rows,features)):
        assert feature=={'id':row['id'],'prompt':row['source']}
        g=row['generated'];ids=g['token_ids'];stop=ids.index(256);b=bytes(ids[:stop])
        assert ids[stop+1:]==[257]*(129-stop-1) and b.hex()==g['raw_hex'] and b.decode('utf-8')==g['text']
        assert g['utf8_valid'] and g['eos'] and g['well_formed'] and not g['special_before_eos']
        exact=g['text']==row['source'];assert exact==row['reconstruction_exact']
        item={'index':i,'id':row['id'],'source':row['source'],'reconstruction':g['text'],'exact':exact,**metrics(row['source'],g['text'])}
        if comments:item.update(ROOT_reading=comments[i],whole_faithful=False,confirmed_partial_proposition=False)
        else:item.update(ROOT_read=True,whole_source_span_faithful=exact,meaning_grade='exact_source_span' if exact else 'incomplete_before_focused_fit')
        scored.append(item)
    assert len(scored)==len(features)
    summary={k:statistics.mean(x[k] for x in scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
    summary.update(rows=len(rows),exact=sum(x['exact'] for x in scored),UTF8=len(rows),EOS=len(rows))
    return scored,summary

def saved_metrics(path,targets,expected):
    t=torch.load(path,map_location='cpu',weights_only=True);ids=t['target_ids'];active=ids!=257;out={}
    for i,target in enumerate(targets):assert ids[i].tolist()==target['target_ids']+[257]*(129-len(target['target_ids']))
    for mode in ('normal','swap','zero'):
        x=t[mode].double();nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1)
        correct=int((x.argmax(-1)[active]==ids[active]).sum());value=float(nll[active].mean())
        eos=int(((ids==256)&(x.argmax(-1)==256)).sum())
        assert correct==expected[mode]['correct'] and int(active.sum())==expected[mode]['symbols'] and eos==expected[mode]['sentence_EOS_correct']
        assert abs(value-expected[mode]['NLL'])<.0001
        out[mode]={'NLL':value,'correct':correct,'symbols':int(active.sum()),'byte_accuracy':correct/int(active.sum()),'EOS_correct':eos}
    return out

def main():
    torch.set_num_threads(2);run=R/'runs/owt_fuzzy_reconstruction_20261029_v31'
    p,r=read(run/'plan.json'),read(run/'result.json');end=read(run/'supervisor/final_status.json');native=read(run/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python']==[]
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    live=read(run/'code_snapshot/root_native_launch_observation.json')
    assert live['progress']['native_pid']==22104 and any(x['ProcessId']==22104 and x['ParentProcessId']==8808 and x['CommandLine'] for x in live['native_python'])
    parent=torch.load(p['source'],map_location='cpu',weights_only=True);initial=torch.load(run/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    assert support.state_sha(parent)==p['source_algorithm']
    components=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    source_digest=support.state_sha({k:parent[k] for k in components})
    assert source_digest==support.state_sha({k:initial[k] for k in components})
    assert initial['ancestry']['imported_component_sha256']==source_digest and initial['migration']==parent['migration']
    for k in ('history','orders','exposure','evaluations','ancestry','actual_calls'):assert support.state_sha(initial['ancestry'][k])==support.state_sha(parent[k])
    old_names={n for group in initial['optimizer_parameter_names'][:2] for n in group};added=set(initial['optimizer_parameter_names'][2]);trained=old_names|added
    assert len(old_names)==63 and len(added)==18
    scaler=dict(parent['scaler']);source_calls=dict(parent['actual_calls']);del parent
    full=[]
    for local,path in ((0,run/'import/checkpoints/state_0000.pt'),(1024,run/'model/checkpoints/state_1024.pt'),(2048,run/'model/checkpoints/state_2048.pt')):
        s=torch.load(path,map_location='cpu',weights_only=True);mark=read(path.with_suffix('.json'));alg=support.state_sha(s);h=support.sha(path)
        assert h==mark['sha256'] and alg==mark['algorithm_sha256'] and s['schema']=='fdt_owt_fuzzy_AE_v31_fullstate'
        assert s['attempts']==s['updates']==local and s['skips']==0 and s['cursor']==8*local and not s['gradients']
        assert s['training_objective']==p['training_objective'] and s['trainability']==p['trainability'] and s['ownership_policy']==p['ownership_policy'] and s['resource_policy']==p['resource_policy']
        assert s['migration']==initial['migration'] and s['ancestry']==initial['ancestry'] and s['optimizer_parameter_names']==initial['optimizer_parameter_names']
        assert [h['rows'] for h in s['history']]==s['orders'][:local] and len(s['history'])==local
        assert all(h['attempt']==i+1 and h['hold_attempt']==i+1 and h['mask_probability']==.5 and h['outcome']=='UPDATED' and not h['AMP_skip'] for i,h in enumerate(s['history']))
        assert all(abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5 for h in s['history'])
        for gi,group in enumerate(s['optimizer']['param_groups']):
            assert len(group['params'])==len(s['optimizer_parameter_names'][gi])
            assert all(int(s['optimizer']['state'][idx]['step'])==(100287 if gi<2 else 12282)+local for idx in group['params'])
        assert all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in trained)
        assert s['frozen_sha']==initial['frozen_sha']
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        scale,tracker=scaler['scale'],scaler['_growth_tracker']
        for rec in s['history']:
            tracker+=1
            if tracker==scaler['growth_interval']:scale*=scaler['growth_factor'];tracker=0
            assert rec['scale']==scale
        assert s['scaler']==dict(scaler,scale=scale,_growth_tracker=tracker)
        for label,digest in s['evaluations'].items():
            suffix='.pt' if label.endswith('_logits') else '.json'
            assert support.sha(run/(label+suffix))==digest
        full.append({'attempt':local,'updates':local,'bytes':path.stat().st_size,'sha256':h,'algorithm_sha256':alg,'worker_stage':s['worker_stage'],'old63_Adam_step':100287+local,'added18_Adam_step':12282+local,'scaler':s['scaler']})
        if local==2048:state=s
        else:del s
    assert full[-1]['sha256']==r['latest']['sha256'] and full[-1]['algorithm_sha256']==r['latest']['algorithm_sha256']
    assert count(state,'attempts')==102372 and count(state,'updates')==102335
    for epoch in range(256):assert sorted(i for b in state['orders'][epoch*8:(epoch+1)*8] for i in b)==list(range(64))
    assert state['exposure']==r['exposure'] and state['allocation_calls']==r['actual_calls'] and state['actual_calls']==r['cumulative_calls']
    for key in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):assert sum(h[key] for h in state['history'])==r['exposure'][key]
    assert r['exposure']['target_UTF8_bytes']==1101824 and r['exposure']['selected_rows']==16384
    online=[{'epochs':str(start//8+1)+'-'+str((start+512)//8),**{k:statistics.mean(h[k] for h in state['history'][start:start+512]) for k in ('loss','clean_CE','corrupted_prefix_CE')}} for start in (0,512,1024,1536)]
    external=state['external_closed_assessment_calls'];assert external==r['external_closed_assessment_calls'] and external['total']==1936 and external['included_in_source_ledger'] is False
    del state,s,initial
    q=Path(p['focused_artifact']);f,y=read(q/'features.json'),read(q/'targets.json')
    initial_raw=read(run/'focused_initial_reconstruction.json');final_raw=read(run/'focused_final_reconstruction.json');held_raw=read(run/'held_final_reconstruction.json')
    initial_rows,initial_summary=verify_rows(initial_raw,f);final_rows,final_summary=verify_rows(final_raw,f)
    broadf=read(Path(p['artifact'])/'features.json');broady=read(Path(p['artifact'])/'targets.json')
    held_rows,held_summary=verify_rows(held_raw,[broadf['dev'][i] for i in p['generation_indices']],HELD_COMMENTS)
    assert initial_summary['exact']==4 and final_summary['exact']==64 and held_summary['exact']==0
    measured={
        'focused_initial':saved_metrics(run/'focused_initial_logits.pt',y,r['focused_initial']),
        'focused_final':saved_metrics(run/'focused_final_logits.pt',y,r['focused_final']),
        'held_final':saved_metrics(run/'held_final_logits.pt',broady['dev'],r['held_final'])}
    calls={'encode':2048+8+16+8+16+32+4,'denoise':0,'decoder':4096+24+24+96+stop_count(initial_raw)+stop_count(final_raw)+stop_count(held_raw)}
    assert calls==r['actual_calls'] and sum(calls.values())==9606
    baseline_calls={'encode':24,'denoise':0,'decoder':24+stop_count(initial_raw)}
    import_state=torch.load(run/'import/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    assert import_state['allocation_calls']==baseline_calls
    assert import_state['actual_calls']=={k:source_calls[k]+baseline_calls[k] for k in source_calls}
    del import_state
    out={'status':'V31_FOCUSED64_COMPLETE_ROOT_REVIEWED','new_attempts':2048,'new_updates':2048,'AMP_skips':0,'global_attempts':102372,'global_updates':102335,
        'source_component_sha256':source_digest,'ALL81_model_Adam_scaler_RNG_exact_after_new_baseline':True,'source_call_ledger_preserved_in_ancestry':True,'import0_includes_new_focused_baseline_calls':baseline_calls,
        'fullstates':full,'model_Adam_finite':True,'frozen_reasoning_prefix_basis_unchanged':True,'all_RNG_nested_history_orders_and_first_thaw_provenance_preserved':True,
        'focused_initial_reconstruction':initial_summary,'focused_final_reconstruction':final_summary,'held16_reconstruction':held_summary,
        'ROOT_read_all64_initial_final_and16held':True,'ROOT_training_final_source_span_faithful':64,'ROOT_held_whole_faithful':0,'ROOT_held_confirmed_partial':0,
        'saved_logit_metrics':measured,'online_64epoch_blocks':online,'supervision':r['exposure'],'actual_calls':calls,'total_new_calls':9606,'seconds':r['seconds'],
        'external_closed_assessment_calls':external,'resource_scans':r['resource_scans'],'plan_sha256':end['plan_sha256'],'ROOT_direct_live_native_observation':True,
        'all_owned_gone_before_analysis':True,'additional_model_construction_forwards':0,'actual_CUDA_replay_performed':False,'E2_final_accessed':False,
        'scope':'Unchanged raw-input -> original fuzzy extracted state -> same-span BOS/selfprefix reconstruction.64existing training spans repeated256epochs; no rawcontext bypass/labels/length/templates at state-only inference; reasoning denoising frozen.',
        'conclusion':'Training reconstruction capacity demonstrated on64/64 exact spans; generalization deteriorated sharply on same observed held panel. Preserve V31 as a capacity/overfit branch, not an automatic broad-model promotion.',
        'limitations':['Training memorization is included;64exact spans do not establish abstract semantics, a unique latent gold, independent reasoning, or broad LM completion.','Focused spans were selected as short existing train spans; source malformed grammar/fragments remain faithfully copied.','Held metrics are repeatedly observed and unblinded; no new blind benchmark.','Joint encoder/fuzzy/decoder optimization does not isolate a sole causal module.']}
    write(run/'root_result.json',out);write(run/'root_reconstruction_similarity.json',{'focused_initial_summary':initial_summary,'focused_final_summary':final_summary,'held_summary':held_summary,'rows':{'focused_initial':initial_rows,'focused_final':final_rows,'held':held_rows},'additional_model_forwards':0})
    lines=['# V31: ROOT read all64 focused before/after and16held','',out['scope'],'','All final64 are exact original SOURCE SPANS, including original grammar/typos; this is trained reconstruction, not blind generalization.']
    for i,(a,b) in enumerate(zip(initial_rows,final_rows)):
        lines+=['',f'## Focused training {i}',f"Source: {a['source']}",f"Before: {a['reconstruction']}",f"After: {b['reconstruction']}",f"ROOT: after is exact source span; all source wording, negation, names, dates and numbers retained. Before exact={a['exact']}; after exact={b['exact']}."]
    for row in held_rows:lines+=['',f"## Held {row['index']}",f"Source: {row['source']}",f"After: {row['reconstruction']}",f"ROOT: {row['ROOT_reading']}"]
    with (run/'root_all144_before_after_and_held_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('status','fullstates','focused_initial_reconstruction','focused_final_reconstruction','held16_reconstruction','saved_logit_metrics','online_64epoch_blocks','total_new_calls')},ensure_ascii=True))
if __name__=='__main__':main()
