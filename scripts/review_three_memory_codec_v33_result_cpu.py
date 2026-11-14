"""ROOT saved-only V33 result review. No model construction or forward."""
import hashlib,json,statistics,sys
from pathlib import Path
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support
from review_owt_free_similarity_20261028_cpu import metrics
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count

HELD_COMMENTS=[
"He/people remain, but the negated acknowledgement of culpability and trickery/deceit are absent; no complete source proposition.",
"Fans are replaced with Felernick; light-up flower accessories, waving/anticipation and standing are absent.",
"Uppercase fragments do not preserve Disney's donut sundae or the surprised lack-of-knowledge question.",
"The output does not preserve the few ransom payments or victims restoring systems from backups.",
"Named people/place and the assault team facing Slade Wilson in Moscow are absent.",
"The output does not preserve Israel/Syria/Assad or the contrast between war and stable ceasefire.",
"The sobriety test/station lobby/DUI charge are lost; Danil/fake appear from a focused training sentence.",
"I remains, but taking goods from Yanick and leaving without paying is not conveyed; negative-looking bytes alone are insufficient.",
"But more remains; negation of explosion and the workplace speech restriction relation are lost.",
"Robby Fabbri/Chiarelli/Bowman and barely-on-radar timing are replaced with April Ryan/20-year material from training.",
"We remains, but requesting a second academic's anonymous review of the article is absent.",
"Report/confirmed fragments do not preserve Wubbels/lawyer and the concern about others' inaction.",
"The six-country vessel-hostage relation is absent; April Ryan-like training content intrudes.",
"The source's Jamaican bobsled Olympic qualification, Sochi and two-man detail are absent.",
"Janissa preventing assassination with her wall is not recoverable from Lans sn/word fragments.",
"The temporary string/array-of-pointers relationship and both function identifiers are absent."
]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def digest_parts(s,keys):return support.state_sha({k:s[k] for k in keys})
def total(s,k):
    n=0
    while k in s:n+=s[k];s=s.get('ancestry',{})
    return n

def raw_rows(rows,features,comments=None):
    scored=[]
    assert len(rows)==len(features)
    for i,(r,f) in enumerate(zip(rows,features)):
        assert f=={'id':r['id'],'prompt':r['source']} and r['copy_bytes']==r['denoise_calls']==0
        g=r['generated'];ids=g['token_ids'];ended=256 in ids
        prefix=ids[:ids.index(256)] if ended else ids
        raw=bytes(x for x in prefix if 0<=x<256);special=any(x>=256 for x in prefix)
        try:text=raw.decode('utf-8');valid=True
        except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');valid=False
        assert raw.hex()==g['raw_hex'] and text==g['text']
        assert valid==g['utf8_valid'] and ended==g['eos'] and special==g['special_before_eos']
        assert g['well_formed']==(valid and ended and not special)
        if ended:assert all(x==257 for x in ids[ids.index(256)+1:])
        exact=raw==r['source'].encode('utf-8') and valid and not special
        assert exact==r['reconstruction_exact']
        q={'index':i,'id':r['id'],'source':r['source'],'reconstruction':text,'exact':exact,
           'UTF8':valid,'EOS':ended,'well_formed':g['well_formed'],**metrics(r['source'],text)}
        if comments:q.update(ROOT_reading=comments[i],whole_faithful=False,confirmed_partial_proposition=False)
        else:q.update(ROOT_read=True,whole_source_span_faithful=exact)
        scored.append(q)
    summary={k:statistics.mean(x[k] for x in scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
    summary.update(rows=len(rows),**{k:sum(x[k] for x in scored) for k in ('exact','UTF8','EOS','well_formed')})
    return scored,summary

def logits_review(path,targets,expected,modes=('normal','swap','zero')):
    a=torch.load(path,map_location='cpu',weights_only=True);ids=a['target_ids'];active=ids!=257;out={}
    for i,t in enumerate(targets):assert ids[i].tolist()==t['target_ids']+[257]*(129-len(t['target_ids']))
    for mode in modes:
        z=a[mode].double();n=z.logsumexp(-1)-z.gather(-1,ids[...,None]).squeeze(-1);arg=z.argmax(-1)
        correct=int((arg[active]==ids[active]).sum());eos=int(((ids==256)&(arg==256)).sum());v=float(n[active].mean())
        e=expected[mode];assert correct==e['correct'] and eos==e['sentence_EOS_correct'] and int(active.sum())==e['symbols']
        # Stored logits are FP16, worker sums FP32. Retain measured rounding delta.
        delta=v-e['NLL'];assert abs(delta)<.001
        out[mode]={'NLL_from_saved_FP16':v,'worker_FP32_NLL':e['NLL'],'NLL_rounding_delta':delta,
                   'correct':correct,'symbols':int(active.sum()),'byte_accuracy':correct/int(active.sum()),'EOS_correct':eos}
    return a,out

def trace_review(traces):
    reviews=[]
    for t in traces:
        raw=t['source'].encode('utf-8');assert hashlib.sha256(raw).hexdigest()==t['source_sha256']
        spans=[];offset=0
        for c in t['source']:
            end=offset+len(c.encode('utf-8'));spans.append((offset,end));offset=end
        assert tuple(spans)==t['spans'] and len(raw)<=128
        assert t['copy_bytes']==t['denoise_calls']==0
        q=F.normalize(t['exact_query'],dim=-1);k=F.normalize(t['exact_keys'],dim=-1)
        score=(q@k.T).masked_fill(~t['exact_valid'][None],-torch.inf)
        indices=score.argsort(dim=-1,descending=True,stable=True)[...,:min(16,score.shape[-1])]
        assert torch.equal(indices,t['candidate_ids'])
        w=torch.cat((score.gather(-1,indices),t['exact_null_logit'].expand(q.shape[0],1)),-1).softmax(-1)
        werr=float((w[:,:-1]-t['candidate_weights']).abs().max());nerr=float((w[:,-1]-t['null_weight']).abs().max())
        assert max(werr,nerr)<2e-6 and torch.allclose(w.sum(-1),torch.ones(q.shape[0]))
        # Independent sequential FP64 accumulation, unlike worker's vectorized einsum.
        mu=t['token_mu'].double();val=t['token_values'].double();decay=t['decay'].double()
        ln=torch.zeros_like(t['long_N']);ld=torch.zeros_like(t['long_D']);rn=ln.clone();rd=ld.clone()
        for i in range(len(raw)):
            weight=mu[i,:,None];v=val[i,None,:];ln+=weight*v;ld+=weight
            rn=decay[:,None]*rn+weight*v;rd=decay[:,None]*rd+weight
        errors={name:float((value-t[name]).abs().max()) for name,value in [('long_N',ln),('long_D',ld),('recent_N',rn),('recent_D',rd)]}
        assert max(errors.values())<1e-8
        query_mu=t['query_mu'];rho=query_mu/query_mu.sum(-1,keepdim=True).clamp_min(1e-8)
        assert torch.allclose(rho,t['rho'],atol=2e-7) and torch.equal(query_mu.amax(-1),t['gamma'])
        reviews.append({'id':t['id'],'UTF8_bytes':len(raw),'codepoints':len(spans),
            'Exact_top16_indices_match':True,'Exact_weights_max_abs_error':werr,'NULL_max_abs_error':nerr,
            'NULL_mean':float(t['null_weight'].mean()),'exact_gate_mean':float(t['exact_gate'].mean()),
            'long_gate_mean':float(t['long_gate'].mean()),'decay_min':float(decay.min()),'decay_max':float(decay.max()),
            'sequential_FP64_errors':errors,'model_epoch':t['model_epoch'],'copy_bytes':0,'denoise_calls':0})
    assert len(reviews)==16
    return reviews

def main():
    torch.set_num_threads(2);run=R/'runs/three_memory_codec_20261029_v33'
    p,r,end=read(run/'plan.json'),read(run/'result.json'),read(run/'supervisor/final_status.json')
    native=read(run/'code_snapshot/root_terminal_native_and_summary_20261029_1808.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_count']==0
    assert end['launches']==1 and end['actual']['identity_verified'] and end['exit_code']==0
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    assert support.core_sha(R)==support.CORE_SHA and all(support.sha(Path(file))==h for file,h in p['files'].items())
    parent=torch.load(p['source'],weights_only=True,map_location='cpu');initial=torch.load(run/'import/checkpoints/state_0000.pt',weights_only=True,map_location='cpu')
    assert support.state_sha(parent)==p['source_algorithm']
    assert set(parent['model'])=={n for n in initial['model'] if not n.startswith('memory_codec.')}
    assert support.state_sha(parent['model'])==support.state_sha({k:initial['model'][k] for k in parent['model']})
    assert initial['optimizer_parameter_names'][:3]==parent['optimizer_parameter_names']
    ids={i for g in initial['optimizer']['param_groups'][:3] for i in g['params']}
    old_opt={'param_groups':initial['optimizer']['param_groups'][:3],'state':{i:initial['optimizer']['state'][i] for i in ids}}
    assert len(ids)==81 and support.state_sha(old_opt)==support.state_sha(parent['optimizer'])
    assert not any(i in initial['optimizer']['state'] for i in initial['optimizer']['param_groups'][3]['params'])
    components=('scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    assert digest_parts(initial,components)==digest_parts(parent,components)
    assert initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0} and not initial['evaluations']
    for name in ('history','orders','exposure','evaluations','ancestry','actual_calls'):
        assert support.state_sha(initial['ancestry'][name])==support.state_sha(parent[name])
    parent_model_sha=support.state_sha(parent['model']);scaler=dict(parent['scaler']);source_calls=dict(parent['actual_calls']);del parent,old_opt
    trained={n for group in initial['optimizer_parameter_names'] for n in group};assert len(trained)==91
    full=[];online=[]
    for local,path in ((0,run/'import/checkpoints/state_0000.pt'),(1024,run/'model/checkpoints/state_1024.pt'),(2048,run/'model/checkpoints/state_2048.pt')):
        s=torch.load(path,weights_only=True,map_location='cpu');mark=read(path.with_suffix('.json'));h=support.sha(path);alg=support.state_sha(s)
        assert h==mark['sha256'] and alg==mark['algorithm_sha256'] and s['schema']=='fdt_three_memory_language_codec_v33_fullstate'
        assert s['attempts']==s['updates']==local and s['skips']==0 and s['cursor']==8*local and not s['gradients']
        assert s['migration']==initial['migration'] and s['ancestry']==initial['ancestry'] and s['optimizer_parameter_names']==initial['optimizer_parameter_names']
        assert s['frozen_sha']==initial['frozen_sha'] and all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in trained)
        assert s['memory_epoch']==end['plan_sha256']+f':attempt{local}:updates{local}'
        assert [h['rows'] for h in s['history']]==s['orders'][:local] and len(s['history'])==local
        for gi,g in enumerate(s['optimizer']['param_groups']):
            expected=(102335 if gi<2 else 14330 if gi==2 else 0)+local
            for i in g['params']:
                if i in s['optimizer']['state']:assert int(s['optimizer']['state'][i]['step'])==expected
                else:assert gi==3 and local==0
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        scale,tracker=scaler['scale'],scaler['_growth_tracker']
        for i,rec in enumerate(s['history']):
            assert rec['attempt']==i+1 and not rec['AMP_skip'] and rec['outcome']=='UPDATED'
            assert abs(rec['loss']-.5*(rec['clean_CE']+rec['corrupted_prefix_CE']))<1e-5
            tracker+=1
            if tracker==scaler['growth_interval']:scale*=scaler['growth_factor'];tracker=0
            assert rec['scale']==scale
        assert s['scaler']==dict(scaler,scale=scale,_growth_tracker=tracker)
        for label,value in s['evaluations'].items():
            filename=label if label.endswith(('.pt','.json')) else label+'.json'
            assert support.sha(run/filename)==value
        full.append({'attempt':local,'bytes':path.stat().st_size,'sha256':h,'algorithm_sha256':alg,'stage':s['worker_stage'],'scaler':s['scaler']})
        if local==2048:state=s
        else:del s
    assert full[-1]['sha256']==r['latest']['sha256'] and full[-1]['algorithm_sha256']==r['latest']['algorithm_sha256']
    assert total(state,'attempts')==r['global_attempts']==104420 and total(state,'updates')==r['global_updates']==104383
    for epoch in range(256):assert sorted(i for b in state['orders'][epoch*8:(epoch+1)*8] for i in b)==list(range(64))
    assert state['exposure']==r['exposure'] and state['allocation_calls']==r['actual_calls'] and state['actual_calls']==r['cumulative_calls']
    for key in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):assert sum(h[key] for h in state['history'])==r['exposure'][key]
    assert r['exposure']['target_UTF8_bytes']==1101824 and r['exposure']['selected_rows']==16384
    for start in (0,512,1024,1536):online.append({'epochs':[start//8+1,(start+512)//8],**{k:statistics.mean(h[k] for h in state['history'][start:start+512]) for k in ('loss','clean_CE','corrupted_prefix_CE')}})
    external=state['external_closed_assessment_calls'];assert external==r['external_closed_assessment_calls'] and external['total']==1936 and external['included_in_source_ledger'] is False
    changed={gi:sum(not torch.equal(state['model'][n],initial['model'][n]) for n in group) for gi,group in enumerate(state['optimizer_parameter_names'])}
    memory_weights={n:v.clone() for n,v in state['model'].items() if n.startswith('memory_codec.')}
    del state,s,initial
    f,y=read(Path(p['focused_artifact'])/'features.json'),read(Path(p['focused_artifact'])/'targets.json')
    bf,by=read(Path(p['artifact'])/'features.json'),read(Path(p['artifact'])/'targets.json')
    ar,br,hr=[read(run/(label+'_reconstruction.json')) for label in ('focused_initial','focused_final','held_final')]
    a,sa=raw_rows(ar,f);b,sb=raw_rows(br,f);held,sh=raw_rows(hr,[bf['dev'][i] for i in p['generation_indices']],HELD_COMMENTS)
    assert sa['exact']==0 and sb['exact']==64 and sh['exact']==0
    measured={}
    for label,targets in [('focused_initial',y),('focused_final',y),('held_final',by['dev'])]:
        t,measured[label]=logits_review(run/(label+'_logits.pt'),targets,r[label])
        if label=='held_final':held_t=t
    traces=trace_review(held_t['memory_trace'])
    chosen=p['generation_indices'];normal=held_t['normal'][chosen].double();labels=held_t['target_ids'][chosen];active=labels!=257
    held16normal=float((normal.logsumexp(-1)-normal.gather(-1,labels[...,None]).squeeze(-1))[active].mean())
    ablations={}
    for memory in ('persistent','contextual','exact'):
        _,m=logits_review(run/('held16_without_'+memory+'_logits.pt'),[by['dev'][i] for i in chosen],r['held16_ablations'][memory],modes=('normal',))
        ablations[memory]={**m['normal'],'NLL_delta_vs_same16_normal':m['normal']['NLL_from_saved_FP16']-held16normal}
    calls={'encode':2048+2*(8+16)+32+4+6,'denoise':0,'decoder':4096+2*24+96+6+stop_count(ar)+stop_count(br)+stop_count(hr)}
    assert calls==r['actual_calls'] and sum(calls.values())==9877
    assert r['cumulative_calls']=={k:source_calls[k]+calls[k] for k in calls}
    out={'status':'V33_COMPLETE_ROOT_SAVED_ONLY_REVIEWED','plan_sha256':end['plan_sha256'],'fullstates':full,
         'source_model_sha256':parent_model_sha,'source_old264tensors_ALL81Adam_scaler_ALLstoredRNG_exact':True,
         'new10_moments_empty_at_import':True,'all_four_groups_changed_tensor_counts':changed,'all_model_Adam_finite':True,
         'frozen_reasoning_language_prefix_basis_unchanged':True,'all_source_ancestry_preserved':True,'import0_before_baseline':True,
         'new_attempts':2048,'new_updates':2048,'AMP_skips':0,'global_attempts':r['global_attempts'],'global_updates':r['global_updates'],
         'actual_calls':calls,'total_calls':sum(calls.values()),'external_closed_assessment_calls':external,
         'focused_initial_reconstruction':sa,'focused_final_reconstruction':sb,'held16_reconstruction':sh,
         'saved_logit_metrics':measured,'held16_normal_NLL_from_FP16':held16normal,'held16_memory_ablations':ablations,
         'memory_trace_independently_reconstructed':traces,'online_epochs':online,'exposure':r['exposure'],'worker_seconds':r['seconds'],
         'ROOT_read_all144_reconstructions':True,'held_whole_faithful':0,'held_confirmed_partial_proposition':0,
         'ROOT_direct_owner_launch_observed':True,'ROOT_direct_actual_worker_live_observed':False,
         'OS_owner_actual_worker_identity_verified':True,'owner_actual_native':end['actual'],
         'all_owned_and_proxy_gone_before_analysis':True,'additional_model_construction_or_forward':0,'CUDA_replay_performed':False,
         'denoise_calls':0,'literal_copy_bytes':0,'E2_final_accessed':False,
         'conclusion':'New three-memory function reacquired SAME64 trained spans exactly; held reconstruction is incomplete and broad teacher metrics worsened versus V31. Each memory ablation is a same-panel perturbation, not certified semantics/copy intent/full vNext. Training on64 repeated spans cannot establish general reconstruction.'}
    write(run/'root_result.json',out)
    write(run/'root_reconstruction_similarity.json',{'initial':a,'final':b,'held':held,'summary':{'initial':sa,'final':sb,'held':sh}})
    lines=['# V33 ROOT direct reading: all64 before/after and16held','',out['conclusion']]
    for x,y in zip(a,b):
        lines+=['',f"## TRAIN {x['index']}",f"Source: {x['source']}",f"Initial: {x['reconstruction']}",f"Final: {y['reconstruction']}",
                'ROOT: final is identical to the source span, including names, negation, dates, numbers, punctuation and original grammatical defects. This is trained reconstruction including memorization.']
    for x in held:lines+=['',f"## HELD {x['index']}",f"Source: {x['source']}",f"Reconstruction: {x['reconstruction']}",f"ROOT: {x['ROOT_reading']}"]
    with (run/'root_all144_reading.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('status','fullstates','focused_initial_reconstruction','focused_final_reconstruction','held16_reconstruction','held16_normal_NLL_from_FP16','held16_memory_ablations','online_epochs','total_calls')},ensure_ascii=True))
if __name__=='__main__':main()
