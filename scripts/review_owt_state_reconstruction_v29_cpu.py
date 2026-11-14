"""ROOT saved-state and text review. No model construction, forward or new evaluation."""
import json,statistics,sys
from pathlib import Path
from collections import Counter
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support
from review_owt_free_similarity_20261028_cpu import metrics,chrf
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_owt_state_reconstruction_v22_cpu import READINGS as DEV_FIELDS,TRAIN_READINGS as TRAIN_FIELDS

DEV_COMMENTS=[
"He and a malformed negative opening remain; acknowledgement, personal culpability and deception against the American public are lost.",
"Fentaryland is not Fans; waving light-up flower accessories, anticipation and standing are absent.",
"Uppercase repetition and know fragments do not express Disney having a donut sundae or the surprised lack-of-knowledge question.",
"Global lefty is not Globally; ransom payments, victims and restoring systems from backups are absent.",
"Dethat short is not Deathstroke; actual panel, assault team, Slade Wilson and Moscow are not preserved.",
"Syria alone survives. Pakistan and University are substitutions; Israel, stable ceasefire and unchallenged Assad governance are absent.",
"She fared is not she failed; sobriety-test failure and DUI charge are absent.",
"I/taking/down are isolated fragments. Yanick's stall, taken goods and leaving without payment are not preserved.",
"But more survives, but no explosion, workplace speech restriction and causal effectiveness are absent.",
"Name-like fragments do not preserve Fabbri/Chiarelli/Bowman, barely on radar or three-month timing.",
"We and review-like fragments do not express asking a second academic to review an anonymous article.",
"We's a bulls is not Wubbels; lawyer, others' inaction and primary concern are absent.",
"American/Bengally/name-like fragments do not express gunmen holding vessels from the six actual countries.",
"CBS/Obama and unrelated names replace Jamaican bobsled qualification for Sochi Winter Olympics.",
"Japanese is not Janissa; preventing assassination with a wall is absent.",
"The unfinger is not the named function; temporary-string lifetime and pointer-array relationship are lost."
]
TRAIN_COMMENTS=[
"More people and some/condition remain; criticing is not critical, arriving and late Friday are absent. No confirmed proposition.",
"Delay, definitive answer, Clinton, uncertainty and five-primary sweep are absent; different medical/company story substituted.",
"Outside survives, but the exact Outside the court lead is lost; Brazau, appeal intention and sentence absent.",
"Isrando is not Eduardo: prior exact entity is lost; requesting cash to avoid bank/card fees is absent.",
"Nor ATOD is not NATO: prior exact entity is lost; thumb-drive purchase/use and Russian network entry are absent.",
"Will have and problem-like opening do not preserve probably having opinions about the bike; added don't does not restore source qualifiers.",
"The tweets were remains but golf is not flagged; source and forwarding to narrator absent.",
"Once the first season is remains; revolutionaries is not over, and the no-more qualifying-tournaments condition is lost.",
"Bill Hasplan is not Bill Haslam; taking part in panel discussion with Vermont governor absent. Source is a selected fragment.",
"Donations and correspondence is exactly retained, but support, sending, family recipient and postal address are missing; BBC/NBC and10 invented.",
"It just/goes/you are fragments; showing their exact moral compass is absent.",
"forces and diplomatic is a near lexical opening; diplomats, Iraq/U.S. references and reaffirming absent. Input fragment has no invented complete gold.",
"I'm survives but agreement and My Other Home screening in China on August11 are lost;10.1/Jine invented.",
"We'll be survives but taking orders, exact March interval and November/December2012 shipment are lost;2008/2003/2016 invented.",
"Craig Charles is exactly retained; David Emanuel/Richard Coles/Aasmah Mir and joining relation absent. Entity fragment only.",
"She then told the survives but25-year-old becomes20; cop identity and not-to-move instruction absent."
]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def cumulative(s,key):
    total=0
    while key in s:total+=s[key];s=s.get('ancestry',{})
    return total
def adam_by_name(s):
    return {n:s['optimizer']['state'].get(i) for names,g in zip(s['optimizer_parameter_names'],s['optimizer']['param_groups']) for n,i in zip(names,g['params'])}

def main():
    torch.set_num_threads(2);run=R/'runs/owt_fuzzy_reconstruction_20261029_v29'
    p,r=read(run/'plan.json'),read(run/'result.json');end=read(run/'supervisor/final_status.json')
    native=read(run/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python']==[]
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    live=read(run/'code_snapshot/root_native_launch_observation.json')
    assert live['progress']['native_pid']==25924 and any(x['ProcessId']==25924 and x['ParentProcessId']==31616 and x['CommandLine'] for x in live['native_python'])
    parent=torch.load(p['source'],weights_only=True,map_location='cpu')
    initial=torch.load(run/'import/checkpoints/state_0000.pt',weights_only=True,map_location='cpu')
    assert support.state_sha(parent)==p['source_algorithm']
    assert support.state_sha(parent['model'])==support.state_sha(initial['model'])
    assert initial['optimizer_parameter_names'][:2]==parent['optimizer_parameter_names']
    assert initial['optimizer']['param_groups'][:2]==parent['optimizer']['param_groups']
    a,b=adam_by_name(parent),adam_by_name(initial)
    assert len(a)==63 and len(b)==81 and all(support.state_sha(a[n])==support.state_sha(b[n]) for n in a)
    added=initial['optimizer_parameter_names'][2];assert len(added)==18 and all(b[n] is None for n in added)
    assert all(n.startswith(('core.language.blocks.5.','core.language.blocks.6.')) for n in added)
    for k in ('scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):assert support.state_sha(parent[k])==support.state_sha(initial[k])
    for k in ('history','orders','exposure','evaluations','ancestry','actual_calls'):assert support.state_sha(initial['ancestry'][k])==support.state_sha(parent[k])
    buffers={'core.basis','core.frequencies'}
    old_names=set(a);phase_names=set(b)
    assert support.state_sha({n:v for n,v in parent['model'].items() if n not in old_names|buffers})==parent['frozen_sha']==initial['migration']['source_frozen_sha']
    assert support.state_sha({n:v for n,v in initial['model'].items() if n not in phase_names|buffers})==initial['frozen_sha']==initial['migration']['phase_frozen_sha']
    source_scaler=dict(parent['scaler']);source_components=support.state_sha({k:parent[k] for k in ('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')})
    assert source_components==initial['migration']['source_all_component_sha256']
    del parent,a,b
    full=[]
    for local,path in ((0,run/'import/checkpoints/state_0000.pt'),(2048,run/'model/checkpoints/state_2048.pt'),(4096,run/'model/checkpoints/state_4096.pt')):
        s=torch.load(path,weights_only=True,map_location='cpu');marker=read(path.with_suffix('.json'))
        digest=support.state_sha(s);file_sha=support.sha(path)
        assert digest==marker['algorithm_sha256'] and file_sha==marker['sha256']
        assert s['schema']=='fdt_owt_fuzzy_AE_v29_fullstate' and s['attempts']==local and s['cursor']==local*8 and not s['gradients']
        assert s['constant_mask_probability']==.5 and s['trainability']==p['trainability'] and s['ownership_policy']==p['ownership_policy'] and s['resource_policy']==p['resource_policy']
        assert s['training_objective']==p['training_objective'] and s['migration']==initial['migration'] and s['ancestry']==initial['ancestry']
        assert s['optimizer_parameter_names']==initial['optimizer_parameter_names']
        assert len(s['history'])==local and [h['rows'] for h in s['history']]==s['orders'][:local]
        assert [h['attempt'] for h in s['history']]==list(range(1,local+1))
        assert all(h['hold_attempt']==h['attempt'] and h['mask_probability']==.5 for h in s['history'])
        assert all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in phase_names)
        assert s['frozen_sha']==initial['frozen_sha']
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        states=adam_by_name(s)
        assert all(int(states[n]['step'])==88005+s['updates'] for n in old_names)
        assert all((states[n] is None if local==0 else int(states[n]['step'])==s['updates']) for n in added)
        scale,tracker=source_scaler['scale'],source_scaler['_growth_tracker']
        for h in s['history']:
            if h['AMP_skip']:scale*=source_scaler['backoff_factor'];tracker=0
            else:
                tracker+=1
                if tracker==source_scaler['growth_interval']:scale*=source_scaler['growth_factor'];tracker=0
            assert h['scale']==scale
        assert s['scaler']==dict(source_scaler,scale=scale,_growth_tracker=tracker)
        full.append({'attempt':local,'updates':s['updates'],'sha256':file_sha,'bytes':path.stat().st_size,'algorithm_sha256':digest,'old_Adam_step':88005+s['updates'],'new_Adam_step':s['updates'] if local else None,'scaler':s['scaler']})
        if local==4096:state=s
        else:del s
    assert full[-1]['sha256']==r['latest']['sha256'] and full[-1]['algorithm_sha256']==r['latest']['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'],state['skips'])==(4096,4094,32768,2)
    assert cumulative(state,'attempts')==92132 and cumulative(state,'updates')==92099
    assert sorted(i for b in state['orders'] for i in b)==list(range(32768))
    history=state['history'];skips=[h['attempt'] for h in history if h['AMP_skip']]
    assert all(abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5 for h in history)
    assert all(h['outcome']==('AMP_SKIP' if h['AMP_skip'] else 'UPDATED') for h in history)
    assert state['exposure']==r['exposure'] and state['allocation_calls']==r['actual_calls'] and state['actual_calls']==r['cumulative_calls']
    for k in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):assert sum(h[k] for h in history)==r['exposure'][k]
    assert r['exposure']['target_UTF8_bytes']==2926478 and r['exposure']['selected_rows']==32768
    external=state['external_closed_assessment_calls'];assert external==r['external_closed_assessment_calls'] and external['total']==1936 and external['included_in_source_ledger'] is False
    online=[{k:statistics.mean(h[k] for h in history[i:i+2048]) for k in ('loss','clean_CE','corrupted_prefix_CE')} for i in (0,2048)]
    changed={n:not torch.equal(state['model'][n],initial['model'][n]) for n in added};assert all(changed.values())
    del state,initial,s,history
    features=read(Path(p['artifact'])/'features.json');raw=read(run/'final_generation.json');old=read(run/'parent_reference_generation.json')
    allrows={};summaries={};keys=('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')
    panels=dict(raw,train_normal=read(run/'final_train_reconstruction.json'))
    for mode,rows in panels.items():
        scored=[]
        for i,row in enumerate(rows):
            panel='train' if mode=='train_normal' else 'dev';idx=p['train_reconstruction_indices' if panel=='train' else 'generation_indices'][i]
            assert features[panel][idx]=={'id':row['id'],'prompt':row['source']}
            if panel=='dev':assert (row['id'],row['source'])==(old[mode][i]['id'],old[mode][i]['source'])
            g=row['generated'];ids=g['token_ids'];stop=ids.index(256);value=bytes(ids[:stop])
            assert ids[stop+1:]==[257]*(129-stop-1) and value.hex()==g['raw_hex'] and value.decode('utf-8')==g['text']
            assert g['utf8_valid'] and g['eos'] and g['well_formed'] and not g['special_before_eos']
            assert row['reconstruction_exact']==(row['source']==g['text'])
            item={'index':i,'source':row['source'],'reconstruction':g['text'],**metrics(row['source'],g['text'])}
            if mode in ('normal','train_normal'):
                fields=TRAIN_FIELDS if panel=='train' else DEV_FIELDS;comment=(TRAIN_COMMENTS if panel=='train' else DEV_COMMENTS)[i]
                subject,action,obj,polarity,relations,_=fields[i]
                item.update(subject=subject,action=action,object=obj,source_polarity=polarity,source_relations=relations,ROOT_reading=comment,meaning_grade='meaning_not_preserved',faithful_paraphrase=False,partial_proposition=False)
                if panel=='train' and i==14:item.update(meaning_grade='entity_fragment_only',preserved_named_entities=['Craig Charles'])
            scored.append(item)
        summary={k:statistics.mean(x[k] for x in scored) for k in keys}
        unique=0
        for i,row in enumerate(scored):
            vals=[chrf(other['source'],row['reconstruction']) for other in scored];unique+=vals[i]==max(vals) and vals.count(max(vals))==1
        summary.update(own_source_unique_top1_of16=unique,exact=sum(r['reconstruction_exact'] for r in rows),UTF8=16,EOS=16)
        if mode in ('normal','train_normal'):summary.update(ROOT_read_all16=True,ROOT_faithful_paraphrase=0,ROOT_partial_proposition=0)
        summaries[mode]=summary;allrows[mode]=scored
    calls={'encode':4096+32+4+4,'denoise':0,'decoder':8192+96+sum(stop_count(rows) for rows in raw.values())+stop_count(panels['train_normal'])}
    assert calls==r['actual_calls']
    logits=torch.load(run/'final_logits.pt',weights_only=True,map_location='cpu');ids=logits['target_ids'];active=ids!=257;targets=read(Path(p['artifact'])/'targets.json')['dev']
    assert int(active.sum())==23207
    for i,target in enumerate(targets):assert ids[i].tolist()==target['target_ids']+[257]*(129-len(target['target_ids']))
    saved={}
    for mode in ('normal','swap','zero'):
        x=logits[mode].double();nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1)
        correct=int((x.argmax(-1)[active]==ids[active]).sum());value=float(nll[active].mean())
        assert correct==r['final'][mode]['correct'] and abs(value-r['final'][mode]['NLL'])<.001
        saved[mode]={'NLL':value,'correct':correct,'symbols':23207,'byte_accuracy':correct/23207}
    out={'status':'V29_STATE_RECONSTRUCTION_COMPLETE_ROOT_REVIEWED','new_attempts':4096,'new_updates':4094,'global_attempts':92132,'global_updates':92099,'AMP_skip_attempts':skips,
        'source_import_component_sha256':source_components,'source_old63Adam_model_scaler_allRNG_exact':True,'source_whole_optimizer_structure_unchanged':False,'new18_initial_Adam_empty':True,
        'all18_newly_trainable_tensors_changed':changed,'source_and_phase_frozen_scopes_verified':True,'fullstates':full,'frozen_reasoning_prefix_basis_unchanged':True,
        'all_RNG_and_nested_ancestry_preserved':True,'model_Adam_finite':True,'actual_CUDA_replay_performed':False,'external_closed_assessment_calls':external,
        'saved_logit_metrics':saved,'state_conditioned_reconstruction':{k:v for k,v in summaries.items() if k!='train_normal'},'fixed_train_reconstruction':summaries['train_normal'],
        'online_half_epoch_means':online,'supervision':r['exposure'],'actual_calls':calls,'total_new_calls':sum(calls.values()),'seconds':r['seconds'],'plan_sha256':end['plan_sha256'],
        'ROOT_read_all_dev16_and_train16':True,'ROOT_faithful_paraphrase':0,'ROOT_partial_proposition':0,'ROOT_nonblind':True,'ROOT_direct_live_native_observation':True,
        'independent_owner_verified_native_identity':True,'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False,
        'scope':'Same input X -> original fuzzy extracted Z(X) -> Xhat, BOS/selfprefix. Reasoning and output-specific meaning denoising remain separate/frozen.',
        'limitations':['Correct-prefix teacher accuracy is auxiliary, not percentage of meaning preserved.','Same observed dev and fixed16train; qualitative lexical reading is not blind semantic scoring.','Extra training/order and gradual unfreezing are confounded; no sole extractor/decoder cause established.','An initial one-epoch thaw does not establish whole FDT/fuzzy failure or completion.']}
    write(run/'root_result.json',out);write(run/'root_state_reconstruction_similarity.json',{'summary':out['state_conditioned_reconstruction'],'train_summary':summaries['train_normal'],'rows':allrows,'additional_model_forwards':0})
    lines=['# V29 same-input reconstruction: ROOT all32 reading','',out['scope'],'','Saved text only; no additional model calls; unblinded same dev16 and fixed train16.']
    for panel in ('normal','train_normal'):
        for row in allrows[panel]:lines+=['',f"## {panel} {row['index']}",f"Source: {row['source']}",f"Reconstruction: {row['reconstruction']}",f"ROOT: {row['ROOT_reading']}",f"Subject/action/object: {row['subject']} / {row['action']} / {row['object']}",f"Grade: {row['meaning_grade']}; chrF={row['chrF2_1to6']:.3f}"]
    with (run/'root_all32_state_reconstruction_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('status','global_updates','AMP_skip_attempts','fullstates','saved_logit_metrics','state_conditioned_reconstruction','fixed_train_reconstruction','online_half_epoch_means','total_new_calls')},ensure_ascii=True))
if __name__=='__main__':main()
