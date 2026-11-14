"""Saved-only ROOT fullstate and all80 source/reconstruction review for V38."""
import json,math,statistics,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support
from review_three_memory_codec_v33_result_cpu import raw_rows,logits_review
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count

PARTIAL={4,8,10,20,31,59}
COMMENTS={
0:'Mr_Ogunjobi/team/Panthers wish is replaced with unrelated names/objects.',
1:'Crops/farm/Tillsonburg/Ontario are lost.',
2:'Recent events reversing his plan is garbled, not a clear preserved proposition.',
3:'Probable token remains; listed/still-listed assertion is garbled.',
4:'Reversal of U.S. policy remains; Israel/founding/1948 are lost.',
5:'RE24 recalculation, no park adjustment and SB/CS alternative are lost.',
6:'WordPress/functionality wording remains but the installation/plugin claim is lost.',
7:'April2015/hive boxes/frames/new locations are lost.',
8:'No one seems to be answering that question is preserved; last becomes malformed altest. Near-complete core with modifier corruption, not exact full recovery.',
9:'Gameplay itself is replaced with healing mage fragments.',
10:'The12 teams combined35-37 result remains; one-score-game context is replaced with one-case gross.',
11:'Sealed container/take-it-anywhere relation is lost.',
14:'Swansea/passes and expected style are garbled.',
15:'April Ryan name, press corps, negated agenda and attribution are lost.',
18:'July2/scientists/sunflowers remain, but sprouting event is lost.',
19:'Bride/alcohol poisoning/dehydration treatment is replaced with unrelated relationship wording.',
20:'Setback for a22-year-old remains; preseason return/Euro2016 exploits are lost.',
21:'Conscious becomes consumptions; target of the phenomenal-feeling statement is changed.',
22:'Anti-Semitic lead remains; behaviour and Baroness/report attribution are lost.',
27:'Bears falling to bottom/lowest scores across board is garbled.',
28:'Too-early/jump-the-gun relation is lost.',
29:'Financial complication/tea-garden labourer comprehension relation is lost.',
30:'Baby shower gift to a friend becomes baby showing fragments.',
31:'Photos by Robert Oliver and war survival remain; source1899 is changed to1990. Partial faithful propositions with a factual date error.',
32:'Radio etiquette learned from experience is malformed.',
36:'Grin on his face/when he said it relation is lost.',
41:'McManus did-not-say lead remains but shooting/cause object is lost.',
43:'David Cameron/big-society relation is lost.',
44:'Playing in the dirt, his favourite activity, becomes playing the direct.',
45:'Defines/linker/compiler options are lost.',
46:'Oakland refusal/NFL will/noble assertion is lost.',
47:'Belichick remains but praise of team and liberal smiles are lost.',
48:'Seven-year delays/cost overruns are lost.',
49:'Psaki/Fox News/James Rosen and reply attribution are lost.',
50:'Good-intentioned people bringing it to reality is garbled.',
52:'Portuguese150companies/art-collecting propositions are lost.',
53:'Meads being more well-rounded becomes world-since.',
54:'Self-employment/stability/late90s timing are lost.',
55:'OPI My Vampire is Buff/base relation is lost.',
56:'FDA serious complications not rare becomes unrelated social resumes wording.',
58:'Built-in Bluetooth/speaker endpoint relation is lost.',
59:'Works on all systems sold worldwide is preserved; regional account/online-feature requirement is lost.',
62:'Self-trained element of punk music is lost.',
63:'Appointment-strategy conditional and consequent big win are lost.'}
HELD=[
'Personal culpability, American victims, trickery and deceit are lost.',
'Fans/light-up flowers/waving/anticipation/standing are lost.',
'Disney/donut sundae/surprised speaker question are lost despite uppercase output.',
'Ransom payments/victims/backup restoration propositions are lost.',
'Deathstroke/comic panel/assault team/Slade Wilson/Moscow are lost.',
'Israel/Syria/war-versus-ceasefire/Assad relations are lost.',
'Failed sobriety test/station lobby/DUI charge are lost.',
'Yanick/goods taken/leaving without paying are lost.',
'But more important lead remains; no explosion and effective workplace speech restriction are lost.',
'Fabbri/Chiarelli/Bowman/three-month radar relation are lost.',
'Second academic reviewing anonymous article copy is lost.',
'Wubbels/lawyer/inaction/primary-concern attribution are lost.',
'Gunmen holding vessels from all six listed countries are lost; Irania is not faithful Iran recovery.',
'Jamaica/two-man bobsled/Winter Olympics/Sochi are lost.',
'Janissa preventing assassination with her wall is lost.',
'get_input_string temporary string and tokenize_string2 pointer-array relation are lost.'
]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2)
def count(s,k):
    n=0
    while k in s:n+=s[k];s=s.get('ancestry',{})
    return n
def prefix_alignment(t,rows,indices):
    result=[]
    for raw,i in zip(rows,indices):
        y=t['target_ids'][i];a=t['normal'][i].argmax(-1);n=int((y!=257).sum());b=raw['generated']['token_ids']
        first_t=next((j for j in range(n) if int(a[j])!=int(y[j])),None)
        first_b=next((j for j in range(n) if b[j]!=int(y[j])),None)
        result.append({'id':raw['id'],'teacher_first_error':first_t,'BOS_first_error':first_b})
    assert all(x['teacher_first_error']==x['BOS_first_error'] for x in result)
    return result

def main():
    torch.set_num_threads(2);run=R/'runs/three_memory_codec_20261029_v38'
    p,r,end=[read(run/file) for file in ('plan.json','result.json','supervisor/final_status.json')]
    terminal=read(run/'code_snapshot/root_terminal_native_observation.json');live=read(run/'code_snapshot/root_actual_training_observation.json')
    assert end['status']=='COMPLETE' and end['exit_code']==0 and end['exit']['all_owned_processes_gone'] and len(terminal['native'])==0
    assert any(x['ProcessId']==42704 and x['ParentProcessId']==15172 and x['CommandLine'] for x in live['native']) and live['progress']['updates']>0
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    assert support.core_sha(R)==support.CORE_SHA and all(support.sha(file)==sha for file,sha in p['files'].items())
    parent=torch.load(p['source'],weights_only=True,map_location='cpu');initial=torch.load(run/'import/checkpoints/state_0000.pt',weights_only=True,map_location='cpu')
    assert support.state_sha(parent)==p['source_algorithm']
    parts=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    digest=support.state_sha({k:parent[k] for k in parts})
    assert digest==support.state_sha({k:initial[k] for k in parts})==initial['continuation']['imported_component_sha256']
    assert len(initial['optimizer']['state'])==91 and initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0} and not initial['evaluations']
    assert initial['ancestry']['source_schema']==parent['schema'] and initial['ancestry']['continuation']==parent['continuation']
    for k in ('history','orders','exposure','evaluations','ancestry','actual_calls'):
        assert support.state_sha(initial['ancestry'][k])==support.state_sha(parent[k])
    scaler=dict(parent['scaler']);steps={i:int(v['step']) for i,v in parent['optimizer']['state'].items()};source_calls=dict(parent['actual_calls']);del parent
    trained={n for group in initial['optimizer_parameter_names'] for n in group};full=[]
    for local,path in ((0,run/'import/checkpoints/state_0000.pt'),(2048,run/'model/checkpoints/state_2048.pt'),(4096,run/'model/checkpoints/state_4096.pt')):
        s=torch.load(path,weights_only=True,map_location='cpu');mark=read(path.with_suffix('.json'));sha=support.sha(path);alg=support.state_sha(s)
        assert sha==mark['sha256'] and alg==mark['algorithm_sha256'] and s['schema']=='fdt_three_memory_language_codec_v38_fullstate'
        assert s['attempts']==local and s['updates']==local-s['skips'] and s['cursor']==8*local and not s['gradients']
        assert s['continuation']==initial['continuation'] and s['migration']==initial['migration'] and s['ancestry']==initial['ancestry']
        assert [h['rows'] for h in s['history']]==s['orders'][:local] and len(s['history'])==local
        assert s['frozen_sha']==initial['frozen_sha'] and all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in trained)
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        for i,v in s['optimizer']['state'].items():assert int(v['step'])==steps[i]+s['updates']
        scale,tracker=scaler['scale'],scaler['_growth_tracker'];skips=[]
        for i,h in enumerate(s['history']):
            assert h['attempt']==i+1 and h['denoise_calls']==h['copy_bytes']==0 and h['mask_probability']==.5
            assert abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5
            if h['AMP_skip']:scale*=scaler['backoff_factor'];tracker=0;skips.append(i+1)
            else:
                tracker+=1
                if tracker==scaler['growth_interval']:scale*=scaler['growth_factor'];tracker=0
            assert h['scale']==scale
        assert s['scaler']==dict(scaler,scale=scale,_growth_tracker=tracker)
        if local:
            factor=.1+.9*(1+math.cos(math.pi*(local-1)/4096))/2
            assert all(g['lr']==g['base_lr']*factor for g in s['optimizer']['param_groups'])
        for label,value in s['evaluations'].items():assert support.sha(run/(label if label.endswith(('.pt','.json')) else label+'.json'))==value
        full.append({'attempt':local,'updates':s['updates'],'AMP_skips':skips,'bytes':path.stat().st_size,'sha256':sha,'algorithm_sha256':alg,'stage':s['worker_stage'],'scaler':s['scaler']})
        if local==4096:state=s
        else:del s
    assert full[-1]['sha256']==r['latest']['sha256'] and full[-1]['algorithm_sha256']==r['latest']['algorithm_sha256']
    assert count(state,'attempts')==r['global_attempts']==120804 and count(state,'updates')==r['global_updates']==120759
    assert len(state['orders'])==4096 and sorted(i for b in state['orders'] for i in b)==list(range(32768))
    assert state['exposure']==r['exposure'] and state['exposure']['target_UTF8_bytes']==2926478 and state['exposure']['selected_rows']==32768
    for k in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):assert state['exposure'][k]==sum(h[k] for h in state['history'])
    assert state['allocation_calls']==r['actual_calls'] and state['actual_calls']==r['cumulative_calls']
    online=[{'attempts':[start+1,start+1024],**{k:statistics.mean(h[k] for h in state['history'][start:start+1024]) for k in ('loss','clean_CE','corrupted_prefix_CE')}} for start in (0,1024,2048,3072)]
    external=state['external_closed_assessment_calls'];assert external==p['external_closed_assessment_calls'] and external['total']==1936
    del state,s,initial
    f,y=read(Path(p['focused_artifact'])/'features.json'),read(Path(p['focused_artifact'])/'targets.json')
    bf,by=read(Path(p['artifact'])/'features.json'),read(Path(p['artifact'])/'targets.json')
    tr,hr=[read(run/(name+'_reconstruction.json')) for name in ('focused_final','held_final')]
    train,ts=raw_rows(tr,f);held,hs=raw_rows(hr,[bf['dev'][i] for i in p['generation_indices']],HELD)
    for i,t in enumerate(train):
        t.update(ROOT_reading='Exact source span; all source details identical.' if t['exact'] else COMMENTS[i],
            meaning_grade='exact_source_span' if t['exact'] else 'partial_proposition' if i in PARTIAL else 'not_faithful',confirmed_partial_proposition=i in PARTIAL)
    assert ts['exact']==20 and hs['exact']==0 and sum(t['confirmed_partial_proposition'] for t in train)==6
    tt,focus_metrics=logits_review(run/'focused_final_logits.pt',y,r['focused_final'])
    ht=torch.load(run/'held_final_logits.pt',weights_only=True,map_location='cpu')
    align={'focused64':prefix_alignment(tt,tr,range(64)),'held16':prefix_alignment(ht,hr,p['generation_indices'])}
    memory=read(run/'root_three_memory_readouts_review.json')
    assert memory['status']=='V38_SAVED_THREE_MEMORY_READ_PATHS_VERIFIED' and memory['plan_sha256']==end['plan_sha256']
    calls={'encode':4096+8+16+32+4+6,'denoise':0,'decoder':8192+24+96+6+stop_count(tr)+stop_count(hr)}
    assert calls==r['actual_calls'] and sum(calls.values())==14347 and r['cumulative_calls']=={k:source_calls[k]+calls[k] for k in calls}
    old=read(R/'runs/three_memory_codec_20261029_v37/root_reconstruction_similarity.json')
    gains=[i for i,t in enumerate(train) if t['exact'] and not old['focused64'][i]['exact']]
    losses=[i for i,t in enumerate(train) if not t['exact'] and old['focused64'][i]['exact']]
    out={'status':'V38_COMPLETE_FOURTH_BROAD_EPOCH_ROOT_REVIEWED','plan_sha256':end['plan_sha256'],'fullstates':full,
        'source_all91_model_Adam_scaler_ALLstoredRNG_exact':True,'source_component_sha256':digest,'all_model_Adam_finite':True,
        'frozen_original_reasoning_language_prefix_basis_unchanged':True,'new_finite_cosine_without_Adam_reset_verified':True,
        'new_attempts':4096,'new_updates':4094,'AMP_skips':full[-1]['AMP_skips'],'global_attempts':120804,'global_updates':120759,
        'actual_calls':calls,'total_new_calls':14347,'cumulative_calls':r['cumulative_calls'],'external_closed_assessment_calls':external,
        'exposure':r['exposure'],'online_loss_blocks':online,'focused64_reconstruction':ts,'held16_reconstruction':hs,
        'focused64_partial_propositions':6,'held_whole_faithful':0,'held_confirmed_partial_propositions':0,'ROOT_read_all80':True,
        'focused64_exact_gains_vs_V37':gains,'focused64_exact_losses_vs_V37':losses,'same_first_teacher_BOS_error_position_all80':True,
        'saved_teacher_BOS_prefix_alignment':align,'saved_logit_metrics':{'focused_final':focus_metrics,'held_final':memory['held256_saved_logit_metrics']},
        'memory_readout_review_sha256':support.sha(run/'root_three_memory_readouts_review.json'),
        'held16_normal_NLL':memory['normal_same16_NLL'],'held16_memory_ablations':memory['each_memory_disabled_same16'],
        'all_owned_proxy_gone_before_analysis':True,'ROOT_worker_live_observed':True,'additional_model_construction_forward':0,
        'CUDA_replay_performed':False,'denoise_calls':0,'literal_copy_bytes':0,'E2_final_accessed':False,
        'conclusion':'Observed held teacher byte accuracy improved76.1279->76.8475%; held whole faithful0/16 and focused64exact21->20. Exact date/name/negation/event details still fail, while broad teacher learning continues. All80teacher and BOS first-error positions agree on saved data; this does not isolate extractor vs decoder or prove full semantics. Broad codec learning remains appropriate; no global failure/perfection claim.'}
    write(run/'root_result.json',out);write(run/'root_reconstruction_similarity.json',{'focused64':train,'held16':held,'summary':{'focused64':ts,'held16':hs}})
    lines=['# V38 ROOT reads all80 SAME-input source/reconstructions','',out['conclusion']]
    for label,rows in [('TRAIN_REFERENCE',train),('HELD',held)]:
        for t in rows:lines+=['',f"## {label} {t['index']}",f"Source: {t['source']}",f"Reconstruction: {t['reconstruction']}",f"ROOT: {t['ROOT_reading']}"]
    with (run/'root_all80_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('status','fullstates','focused64_reconstruction','held16_reconstruction','focused64_exact_gains_vs_V37','focused64_exact_losses_vs_V37','online_loss_blocks','total_new_calls')},ensure_ascii=True))
if __name__=='__main__':main()
