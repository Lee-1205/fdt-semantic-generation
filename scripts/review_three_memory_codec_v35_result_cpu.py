"""ROOT's saved-only fullstate/meaning review after all V35 processes exited."""
import json,math,statistics,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support
from review_three_memory_codec_v33_result_cpu import raw_rows,logits_review,trace_review
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count

TRAIN_COMMENTS={
0:'Wanting Mr_Ogunjobi to land on a team remains; Panthers is changed to partnership, so the alternative is wrong.',
4:'A reversal of U.S. policy and1948 remain; founding of Israel is lost.',
5:'RE24 recalculation and exclusion of a park adjustment remain; the SB/CS alternative is replaced with invented CBS Sericane.',
6:'WordPress extension topic remains, but installing a good plugin becomes installing good plan; no faithful full comparison.',
7:'April2015 and hive boxes/frames/new locations are replaced; no source instruction is recovered.',
9:'Gameplay is recognizable, but the sentence ending becomes malformed; exact referent and full source wording are not preserved.',
11:'Sealed container becomes second learn and wherever you go is garbled; no faithful disc/container assertion.',
12:'The distinction between definitions requiring movement and those not requiring it is lost.',
14:'Swansea and passes collapse into Swans passen; the specific promised style of play is not preserved.',
15:'April Ryan and20-year veteran remain; press-corps context and negated agenda are lost and unrelated material is added.',
17:'The story on Wall Street remains; over a century becomes very content and the duration is lost.',
18:'July2/scientists/sunflowers are retained, but sprouting becomes are should; this is a factual scaffold without the event.',
19:'Emergency workers remains; treatment of the bride for alcohol poisoning/dehydration is lost.',
20:'The setback for a22-year-old remains; preseason timing and Euro2016 details are altered or omitted.',
21:'Country replaces conscious; the central consciousness concept is changed despite the long matching prefix.',
22:'The quoted anti-Semitic incidents proposition is retained; Baroness Royall/report attribution is lost.',
24:'Ceramic Workers Union and self-management remain; played a fundamental role becomes release on, losing the assertion.',
26:'There were millions remains; the selection/choice relation changes to cool.',
27:'Bears falling to the bottom of rankings remains; lowest scores across the board is replaced.',
28:'The too-early/jump-the-gun relation is lost; camp lead words alone do not preserve the event.',
30:'Baby shower gift to a friend is lost in baby grow fish/offer fragments.',
32:'Radio etiquette and experience fragments remain but their learning relation is malformed.',
34:'The remanufactured model and the woman not liking it are both lost.',
35:'Core negated personal-comment assertion is nearly preserved; plural reports becomes singular report, so it is not perfect identity.',
36:'The grin on his face and what he said are garbled; inserted spacing does not restore the proposition.',
42:'Resilience, continued fighting and Frazier attribution are lost.',
45:'Defines/linker/compiler options become defense/linkin politics; the technical items are changed.',
47:'Effusive praise of the team and liberal smiles are garbled; mere Belichick identity is insufficient.',
48:'Seven-year delays and cost overruns are lost.',
49:'Psaki replying to a Fox News reporter is retained; James Rosen becomes Reseas, so recipient identity is wrong.',
50:'Confidence/good-intentioned people remain, but bringing it to reality becomes unrelated training/attraction wording.',
52:'Portuguese clause that Lewis owns more than150 companies is retained; collecting artworks is replaced with English fragments.',
53:'The rounded-mead comparison is lost in would re-in.',
55:'OPI My Vampire is Buff and the stated starting base are lost.',
57:'She turned herself in remains; the marriage-to-Danil-was-fake assertion becomes was found and is lost.',
58:'Ability to use built-in Bluetooth remains; speaker output endpoint is lost.',
59:'Works on all systems sold worldwide remains; regional-account requirement for online features is lost.',
62:'Self-trained element of punk music is replaced with unrelated police/defend fragments.',
63:'The appointment-strategy success conditional and resulting big win are not faithfully expressed.'}
PARTIAL={0,4,5,15,17,20,22,26,27,49,52,57,58,59}
HELD_COMMENTS=[
"He didn't is retained, but acknowledgement of personal culpability and trickery/deceit are replaced; the original negated relation is not recovered.",
"Fans, waving light-up flower accessories, anticipation and standing are absent; unrelated economic/service wording appears.",
"Uppercase style and know/do fragments do not preserve Disney/donut sundae or the surprised question.",
"Globally is retained; ransom payments/victims/restoring systems from backups are lost.",
"Death-like lead is retained; third-panel assault team, Slade Wilson and Moscow are replaced by unrelated names/locations.",
"Syria/Israel/Assad and stable ceasefire versus war are lost; surface words are insufficient.",
"She remains but the failed sobriety test/station lobby/DUI charge are not conveyed.",
"I/pay fragments remain; Yanick, taking goods and leaving without paying are lost, with no preserved negation of payment.",
"But more important remains; no explosion and effective workplace speech restrictions are lost. Topic fragments do not establish the proposition.",
"Robby Fabbri/Chiarelli/Bowman and barely-on-radar three months ago are lost.",
"We/second fragments do not preserve requesting a second academic's anonymous article review.",
"Wubbels/her lawyer and concern about others' inaction are replaced with unrelated service/computer wording.",
"The six countries are replaced, and holding vessels hostage is absent.",
"Jamaica/two-man bobsled/Olympics qualification/Sochi are absent.",
"Janissa preventing assassination with a wall becomes unrelated Japanese/visits/water wording.",
"Both function names and the temporary-string/pointer-array relationship are lost."
]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def count(s,k):
    n=0
    while k in s:n+=s[k];s=s.get('ancestry',{})
    return n

def main():
    torch.set_num_threads(2);run=R/'runs/three_memory_codec_20261029_v35'
    p,r,end=[read(run/file) for file in ('plan.json','result.json','supervisor/final_status.json')]
    terminal=read(run/'code_snapshot/root_terminal_OS_observation.json');live=read(run/'code_snapshot/root_native_launch_observation.json')
    assert end['status']=='COMPLETE' and end['exit_code']==0 and end['exit']['all_owned_processes_gone'] and terminal['native_count']==0
    assert any(row['ProcessId']==4824 and row['ParentProcessId']==16404 and row['CommandLine'] for row in live['native_rows']) and live['progress']['updates']>0
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    assert support.core_sha(R)==support.CORE_SHA and all(support.sha(path)==sha for path,sha in p['files'].items())
    parent=torch.load(p['source'],weights_only=True,map_location='cpu');initial=torch.load(run/'import/checkpoints/state_0000.pt',weights_only=True,map_location='cpu')
    assert support.state_sha(parent)==p['source_algorithm']
    parts=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    digest=support.state_sha({k:parent[k] for k in parts})
    assert digest==support.state_sha({k:initial[k] for k in parts})==initial['continuation']['imported_component_sha256']
    assert initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0} and not initial['evaluations']
    assert len(initial['optimizer']['state'])==91 and len(parent['orders'])==4096 and initial['orders']==parent['orders'][3407:]
    assert initial['ancestry']['source_schema']==parent['schema'] and initial['ancestry']['continuation']==parent['continuation']
    for k in ('history','orders','exposure','evaluations','ancestry','actual_calls'):
        assert support.state_sha(initial['ancestry'][k])==support.state_sha(parent[k])
    scaler=dict(parent['scaler']);steps={i:int(v['step']) for i,v in parent['optimizer']['state'].items()};source_calls=dict(parent['actual_calls'])
    epoch_prefix=[i for b in parent['orders'][:3407] for i in b];parent_exposure=dict(parent['exposure']);del parent
    trained={n for g in initial['optimizer_parameter_names'] for n in g};full=[]
    for local,path in ((0,run/'import/checkpoints/state_0000.pt'),(689,run/'model/checkpoints/state_0689.pt')):
        s=torch.load(path,weights_only=True,map_location='cpu');mark=read(path.with_suffix('.json'));sha=support.sha(path);alg=support.state_sha(s)
        assert sha==mark['sha256'] and alg==mark['algorithm_sha256'] and s['schema']=='fdt_three_memory_language_codec_v35_fullstate'
        assert s['attempts']==s['updates']==local and s['skips']==0 and s['cursor']==8*local and not s['gradients']
        assert s['epoch_schedule']==initial['epoch_schedule'] and s['migration']==initial['migration'] and s['continuation']==initial['continuation'] and s['ancestry']==initial['ancestry']
        assert [h['rows'] for h in s['history']]==s['orders'][:local] and len(s['history'])==local
        assert all(torch.equal(v,initial['model'][n]) for n,v in s['model'].items() if n not in trained) and s['frozen_sha']==initial['frozen_sha']
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        for i,v in s['optimizer']['state'].items():assert int(v['step'])==steps[i]+local
        scale,tracker=scaler['scale'],scaler['_growth_tracker']
        for i,h in enumerate(s['history']):
            assert h['attempt']==i+1 and h['epoch_attempt']==3408+i and not h['AMP_skip'] and h['denoise_calls']==h['copy_bytes']==0
            assert h['schedule_factor']==.1+.9*(1+math.cos(math.pi*(3407+i)/4096))/2
            assert abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5
            tracker+=1
            if tracker==scaler['growth_interval']:scale*=scaler['growth_factor'];tracker=0
            assert h['scale']==scale
        assert s['scaler']==dict(scaler,scale=scale,_growth_tracker=tracker)
        for label,sha_eval in s['evaluations'].items():assert support.sha(run/(label if label.endswith(('.pt','.json')) else label+'.json'))==sha_eval
        full.append({'attempt':local,'bytes':path.stat().st_size,'sha256':sha,'algorithm_sha256':alg,'worker_stage':s['worker_stage'],'scaler':s['scaler']})
        if local==689:state=s
        else:del s
    assert count(state,'attempts')==r['global_attempts']==108516 and count(state,'updates')==r['global_updates']==108478
    combined_rows=epoch_prefix+[i for b in state['orders'] for i in b]
    assert len(combined_rows)==len(set(combined_rows))==32768 and sorted(combined_rows)==list(range(32768))
    assert state['exposure']==r['exposure'] and r['exposure']['target_UTF8_bytes']==492660 and r['exposure']['selected_rows']==5512
    combined={k:parent_exposure.get(k,0)+state['exposure'].get(k,0) for k in set(parent_exposure)|set(state['exposure'])}
    assert combined['target_UTF8_bytes']==2926478 and combined['selected_rows']==32768 and combined['decoder_supervised_symbol_occurrences']==5918492
    assert state['allocation_calls']==r['actual_calls'] and state['actual_calls']==r['cumulative_calls']
    online={k:statistics.mean(h[k] for h in state['history']) for k in ('loss','clean_CE','corrupted_prefix_CE')}
    external=state['external_closed_assessment_calls'];assert external==p['external_closed_assessment_calls'] and external['total']==1936
    del state,s,initial
    f,y=read(Path(p['focused_artifact'])/'features.json'),read(Path(p['focused_artifact'])/'targets.json')
    bf,by=read(Path(p['artifact'])/'features.json'),read(Path(p['artifact'])/'targets.json')
    raw_train=read(run/'focused_final_reconstruction.json');raw_held=read(run/'held_final_reconstruction.json')
    train,ts=raw_rows(raw_train,f);held,hs=raw_rows(raw_held,[bf['dev'][i] for i in p['generation_indices']],HELD_COMMENTS)
    for i,t in enumerate(train):
        if t['exact']:t.update(ROOT_reading='Exact source span: all source propositions, negation, names, numbers, punctuation and original defects identical.',meaning_grade='exact_source_span',confirmed_partial_proposition=False)
        else:
            t.update(ROOT_reading=TRAIN_COMMENTS[i],meaning_grade='near_core_with_number_change' if i==35 else 'partial_proposition' if i in PARTIAL else 'not_faithful',confirmed_partial_proposition=i in PARTIAL)
    assert ts['exact']==25 and hs['exact']==0 and sum(t['confirmed_partial_proposition'] for t in train)==14
    measured={}
    for label,targets in [('focused_final',y),('held_final',by['dev'])]:
        tensors,measured[label]=logits_review(run/(label+'_logits.pt'),targets,r[label])
        if label=='held_final':held_t=tensors
    trace=trace_review(held_t['memory_trace'])
    idx=p['generation_indices'];z=held_t['normal'][idx].double();ids=held_t['target_ids'][idx];active=ids!=257
    base=float((z.logsumexp(-1)-z.gather(-1,ids[...,None]).squeeze(-1))[active].mean());ablations={}
    for memory in ('persistent','contextual','exact'):
        _,m=logits_review(run/('held16_without_'+memory+'_logits.pt'),[by['dev'][i] for i in idx],r['held16_ablations'][memory],modes=('normal',))
        ablations[memory]={**m['normal'],'NLL_delta_vs_same16_normal':m['normal']['NLL_from_saved_FP16']-base}
    calls={'encode':689+8+16+32+4+6,'denoise':0,'decoder':2*689+24+96+6+stop_count(raw_train)+stop_count(raw_held)}
    assert calls==r['actual_calls'] and sum(calls.values())==4130 and r['cumulative_calls']=={k:source_calls[k]+calls[k] for k in calls}
    baseline=read(R/'runs/three_memory_codec_20261029_v33/root_result.json')
    comparison={k:{'V33':baseline['saved_logit_metrics']['held_final']['normal'][k],
                   'V35':measured['held_final']['normal'][k]} for k in ('NLL_from_saved_FP16','byte_accuracy')}
    out={'status':'V35_COMPLETE_ONE_FINITE_BROAD_EPOCH_ROOT_REVIEWED','plan_sha256':end['plan_sha256'],'fullstates':full,
        'source_all91_model_Adam_scaler_ALLstoredRNG_exact':True,'source_component_sha256':digest,'frozen_original_reasoning_prefix_unchanged':True,
        'original_epoch_suffix_and_cosine_position_exact':True,'combined_epoch_attempts':4096,'combined_epoch_updates':4095,'combined_epoch_AMP_skips':1,
        'combined_epoch_exposure':combined,'new_attempts':689,'new_updates':689,'AMP_skips':0,'global_attempts':108516,'global_updates':108478,
        'actual_calls':calls,'total_new_calls':4130,'external_closed_assessment_calls':external,'cumulative_calls':r['cumulative_calls'],
        'focused64_reconstruction':ts,'held16_reconstruction':hs,'focused64_partial_propositions':14,'focused64_near_core_number_changed':1,
        'held16_whole_faithful':0,'held16_confirmed_partial_propositions':0,'ROOT_read_all80_source_reconstructions':True,
        'saved_logit_metrics':measured,'held16_normal_NLL_from_saved_FP16':base,'held16_memory_ablations':ablations,
        'memory_trace_independently_reconstructed':trace,'V33_to_V35_held_comparison':comparison,'V35_online_loss':online,'worker_seconds':r['seconds'],
        'all_owned_proxy_gone_before_analysis':True,'ROOT_actual_worker_live_observed':True,'additional_model_construction_forward':0,
        'CUDA_replay_performed':False,'denoise_calls':0,'literal_copy_bytes':0,'E2_final_accessed':False,
        'conclusion':'One broad three-memory epoch greatly improved observed teacher byte loss, but held selfprefix whole reconstruction remains0/16; focused64exact fell64->25 with14partial and1near-core. Continue language codec learning; no denoising/perfect semantics/fullvNext claim.'}
    write(run/'root_result.json',out);write(run/'root_reconstruction_similarity.json',{'focused64':train,'held16':held,'summary':{'focused64':ts,'held16':hs}})
    lines=['# V35 ROOT read all80 same-input reconstructions','',out['conclusion']]
    for label,rows in [('TRAIN_REFERENCE',train),('HELD',held)]:
        for t in rows:lines+=['',f"## {label} {t['index']}",f"Source: {t['source']}",f"Reconstruction: {t['reconstruction']}",f"ROOT: {t['ROOT_reading']}"]
    with (run/'root_all80_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('status','fullstates','combined_epoch_exposure','focused64_reconstruction','held16_reconstruction','held16_normal_NLL_from_saved_FP16','held16_memory_ablations','V35_online_loss','total_new_calls')},ensure_ascii=True))
if __name__=='__main__':main()
