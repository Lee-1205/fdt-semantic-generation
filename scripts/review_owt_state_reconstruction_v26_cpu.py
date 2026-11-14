"""Saved-state/text review only; no model forward or new evaluation.

Legacy worker names 'free/generation' mean reconstruction conditioned on the
state extracted from the SAME input sentence, without gold previous bytes.
They do not mean unconditional creation, reasoning, or new output-meaning tasks.
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import text_generation_support_v1 as support
from owt_fuzzy_engine_v26 import COMPONENTS, cumulative_count
from research.owt_prefix_reconstruction_v19 import OBJECTIVE
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_owt_free_similarity_20261028_cpu import metrics, chrf, REVIEWS

# Source proposition fields are reused from the identical panel; comments below
# are ROOT's NEW reading of the actual V24 saved outputs, not old evaluations.
from review_owt_state_reconstruction_v22_cpu import READINGS as OLD_DEV,TRAIN_READINGS as OLD_TRAIN
DEV_COMMENTS=[
'He did not remains, but played is not acknowledge; culpability, deception and the public relation are absent.',
'Featurelance is not Fans; waving flowers, standing and anticipation are absent.',
'Uppercase fragments do not preserve Disney, donut sundae and the quoted surprised lack of knowledge.',
'Globally is now exactly retained, but many few is confused and ransom payments/victims/backup restoration are absent.',
'Departs is not Deathstroke; assault team, Slade Wilson, Moscow and the panel relation are absent.',
'Countries/name-like words replace the actual Israel-Syria-Assad stable ceasefire/governance relation; prior Israel name not retained.',
'She fared is not she failed; sobriety test and DUI charge are absent.',
'I and not are fragments; taking items from the stall and leaving without paying are absent.',
'But more remains; no explosion, speech restrictions and causal effectiveness are absent.',
'Carolina is not the preserved scout/player names; radar, barely and three-month relation are absent.',
'We and activity are generic pieces; second academic, anonymous article and request for review are absent.',
'Weubly is not Wubbels; lawyer, inaction and main concern relation are absent.',
'Country-like/list structure does not preserve gunmen holding vessels from the listed countries.',
'No preserved Jamaica, bobsled qualification, Sochi or Winter Olympics relation.',
'Prevent survives as a word but Japanes is not Janissa; potential assassination and wall are missing.',
'Types/ integrating are domain hints; named functions and temporary-string/pointer relation are absent.'
]
TRAIN_COMMENTS=[
'More people and critical are retained words, but critical condition, arriving and late Friday are absent.',
'Deployed is not delay; five/president fragments do not preserve Clinton, postponed answer or primary sweep uncertainty.',
'Court, Brazau, future appeal intention and sentence are absent.',
'Avoid survives as a word; Eduardo, cash-payment request and bank/card fees are missing.',
'Norma AROsk is not NATO; thumb-drive purchase/use and Russian network entry are absent.',
'They will probably have is retained but someone is not some opinions; bike and the opinion target are absent.',
'The tweets were survives, but go fasted is not flagged; source and forwarding are absent.',
'No return is not no more final qualifying tournaments; season-end conditional and qualification relation are absent.',
'Bill survives but Hampson is not Haslam; panel discussion and governor reference are absent.',
'Donations survives; support correspondence, recipient family and the actual postal address are absent.',
'It just goes to remains; show-you and the exact moral-compass judgment are absent.',
'forces and and U.S. are retained, but diplomats, Iraq and reaffirming are absent. Source is fragmentary; no invented complete gold proposition.',
'I am own is not I am down for it; film, screening in China and August11 are absent.',
'We will be taking order retains a longer lexical action lead; the actual2012 order interval and shipment schedule are replaced by1999/2015 and unrelated material. No complete schedule proposition.',
'Craig Edual Charlis is not Craig Charles; David Emanuel and the two other names are not preserved. The two exact names seen in V25 have regressed.',
'She then told the 25-year-old is an exact subject-action-recipient fragment, including the correct age. This minimal telling fragment is credited separately; cop identity and do-not-move instruction are still absent.'
]
READINGS=[tuple(x[:3])+('source polarity and qualifiers not faithfully retained',x[4],DEV_COMMENTS[i]) for i,x in enumerate(OLD_DEV)]
TRAIN_READINGS=[tuple(x[:3])+('source polarity and qualifiers not faithfully retained',x[4],TRAIN_COMMENTS[i]) for i,x in enumerate(OLD_TRAIN)]

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    torch.set_num_threads(2)
    run = ROOT/'runs/owt_fuzzy_reconstruction_20261029_v26'
    p, result = read(run/'plan.json'), read(run/'result.json')
    end = read(run/'supervisor/final_status.json')
    native = read(run/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==p['source_sha']
    initial = torch.load(run/'import/checkpoints/state_0000.pt', map_location='cpu', weights_only=True)
    parent = torch.load(p['source'], map_location='cpu', weights_only=True)
    component_sha = support.state_sha({k: parent[k] for k in COMPONENTS})
    assert component_sha==support.state_sha({k: initial[k] for k in COMPONENTS})
    assert initial['ancestry']['history']==parent['history']
    source_scaler = dict(parent['scaler'])
    assert initial['ancestry']['source_global_updates']==63438 and initial['ancestry']['source_global_attempts']==63460
    assert initial['ancestry']['ancestry']==parent['ancestry']
    assert initial['ancestry']['external_closed_assessment_calls']==parent['external_closed_assessment_calls']
    assert initial['ancestry']['orders']==parent['orders'] and initial['ancestry']['exposure']==parent['exposure']
    del parent
    fullstates = []
    for local, path in ((0, run/'import/checkpoints/state_0000.pt'), (4096, run/'model/checkpoints/state_4096.pt'), (8192, run/'model/checkpoints/state_8192.pt')):
        s = torch.load(path, map_location='cpu', weights_only=True)
        assert s['schema']=='fdt_owt_fuzzy_AE_v26_fullstate' and s['training_objective']==OBJECTIVE
        assert s['loss_reduction']=='elementwise_none_then_sum' and s['attempts']==local and not s['gradients']
        assert s['ancestry']==initial['ancestry']
        steps = Counter(int(v['step']) for v in s['optimizer']['state'].values())
        assert steps=={63438+s['updates']:63}
        assert s['cursor']==local*8 and len(s['history'])==local
        assert [h['rows'] for h in s['history']]==s['orders'][:local]
        scale, tracker = source_scaler['scale'], source_scaler['_growth_tracker']
        for h in s['history']:
            if h['AMP_skip']:
                scale *= source_scaler['backoff_factor']; tracker=0
            else:
                tracker += 1
                if tracker==source_scaler['growth_interval']:
                    scale *= source_scaler['growth_factor']; tracker=0
            assert scale==h['scale']
        expected = dict(source_scaler, scale=scale, _growth_tracker=tracker)
        assert s['scaler']==expected
        trained = {n for group in s['optimizer_parameter_names'] for n in group}
        assert all(torch.equal(v, initial['model'][n]) for n,v in s['model'].items() if n not in trained)
        assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in s['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        fullstates.append({'attempt':local,'updates':s['updates'],'sha256':support.sha(path),'algorithm_sha256':support.state_sha(s),'Adam_steps':dict(steps),'scaler':expected})
        if local==8192: state=s
        else: del s
    assert state['external_closed_assessment_calls']==result['external_closed_assessment_calls']
    assert result['external_closed_assessment_calls']['total']==1936 and result['external_closed_assessment_calls']['included_in_source_ledger'] is False
    assert fullstates[-1]['sha256']==result['latest']['sha256'] and fullstates[-1]['algorithm_sha256']==result['latest']['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'],state['skips'])==(8192,8189,65536,3)
    assert cumulative_count(state,'attempts')==71652 and cumulative_count(state,'updates')==71627
    history = state['history']; skips=[h['attempt'] for h in history if h['AMP_skip']]
    for epoch in range(2):
        assert sorted(i for b in state['orders'][epoch*4096:(epoch+1)*4096] for i in b)==list(range(32768))
    assert all(h['outcome']==('AMP_SKIP' if h['AMP_skip'] else 'UPDATED') for h in history)
    assert all(abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5 for h in history)
    assert state['exposure']==result['exposure']
    for key, expected in (('eligible_prefix_bytes',5852956),('masked_prefix_bytes',877944),('decoder_supervised_symbol_occurrences',11836984)):
        assert sum(h[key] for h in history)==state['exposure'][key]==expected
    assert state['actual_calls']==result['cumulative_calls'] and state['allocation_calls']==result['actual_calls']
    epoch_means=[{k:statistics.mean(h[k] for h in history[i:i+4096]) for k in ('loss','clean_CE','corrupted_prefix_CE')} for i in (0,4096)]
    del state, initial, history, s
    raw, old = read(run/'final_generation.json'), read(run/'parent_reference_generation.json')
    summaries, allrows = {}, {}
    for mode, rows in raw.items():
        assert len(rows)==16
        scored=[]
        for i,(row,previous) in enumerate(zip(rows,old[mode])):
            assert row['id']==previous['id'] and row['source']==previous['source']
            g=row['generated']; assert bytes.fromhex(g['raw_hex']).decode('utf-8')==g['text']
            assert row['reconstruction_exact']==(g['well_formed'] and row['source']==g['text'])
            item={'index':i,'id':row['id'],'source':row['source'],'reconstruction':g['text'],**metrics(row['source'],g['text'])}
            if mode=='normal':
                subject,action,obj,polarity,relations,comment=READINGS[i]
                item.update(source_proposition=('Failure to acknowledge personal culpability for deceiving the American public' if i==0 else REVIEWS[i][0]),subject=subject,action=action,object=obj,polarity=polarity,relations=relations,ROOT_reading=comment,meaning_grade='meaning_not_preserved',faithful_paraphrase=False,partial_proposition=False)
            scored.append(item)
        summary={k:statistics.mean(x[k] for x in scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
        top=0
        for i,row in enumerate(scored):
            vals=[chrf(other['source'],row['reconstruction']) for other in scored]
            top+=vals[i]==max(vals) and vals.count(max(vals))==1
        summary.update(own_source_unique_top1_of16=top,exact=sum(r['reconstruction_exact'] for r in rows),UTF8=sum(r['generated']['utf8_valid'] for r in rows),EOS=sum(r['generated']['eos'] for r in rows))
        summaries[mode]=summary; allrows[mode]=scored
    train_raw=read(run/'final_train_reconstruction.json')
    train_scored=[]
    for i,row in enumerate(train_raw):
        g=row['generated'];assert bytes.fromhex(g['raw_hex']).decode('utf-8')==g['text']
        feature=read(Path(p['artifact'])/'features.json')['train'][p['train_reconstruction_indices'][i]]
        assert feature=={'id':row['id'],'prompt':row['source']}
        assert row['reconstruction_exact']==(g['well_formed'] and row['source']==g['text'])
        subject,action,obj,polarity,relations,comment=TRAIN_READINGS[i]
        train_scored.append({'index':i,'source':row['source'],'reconstruction':g['text'],**metrics(row['source'],g['text']),'subject':subject,'action':action,'object':obj,'polarity':polarity,'relations':relations,'ROOT_reading':comment,'meaning_grade':'meaning_not_preserved','faithful_paraphrase':False,'partial_proposition':False})
    train_summary={k:statistics.mean(r[k] for r in train_scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
    train_scored[15].update(meaning_grade='minimal_subject_action_recipient_fragment',partial_proposition=True,partial_scope='She told the25-year-old only; police identity and prohibition NOT preserved')
    train_summary.update(exact=sum(r['reconstruction_exact'] for r in train_raw),UTF8=sum(r['generated']['utf8_valid'] for r in train_raw),EOS=sum(r['generated']['eos'] for r in train_raw),ROOT_read_all16=True,ROOT_faithful_paraphrase=0,ROOT_partial_proposition=1)
    allrows['train_normal']=train_scored
    calls={'encode':8192+32+4+4,'denoise':0,'decoder':16384+96+sum(stop_count(rows) for rows in raw.values())+stop_count(train_raw)}
    assert calls==result['actual_calls']
    data=torch.load(run/'final_logits.pt',map_location='cpu',weights_only=True)
    targets=read(Path(p['artifact'])/'targets.json')['dev']; ids=data['target_ids']; active=ids!=257
    for i,target in enumerate(targets):
        assert ids[i].tolist()==target['target_ids']+[257]*(129-len(target['target_ids']))
    saved_metrics={}
    for mode in raw:
        x=data[mode].double(); nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1)
        correct=int((x.argmax(-1)[active]==ids[active]).sum()); score=float(nll[active].mean())
        assert correct==result['final'][mode]['correct'] and abs(score-result['final'][mode]['NLL'])<.001
        saved_metrics[mode]={'NLL':score,'correct':correct,'symbols':int(active.sum()),'byte_accuracy':correct/int(active.sum())}
    mu=data['memberships'].double()
    out={'status':'V26_STATE_CONDITIONED_RECONSTRUCTION_COMPLETE_ROOT_REVIEWED',
         'new_attempts':8192,'new_updates':8189,'AMP_skip_attempts':skips,'global_attempts':71652,'global_updates':71627,
         'source_import_component_sha256':component_sha,'source_all_components_exact':True,'fullstates':fullstates,
         'frozen_reasoning_prefix_basis_unchanged':True,'model_Adam_finite':True,'all_RNG_and_nested_ancestry_preserved':True,
         'external_closed_assessment_calls':result['external_closed_assessment_calls'],'resource_scans':result['resource_scans'],
         'online_epoch_means':epoch_means,'supervision':result['exposure'],'masked_fraction':877944/5852956,
         'saved_logit_metrics':saved_metrics,'state_conditioned_reconstruction':summaries,'fixed_train_reconstruction':train_summary,
         'fuzzy_statistics':{'coverage_anchors':int((mu.sum((0,1))>0).sum()),'mean_membership_mass':float(mu.sum(-1).mean())},
         'ROOT_read_all_dev16_and_train16':True,'ROOT_faithful_paraphrase':0,'ROOT_partial_proposition':0,'ROOT_train_minimal_subject_action_recipient_fragments':1,'ROOT_nonblind':True,
         'actual_calls':calls,'total_new_calls':sum(calls.values()),'seconds':result['seconds'],'plan_sha256':end['plan_sha256'],
         'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False,
         'scope':'Same sentence X -> original fuzzy extracted Z(X) -> reconstruction Xhat. No reasoning or output-specific meaning denoising. Legacy free naming is not unconditional or open-ended generation.',
         'limitations':['Teacher-prefix metrics allow correct past bytes; do not alone establish whole-sentence reconstruction.','State-only decoding starts at BOS with the same input-derived state; no gold prefix/length or raw context bypass.','Repeatedly observed dev and unblinded lexical/manual scores; not blind semantic accuracy.','Small data and incomplete learned reconstruction do not establish whole FDT/fuzzy failure or a sole extractor/decoder cause.']}
    for filename,obj in (('root_result.json',out),('root_state_reconstruction_similarity.json',{'summary':summaries,'train_summary':train_summary,'rows':allrows,'additional_model_forwards':0})):
        with (run/filename).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    lines=['# V26: same-input state-conditioned reconstruction / ROOT reading','',out['scope'],'','No new model forward. Same observed16 dev documents and fixed16 training documents, kept separate; unblinded qualitative reading.']
    for row in allrows['normal']+allrows['train_normal']:
        lines += ['',f"## {'train' if row in allrows['train_normal'] else 'dev'} {row['index']}",f"Source: {row['source']}",f"Reconstruction: {row['reconstruction']}",f"Subject/action/object: {row['subject']} / {row['action']} / {row['object']}",f"Polarity: {row['polarity']}",f"Relations: {row['relations']}",f"ROOT: {row['ROOT_reading']}",f"Grade: {row['meaning_grade']}; chrF={row['chrF2_1to6']:.3f}"]
    with (run/'root_all32_state_reconstruction_reading.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps(out,ensure_ascii=True))

if __name__=='__main__': main()
