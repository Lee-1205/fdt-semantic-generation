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
from owt_fuzzy_engine_v20 import COMPONENTS, cumulative_count
from research.owt_prefix_reconstruction_v19 import OBJECTIVE
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_owt_free_similarity_20261028_cpu import metrics, chrf, REVIEWS

# ROOT's unblinded reading of the actual sixteen saved normal reconstructions.
# Similar spellings/pronouns alone are not credited as a preserved proposition.
READINGS = [
 ('He', 'acknowledging culpability', 'American public/deceit', 'source denial missing', 'responsibility for deceptive act missing', 'He survives only; continued/playing/lawmaking do not restore the claim.'),
 ('fans', 'waving and standing', 'light-up flowers', 'anticipation missing', 'actor/actions/accessory links missing', 'Fanstine is a spelling fragment; no correct action is expressed.'),
 ('Disney and speaker', 'having/knowing', 'donut sundae', 'surprise and lack of knowledge missing', 'quoted question missing', 'Uppercase and negation fragments do not retain Disney or the sundae; Texas is invented.'),
 ('victims', 'restoring systems', 'backups and ransom', 'few payments missing', 'causal explanation missing', 'Neither ransomware payments nor recovery from backups survives.'),
 ('assault team and Slade Wilson', 'facing off', 'Moscow panel', 'no relevant negation', 'panel/team/person/location missing', 'Names and confrontation are absent; unrelated years/numbers appear.'),
 ('Israel, Syria, Assad family', 'maintaining ceasefire/ruling', 'stable ceasefire', 'war versus ceasefire contrast missing', 'time/governance relation missing', 'Political-sounding words do not preserve the named countries or claim.'),
 ('she', 'failing test/being charged', 'sobriety test and DUI', 'source failure missing', 'test result and charge missing', 'She and officers are surface/topic hints; the DUI proposition is not reconstructed.'),
 ('I', 'taking items/leaving without paying', 'Yanick stall and items', 'possible ambiguous reversal: did not pay versus did not take', 'taking and nonpayment must remain distinct', 'Output I didn\u2019t taken package is ungrammatical; negation may attach to taking instead of paying. Not a proven formal contradiction, and not faithful.'),
 ('speech restrictions', 'being effective', 'speech at work', 'no explosion missing', 'causal explanation missing', 'But survives; speech, restriction, negation and cause are absent.'),
 ('Robby Fabbri, Chiarelli, Bowman', 'being barely on radar', 'player', 'barely qualifier missing', 'scouting and three-month relation missing', 'Named people and relation are absent; unrelated 19999 appears.'),
 ('we and second academic', 'asking to review', 'anonymous article copy', 'no relevant negation', 'request/reviewer/anonymity missing', 'We alone does not preserve an academic review request.'),
 ('Wubbels and lawyer', 'naming as concern', 'others\u2019 inaction', 'inaction missing', 'person/concern relationship missing', 'The people and concern about inaction are absent; will not is unrelated.'),
 ('gunmen', 'holding', 'vessels from six countries', 'no relevant negation', 'holder/vessels/origin links missing', 'Name-like tokens do not preserve gunmen, ships or the country list.'),
 ('Jamaican two-man bobsled team', 'qualifying', 'Sochi Winter Olympics', 'no relevant negation', 'team/event/qualification missing', 'The qualification claim is absent; U.S. is invented.'),
 ('Janissa', 'preventing', 'assassination with wall', 'prevention missing', 'agent/action/instrument missing', 'Japanish resembles a name but is not preserved Janissa; wall and prevention are absent.'),
 ('get_input_string and tokenize_string2', 'returning/building', 'temporary string and pointer array', 'no relevant negation', 'pointers into temporary string missing', 'Function-like spelling does not retain function names or the lifetime/pointer relation; 2016 is invented.'),
]

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    torch.set_num_threads(2)
    run = ROOT/'runs/owt_fuzzy_reconstruction_20261028_v20'
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
    del parent
    fullstates = []
    for local, path in ((0, run/'import/checkpoints/state_0000.pt'), (2048, run/'model/checkpoints/state_2048.pt'), (4096, run/'model/checkpoints/state_4096.pt')):
        s = torch.load(path, map_location='cpu', weights_only=True)
        assert s['schema']=='fdt_owt_fuzzy_AE_v20_fullstate' and s['training_objective']==OBJECTIVE
        assert s['loss_reduction']=='elementwise_none_then_sum' and s['attempts']==local and not s['gradients']
        assert s['ancestry']==initial['ancestry']
        steps = Counter(int(v['step']) for v in s['optimizer']['state'].values())
        assert steps=={23246+s['updates']:63}
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
        if local==4096: state=s
        else: del s
    assert fullstates[-1]['sha256']==result['latest']['sha256'] and fullstates[-1]['algorithm_sha256']==result['latest']['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'],state['skips'])==(4096,4095,32768,1)
    assert cumulative_count(state,'attempts')==27348 and cumulative_count(state,'updates')==27341
    history = state['history']; skips=[h['attempt'] for h in history if h['AMP_skip']]
    for epoch in range(2):
        assert sorted(i for b in state['orders'][epoch*2048:(epoch+1)*2048] for i in b)==list(range(16384))
    assert all(h['outcome']==('AMP_SKIP' if h['AMP_skip'] else 'UPDATED') for h in history)
    assert all(abs(h['loss']-.5*(h['clean_CE']+h['corrupted_prefix_CE']))<1e-5 for h in history)
    assert state['exposure']==result['exposure']
    for key, expected in (('eligible_prefix_bytes',2918756),('masked_prefix_bytes',437107),('decoder_supervised_symbol_occurrences',5903048)):
        assert sum(h[key] for h in history)==state['exposure'][key]==expected
    assert state['actual_calls']==result['cumulative_calls'] and state['allocation_calls']==result['actual_calls']
    epoch_means=[{k:statistics.mean(h[k] for h in history[i:i+2048]) for k in ('loss','clean_CE','corrupted_prefix_CE')} for i in (0,2048)]
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
                item.update(source_proposition=REVIEWS[i][0],subject=subject,action=action,object=obj,polarity=polarity,relations=relations,ROOT_reading=comment,meaning_grade='meaning_not_preserved',faithful_paraphrase=False,partial_proposition=False)
            scored.append(item)
        summary={k:statistics.mean(x[k] for x in scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
        top=0
        for i,row in enumerate(scored):
            vals=[chrf(other['source'],row['reconstruction']) for other in scored]
            top+=vals[i]==max(vals) and vals.count(max(vals))==1
        summary.update(own_source_unique_top1_of16=top,exact=sum(r['reconstruction_exact'] for r in rows),UTF8=sum(r['generated']['utf8_valid'] for r in rows),EOS=sum(r['generated']['eos'] for r in rows))
        summaries[mode]=summary; allrows[mode]=scored
    calls={'encode':4096+32+4,'denoise':0,'decoder':8192+96+sum(stop_count(rows) for rows in raw.values())}
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
    out={'status':'V20_STATE_CONDITIONED_RECONSTRUCTION_COMPLETE_ROOT_REVIEWED',
         'new_attempts':4096,'new_updates':4095,'AMP_skip_attempts':skips,'global_attempts':27348,'global_updates':27341,
         'source_import_component_sha256':component_sha,'source_all_components_exact':True,'fullstates':fullstates,
         'frozen_reasoning_prefix_basis_unchanged':True,'model_Adam_finite':True,'all_RNG_and_nested_ancestry_preserved':True,
         'online_epoch_means':epoch_means,'supervision':result['exposure'],'masked_fraction':437107/2918756,
         'saved_logit_metrics':saved_metrics,'state_conditioned_reconstruction':summaries,
         'fuzzy_statistics':{'coverage_anchors':int((mu.sum((0,1))>0).sum()),'mean_membership_mass':float(mu.sum(-1).mean())},
         'ROOT_read_all16':True,'ROOT_faithful_paraphrase':0,'ROOT_partial_proposition':0,'ROOT_nonblind':True,
         'actual_calls':calls,'total_new_calls':sum(calls.values()),'seconds':result['seconds'],'plan_sha256':end['plan_sha256'],
         'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False,
         'scope':'Same sentence X -> original fuzzy extracted Z(X) -> reconstruction Xhat. No reasoning or output-specific meaning denoising. Legacy free naming is not unconditional or open-ended generation.',
         'limitations':['Teacher-prefix metrics allow correct past bytes; do not alone establish whole-sentence reconstruction.','State-only decoding starts at BOS with the same input-derived state; no gold prefix/length or raw context bypass.','Repeatedly observed dev and unblinded lexical/manual scores; not blind semantic accuracy.','Small data and incomplete learned reconstruction do not establish whole FDT/fuzzy failure or a sole extractor/decoder cause.']}
    for filename,obj in (('root_result.json',out),('root_state_reconstruction_similarity.json',{'summary':summaries,'rows':allrows,'additional_model_forwards':0})):
        with (run/filename).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    lines=['# V20: same-input state-conditioned reconstruction / ROOT reading','',out['scope'],'','No new model forward. Same observed16 dev documents; unblinded qualitative reading.']
    for row in allrows['normal']:
        lines += ['',f"## {row['index']}",f"Source: {row['source']}",f"Reconstruction: {row['reconstruction']}",f"Subject/action/object: {row['subject']} / {row['action']} / {row['object']}",f"Polarity: {row['polarity']}",f"Relations: {row['relations']}",f"ROOT: {row['ROOT_reading']}",f"Grade: {row['meaning_grade']}; chrF={row['chrF2_1to6']:.3f}"]
    with (run/'root_all16_state_reconstruction_reading.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps(out,ensure_ascii=True))

if __name__=='__main__': main()
