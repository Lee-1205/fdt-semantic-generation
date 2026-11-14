"""Source-bound long fields and rejection receipts, no NN or new trace replay."""
import collections,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN

def main():
    run=ROOT/'runs/critical_repair_20261113_v88';p=io.load_json(run/'plan.json');review=io.load_json(run/'root_saved_review.json');panels=io.load_json(run/'root_decoded_panels.json');items=io.load_json(p['eval_inputs']);longs=[x for x in items if x['panel']=='long']
    old=io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json');targets=io.load_json(Path(old['old_eval_data'])/'long_targets.json')[:2];metrics={};snippets=[]
    for phase in ('baseline','final'):
        counts=collections.Counter();bad=[]
        for i,(info,t,r) in enumerate(zip(longs,targets,panels[phase+'_long'])):
            source=info['row']['prompt'].encode();out=r['output'].encode();assert len(source)==len(out) and info['row']['id']==r['id']
            counts['source_bytes']+=len(source);counts['decoded_byte_errors']+=sum(a!=b for a,b in zip(source,out))
            for event in t['events']:
                a,b=event['byte_start'],event['byte_end'];text=source[a:b].decode();actual=out[a:b].decode();match=PATTERN.fullmatch(text);assert match
                for group in ('quantity','reserve','actor','checker','item','order'):
                    start,end=match.span(group);counts[group+'_positions']+=1
                    if text[start:end]!=actual[start:end]:counts[group+'_errors']+=1
                pred=text.index(' moved ')+1;counts['predicate_positions']+=1;counts['predicate_errors']+=actual[pred:pred+5]!='moved'
            for m in re.finditer(rb'episode (\d+)',source):counts['episode_positions']+=1;counts['episode_errors']+=source[m.start(1):m.end(1)]!=out[m.start(1):m.end(1)]
            for m in re.finditer(rb'\b(?:shelf|drawer|box|gate)\b',source):counts['location_noun_positions']+=1;counts['location_noun_errors']+=source[m.start():m.end()]!=out[m.start():m.end()]
            if phase=='final':
                for pos in (0,len(source)//2,max(0,len(source)-260)):snippets.append(dict(row=i,offset=pos,source=source[pos:pos+260].decode(),output=out[pos:pos+260].decode()))
        metrics[phase]=dict(counts)
    state=torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True)['runtime'];alphas=collections.Counter();lowest=collections.Counter();allcheck=collections.Counter()
    for h in state['history']:
        receipt=h['update_receipt']
        if receipt['accepted']:alphas[str(receipt['alpha'])]+=1
        else:
            m=receipt['reference_checks'][-1]['measurements'];key=','.join(k for k in ('new_critical_flips','new_noncritical_EOS_flips','weak_critical_margin_violations') if m[k]>0);lowest[key]+=1
        for check in receipt['reference_checks']:
            for key in ('new_critical_flips','new_noncritical_EOS_flips','weak_critical_margin_violations'):allcheck[key+'_positive_checks']+=check['measurements'][key]>0
    target=torch.load(state['bank_ref']['targets'],map_location='cpu',weights_only=True);features=torch.load(state['bank_ref']['features'],map_location='cpu',weights_only=True);cal=target['bank_runtime']['calibration'];final=torch.load(state['reference_final']['path'],map_location='cpu',weights_only=True)
    headroom=(final['margin']-cal['required_margin']).masked_fill(~cal['weak_floor_mask'],torch.inf);values,indices=headroom.flatten().sort();near=[];rows=io.load_json(Path(p['data'])/'train_features.json')
    for value,index in zip(values[:5],indices[:5]):
        patch,role=divmod(int(index),16);rg=next(x for x in features['source_ranges'] if x['start']<=patch<x['end']);row=rows[rg['source_index']];offset=rg['source_patch_indices'][patch-rg['start']]*16+role
        near.append(dict(source_index=rg['source_index'],source_id=row['id'],offset=offset,headroom=float(value),initial_margin=float(cal['initial_margin'][patch,role]),required=float(cal['required_margin'][patch,role]),
            final_margin=float(final['margin'][patch,role]),gold=int(cal['labels'][patch,role]),source_context=row['prompt'].encode()[max(0,offset-45):offset+60].decode(errors='replace')))
    manual=dict(ROOT_read_all5_changed_long_contexts=True,ROOT_read_start_middle_end_6_snippets=True,short192_before_after_byte_identical=True,
        changed_source_word_locations=review['changed_outputs']['long'],interpretation='4 corrected source bytes:3 repeated remove; boundaries and1episode1359. 1 new byte corruption:drawer->draber. Numericquantity/name/predicate residuals remain. Count4does not mean4independent semantic facts; no general promotion.',
        limitation='Not all524071long bytes read manually. Source-bound templates/fields are correlated and already observed; natural unseen semantic generalization not certified.')
    report=dict(status='ROOT_V88_MEANING_REJECTION_REVIEW_COMPLETE',long_source_bound_fields=metrics,all5_actual_changed_contexts=review['changed_outputs']['long'],start_mid_end=snippets,
        accepted_alpha=dict(alphas),rejected_lowest_alpha_causes=dict(lowest),all_candidate_check_counts=dict(allcheck),last_accepted_attempt=max(h['attempt'] for h in state['history'] if h['update_receipt']['accepted']),
        final_weak_reference_closest_to_floor=near,manual_reading=manual,extra_model_calls=0,
        cause_limit='Saved rejected-candidate receipts contain counts, not blocking position IDs or candidate weights/logits. Closest final positions are candidates, not proof that the same position blocked every rejection. Do not rerun same trial just to add logs.',
        next='Do not repeat same192 or lower floor automatically. Consider a TRAIN-only feasible-direction/actual-Adam-delta constraint correction with original V80 anchor retained; first prove finite convex constraint and exact rollback on CPU. A new run must not reset margin floors from the already reduced V88 values (ratchet). Diffusion not a remedy for this confirmed readout constraint conflict; future thought-only trial freezes entire codec.')
    io.atomic_json(run/'root_meaning_and_rejection_review.json',report);print(json.dumps(dict(metrics=metrics,accepted_alpha=dict(alphas),rejected_lowest_alpha_causes=dict(lowest),candidate_checks=dict(allcheck),closest=near[:2],snippets=snippets),ensure_ascii=False))

if __name__=='__main__':main()
