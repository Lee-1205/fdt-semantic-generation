"""Saved TRAIN anchors, actual loss participation, and adjacent numeral risks.

No model forward, no revised historical scores, no changes to TRAIN/inference.
"""
import collections,json,re,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN


def main():
    run=ROOT/'runs/last_linear_20261113_v79';assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V79_SAVED_REVIEW_PASS'
    p=io.load_json(run/'plan.json');env=torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True);runtime=env['runtime']
    anchors=collections.Counter();groups=collections.Counter();sum_terms=collections.Counter();violations=collections.Counter()
    for ref in runtime['cache_refs']:
        t=torch.load(ref['targets'],map_location='cpu',weights_only=True);body=t['labels']<256;correct=t['teacher_correct'];mask=t['critical_mask'];weak=t['teacher_margin']<=3.
        anchors.update(correct_critical=int((body&mask&correct).sum()),weak_correct_critical=int((body&mask&correct&weak).sum()),
            correct_noncritical=int((body&~mask&correct).sum()),weak_correct_noncritical=int((body&~mask&correct&weak).sum()),incorrect_critical=int((mask&~correct).sum()))
    targets=io.load_json(Path(p['data'])/'train_targets.json')
    for h in runtime['history']:
        for m in h['micros']:
            group=targets[m['row']].get('mode','OWT');groups[group]+=1
            for key in ('critical_competitor_margin','noncritical_competitor_margin','correct_critical_KL','correct_noncritical_KL'):
                sum_terms[key]+=.2*m[key]
            for key in ('critical_margin_violations','noncritical_margin_violations','critical_margin_eligible','noncritical_margin_eligible'):violations[key]+=m[key]
    assert len(runtime['history'])==384 and sum(groups.values())==1920
    training=dict(unique_teacher_anchors=dict(anchors),head_source_groups=dict(groups),actual_group_mean_terms={k:v/384 for k,v in sum_terms.items()},
        head_position_occurrence_counts=dict(violations),groups_clipped=sum(h['grad_norm']>1. for h in runtime['history']),optimizer_groups=384,
        interpretation='Anchor was numerically active on TRAIN; this is not a bound on unseen Commit margins. Counts are repeated byte occurrences, not independent facts or parameter-gradient mass.')
    rows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2];labels=io.load_json(Path(p['old_eval_data'])/'long_targets.json')[:2]
    boundary={}
    for name,folder in [('V75',Path(p['baseline_run'])),('V77',ROOT/'runs/update_balance_20261113_v77'),('V78',ROOT/'runs/competing_margin_20261113_v78'),('V79',run)]:
        records=io.load_json(folder/'long_reconstructions.json');cards=[]
        for ri,(row,t,record) in enumerate(zip(rows,labels,records)):
            source=row['prompt'].encode();out=record['output'].encode()
            for j,event in enumerate(t['events']):
                a,b=event['byte_start'],event['byte_end'];s=source[a:b].decode();output=out[a:b].decode();match=PATTERN.fullmatch(s);assert match
                lo,hi=match.span('episode');gold=match['episode'];fixed=output[lo:hi]
                token=re.match(r'[^\s,;]+',output[lo:])
                if fixed==gold and token and token.group()!=gold:
                    actual=token.group();numeric=re.fullmatch(r'[+-]?\d+',actual)
                    tier='clear_merged_different_numeric_value' if numeric and int(actual)!=int(gold) else 'trailing_symbol_or_malformed_token_not_certified'
                    cards.append(dict(row=ri,event=j+1,source_number=gold,unchanged_fixed_span=fixed,actual_episode_token=actual,tier=tier,source=s,output=output))
        boundary[name]=dict(counts=dict(collections.Counter(c['tier'] for c in cards)),cards=cards)
    assert not torch.cuda.is_initialized()
    proof=dict(status='ROOT_V79_TRAIN_AND_ADJACENT_TOKEN_REVIEW_PASS',training=training,episode_adjacent_risks_not_in_fixed_span_counts=boundary,
        extra_model_calls=0,historical_fixed_span_scores_unchanged=True,new_training_data_or_inference_rules=False,
        limitations='Source-bound authored episode token scan is score-only; excludes already changed fixed spans, not all semantic errors. No normalization/repair of any output.')
    io.atomic_json(run/'root_training_and_boundary_review.json',proof)
    print(json.dumps(dict(status=proof['status'],training=training,adjacent={k:dict(counts=v['counts'],cards=[{a:c[a] for a in ('row','event','source_number','actual_episode_token','tier')} for c in v['cards']]) for k,v in boundary.items()})))


if __name__=='__main__':main()
