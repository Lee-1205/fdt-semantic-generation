"""Severity/rollback analysis on saved outputs, loss weights, no model calls."""
import collections,json,re,statistics,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
from review_context_v74_meaning_cpu import kg_values
RUN=ROOT/'runs/critical_retention_20261112_v75'
def classify(source,output):
    expected=Decimal(source[:-3]);strict=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?) kg',output)
    if strict:
        return 'same_value_unit_clear' if Decimal(strict[1])==expected else 'clear_value_or_sign_change'
    # Interpretability label only, NOT a semantic-correctness oracle or repair.
    unsigned=source[1:] if source.startswith('+') else source
    if re.fullmatch(r'[^0-9+\-]+'+re.escape(unsigned),output):
        return 'numeric_core_and_kg_readable_extra_symbol_not_certified'
    if output==source[:-3]:
        return 'number_retained_unit_missing'
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)? \+g',output):
        return 'clear_value_and_unit_damage' if Decimal(output.split()[0])!=expected else 'unit_damage'
    return 'unresolved_needs_manual'
def stock_tiers(rows,targets,records):
    counts=collections.Counter();cards=[]
    for i,(f,t,r) in enumerate(zip(rows,targets,records)):
        if t.get('mode')=='unsigned':continue
        n=t.get('reserve',t.get('stored_quantity'));text=f'+{n}.25 kg'
        spans=[s for s in t['critical_spans'] if s['text']==text];assert len(spans)==1
        s=spans[0];actual=r['output'].encode()[s['start']:s['end']].decode(errors='replace');tier=classify(text,actual);counts[tier]+=1
        cards.append(dict(row=i,source=text,output=actual,tier=tier,pair_id=t.get('pair_id')))
    return dict(counts=dict(counts),cards=cards)
def oldkg_tiers(rows,targets,records):
    counts=collections.Counter()
    for f,t,r in zip(rows,targets,records):
        for s in t['critical_spans']:
            if s['text'].startswith('+') and s['text'].endswith(' kg'):
                actual=r['output'].encode()[s['start']:s['end']].decode(errors='replace');counts[classify(s['text'],actual)]+=1
    return dict(counts)
def main():
    torch.set_num_threads(2)
    assert io.load_json(RUN/'root_saved_review.json')['status']=='ROOT_V75_SAVED_REVIEW_PASS'
    p=io.load_json(RUN/'plan.json');d=Path(p['data']);base=Path(p['baseline_run'])
    rows=io.load_json(d/'dev_features.json');targets=io.load_json(d/'dev_targets.json')
    b=stock_tiers(rows,targets,io.load_json(RUN/'new_baseline_reconstructions.json'));a=stock_tiers(rows,targets,io.load_json(RUN/'new_English_reconstructions.json'))
    transitions=collections.Counter();changed=[]
    for x,y in zip(b['cards'],a['cards']):
        assert x['row']==y['row'];transitions[(x['tier'],y['tier'])]+=1
        if x['tier']!=y['tier']:changed.append(dict(row=x['row'],pair_id=x['pair_id'],source=x['source'],before=x['output'],after=y['output'],before_tier=x['tier'],after_tier=y['tier']))
    e=Path(p['old_eval_data']);orows=io.load_json(e/'short_features.json');ot=io.load_json(e/'short_targets.json')
    oldb=oldkg_tiers(orows,ot,io.load_json(base/'old_untrained128_reconstructions.json'));olda=oldkg_tiers(orows,ot,io.load_json(RUN/'old_untrained128_reconstructions.json'))
    envelope=torch.load(RUN/'model/runtime.pt',map_location='cpu',weights_only=True);hist=envelope['runtime']['history'];refs=envelope['runtime']['cache_refs']
    ratios=[];labels=collections.Counter();prob=[];nonempty=0
    for h,order in zip(hist,envelope['runtime']['orders']):
        w=h['wrong_critical_positions'];c=h['critical_positions']
        if w:
            nonempty+=1;ratios.append(c/w)
        t=torch.load(refs[order[0]]['targets'],map_location='cpu',weights_only=True)
        mask=t['critical_mask']&~t['teacher_correct']
        for y in t['labels'][mask].tolist():labels[chr(y)]+=1
        prob.extend(t['teacher_gold_probability'][mask].tolist())
    m=io.load_json(RUN/'root_meaning_review.json')
    assert sum(labels.values())==168
    wrongterm=sum(.25*h['wrong_critical_CE'] for h in hist);total=sum(h['loss'] for h in hist)
    pressure=dict(teacher_wrong_critical_positions=168,rows_with_wrong_critical=nonempty,teacher_wrong_target_characters=dict(labels),
        per_wrong_position_critical_CE_coefficient_vs_previous_allcritical_normalization=dict(min=min(ratios),median=statistics.median(ratios),max=max(ratios)),
        actual_wrong_critical_loss_fraction=wrongterm/total,not_full_parameter_gradient_ratio=True,
        caveat='Per-row .25/wrong_count vs .25/all_critical_count only; clipping/other terms/shared features intervene. Dataset+loss+LR changed jointly, no isolated-cause proof; missing full teacher logits prevents claiming what the wrong competitor was.')
    decision=dict(action='NO_WHOLE_MODEL_ROLLBACK_NOW',weight_files_modified=False,next_research_initialization_candidate='V75 retained, V74 comparison/fallback retained',automatic_promotion=False,
        rationale=['Long quantity29 to16/name5 to2/decoded110 to58 improves.',
                   'New primary64/64 and all144name spans maintained; unsigned32sentences remain exact.',
                   'New stock strict20 to14 includes extra-symbol cases; clear content/sign/unit changes8 to10 across32rows, the two newly bad value rows share one counterfactual-pair context.',
                   'Old72 positivekg strict64 to59 does not imply five new clear numeric changes; tiers separate extra-symbol damage.',
                   'Early end-of-unit omission already exists before V75, so rollback alone does not fix it.'],
        unresolved='V75 has real local stock errors. Neither version is certified complete/lossless. No deployment model registry is changed.')
    report=dict(status='ROOT_V75_REGRESSION_SEVERITY_AND_ROLLBACK_REVIEW_COMPLETE',new_stock_before=b,new_stock_after=a,
        transitions=[dict(before=k[0],after=k[1],rows=n) for k,n in transitions.items()],newly_changed_tiers=changed,
        old_positive72_before=oldb,old_positive72_after=olda,loss_normalization_pressure=pressure,
        actual_max_long=dict(rows=2,bytes=[262042,262029],TRAIN_max_bytes=16384,quantity_before=29,quantity_after=16,names_before=5,names_after=2,
            output_bytes_unchanged=True,events=2718,actual_raw_review_completed=True,
            limitation='Existing observed synthetic document and single-run-on sentence, repeated grammar/local memory grows with length; not a fresh natural-document or 1MiB evaluation.'),
        rollback_decision=decision,classification_is_manual_interpretability_audit_not_new_NN_semantic_accuracy=True,model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
    assert not report['CUDA_initialized']
    io.atomic_json(RUN/'root_regression_severity_rollback_review.json',report)
    print(json.dumps(dict(status=report['status'],new_stock_before=b['counts'],new_stock_after=a['counts'],changed_tiers=changed,old72before=oldb,old72after=olda,pressure=pressure,decision=decision)))
if __name__=='__main__':main()

