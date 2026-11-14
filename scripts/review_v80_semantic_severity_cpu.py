"""Additional source-bound predicate and numeric severity; no output repair/NN."""
import collections,json,re,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN


def numeric_severity(targets,records):
    counts=collections.Counter();cards=[]
    assert len(targets)==len(records) and [t['id'] for t in targets]==[r['id'] for r in records],'severity requires exact same source IDs'
    for i,(t,r) in enumerate(zip(targets,records)):
        s=t['B_span'];actual=r['output'].encode()[s['start']:s['end']].decode();gold=s['text'];unit=t['unit']
        gold_number=Decimal(gold if unit=='items' else gold[:-(len(unit)+1)])
        tail='' if unit=='items' else ' '+re.escape(unit)
        strict=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)'+tail,actual)
        core=re.fullmatch(r'[^\w\d\s+-]([+-]?\d+(?:\.\d+)?)'+tail,actual)
        if strict and Decimal(strict[1])==gold_number:tier='value_unit_clear'
        elif strict:tier='clear_different_numeric_value'
        elif core and Decimal(core[1])==gold_number:tier='numeric_core_and_unit_retained_nonstandard_prefix_UNCERTIFIED'
        elif unit=='items' and actual in ('7:','7;') and gold=='75':tier='digit_content_damaged_75_to_7punct_MANUAL'
        else:tier='other_uncertified'
        counts[tier]+=1
        if tier!='value_unit_clear':cards.append(dict(row=i,gold=gold,output=actual,tier=tier))
    return dict(counts=dict(counts),cards=cards,scope='B fields only; A64 all retained under existing score. New score does not replace historical strict51/64 or assert whole-sentence accuracy.')


def main():
    run=ROOT/'runs/update_retention_20261113_v80';p=io.load_json(run/'plan.json');rows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2];targets=io.load_json(Path(p['old_eval_data'])/'long_targets.json')[:2]
    base=Path(p['baseline_run']);versions=[('V75',base),('V79',ROOT/'runs/last_linear_20261113_v79'),('V80',run)];predicates={};symbols={}
    for version,folder in versions:
        records=io.load_json(folder/'long_reconstructions.json');cards=[];ct=collections.Counter()
        for ri,(row,t,r) in enumerate(zip(rows,targets,records)):
            raw=row['prompt'].encode();out=r['output'].encode()
            for j,event in enumerate(t['events']):
                a,b=event['byte_start'],event['byte_end'];source=raw[a:b].decode();actual=out[a:b].decode();assert PATTERN.fullmatch(source)
                start=source.index(' moved ')+1;word=actual[start:start+5];ct['source_moved_events']+=1
                if word!='moved':cards.append(dict(row=ri,event=j+1,source_predicate='moved',output_predicate=word,source=source,output=actual))
            for a,b in zip(raw,out):
                if a!=b:ct['different_bytes']+=1;ct['wrong_plus']+=int(b==43);ct['digit_to_plus']+=int(48<=a<=57 and b==43)
        predicates[version]=dict(changed_predicates=len(cards),cards=cards);symbols[version]=dict(ct)
    selected=io.load_json(Path(p['eval_data'])/'targets.json')
    numeric={v:numeric_severity(selected,io.load_json((run/'new_baseline_reconstructions.json') if v=='V75' else folder/'new_English_reconstructions.json')) for v,folder in versions}
    env=torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True);s=env['runtime'];bank=torch.load(s['bank_ref']['targets'],map_location='cpu',weights_only=True);final=torch.load(s['reference_final_ref']['path'],map_location='cpu',weights_only=True)
    valid=bank['labels']!=257;before=bank['initial_correct'];after=(final['argmax'].long()==bank['labels'])&valid;critical=bank['critical_mask']
    learning=dict(valid_positions=int(valid.sum()),initial_correct=int(before.sum()),final_correct=int(after.sum()),initial_wrong_corrected=int((~before&after&valid).sum()),initial_correct_lost=int((before&~after).sum()),
        corrected_critical=int((~before&after&valid&critical).sum()),corrected_noncritical_EOS=int((~before&after&valid&~critical).sum()),TRAIN_only_not_unseen_generalization=True,
        initial_and_final_classes_from_saved_argmax_NOT_full_logits_recomputed=True)
    report=dict(status='ROOT_V80_SEMANTIC_SEVERITY_AND_PREDICATE_REVIEW_COMPLETE',new64_B_numeric_severity=numeric,long_action_predicate_scan=predicates,long_symbols=symbols,TRAIN_reference_learning=learning,
        extra_model_calls=0,changed_any_model_output=False,old_strict_field_scores_preserved=True,CUDA_initialized=False,
        limitation='All long predicates share authored source word moved and source-bound alignment; this is not full natural NLU or paraphrase accuracy. Roved is treated as predicate fidelity risk, not forgiven as typographical equivalence.')
    io.atomic_json(run/'root_semantic_severity_review_v2.json',report)
    print(json.dumps(dict(status=report['status'],numeric={k:v['counts'] for k,v in numeric.items()},predicates={k:v['changed_predicates'] for k,v in predicates.items()},symbols=symbols,TRAIN_learning=learning)))


if __name__=='__main__':main()
