"""Saved decoded values/identity comparisons, no extra model calls."""
import collections,json,re,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_extraction_v73_meaning_cpu import short_fields,long_fields
def kg_values(rows,labels,records):
    counts=collections.Counter();cards=[]
    for i,(row,t,r) in enumerate(zip(rows,labels,records)):
        output=r['output'].encode()
        for span in t['critical_spans']:
            text=span['text']
            if text.startswith('+') and text.endswith(' kg'):
                counts['fields']+=1;actual=output[span['start']:span['end']].decode(errors='replace');match=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?) kg',actual)
                category='same_numeric_value' if match and Decimal(match[1])==Decimal(text[:-3]) else 'valid_but_different_numeric_value' if match else 'malformed_requires_reading';counts[category]+=1
                if match and match[1].startswith('-'):counts['explicit_sign_flip']+=1
                if category!='same_numeric_value':cards.append(dict(row=i,source=text,output=actual,category=category))
    return dict(counts=counts,risks=cards)
def primary_values(rows,labels,records):
    cells=collections.defaultdict(collections.Counter);cards=[]
    for i,(row,t,r) in enumerate(zip(rows,labels,records)):
        text=f"{t['sign']}{t['quantity']}.50 {t['unit']}";b=row['prompt'].encode();a=b.index(text.encode(),t['clause_start']);output=r['output'].encode();actual=output[a:a+len(text.encode())].decode(errors='replace')
        match=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?) '+re.escape(t['unit']),actual);expected=Decimal(t['sign']+str(t['quantity'])+'.50')
        category='same_numeric_value' if match and Decimal(match[1])==expected else 'valid_but_different_numeric_value' if match else 'malformed_requires_reading'
        cells[t['sign']+t['unit']]['fields']+=1;cells[t['sign']+t['unit']][category]+=1
        if category!='same_numeric_value':cards.append(dict(row=i,source=text,output=actual,category=category))
    return dict(cells=dict(cells),risks=cards)
def main():
    run=ROOT/'runs/context_verbalizer_20261112_v74';p=io.load_json(run/'plan.json');d=Path(p['data']);old=Path(p['baseline_run']);e=Path(p['old_eval_data'])
    assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V74_SAVED_REVIEW_PASS'
    rows=io.load_json(d/'dev_features.json');labels=[dict(t,origin='new_DEV64') for t in io.load_json(d/'dev_targets.json')];before=io.load_json(run/'new_baseline_reconstructions.json');after=io.load_json(run/'new_English_reconstructions.json')
    newbefore,bc=short_fields(rows,labels,before);newafter,ac=short_fields(rows,labels,after)
    orows=io.load_json(e/'short_features.json');ot=io.load_json(e/'short_targets.json');prior=io.load_json(old/'new_English_reconstructions.json');now=io.load_json(run/'old_untrained128_reconstructions.json')
    oldbefore,ocb=short_fields(orows,ot,prior);oldafter,oca=short_fields(orows,ot,now)
    beforekg=kg_values(orows,ot,prior);afterkg=kg_values(orows,ot,now)
    longrows=io.load_json(e/'long_features.json')[:2];lt=io.load_json(e/'long_targets.json')[:2];long_before,lbc,_=long_fields(longrows,lt,io.load_json(old/'long_reconstructions.json')[:2]);long_after,lac,regions=long_fields(longrows,lt,io.load_json(run/'long_reconstructions.json'))
    newcards=[]
    for i,(row,t,r) in enumerate(zip(rows,labels,after)):
        a=t['clause_start'];body=row['prompt'].encode()[a:].decode();out=r['output'].encode()[a:].decode(errors='replace')
        if body!=out:newcards.append(dict(row=i,source=body,output=out))
    report=dict(status='ROOT_V74_RESTRICTED_MEANING_CARDS_READY',new64_before=newbefore,new64_after=newafter,old128_before=oldbefore,old128_after=oldafter,
        old_positive_kg_before=beforekg,old_positive_kg_after=afterkg,new64_primary_before=primary_values(rows,labels,before),new64_primary_after=primary_values(rows,labels,after),
        observed_long2_before=long_before,observed_long2_after=long_after,extra_model_calls=0,
        scope='limited source-bound field/value fidelity on actual decoded output; old panels now observed, newDEV64 untrained; cache readout plus two fresh long paths, not new encoder training or abstract meaning certificate')
    io.atomic_json(run/'root_meaning_review.json',report);io.atomic_json(run/'root_manual_reading_cards.json',dict(new64_after_all_nonliteral=ac,new64_after_changed_clauses=newcards,old128_after_nonliteral=oca,long_changed_before=lbc,long_changed_after=lac,long_start_middle_end=regions))
    small=lambda s:{k:v for k,v in s.items() if k!='errors'}
    print(json.dumps(dict(new64_before=small(newbefore),new64_after=small(newafter),old128_before=small(oldbefore),old128_after=small(oldafter),positive_kg_before=beforekg['counts'],positive_kg_after=afterkg['counts'],
        new64_primary_before=report['new64_primary_before']['cells'],new64_primary_after=report['new64_primary_after']['cells'],oldlong_before=long_before['rows'],oldlong_after=long_after['rows'])))
if __name__=='__main__':main()
