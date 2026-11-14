"""Restricted source-bound meanings/Decimal values and cards on saved outputs."""
import collections,json,re,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_extraction_v73_meaning_cpu import short_fields,long_fields
from review_context_v74_meaning_cpu import kg_values,primary_values

def primary75(rows,labels,records):
    counts=collections.defaultdict(collections.Counter);risks=[]
    for i,(row,t,r) in enumerate(zip(rows,labels,records)):
        text=t['primary_text'];a=t['primary_start'];actual=r['output'].encode()[a:a+len(text.encode())].decode(errors='replace')
        assert row['prompt'].encode()[a:a+len(text.encode())].decode()==text
        key='unsigned' if t['mode']=='unsigned' else t['sign']+t['unit'];suffix='' if key=='unsigned' else ' '+re.escape(t['unit'])
        match=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)'+suffix,actual);expected=Decimal(str(t['primary_value'])) if key=='unsigned' else Decimal(t['sign']+str(t['primary_value'])+'.75')
        category='same_numeric_value' if match and Decimal(match[1])==expected else 'valid_different_value' if match else 'malformed_or_missing'
        counts[key]['fields']+=1;counts[key][category]+=1
        if category!='same_numeric_value':risks.append(dict(row=i,source=text,output=actual,category=category))
    return dict(cells=dict(counts),risks=risks)

def main():
    run=ROOT/'runs/critical_retention_20261112_v75';assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V75_SAVED_REVIEW_PASS'
    p=io.load_json(run/'plan.json');d=Path(p['data']);base=Path(p['baseline_run']);e=Path(p['old_eval_data']);olddata=Path(p['observed_V74_data'])
    rows=io.load_json(d/'dev_features.json');labels=[dict(t,origin='new_V75_DEV64') for t in io.load_json(d/'dev_targets.json')];before=io.load_json(run/'new_baseline_reconstructions.json');after=io.load_json(run/'new_English_reconstructions.json')
    nb,cb=short_fields(rows,labels,before);na,ca=short_fields(rows,labels,after)
    oldrows=io.load_json(e/'short_features.json');ot=io.load_json(e/'short_targets.json');ob=io.load_json(base/'old_untrained128_reconstructions.json');oa=io.load_json(run/'old_untrained128_reconstructions.json');osb,_=short_fields(oldrows,ot,ob);osa,oc=short_fields(oldrows,ot,oa)
    vrows=io.load_json(olddata/'dev_features.json');vt=[dict(t,origin='observed_V74_DEV64') for t in io.load_json(olddata/'dev_targets.json')];vb=io.load_json(base/'new_English_reconstructions.json');va=io.load_json(run/'observed_V74_DEV_reconstructions.json');vsb,_=short_fields(vrows,vt,vb);vsa,vc=short_fields(vrows,vt,va)
    lr=io.load_json(e/'long_features.json')[:2];lt=io.load_json(e/'long_targets.json')[:2];lb,lbc,_=long_fields(lr,lt,io.load_json(base/'long_reconstructions.json'));la,lac,regions=long_fields(lr,lt,io.load_json(run/'long_reconstructions.json'))
    conf={}
    for label,records in [('V74',io.load_json(base/'long_reconstructions.json')),('V75',io.load_json(run/'long_reconstructions.json'))]:
        changed=collections.Counter();length=[]
        for r,f in zip(records,lr):
            a,b=f['prompt'].encode(),r['output'].encode();length.append((len(a),len(b)))
            for x,y in zip(a,b):
                if x!=y:changed[(chr(x),chr(y))]+=1
        conf[label]=dict(wrong_plus=sum(n for (a,b),n in changed.items() if b=='+'),digit_to_plus=sum(n for (a,b),n in changed.items() if b=='+' and a.isdigit()),lengths=length)
    report=dict(status='ROOT_V75_RESTRICTED_MEANING_CARDS_READY',new64_before=nb,new64_after=na,new64_primary_before=primary75(rows,labels,before),new64_primary_after=primary75(rows,labels,after),
        old128_before=osb,old128_after=osa,old_positive_kg_before=kg_values(oldrows,ot,ob),old_positive_kg_after=kg_values(oldrows,ot,oa),
        observed_V74_DEV_before=vsb,observed_V74_DEV_after=vsa,observed_V74_primary_before=primary_values(vrows,vt,vb),observed_V74_primary_after=primary_values(vrows,vt,va),
        observed_long2_before=lb,observed_long2_after=la,long_confusions=conf,extra_model_calls=0,
        scope='source-aligned authored fields and Decimal quantity values plus manual reading, not a general semantic percentage, abstract latent validation or blind natural benchmark')
    io.atomic_json(run/'root_meaning_review.json',report);io.atomic_json(run/'root_manual_reading_cards.json',dict(new64_before=cb,new64_after=ca,old128_after=oc,observed_V74_after=vc,long_before=lbc,long_after=lac,long_regions=regions))
    small=lambda v:{k:z for k,z in v.items() if k not in ('errors','tail_length_changes')}
    print(json.dumps(dict(new64_before=small(nb),new64_after=small(na),new64_primary_before=report['new64_primary_before']['cells'],new64_primary_after=report['new64_primary_after']['cells'],
        old128_before=small(osb),old128_after=small(osa),positivekg_before=report['old_positive_kg_before']['counts'],positivekg_after=report['old_positive_kg_after']['counts'],
        observed_V74_before=small(vsb),observed_V74_after=small(vsa),V74_primary_before=report['observed_V74_primary_before']['cells'],V74_primary_after=report['observed_V74_primary_after']['cells'],long_before=lb['rows'],long_after=la['rows'],confusions=conf)))
if __name__=='__main__':main()
