"""Specific stock/end-of-unit residual: TRAIN coverage and saved logit margins."""
import collections,json,re,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def stock(rows,labels,records):
    counts=collections.Counter();cards=[]
    for i,(row,t,r) in enumerate(zip(rows,labels,records)):
        if t.get('mode')=='unsigned':continue
        value=t.get('reserve',t.get('stored_quantity'));text=f'+{value}.25 kg'
        candidates=[s for s in t['critical_spans'] if s['text']==text];assert len(candidates)==1;s=candidates[0];a,b=s['start'],s['end'];actual=r['output'].encode()[a:b].decode(errors='replace')
        match=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?) kg',actual);category='same_numeric_value' if match and Decimal(match[1])==Decimal(text[:-3]) else 'valid_different_value' if match else 'malformed_or_missing_unit'
        counts['fields']+=1;counts[category]+=1
        if match and match[1].startswith('-'):counts['explicit_sign_flip']+=1
        if category!='same_numeric_value':cards.append(dict(row=i,start=a,source=text,output=actual,category=category))
    return dict(counts=dict(counts),risks=cards)

def main():
    torch.set_num_threads(2);run=ROOT/'runs/critical_retention_20261112_v75';assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V75_SAVED_REVIEW_PASS'
    p=io.load_json(run/'plan.json');d=Path(p['data']);train=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');dev=io.load_json(d/'dev_features.json');dt=io.load_json(d/'dev_targets.json');base=Path(p['baseline_run']);vd=Path(p['observed_V74_data'])
    audit={}
    for key,rows,ts in [('TRAIN_authored512',train[:512],labels[:512]),('TRAIN_signed256',[f for f,t in zip(train[:512],labels[:512]) if t['mode']=='signed'],[t for t in labels[:512] if t['mode']=='signed']),('oldOWTTRAIN128',train[512:],labels[512:]),('newDEV64',dev,dt)]:
        audit[key]=dict(rows=len(rows),ends_kg_period=sum(r['prompt'].endswith('kg.') for r in rows),ends_temperature_period=sum(r['prompt'].endswith('degrees C.') for r in rows),EOS_byte_position_counts=dict(collections.Counter(len(r['prompt'].encode())%16 for r in rows)))
    before=io.load_json(run/'new_baseline_reconstructions.json');after=io.load_json(run/'new_English_reconstructions.json');newbefore=stock(dev,dt,before);newafter=stock(dev,dt,after)
    vr=io.load_json(vd/'dev_features.json');vt=io.load_json(vd/'dev_targets.json');vbefore=stock(vr,vt,io.load_json(base/'new_English_reconstructions.json'));vafter=stock(vr,vt,io.load_json(run/'observed_V74_DEV_reconstructions.json'))
    same_commit=0;tail=[];margins=[]
    for i,(row,t,b,a) in enumerate(zip(dev,dt,before,after)):
        x=torch.load(run/f'raw_new_baseline_{i:02d}.pt',map_location='cpu',weights_only=True);y=torch.load(run/f'raw_new_English_{i:02d}.pt',map_location='cpu',weights_only=True)
        assert torch.equal(x['fuzzy_commit'],y['fuzzy_commit']);same_commit+=1
        if t['mode']=='signed':
            text=f"+{t['reserve']}.25 kg";s=next(s for s in t['critical_spans'] if s['text']==text);j=s['start'];entry=dict(row=i,source=text,output_before=b['output'].encode()[j:s['end']].decode(errors='replace'),output_after=a['output'].encode()[j:s['end']].decode(errors='replace'))
            for label,raw in [('before',x),('after',y)]:
                z=raw['logits'].reshape(-1,258)[j].float();other=z.clone();other[43]=-torch.inf
                entry[label+'_plus_vs_best_other_margin']=float(z[43]-other.max());entry[label+'_raw_argmax']=int(z.argmax())
            margins.append(entry)
        if len(a['output'].encode())!=len(row['prompt'].encode()):
            n=len(row['prompt'].encode());end=len(a['output'].encode());z=y['logits'].reshape(-1,258)
            tail.append(dict(row=i,source_bytes=n,output_bytes=end,missing=row['prompt'].encode()[end:].decode(),source_EOS_position=n%16,predicted_EOS_position=end%16,
                source_tail=row['prompt'][-70:],output_tail=a['output'][-70:],actual_predicted_EOS_argmax=int(z[end].argmax()),expected_byte_at_early_EOS=row['prompt'].encode()[end]))
    assert same_commit==64 and not torch.cuda.is_initialized()
    report=dict(status='ROOT_V75_STOCK_TAIL_SOURCE_COVERAGE_AND_SAVED_MARGIN_REVIEW_COMPLETE',TRAIN_tail_coverage=audit,
        newDEV_stock_before=newbefore,newDEV_stock_after=newafter,observed_V74_stock_before=vbefore,observed_V74_stock_after=vafter,
        all64_newDEV_Commit_bitwise_identical_before_after=True,newDEV_stock_plus_margins=margins,actual_early_EOS_cases=tail,
        inference_changed=False,model_calls=0,CUDA_initialized=False,
        interpretation='Authored TRAIN contains no sentence-final kg/degrees-C endings despite covering EOS byte roles; new signed DEV stock is sentence-final. Actual +2.25 kg. loses the final space/kg/period at an earlier EOS. This supports a joint tail/predicate coverage hypothesis, not a single-cause proof. Same Commit and changed logits locate these observed changes in readout learning, without certifying full abstract meaning formation.')
    io.atomic_json(run/'root_stock_tail_review.json',report);print(json.dumps({k:v for k,v in report.items() if k!='newDEV_stock_plus_margins'}))
if __name__=='__main__':main()
