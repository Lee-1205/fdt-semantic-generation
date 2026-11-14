"""New authored TRAIN coverage; English factual-byte loss labels separate."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRAIN_NAMES=['Calder','Vera','Milo','Selene','Ansel','Petra','Hugo','Tamsin','Bram','Leona','Felix','Cora','Emmett','Sabine','Galen','Nadia']
DEV_NAMES=['Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette']
ITEMS=['copper coins','blue folders','red pencils','white cards','small buttons','paper tickets']

def authored(count,names,seed,held):
    rng=random.Random(seed);features=[];targets=[]
    for i in range(count):
        text=rng.choice(['Morning report. ','During the afternoon shift, ','A note from the workshop: ',''])
        events=[]
        for j in range(1+i%6):
            a,b=rng.sample(names,2);q=rng.randint(3,98);left=rng.randint(1,19);item=rng.choice(ITEMS);order=rng.choice(['before','after'])
            if held:
                phrase=f'{b} received {q} {item} from {a}; {left} {item} remained with {a}, who did not give those remaining items away.'
            elif (i+j)%4==0:
                phrase=f'{a} sent {q} {item} to {b} and kept {left} {item}; {b} did not receive the remaining {left} items.'
            elif (i+j)%4==1:
                phrase=f'{q} {item} were delivered by {a} to {b} {order} the bell rang, while {left} {item} stayed with {a}.'
            elif (i+j)%4==2:
                phrase=f'{order.capitalize()} closing the door, {a} handed {q} {item} to {b}; {a} still had {left} {item} afterward.'
            else:
                phrase=f'Only {q} {item} went from {a} to {b}. The {left} {item} kept by {a} were not part of that delivery.'
            start=len(text.encode());text+=phrase+(' ' if (i+j)%3 else '\n\n')
            event=dict(actor=a,recipient=b,quantity=q,reserve=left,item=item,byte_start=start,byte_end=start+len(phrase.encode()))
            if not held and (i+j)%4 in (1,2):event['temporal_order']=order
            events.append(event)
        text=text.rstrip();rid='fact70:'+hashlib.sha256(text.encode()).hexdigest()[:24]
        # These TRAIN labels never enter the model's feature API. Literal
        # name/numeral spans mark critical content; not a runtime NER system.
        spans=[]
        for match in re.finditer(r'\b(?:'+('|'.join(names))+r'|[0-9]+)\b',text):
            a,b=match.span();spans.append(dict(start=a,end=b,text=match.group(),kind='number' if match.group().isdigit() else 'name'))
        features.append(dict(id=rid,prompt=text));targets.append(dict(id=rid,critical_spans=spans,authored_events=events))
    return features,targets

def main():
    out=ROOT/'research/artifacts/factual_cycle_20261112_v70_cpu_v2';out.mkdir(exist_ok=False)
    train,targets=authored(384,TRAIN_NAMES,70112,False);dev,devtargets=authored(32,DEV_NAMES,70113,True)
    old=ROOT/'research/artifacts/english_memory_cycle_20261112_v68_cpu/OWT_features.json';owt=json.loads(old.read_text())
    for row in owt:
        train.append(row);text=row['prompt'];spans=[]
        for m in re.finditer(r'[0-9]+',text):
            spans.append(dict(start=len(text[:m.start()].encode()),end=len(text[:m.end()].encode()),text=m.group(),kind='number'))
        targets.append(dict(id=row['id'],critical_spans=spans,authored_events=[]))
    assert len(train)==len(targets)==512 and len({r['id'] for r in train})==512
    held66=json.loads((ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json').read_text())
    held69=json.loads((ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu/features.json').read_text())
    assert not any(h['prompt'] in r['prompt'] for h in held66+held69 for r in train)
    assert not {r['prompt'] for r in train}&{r['prompt'] for r in dev}
    orders=list(range(512));random.Random(70112).shuffle(orders);orders=[[i] for i in orders]
    manifest=dict(schema='factual_cycle_v70_data',train_rows=512,authored_train_rows=384,OWT_train_retention_rows=128,
        train_epoch=1,attempts=512,dev_new_authored=32,TRAIN_names=TRAIN_NAMES,DEV_names=DEV_NAMES,
        target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train),max_input_bytes=max(len(r['prompt'].encode()) for r in train),
        critical_bytes=sum(s['end']-s['start'] for t in targets for s in t['critical_spans']),
        training_features=['id','prompt'],critical_targets='separate TRAIN-only loss/score spans; no runtime role/regex/name filter',
        OWT_source=str(old),all_V66_V69_eval_stays_eval_only=True,E2_registry_access=0,new_downloads=0,
        limitations='prospective authored coverage chosen after observed V69 risk; correlated templates and deliberate family overlap; new held-name form not blind design; no abstract-meaning certificate')
    for name,value in [('train_features',train),('train_targets',targets),('dev_features',dev),('dev_targets',devtargets),('orders',orders),('manifest',manifest)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest))
if __name__=='__main__':main()
