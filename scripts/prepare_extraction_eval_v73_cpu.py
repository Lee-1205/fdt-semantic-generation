"""Expanded never-TRAIN English/length panels requested by the human."""
import hashlib,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
NAMES=['Ariadne','Boaz','Calista','Dashiell','Esme','Idris','Noemi','Zephyr']
ITEMS=['brass pins','blue notebooks','glass beads','wooden tokens','green ribbons','silver clips']
COLORS=['amber','blue','green','ivory','red','violet']
def main():
    out=ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu';out.mkdir(exist_ok=False)
    train_dir=ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2';train=io.load_json(train_dir/'train_features.json')
    short=io.load_json(train_dir/'dev_features.json');labels=[dict(t,origin='prepared_DEV32') for t in io.load_json(train_dir/'dev_targets.json')]
    rng=random.Random(73117)
    for i in range(96):
        text='';events=[]
        for j in range(1+i%5):
            a,b=rng.sample(NAMES,2);q=rng.randrange(100);reserve=rng.randrange(20);obj=rng.choice(ITEMS);family=(i+j)%4
            clauses=[f'The crate beside the door belongs to {a}, not {b}; it contains exactly {q} {obj}.',
                f'{a} arrived after {b}, although {a} had departed earlier; neither person took the remaining {reserve} {obj}.',
                f'{a} recorded a temperature of -{q}.35 degrees C at 08:07 on 2024-06-19, while {b} kept +{reserve}.50 kg of flour.',
                f'Only {a} transferred {q} {obj} to {b}; the {reserve} items left with {a} were not part of the delivery.']
            phrase=clauses[family];start=len(text.encode());text+=phrase+(' ' if i%2 else '\n\n');events.append(dict(start=start,end=start+len(phrase.encode()),actor=a,other=b,quantity=q,reserve=reserve,family=family))
        text=text.rstrip();identity='fresh73:'+hashlib.sha256(text.encode()).hexdigest()[:24]
        names=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(NAMES)+r')\b',text)]
        short.append(dict(id=identity,prompt=text));labels.append(dict(id=identity,origin='fresh_supplement96',events=events,critical_spans=complete_expression_spans(text,names)))
    oldlong=ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu';long=io.load_json(oldlong/'features.json')[-2:];longlabels=[dict(t,origin='previous_observed_not_TRAIN') for t in io.load_json(oldlong/'targets.json')[-2:]]
    for maximum in (32768,65536,131072,262144):
        for style in ('document','single_sentence'):
            text='The following account records the Larchhaven depot activities: ';events=[];n=1
            while True:
                a,b=rng.sample(NAMES,2);q=rng.randrange(100);reserve=rng.randrange(20);obj=rng.choice(ITEMS);source,destination=rng.sample(COLORS,2);order=rng.choice(['before','after']);negative=rng.randrange(5)!=0
                phrase=f'in episode {n}, {a} moved {q} {obj} from the {source} shelf to the {destination} drawer {order} {b} closed the {source} gate, and the {destination} box contained {reserve} {obj} that {a} did '+('not ' if negative else '')+'remove'
                if style=='document':phrase=phrase[0].upper()+phrase[1:]
                if len((text+phrase+'.').encode())>maximum-8:break
                start=len(text.encode());text+=phrase+('.\n\n' if style=='document' else '; ')
                events.append(dict(episode=n,actor=a,checker=b,quantity=q,item=obj,source_color=source,destination_color=destination,order=order,reserve=reserve,negative_removal=negative,byte_start=start,byte_end=start+len(phrase.encode())));n+=1
            text=text.rstrip().rstrip(';');text+= '' if text.endswith('.') else '.';identity='freshlong73:'+hashlib.sha256(text.encode()).hexdigest()[:24]
            assert maximum-350<len(text.encode())<=maximum
            if style=='single_sentence':assert text.count('.')==1 and '\n' not in text
            long.append(dict(id=identity,prompt=text));longlabels.append(dict(id=identity,origin='fresh_long8',style=style,requested_max_bytes=maximum,actual_bytes=len(text.encode()),events=events))
    assert len(short)==128 and len(long)==10 and len({r['prompt'] for r in short+long})==138
    assert not any(t['prompt'] in f['prompt'] or f['prompt'] in t['prompt'] for t in train for f in short+long)
    for name in NAMES:assert not any(re.search(r'\b'+name+r'\b',t['prompt']) for t in train)
    old=ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu/train_features.json';oldtrain=io.load_json(old)
    assert not any(t['prompt']==f['prompt'] for t in oldtrain for f in short+long)
    chunks=sum((len(r['prompt'].encode())+2047)//2048 for r in long)
    manifest=dict(schema='expanded_untrained_English_v73_eval',short_rows=128,prepared32=True,fresh_supplement96=True,long_rows=10,fresh_long8=True,previous_observed_long2=True,
        long_lengths=[len(r['prompt'].encode()) for r in long],short_UTF8_bytes=sum(len(r['prompt'].encode()) for r in short),long_UTF8_bytes=sum(len(r['prompt'].encode()) for r in long),
        long_stream_chunks=chunks,whole_TRAIN_max=16384,fresh_names_all_absent_TRAIN=True,features=['id','prompt'],targets_OFFLINE_score_only=True,
        new_downloads=0,E2_registry_access=0,limitations='new authored strings not trained, templates/facts correlated and design informed by previous errors; old32/old2 explicitly observed; not blind natural benchmark, not fixed-memory long reasoning')
    for n,v in [('short_features',short),('short_targets',labels),('long_features',long),('long_targets',longlabels),('manifest',manifest)]:io.atomic_json(out/(n+'.json'),v)
    print(json.dumps(manifest))
if __name__=='__main__':main()
