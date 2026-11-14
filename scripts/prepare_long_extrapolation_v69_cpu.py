"""English meaning-retention stress beyond the learned 16384-byte maximum."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=['Aveline','Bastian','Cyril','Dorothea','Eamon','Fiora','Hector','Ilona']
OBJECTS=['brass pins','blue notebooks','glass beads','wooden tokens','green ribbons','silver clips']
COLORS=['amber','blue','green','ivory','red','violet']

def main():
    out=ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu';out.mkdir(exist_ok=True)
    assert not (out/'features.json').exists()
    rng=random.Random(69112);features=[];targets=[]
    for maximum in (16384,32768,65536,131072,262144):
        for style in ('document','single_sentence'):
            limit=maximum if style=='document' else maximum-7
            head='This is the complete account of the Alderwick workshop activities: '
            text=head;events=[];n=1
            while True:
                actor,checker=rng.sample(NAMES,2);quantity=rng.randint(3,29);obj=rng.choice(OBJECTS);a,b=rng.sample(COLORS,2);order=rng.choice(['before','after']);reserve=rng.randint(1,9)
                phrase=(f'in episode {n}, {actor} moved {quantity} {obj} from the {a} shelf to the {b} drawer {order} {checker} closed the {a} gate, '
                    f'and the {b} box contained {reserve} {obj} that {actor} did not remove')
                phrase=phrase[0].upper()+phrase[1:] if style=='document' else phrase
                sep='.\n\n' if style=='document' else '; '
                if len((text+phrase+'.').encode())>limit:break
                start=len(text.encode());text+=phrase+sep
                events.append(dict(episode=n,actor=actor,checker=checker,quantity=quantity,item=obj,source_color=a,destination_color=b,order=order,reserve=reserve,
                    byte_start=start,byte_end=start+len(phrase.encode()),negative_removal=True))
                n+=1
            text=text.rstrip()
            text=text.rstrip(';') if style=='single_sentence' else text
            if not text.endswith('.'):text+='.'
            raw=text.encode();rid='long69:'+hashlib.sha256(raw).hexdigest()[:24]
            assert len(raw)<=limit and len(raw)>limit-300
            if style=='single_sentence':assert text.count('.')==1 and '\n' not in text
            features.append(dict(id=rid,prompt=text))
            targets.append(dict(id=rid,style=style,requested_max_bytes=maximum,actual_bytes=len(raw),events=events,
                semantic_fields=['actor','checker','quantity','item','source_color','destination_color','order','reserve','negative_removal'],
                boundary_crossing_events=sum(e['byte_start']//2048!=(e['byte_end']-1)//2048 for e in events)))
    assert len(features)==10
    parity=[dict(id='parity69:'+str(i),prompt=features[i]['prompt'][:4096]) for i in (0,1)]
    manifests=dict(schema='long_extrapolation_v69_data',trained_input_max_UTF8_bytes=16384,test_input_max_UTF8_bytes=262144,
        lengths=[t['actual_bytes'] for t in targets],styles=[t['style'] for t in targets],documents=10,parity_documents=2,parity_bytes=4096,
        source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in features),meaning_event_bundles=sum(len(t['events']) for t in targets),
        source_model='V67 frozen b47c9bdf920605b153c1ebfaa31a28579eb35c1784de6037ba54f4674cdc18b9',
        model_input_keys=['id','prompt'],all_gold_events_OFFLINE_SCORE_ONLY=True,new_training_rows=0,new_downloads=0,E2_registry_access=0,
        execution='per-layer finite-window KV carry; original whole-document N/D and global pool; not independent-chunk concatenation',
        limitations='new authored correlated English event prose; one run-on sentence per length; byte symbols not BPE tokens; no official RULER/RHELM score or RLM recursion; local Commit storage grows linearly')
    for name,value in [('features',features),('targets',targets),('parity_features',parity),('manifest',manifests)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifests))
if __name__=='__main__':main()
