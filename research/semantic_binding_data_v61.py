"""More independent authored facts; unchanged, already-observed held panel."""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.semantic_binding_data_v59 import TRAIN_NAMES,DEV_NAMES,render,pattern,fact_key
from research.semantic_binding_data_v60 import new_pattern


def build():
    old=ROOT/'research/artifacts/semantic_binding_20261111_v60_cpu'
    out=ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu'
    if out.exists():raise FileExistsError('immutable V61 data')
    names=TRAIN_NAMES+('Adrian','Beatriz','Camilo','Daphne','Enzo','Fatima','Gustav','Helena')
    gifts=('silver coin','blue envelope','wooden spoon','green ribbon','small drum','yellow scarf',
           'brass key','white notebook','black pen','paper lantern','ceramic cup','wool blanket')
    gates=('wooden gate','green door','iron hatch','small window','bronze hatch','side window',
           'front door','round window','silver gate','blue door','ceiling hatch','back gate')
    boxes=('amber box','blue basket','green chest','wooden bin','silver crate','navy barrel',
           'white box','brass tray','brown basket','small chest','wide bin','black crate')
    objects=('red tokens','blue tokens','green tokens','yellow tokens','white beads','black beads',
             'golden counters','silver counters','brass coins','wooden blocks','paper stars','ceramic discs')
    rng=random.Random(61111);features=[];targets=[];groups=[]
    prefixes=('','Log header.\n','This is a separate recorded event.\n','Please read the following entry.\n','Record details follow.\n')
    facts_by_kind=[]
    facts_by_kind.append([dict(relation=0,polarity=1,quantity=0,roles=dict(primary=a,recipient=b,theme=c))
        for a in names for b in names if b!=a for c in gifts])
    facts_by_kind.append([dict(relation=1,polarity=i%2,quantity=0,roles=dict(primary=a,recipient=None,theme=c))
        for i,(a,c) in enumerate((a,c) for a in names for c in gates)])
    facts_by_kind.append([dict(relation=2,polarity=1,quantity=n,roles=dict(primary=a,recipient=None,theme=c))
        for a in boxes for c in objects for n in range(1,25)])
    for kind,wanted in enumerate((342,341,341)):
        candidates=facts_by_kind[kind];rng.shuffle(candidates)
        for i,fact in enumerate(candidates[:wanted]):
            other=json.loads(json.dumps(fact))
            if kind==0:other['roles']['primary'],other['roles']['recipient']=other['roles']['recipient'],other['roles']['primary']
            elif kind==1:other['polarity']=1-other['polarity']
            else:other['quantity']=other['quantity']%24+1
            families=rng.sample((0,1,2,6,7,8),3);group=[];gid=f'train61:{kind}:{i:04d}'
            for j in range(4):
                selected=other if j==3 else fact;family=families[0] if j==3 else families[j]
                template,fields=pattern(selected,family) if family<3 else new_pattern(selected,family)
                lead=prefixes[(i+j)%len(prefixes)]*((i+j)%3)
                tail='\nRecord ends.' if (i+j)%2 else ''
                text,spans=render(lead+template+tail,fields);rid=gid+':'+str(j)
                group.append(len(features));features.append(dict(id=rid,prompt=text))
                targets.append(dict(id=rid,spans=spans,meaning=selected,meaning_id=fact_key(selected),family=family,group=gid))
            groups.append(group)
    orders=[]
    for epoch in range(4):
        shuffled=list(groups);random.Random(61211+epoch).shuffle(shuffled)
        orders.extend([sum(shuffled[i:i+4],[]) for i in range(0,len(shuffled),4)])
    dev=json.loads((old/'dev_features.json').read_text(encoding='utf-8-sig'))
    dy=json.loads((old/'dev_targets.json').read_text(encoding='utf-8-sig'))
    assert not set(names)&set(DEV_NAMES)
    assert not {f['prompt'] for f in features}&{f['prompt'] for f in dev}
    assert not {t['meaning_id'] for t in targets}&{t['meaning_id'] for t in dy}
    out.mkdir()
    for name,value in [('train_features',features),('train_targets',targets),('train_groups',groups),('orders',orders)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name in ('dev_features','dev_targets','dev_groups','OWT_train_reconstruction_windows'):
        (out/(name+'.json')).write_bytes((old/(name+'.json')).read_bytes())
    width=max(len(f['prompt'].encode()) for f in features)
    manifest=dict(schema='semantic_binding_v61_CPU',TRAIN_rows=len(features),TRAIN_groups=len(groups),
        TRAIN_unique_meanings=len({t['meaning_id'] for t in targets}),DEV_rows=len(dev),epochs=4,attempts=len(orders),
        row_selections=sum(map(len,orders)),TRAIN_input_bytes=sum(len(f['prompt'].encode()) for f in features),
        max_TRAIN_bytes=width,min_TRAIN_bytes=min(len(f['prompt'].encode()) for f in features),
        TRAIN_families=[0,1,2,6,7,8],DEV_families=[3,4,5],TRAIN_names=list(names),
        expected_packed_cache_tensor_bytes=len(features)*width*(704*4+3),
        duplicate_TRAIN_prompts=len(features)-len({f['prompt'] for f in features}),adaptive_not_blind=True,
        feature_keys=['id','prompt'],gold_targets='loss/score only; no gold span or lexical dictionary candidate filter',
        OWT_windows_used=0,new_downloads=0,E2_access=0,limitations='authored restricted give/open/contain semantics; not general document understanding')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':build()
