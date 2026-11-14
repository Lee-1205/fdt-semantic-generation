"""Additional authored TRAIN forms; old held DEV rows remain byte-for-byte held."""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.semantic_binding_data_v59 import render,WORDS

def new_pattern(fact,family):
    fields={k:v or '' for k,v in fact['roles'].items()};kind=fact['relation'];positive=fact['polarity']
    if kind==0:
        forms=('A {theme} was a gift for {recipient}, sent by {primary}.','{primary} presented a {theme} to {recipient}.',
            '{recipient} got a {theme}; the giver was {primary}.')
    elif kind==1:
        forms=(('{primary} was the person who did open the {theme}.' if positive else '{primary} was the person who did not open the {theme}.'),
            ('It is the case that {primary} opened the {theme}.' if positive else 'It is not the case that {primary} opened the {theme}.'),
            ('The opening of the {theme} by {primary} occurred.' if positive else 'The opening of the {theme} by {primary} did not occur.'))
    else:
        n=fact['quantity'];fields.update(number=str(n),word=WORDS[n-1],word_cap=WORDS[n-1].capitalize())
        forms=('Of the {theme}, there are {word} in the {primary}.','The {primary} holds exactly {number} {theme}.',
            '{word_cap} {theme} are present inside the {primary}.')
    return forms[family-6],fields

def build(root):
    root=Path(root);old=root/'research/artifacts/semantic_binding_20261111_v59_cpu';out=root/'research/artifacts/semantic_binding_20261111_v60_cpu'
    if out.exists():raise FileExistsError('immutable expanded data')
    read=lambda name:json.loads((old/(name+'.json')).read_text(encoding='utf-8-sig'))
    features,targets,groups=read('train_features'),read('train_targets'),read('train_groups');new_features=[];new_targets=[];new_groups=[]
    for group in groups:
        ids=[]
        for i in group:
            ids.append(len(new_features));new_features.append(features[i]);new_targets.append(targets[i])
        for origin in (group[0],group[3]):
            fact=targets[origin]['meaning']
            for family in (6,7,8):
                text,spans=render(*new_pattern(fact,family));rid=targets[group[0]]['group']+f':expanded{origin==group[3]}:{family}'
                ids.append(len(new_features));new_features.append(dict(id=rid,prompt=text))
                new_targets.append(dict(id=rid,spans=spans,meaning=fact,meaning_id=targets[origin]['meaning_id'],family=family,group=targets[group[0]]['group']))
        new_groups.append(ids)
    orders=[]
    for epoch in range(4):
        shuffled=list(new_groups);random.Random(60311+epoch).shuffle(shuffled)
        orders.extend([sum(shuffled[i:i+2],[]) for i in range(0,len(shuffled),2)])
    dev=read('dev_features');assert not {r['prompt'] for r in new_features}&{r['prompt'] for r in dev}
    out.mkdir()
    for name,value in [('train_features',new_features),('train_targets',new_targets),('train_groups',new_groups),('orders',orders)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('dev_features','dev_targets','dev_groups','OWT_train_reconstruction_windows'):
        (out/(name+'.json')).write_bytes((old/(name+'.json')).read_bytes())
    manifest=dict(schema='semantic_binding_v60_CPU',train_rows=len(new_features),dev_rows=len(dev),train_groups=len(groups),epochs=4,attempts=len(orders),
        row_selections=sum(map(len,orders)),rows_per_batch=20,TRAIN_input_bytes=sum(len(r['prompt'].encode()) for r in new_features),
        max_TRAIN_bytes=max(len(r['prompt'].encode()) for r in new_features),max_DEV_bytes=max(len(r['prompt'].encode()) for r in dev),
        TRAIN_families=[0,1,2,6,7,8],DEV_families=[3,4,5],DEV_exact_old_file_SHA=hashlib.sha256((out/'dev_features.json').read_bytes()).hexdigest(),
        adaptive_design=True,limitation='held failures informed TRAIN lexical coverage; old DEV literals/labels never added to TRAIN; restricted one-event curriculum',
        source_encodes=(len(new_features)+15)//16+(len(dev)+15)//16,cost=832,OWT_windows_prepared_not_trained=733,new_downloads=0,E2_access=0)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return manifest

if __name__=='__main__':print(json.dumps(build(ROOT)))
