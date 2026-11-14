"""Prospective authored construction coverage, retaining one old view/group.

Old observed DEV literals/facts never enter TRAIN. Broad construction families
now overlap (questions/passives/nominal quantities), so DEV is no longer held
at that broad construction-family level. It remains observed lexical DEV.
"""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.semantic_binding_data_v59 import render,WORDS,fact_key


def new_pattern(fact,family):
    fields={k:v or '' for k,v in fact['roles'].items()};kind=fact['relation'];positive=fact['polarity']
    if kind==0:
        forms=('From {primary}, a {theme} was sent to {recipient} as a gift.',
               'Who got the {theme} from {primary}? {recipient} received the gift.',
               'The record names {primary} as the giver of a {theme} and {recipient} as the recipient.')
    elif kind==1:
        forms=(('The {theme} has been opened by {primary}.' if positive else 'The {theme} has not been opened by {primary}.'),
               ('Has {primary} opened the {theme}? Yes, that happened.' if positive else 'Has {primary} opened the {theme}? No, that did not happen.'),
               ('The statement that {primary} opened the {theme} is true.' if positive else 'The statement that {primary} opened the {theme} is false.'))
    else:
        n=fact['quantity'];fields.update(number=str(n),word=WORDS[n-1],word_cap=WORDS[n-1].capitalize())
        forms=('In the {primary}, a count of {word} {theme} was recorded.',
               'What is the number of {theme} inside the {primary}? It is {number}.',
               '{word_cap} {theme} are being held by the {primary}.')
    return forms[family-9],fields


def build():
    old=ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu';out=ROOT/'research/artifacts/semantic_binding_20261112_v63_cpu'
    if out.exists():raise FileExistsError('immutable new construction data')
    read=lambda n:json.loads((old/(n+'.json')).read_text(encoding='utf-8-sig'))
    f,t,g=read('train_features'),read('train_targets'),read('train_groups');features=[];targets=[];groups=[]
    for i,group in enumerate(g):
        ids=[len(features)];features.append(f[group[0]]);targets.append(t[group[0]])
        for j in (1,2,3):
            fact=t[group[3] if j==3 else group[0]]['meaning'];family=9+(i+j-1)%3
            template,fields=new_pattern(fact,family)
            lead=('', 'A distinct event is recorded here.\n','Please inspect this record.\n')[(i+j)%3]
            tail='\nEnd of record.' if (i+j)%2 else ''
            text,spans=render(lead+template+tail,fields);rid=f'train63:{i:04d}:{j}'
            ids.append(len(features));features.append(dict(id=rid,prompt=text))
            targets.append(dict(id=rid,spans=spans,meaning=fact,meaning_id=fact_key(fact),family=family,group='train63:'+str(i)))
        groups.append(ids)
    orders=[]
    for epoch in range(8):
        shuffled=list(groups);random.Random(63311+epoch).shuffle(shuffled)
        orders.extend([sum(shuffled[i:i+4],[]) for i in range(0,len(shuffled),4)])
    dev,dy=read('dev_features'),read('dev_targets')
    assert not {r['prompt'] for r in features}&{r['prompt'] for r in dev}
    assert not {r['meaning_id'] for r in targets}&{r['meaning_id'] for r in dy}
    out.mkdir()
    for name,value in [('train_features',features),('train_targets',targets),('train_groups',groups),('orders',orders)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name in ('dev_features','dev_targets','dev_groups','OWT_train_reconstruction_windows'):
        (out/(name+'.json')).write_bytes((old/(name+'.json')).read_bytes())
    width=max(len(r['prompt'].encode()) for r in features)
    manifest=dict(schema='semantic_binding_v63_CPU',TRAIN_rows=4096,retained_old_rows=1024,new_authored_views=3072,groups=1024,
        unique_meanings=len({r['meaning_id'] for r in targets}),DEV_rows=256,epochs=8,attempts=len(orders),row_selections=sum(map(len,orders)),
        TRAIN_input_bytes=sum(len(r['prompt'].encode()) for r in features),max_TRAIN_bytes=width,min_TRAIN_bytes=min(len(r['prompt'].encode()) for r in features),
        expected_packed_cache_tensor_bytes=4096*width*(704*4+3),new_downloads=0,OWT_used=0,E2_access=0,
        TRAINDEV_literal_overlap=0,TRAINDEV_meaning_overlap=0,DEV_files_unchanged=True,
        broad_construction_family_overlap_NOW=True,DEV_no_longer_broad_family_held=True,
        limitations='adaptive practical coverage repair; restricted give/open/contain; same facts, new correlated wording, observed lexical DEV; not blind or general semantics')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');print(json.dumps(manifest))

if __name__=='__main__':build()
