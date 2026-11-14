"""Authored one-event semantic supervision; source features never contain labels.

This is a restricted extraction curriculum, not gold semantics for arbitrary OWT.
The existing V58 diagnostic and original corpus splits remain untouched.
"""
import hashlib,json,random,re
from pathlib import Path

TRAIN_NAMES=('Alice','Bruno','Celia','Dario','Elise','Felix','Grace','Hugo','Ilana','Jonas','Karla','Leon',
    'Marin','Nadia','Oscar','Petra','Quinn','Rosa','Selim','Talia','Uriel','Vincent','Álvaro','Zoë')
DEV_NAMES=('Iris','Vera','Émile','Søren','Nia','Theo','Eden','Remy')
TRAIN_GIFTS=('silver coin','blue envelope','wooden spoon','green ribbon','small drum','yellow scarf')
DEV_GIFTS=('red badge','glass bottle')
WORDS=('one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen',
    'fifteen','sixteen','seventeen','eighteen','nineteen','twenty','twenty-one','twenty-two','twenty-three','twenty-four')
ROLES=('primary','recipient','theme')

def render(pattern,fields):
    pieces=[];spans={};at=0;cursor=0
    for match in re.finditer(r'\{([a-z_]+)\}',pattern):
        prefix=pattern[cursor:match.start()];pieces.append(prefix);at+=len(prefix.encode())
        key=match[1];value=fields[key];begin=at;pieces.append(value);at+=len(value.encode());cursor=match.end()
        if key in ROLES:
            if key in spans:raise ValueError('one explicit occurrence per semantic role')
            spans[key]=(begin,at)
    pieces.append(pattern[cursor:]);text=''.join(pieces)
    return text,[spans.get(role,(-1,-1)) for role in ROLES]

def pattern(fact,family):
    kind=fact['relation'];polarity=fact['polarity'];fields={k:v or '' for k,v in fact['roles'].items()}
    if kind==0:
        forms=('{primary} gave {recipient} a {theme}.','{recipient} received a {theme} from {primary}.',
            'The {theme} was given to {recipient} by {primary}.','A {theme} went from {primary} to {recipient} as a gift.',
            'As a gift, {primary} handed a {theme} to {recipient}.','The recipient of a {theme} from {primary} was {recipient}.')
    elif kind==1:
        forms=(('{primary} opened the {theme}.' if polarity else '{primary} did not open the {theme}.'),
            ('It is true that {primary} opened the {theme}.' if polarity else 'It is not true that {primary} opened the {theme}.'),
            ('According to the entry, {primary} did open the {theme}.' if polarity else 'According to the entry, {primary} did not open the {theme}.'),
            ('The {theme} was opened by {primary}.' if polarity else 'The {theme} was not opened by {primary}.'),
            ('Did {primary} open the {theme}? Yes.' if polarity else 'Did {primary} open the {theme}? No.'),
            ('Regarding the {theme}, it is true that {primary} opened it.' if polarity else 'Regarding the {theme}, it is false that {primary} opened it.'))
    else:
        n=fact['quantity'];fields.update(number=str(n),word=WORDS[n-1],word_cap=WORDS[n-1].capitalize())
        forms=('The {primary} contains {number} {theme}.','There are {word} {theme} in the {primary}.',
            'Inside the {primary} are {number} {theme}.','The count of {theme} in the {primary} is {word}.',
            '{word_cap} {theme} are held by the {primary}.','In the {primary}, the number of {theme} is {number}.')
    return forms[family],fields

def fact_key(fact):return hashlib.sha256(json.dumps(fact,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def make_split(split):
    rng=random.Random(59111 if split=='train' else 59211);names=TRAIN_NAMES if split=='train' else DEV_NAMES
    gifts=TRAIN_GIFTS if split=='train' else DEV_GIFTS
    gates=('wooden gate','green door','iron hatch','small window') if split=='train' else ('garden gate','glass door')
    containers=('amber box','blue basket','green chest','wooden bin') if split=='train' else ('violet box','steel basket')
    colors=('red','blue','green','yellow') if split=='train' else ('orange','purple')
    wanted=96 if split=='train' else 24;features=[];targets=[];groups=[]
    for kind in range(3):
        candidates=[]
        if kind==0:
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    for obj in gifts:candidates.append(dict(relation=0,polarity=1,quantity=0,roles=dict(primary=a,recipient=b,theme=obj)))
        elif kind==1:
            for i,a in enumerate(names):
                for j,obj in enumerate(gates):candidates.append(dict(relation=1,polarity=(i+j)%2,quantity=0,roles=dict(primary=a,recipient=None,theme=obj)))
        else:
            for box in containers:
                for color in colors:
                    for n in range(1,25):candidates.append(dict(relation=2,polarity=1,quantity=n,roles=dict(primary=box,recipient=None,theme=color+' tokens')))
        rng.shuffle(candidates)
        # DEV negation has only16 distinct facts. Keep its true smaller size.
        for i,fact in enumerate(candidates[:wanted]):
            gid=f'{split}:{kind}:{i:03d}';other=json.loads(json.dumps(fact))
            if kind==0:other['roles']['primary'],other['roles']['recipient']=other['roles']['recipient'],other['roles']['primary']
            elif kind==1:other['polarity']=1-other['polarity']
            else:other['quantity']=other['quantity']%24+1
            indices=[]
            for j in range(4):
                selected=other if j==3 else fact;family=(j if j<3 else i%3)+(0 if split=='train' else 3)
                template,fields=pattern(selected,family)
                if j==2:
                    lead=('Here is the entry.\n','A brief note follows.\n','For the record.\n')[i%3]
                    template=lead+template+'\nEnd of entry.'
                text,spans=render(template,fields);rid=gid+f':{j}'
                raw=text.encode()
                for role,(a,b) in zip(ROLES,spans):
                    if a>=0:assert raw[a:b].decode()==selected['roles'][role]
                target=dict(id=rid,spans=spans,meaning=selected,meaning_id=fact_key(selected),family=family,group=gid)
                indices.append(len(features));features.append(dict(id=rid,prompt=text));targets.append(target)
            groups.append(indices)
    assert all(set(r)=={'id','prompt'} for r in features)
    return features,targets,groups

def make_orders(groups,epochs=8,batch_groups=4):
    orders=[]
    for epoch in range(epochs):
        shuffled=list(groups);random.Random(59311+epoch).shuffle(shuffled)
        orders.extend([sum(shuffled[i:i+batch_groups],[]) for i in range(0,len(shuffled),batch_groups)])
    return orders

def build(root):
    root=Path(root);out=root/'research/artifacts/semantic_binding_20261111_v59_cpu'
    if out.exists():raise FileExistsError('immutable semantic data')
    train,ty,tg=make_split('train');dev,dy,dg=make_split('dev')
    previous=json.loads((root/'research/artifacts/semantic_controls_20261111_v58_cpu/features.json').read_text(encoding='utf-8-sig'))
    assert not {r['prompt'] for r in train}&{r['prompt'] for r in dev+previous}
    assert not {r['meaning_id'] for r in ty}&{r['meaning_id'] for r in dy}
    orders=make_orders(tg);out.mkdir()
    for name,value in [('train_features',train),('train_targets',ty),('train_groups',tg),('dev_features',dev),('dev_targets',dy),('dev_groups',dg),('orders',orders)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Original OWT short/mid/long reconstruction inputs for the NEXT language
    # alignment phase. No invented fact labels and no DEV documents enter TRAIN.
    original=json.loads((root/'research/artifacts/broad_long_codec_20261111_v53_cpu/train.json').read_text(encoding='utf-8-sig'))
    selected=sorted(original,key=lambda r:hashlib.sha256(('v59-owt:'+r['id']).encode()).hexdigest())[:256];windows=[]
    for row in selected:
        text=row['prompt'];ends=[m.end() for m in re.finditer(r'[.!?](?:["”\']?)(?:\s+|$)',text)]
        end=next((n for n in ends if 16<=len(text[:n].encode())<=256),None)
        for label,cut in [('short',end),('mid',next((n for n in ends if 256<=len(text[:n].encode())<=768),None)),('long',len(text))]:
            if cut is None:continue
            source=text[:cut];windows.append(dict(id=row['id']+':'+label,prompt=source,source_id=row['source_id'] if 'source_id' in row else row['id'],
                source_window_byte_start=0,source_window_byte_end=len(source.encode()),selection=label))
    (out/'OWT_train_reconstruction_windows.json').write_text(json.dumps(windows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest=dict(schema='semantic_binding_v59_CPU',train_rows=len(train),dev_rows=len(dev),train_fact_groups=len(tg),dev_fact_groups=len(dg),
        attempts=len(orders),epochs=8,row_selections=sum(map(len,orders)),TRAIN_families=[0,1,2],DEV_families=[3,4,5],
        train_names=list(TRAIN_NAMES),dev_names=list(DEV_NAMES),V58_literal_train_overlap=0,train_dev_meaning_overlap=0,
        max_input_bytes=max(len(r['prompt'].encode()) for r in train+dev),TRAIN_target_bytes=sum(len(r['prompt'].encode()) for r in train),
        OWT_unlabelled_reconstruction_windows=len(windows),OWT_windows_used_for_semantic_supervision=False,
        source='root-authored one-event extraction curriculum + original OWT TRAIN windows',new_downloads=0,E2_access=0,
        feature_keys=['id','prompt'],gold_span_facts='loss/score only',limitations='restricted give/open/contain vocabulary; no general OWT meaning labels or semantic language-generation claim')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return manifest

if __name__=='__main__':print(json.dumps(build(Path(__file__).resolve().parents[1])))
