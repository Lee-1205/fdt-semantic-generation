"""Independent English paraphrase/different-meaning groups plus retained OWT."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRAIN_NAMES=['Calder','Vera','Milo','Selene','Ansel','Petra','Hugo','Tamsin','Bram','Leona','Felix','Cora','Emmett','Sabine','Galen','Nadia']
DEV_NAMES=['Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette']
ITEMS=['a bronze key','a folded map','a blue notebook','a small wooden box','a silver badge','a green envelope']
DOORS=['the bronze door','the garden gate','the eastern window','the small wooden hatch']
PLURALS=['glass beads','brass pins','blue pencils','wooden tokens']

def canonical(fact):return json.dumps(fact,sort_keys=True,separators=(',',':'))
def identity(text):return hashlib.sha256(('v68:'+text).encode()).hexdigest()[:24]

def make_groups(count,names,seed,held):
    rng=random.Random(seed);features=[];targets=[];groups=[];meanings={}
    for g in range(count):
        a,b=rng.sample(names,2);item=rng.choice(ITEMS);door=rng.choice(DOORS);q=rng.randint(3,27);kind=g%4
        if kind==0:
            facts=[dict(kind='transfer',actor=a,recipient=b,item=item),dict(kind='transfer',actor=b,recipient=a,item=item)]
            views=[]
            for fact in facts:
                actor,recipient=fact['actor'],fact['recipient']
                if held:
                    views.extend([f'{recipient} received {item} from {actor}.',f'The person who gave {item} to {recipient} was {actor}.'])
                elif g%8:
                    views.extend([f'{actor} gave {item} to {recipient}.',f'{item[0].upper()+item[1:]} was given to {recipient} by {actor}.'])
                else:
                    views.extend([f'{actor} handed {item} to {recipient}.',f'The recipient of {item} from {actor} was {recipient}.'])
        elif kind==1:
            facts=[dict(kind='opening',actor=a,item=door,negated=False),dict(kind='opening',actor=a,item=door,negated=True)]
            if held:
                views=[f'It was {a} who opened {door}.',f'{door[0].upper()+door[1:]} was opened, and {a} was the person who opened it.',
                    f'It is not true that {a} opened {door}.',f'The statement that {a} opened {door} is false.']
            else:
                views=[f'{a} opened {door}.',f'{door[0].upper()+door[1:]} was opened by {a}.',f'{a} did not open {door}.',f'{door[0].upper()+door[1:]} was not opened by {a}.']
        elif kind==2:
            obj=rng.choice(PLURALS);container=f"{a}'s green box";facts=[dict(kind='contain',owner=a,container=container,item=obj,quantity=q),dict(kind='contain',owner=a,container=container,item=obj,quantity=q+1)]
            views=[]
            for fact in facts:
                n=fact['quantity']
                if held:views.extend([f'The number of {obj} inside {container} is {n}.',f'Inside {container}, the count of {obj} is {n}.'])
                else:views.extend([f'{container} contains {n} {obj}.',f'There are {n} {obj} in {container}.'])
        else:
            facts=[dict(kind='temporal',actor=a,moved=item,closed=door,order='before'),dict(kind='temporal',actor=a,moved=item,closed=door,order='after')];views=[]
            for fact in facts:
                order=fact['order']
                if held:views.extend([f'{a} moved {item}; this happened {order} {a} closed {door}.',f'The movement of {item} by {a} happened {order} the closing of {door} by {a}.'])
                else:views.extend([f'{order.capitalize()} closing {door}, {a} moved {item}.',f'{a} moved {item} {order} closing {door}.'])
        member=[];labels=[]
        for j,text in enumerate(views):
            fact=facts[j//2];key=canonical(fact);meanings.setdefault(key,len(meanings));i=len(features);rid=identity(text)
            features.append(dict(id=rid,prompt=text));targets.append(dict(id=rid,meaning_id=meanings[key],canonical_fact=fact,group=g,view=j));member.append(i);labels.append(meanings[key])
        groups.append(dict(group=g,indices=member,meaning_ids=labels,kind=facts[0]['kind']))
    return features,targets,groups

def main():
    out=ROOT/'research/artifacts/english_memory_cycle_20261112_v68_cpu';out.mkdir(exist_ok=False)
    train,targets,groups=make_groups(64,TRAIN_NAMES,68112,False);dev,devtargets,devgroups=make_groups(16,DEV_NAMES,68113,True)
    assert len(train)==256 and len(dev)==64
    assert not {r['prompt'] for r in train}&{r['prompt'] for r in dev}
    original=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu/train.json';windows=ROOT/'research/artifacts/semantic_binding_20261111_v59_cpu/OWT_train_reconstruction_windows.json'
    long=json.loads(original.read_text(encoding='utf-8-sig'));short=json.loads(windows.read_text(encoding='utf-8-sig'));owt=[]
    choose=lambda rows,n:sorted(rows,key=lambda r:hashlib.sha256(('v68:'+r['id']).encode()).hexdigest())[:n]
    for label in ('short','mid'):owt.extend(choose([r for r in short if r['selection']==label],16))
    for low,high in ((769,2048),(2049,8192),(8193,16383),(16384,16384)):owt.extend(choose([r for r in long if low<=len(r['prompt'].encode())<=high],24))
    assert len(owt)==128;owt=[dict(id=r['id'],prompt=r['prompt']) for r in owt]
    held=json.loads((ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json').read_text())
    assert not any(h['prompt'] in t['prompt'] for h in held for t in train+owt)
    orders=[];docids=list(range(128));random.Random(68114).shuffle(docids);at=0
    for epoch in range(8):
        ids=list(range(64));random.Random(68112+epoch).shuffle(ids)
        for k,g in enumerate(ids):
            orders.append(dict(kind='meaning',group=g,indices=groups[g]['indices'],epoch=epoch))
            if k%4==3:orders.append(dict(kind='OWT',indices=[docids[at]],epoch=0));at+=1
    assert len(orders)==640 and at==128
    manifest=dict(schema='english_memory_cycle_v68_data',TRAIN_features=256,TRAIN_groups=64,TRAIN_meanings=len({canonical(t['canonical_fact']) for t in targets}),
        DEV_features=64,DEV_groups=16,TRAIN_names=TRAIN_NAMES,DEV_names=DEV_NAMES,group_epochs=8,group_attempts=512,OWT_attempts=128,total_attempts=640,
        synthetic_row_selections=2048,OWT_row_selections=128,target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train)*8+sum(len(r['prompt'].encode()) for r in owt),
        feature_keys=['id','prompt'],all_semantic_labels_loss_score_only=True,global_alignment='actual first2 CodecCommit views; no factor probabilities or raw H',
        random_local_slot_dropout='.25 source-Commit-topology only, independent of gold role/span/target',
        source_TRAIN_files=[str(original),str(windows)],new_downloads=0,E2_registry_access=0,Exact_policy='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST',
        limitations='small independently authored correlated groups; held names/forms plus observed development design; semantic inequality negatives not necessarily logical contradiction; no GPU or new model performance yet')
    for name,value in [('train_features',train),('train_targets',targets),('train_groups',groups),('dev_features',dev),('dev_targets',devtargets),('dev_groups',devgroups),('OWT_features',owt),('orders',orders),('manifest',manifest)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest))
if __name__=='__main__':main()
