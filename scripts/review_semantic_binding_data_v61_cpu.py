"""Independent byte offsets, offline authored semantics and finite exposure."""
import collections,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu';old=ROOT/'research/artifacts/semantic_binding_20261111_v60_cpu'
    read=lambda n:json.loads((data/(n+'.json')).read_text(encoding='utf-8-sig'))
    train,ty,groups,orders=map(read,('train_features','train_targets','train_groups','orders'))
    dev,dy=read('dev_features'),read('dev_targets')
    for name in ('dev_features','dev_targets','dev_groups','OWT_train_reconstruction_windows'):
        assert (data/(name+'.json')).read_bytes()==(old/(name+'.json')).read_bytes()
    for fs,ts in ((train,ty),(dev,dy)):
        for f,t in zip(fs,ts):
            assert set(f)=={'id','prompt'} and f['id']==t['id']
            raw=f['prompt'].encode()
            for key,(a,b) in zip(('primary','recipient','theme'),t['spans']):
                if t['meaning']['roles'][key] is None:assert [a,b]==[-1,-1]
                else:assert raw[a:b].decode()==t['meaning']['roles'][key] and raw[:a].decode() is not None and raw[b:].decode() is not None
    words=('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four').split()
    prefixes=('Log header.\n','This is a separate recorded event.\n','Please read the following entry.\n','Record details follow.\n')
    for f,t in zip(train,ty):
        text=f['prompt'].removesuffix('\nRecord ends.')
        while any(text.startswith(prefix) for prefix in prefixes):text=next(text[len(p):] for p in prefixes if text.startswith(p))
        k=t['meaning']['relation'];fam=t['family'];truth=t['meaning'];roles=truth['roles']
        if k==0:
            patterns={0:r'(.+?) gave (.+?) a (.+?)\.',1:r'(.+?) received a (.+?) from (.+?)\.',2:r'The (.+?) was given to (.+?) by (.+?)\.',
                6:r'A (.+?) was a gift for (.+?), sent by (.+?)\.',7:r'(.+?) presented a (.+?) to (.+?)\.',8:r'(.+?) got a (.+?); the giver was (.+?)\.'}
            m=re.fullmatch(patterns[fam],text);assert m
            ordering={0:('primary','recipient','theme'),1:('recipient','theme','primary'),2:('theme','recipient','primary'),
                6:('theme','recipient','primary'),7:('primary','theme','recipient'),8:('recipient','theme','primary')}
            assert dict(zip(ordering[fam],m.groups()))==roles
        elif k==1:
            # Negation cues used only to audit training annotation, never inference.
            assert (' not ' not in ' '+text+' ')==bool(truth['polarity'])
            assert text.count(roles['primary'])==text.count(roles['theme'])==1
        else:
            matches=re.findall(r'\b(?:[0-9]+|'+('|'.join(sorted(words,key=len,reverse=True)))+r')\b',text.lower())
            assert len(matches)==1
            count=int(matches[0]) if matches[0].isdigit() else words.index(matches[0])+1
            assert count==truth['quantity'] and text.count(roles['primary'])==text.count(roles['theme'])==1
    assert not {f['prompt'] for f in train}&{f['prompt'] for f in dev}
    assert not {t['meaning_id'] for t in ty}&{t['meaning_id'] for t in dy}
    for g in groups:
        assert len(g)==4 and ty[g[0]]['meaning']==ty[g[1]]['meaning']==ty[g[2]]['meaning']!=ty[g[3]]['meaning']
    for epoch in range(4):
        batches=orders[epoch*256:(epoch+1)*256];assert sorted(sum(batches,[]))==list(range(4096))
        for b in batches:
            selected=set(b)
            assert all(not(selected&set(g)) or selected.issuperset(g) for g in groups)
    report=dict(all4352_UTF8_offsets_and_source_split_checked=True,TRAIN4096_offline_annotation_checked=True,
        exact4epochs16384rows=True,groups1024_preserved=True,TRAIN_unique_meanings=len({t['meaning_id'] for t in ty}),
        TRAIN_duplicate_prompts=len(train)-len({f['prompt'] for f in train}),held4files_byte_identical=True,
        TRAINDEV_literal_and_meaning_overlap=0,TRAIN_families=sorted({t['family'] for t in ty}),
        cached_UTF8_exposure=4*sum(len(f['prompt'].encode()) for f in train),model_forwards=0,
        caveat='adaptive authored curriculum; existing DEV already observed; not blind or general OWT semantic truth')
    (data/'root_data_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))

if __name__=='__main__':main()
