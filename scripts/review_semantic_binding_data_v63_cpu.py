"""Independent audit: retained old rows, new authored meaning, UTF8, orders."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'research/artifacts/semantic_binding_20261112_v63_cpu';old=ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu'
    read=lambda d,n:json.loads((d/(n+'.json')).read_text(encoding='utf-8-sig'))
    f,t,g,orders=[read(data,n) for n in ('train_features','train_targets','train_groups','orders')]
    oldf,oldt,oldg=[read(old,n) for n in ('train_features','train_targets','train_groups')]
    dev,dy=read(data,'dev_features'),read(data,'dev_targets')
    for name in ('dev_features','dev_targets','dev_groups','OWT_train_reconstruction_windows'):assert (data/(name+'.json')).read_bytes()==(old/(name+'.json')).read_bytes()
    for features,targets in ((f,t),(dev,dy)):
        for a,b in zip(features,targets):
            assert set(a)=={'id','prompt'} and a['id']==b['id'];raw=a['prompt'].encode()
            for key,(x,y) in zip(('primary','recipient','theme'),b['spans']):
                if b['meaning']['roles'][key] is None:assert [x,y]==[-1,-1]
                else:assert raw[x:y].decode()==b['meaning']['roles'][key] and raw[:x].decode() is not None and raw[y:].decode() is not None
    words='one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four'.split()
    prefixes=('A distinct event is recorded here.\n','Please inspect this record.\n')
    for i,group in enumerate(g):
        assert f[group[0]]==oldf[oldg[i][0]] and t[group[0]]==oldt[oldg[i][0]]
        assert t[group[0]]['meaning']==t[group[1]]['meaning']==t[group[2]]['meaning']!=t[group[3]]['meaning']
        for j in group[1:]:
            text=f[j]['prompt'].removesuffix('\nEnd of record.')
            for prefix in prefixes:
                if text.startswith(prefix):text=text[len(prefix):]
            truth=t[j]['meaning'];role=truth['roles'];kind=truth['relation'];family=t[j]['family']
            if kind==0:
                patterns={9:r'From (.+?), a (.+?) was sent to (.+?) as a gift\.',10:r'Who got the (.+?) from (.+?)\? (.+?) received the gift\.',11:r'The record names (.+?) as the giver of a (.+?) and (.+?) as the recipient\.'}
                match=re.fullmatch(patterns[family],text);assert match
                order={9:('primary','theme','recipient'),10:('theme','primary','recipient'),11:('primary','theme','recipient')}[family]
                assert dict(zip(order,match.groups()))==role
            elif kind==1:
                if family==9:
                    match=re.fullmatch(r'The (.+?) has (not )?been opened by (.+?)\.',text);assert match
                    obj,neg,actor=match.groups();positive=neg is None
                elif family==10:
                    match=re.fullmatch(r'Has (.+?) opened the (.+?)\? (Yes|No), that (happened|did not happen)\.',text);assert match
                    actor,obj,answer,event=match.groups();positive=answer=='Yes';assert positive==(event=='happened')
                else:
                    match=re.fullmatch(r'The statement that (.+?) opened the (.+?) is (true|false)\.',text);assert match
                    actor,obj,truth_word=match.groups();positive=truth_word=='true'
                assert actor==role['primary'] and obj==role['theme'] and positive==bool(truth['polarity'])
            else:
                if family==9:
                    match=re.fullmatch(r'In the (.+?), a count of ([a-z-]+) (.+?) was recorded\.',text);assert match
                    box,count,obj=match.groups();count=words.index(count)+1
                elif family==10:
                    match=re.fullmatch(r'What is the number of (.+?) inside the (.+?)\? It is ([0-9]+)\.',text);assert match
                    obj,box,count=match.groups();count=int(count)
                else:
                    match=re.fullmatch(r'([A-Za-z-]+) (.+?) are being held by the (.+?)\.',text);assert match
                    count,obj,box=match.groups();count=words.index(count.lower())+1
                assert box==role['primary'] and obj==role['theme'] and count==truth['quantity']
    for epoch in range(8):
        batches=orders[epoch*256:(epoch+1)*256];assert sorted(sum(batches,[]))==list(range(4096))
        for b in batches:assert all(not(set(b)&set(group)) or set(b).issuperset(group) for group in g)
    assert not {a['prompt'] for a in f}&{a['prompt'] for a in dev}
    assert not {a['meaning_id'] for a in t}&{a['meaning_id'] for a in dy}
    patterns={'question':r'\?','passive_open':r'\bhas (?:not )?been opened by\b','false_scope':r'\bfalse\b','count':r'\bcount\b','number':r'\bnumber\b','held_by':r'\bheld by\b'}
    coverage={key:sum(bool(re.search(pattern,a['prompt'])) for a in f) for key,pattern in patterns.items()}
    assert all(n>0 for n in coverage.values())
    report=dict(status='ROOT_V63_DATA_AUDIT_PASS',all4352_UTF8_spans=True,retained1024exact_old_rows=True,new3072_offline_meanings_parsed=True,
        eight_finite_orders_and_groups=True,TRAINDEV_literal_meaning_overlap=0,held4files_byte_identical=True,TRAIN_coverage=coverage,
        broad_construction_family_overlap_NOW=True,oldDEV_NOT_broadfamilyheld=True,
        cached_UTF8_exposure=8*sum(len(a['prompt'].encode()) for a in f),TRAIN_duplicates=len(f)-len({a['prompt'] for a in f}),model_forwards=0)
    (data/'root_data_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))

if __name__=='__main__':main()
