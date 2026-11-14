"""Newly authored after V65 ended; assessment only, never added to TRAIN."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENGLISH=[
 'Nerine gave Orrin seven copper buttons, but kept the small wooden box for herself.',
 'Orrin gave Nerine seven copper buttons, but kept the small wooden box for himself.',
 'The north gate remained open after the gardener closed the window beside it.',
 'The north gate remained closed after the gardener opened the window beside it.',
 'At 6:45 on Tuesday, Alton moved three blue folders from the lower shelf to the upper drawer.',
 'At 6:45 on Tuesday, Alton moved three blue folders from the upper drawer to the lower shelf.',
 'The orange basket contains eleven pears; the violet basket contains only four.',
 'The orange basket contains four pears; the violet basket contains only eleven.',
 'Although the ferry was delayed, Rhea reached the library before the evening lecture began.',
 'Although Rhea reached the library, she arrived after the evening lecture had already ended.',
 'No visitor entered the greenhouse while the eastern door was unlocked.',
 'One visitor entered the greenhouse while the eastern door was unlocked.',
 'Inez lent the telescope to Dorian, who returned it before the rain started.',
 'Dorian lent the telescope to Inez, who returned it before the rain started.',
 'The longer rope is lighter than the shorter rope because their materials are different.',
 'The longer rope is heavier than the shorter rope because their materials are different.',
 'A payment of 42.75 euros covered the notebook and two replacement pens.',
 'A payment of 47.25 euros covered the notebook and three replacement pens.',
 'The unsigned letter was placed beneath the map, rather than inside the red envelope.',
 'The unsigned letter was placed inside the red envelope, rather than beneath the map.',
 'Before repairing the lantern, Soren recorded the cracked lens and disconnected its battery.',
 'After repairing the lantern, Soren replaced the cracked lens and reconnected its battery.',
 'Only the second photograph shows the white tower reflected in the shallow pond.',
 'Both photographs show the white tower reflected in the shallow pond.'
]
KOREAN=[
 '다온은 노란 상자에 유리구슬 일곱 개를 넣고 나무구슬 세 개는 서랍에 남겼다.',
 '다온은 노란 상자에 나무구슬 세 개를 넣고 유리구슬 일곱 개는 서랍에 남겼다.',
 '아침에 열린 북쪽 문은 저녁에도 열려 있었지만 동쪽 창문은 닫혔다.',
 '아침에 닫힌 북쪽 문은 저녁에도 닫혀 있었지만 동쪽 창문은 열렸다.',
 '유나는 민서에게 파란 공책 두 권을 빌려주었고 금요일에 돌려받았다.',
 '민서는 유나에게 파란 공책 두 권을 빌려주었고 금요일에 돌려받았다.',
 '기차는 오후 세 시 십칠 분에 출발했으며 지수는 출발하기 전에 역에 도착했다.',
 '기차는 오후 세 시 십칠 분에 출발했으며 지수는 출발한 뒤에 역에 도착했다.',
 '작은 배는 다리를 지나기 전에 멈췄고 큰 배는 다리를 지난 뒤에 멈췄다.',
 '작은 배는 다리를 지난 뒤에 멈췄고 큰 배는 다리를 지나기 전에 멈췄다.',
 '연두색 바구니에는 사과가 열한 개 있으며 보라색 바구니에는 네 개뿐이다.',
 '연두색 바구니에는 사과가 네 개 있으며 보라색 바구니에는 열한 개뿐이다.',
 '비가 오지 않았으므로 서윤은 우산을 펼치지 않고 도서관으로 걸어갔다.',
 '비가 내렸으므로 서윤은 우산을 펼친 뒤 도서관으로 걸어갔다.',
 '보고서의 두 번째 표에는 비용이 42,750원이라고 적혀 있고 첫 번째 표에는 날짜만 있다.',
 '보고서의 첫 번째 표에는 비용이 47,250원이라고 적혀 있고 두 번째 표에는 날짜만 있다.'
]

def main():
    run=ROOT/'runs/balanced_fuzzy_cycle_20261112_v65';terminal=json.loads((run/'final_status.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone']
    out=ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu';out.mkdir(exist_ok=False)
    rows=[dict(id=f'fresh66:en:{i}',prompt=text,language='English',kind='single_sentence',group=i//2) for i,text in enumerate(ENGLISH)]
    rows.extend(dict(id=f'fresh66:ko:{i}',prompt=text,language='Korean',kind='single_sentence',group=i//2) for i,text in enumerate(KOREAN))
    names=['Nerine','Orrin','Rhea','Alton','Inez','Dorian','Soren','Elva']
    objects=['brass clips','blue notebooks','glass beads','linen ribbons','wooden tokens','silver pins']
    rng=random.Random(66112)
    for size in (512,1024,2048,4096,8192,16384):
        for variant in range(2):
            text=f'The following account describes a new visit to the Alderwick community workshop, session {size+variant}.\n\n';number=1
            while True:
                a,b=rng.sample(names,2);item=rng.choice(objects);quantity=rng.randint(3,29);left=rng.randint(1,8)
                paragraph=(f'In record {number}, {a} placed {quantity} {item} in the green container and left {left} on the bench. '
                    f'{b} checked the count before moving the container to the west shelf. '
                    f'The east cabinet stayed closed; neither person put the remaining objects in that cabinet.\n\n')
                if len((text+paragraph).encode())>size:break
                text+=paragraph;number+=1
            rows.append(dict(id=f'fresh66:paragraph:{size}:{variant}',prompt=text.rstrip(),language='English',kind='multi_sentence',requested_max_bytes=size,
                group=f'paragraph:{size}:{variant}',templated_authored=True))
    assert len(rows)==52 and max(len(r['prompt'].encode()) for r in rows)<=16384
    # Narrow explicit ancestry checks; never open E2/registry/sealed test paths.
    sources=[ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu/train.json',ROOT/'research/artifacts/fuzzy_roundtrip_20261112_v64_cpu/train.json',
        ROOT/'research/artifacts/balanced_fuzzy_cycle_20261112_v65_cpu/train.json']
    corpus=[r['prompt'] for path in sources for r in json.loads(path.read_text(encoding='utf-8-sig'))]
    for row in rows:assert not any(row['prompt'] in old for old in corpus)
    manifest=dict(schema='fresh_fuzzy_eval_v66_data',rows=52,English_single=24,Korean_single=16,English_contexts=12,
        target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),actual_lengths=[len(r['prompt'].encode()) for r in rows],
        frozen_model_run=str(run),frozen_model_SHA='3351bae2358ed660effbb67c642650b3401b9142f6b9b09e19d823310f9cc7a4',
        authored_after_V65_terminal=True,training_exposure_rows=0,literal_or_substring_overlap_with_checked_TRAIN=0,checked_TRAIN_sources=list(map(str,sources)),
        E2_registry_access=0,new_downloads=0,optimizer_updates=0,
        limitations='new ROOT-authored contrast pairs and templated long contexts, correlated forms; prior pretraining overlap not exhaustively knowable; not a random natural-corpus benchmark')
    for name,value in (('rows',rows),('manifest',manifest)):(out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest))
if __name__=='__main__':main()
