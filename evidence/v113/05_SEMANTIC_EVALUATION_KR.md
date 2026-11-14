2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.

# 05. 실제 의미·일반화 평가

**OBSERVED** V113은제한된TRAIN에서실제정상관계core를생성하기시작했다. ROOT가전체80TRAIN긍정출력을모두읽었다(max149byte). 정상문법으로맞는관계core34/80와 `C is smallertthan F.`라는명확한철자손상1개를별도로확인했다. 최소35/80core를읽을수있었다. 이는전체80개의일반의미정확도를인증한숫자가아니며, 정답뒤잡음이있어서긍정전체문자열exact는0/80이다. 전체TRAIN exact17/134는모두UNKNOWN이다. UNKNOWN은54/54가정답prefix지만뒤noise를완전성공으로숨기지않는다.

|동일조건패널|NLL 전→후|byte accuracy 전→후|자동정상관계core(긍정)|
|---|---:|---:|---:|
|TRAIN_all|1.571820→0.479030|76.842→96.647%|34/80|
|dev_pair|2.487229→2.731941|74.340→83.382%|0/16|
|dev_surface|4.695938→5.686901|47.749→49.648%|0/16|
|dev_position|6.957152→7.663967|36.533→36.821%|0/16|

직접읽은예시:
- `H is larger than B.rly` → H>B core맞음, 뒤잡음.
- `F is smaller than C.CA    .faC a &` → F<C core맞음, 뒤잡음.
- `C is smallertthan F.` → C<F 의미를읽을수있으나철자손상(자동문법은miss).
- 정답F>C에 `C is larger than F..Ceia f` → 관계가반대라실패.
- 정답A>G에 `A is larger than B.Cltnp...` → 객체오류.
- 새DEV정답C>D에 `H is larger than D.Smer iemana` → 관계word는맞지만주체오류.

DEV긍정48개앞150자를직접읽었고확인된정상정답core0이었다. 긴DEV잡음tail전부를수동판독한것은아니다. UNKNOWN상수기준은TRAIN54/134=40.30%, DEV각16/32=50%다. 전체정답률이나byte향상만으로이기준을넘는일반의미이해를주장하지않는다.

32긍정역전쌍정상관계bothcorrect0→8/32, 동일출력2→0, target방향cosine.668333→.845888, target차이에대한투영비율.310958→.524632, 차이MSE1.479526→.861468. 이잠재거리비율들은의미정확도가아니다. 8짝의TRAIN질문역할반응은제한된부분학습의증거이며메모리개별인과·독립추상사고·Composition증거는아니다.

TRAIN만으로맞춘최적상수bodyMSE0.768583보다현재0.582379가낮다. 따라서현재를전부상수출력이라고부를수없다. 반면미관측사실로일반화했다고도할수없다.

**VERIFIED 자료점검** TRAIN134/31factkeys/27answers/family0–3(23/31/50/30),layout0/1=56/78. dev_pair는fact/groupseen0/32이고긍정정답seen0/16이다. 최종DEV긍정16중12개는TRAIN의정형답문core를대신출력했다. 이는새개체조합일반화부족과일치하지만메모리검색답안반환코드가있다는뜻은아니다. 실제추론에답안검색/GT조건은없었다.

dev_surface/position은fact와답문이모두TRAIN에있으나실패했다. 단순미관측답문문제만으로설명할수없고질문표현·위치과의존도남는다. NLL은모든DEV에서악화했으므로새모델을전체개선으로승격하지않는다. 이미관측한개발자료의다중사용·선별bias·짧은A-H자료한계를유지한다.
