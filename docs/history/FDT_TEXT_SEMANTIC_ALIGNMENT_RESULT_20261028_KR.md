# V3 생성 결과와 현재 한계 — 2026-10-28

Root가 에이전트 없이 설계·구현·CPU검사·학습·원문검증을 직접 완료했다. V3는 **원문→동결byte encoder/VP제안→학습된질문정보reader→학습된causal byte decoder→문장/EOS** 경로다. Runtime은정답/외부개체역할/regex해석/고정답변조립/검색을쓰지않는다. 합성자료의정답작성과사후채점에만규칙을썼다. Diffusion단독출력이나범용LM완성으로부르지않는다.

원래typed102,237,197모델/core/checkpoint는보존했다. 별도V3hybrid는동결VP91,633,920+학습reader/decoder18,144,535=총109,778,455parameters다. 언어encoder63,913,728개는이번세분기에서계속동결됐으며폭넓은어휘·문서능력학습을했다고하지않는다.

## 관측 개발 결과

| 패널 | exact | 의미동치 별도 | 정상UTF8/EOS | 긍정문장 exact | 불명상수 기준선 |
|---|---:|---:|---:|---:|---:|
| 기존train probe16 | 6/16 | 7/16 | 16/16 | 1/11 | 5/16 |
| 미관측pair+language | 12/32(37.5%) | 15/32 | 32/32 | 2/22 | 10/32 |
| 미학습wording | 13/32(40.625%) | 16/32 | 32/32 | 3/22 | 10/32 |

정상문장형식은80/80이지만, **75%exact/60%긍정exact의새의미목표는FAIL**이다. 미관측pair의주어/목적어/관계/언어head는59.375/56.25/68.75/100%, 새wording은56.25/56.25/50/100%다. 구조화역할을모델입력으로받지않고일부질문정보를학습했지만안정적원문이해가아니다.

실제출력은 `C is smaller than D.`처럼문장자체는정상이지만방향이반대이거나, `C is smaller than C.`처럼개체결합이잘못될수있다. 정답개체/방향세head가모두맞은미관측pair10개중3개도생성의미가틀렸다. Reader오류뿐아니라예측semanticfactor를문장에충실히옮기는계약도미완료다. 이CPU사후분류는단일원인증명이아니다.

Zero-VPlatent통제exact는4/16·7/32·8/32였고head예측은불변이다. 정상latent6/16·12/32·13/32와차이가있지만작고상관된관측자료의단일학습결과라독립diffusion추론효과를확정하지않는다. V2같은패널exact5/16·10/32·13/32와도구분한다. V3의새입력구조/보조목적/학습이함께바뀌었고V3초기baseline은새reader를포함하므로순수구조/학습단독인과이득은아니다.

## 실제 실행·보존

- ONEhiddenownedlaunch07:50:50.497/.522KST owner24832→8128,07:50:54.263/.279 worker31000→40968의실제생성/parent/command를root가관측했다. Worker/ownerexit0·COMPLETE,종료뒤네프로세스및freshnativePython0확인.
- 512attempt/512update/AMPskip0,1,024고유pool행의4유한순열4,096exposures/target80,632UTF8bytes. 기존VPcache를자료로재사용했고새VPencode/denoise0,reader542+decoder1226=1,768calls/39.745worker초. 새900초/2500call예산CLOSED/미사용분재사용금지.
- PlanSHA254ea38230f1cce1f9a1b31c6e3a87f0284083ca960cb65f6ce99604309089fc. 최종585,034,553B SHAa5e1c0e774c89a22dde794d2cffd138926ae961fd8cde592bf466d25d63a71b3, algorithmSHA52d59bfbd5bb2021982065466602c716555333a98f7aee4bc8b0f376d29daefc. resultSHA62b76e2db37a28835c688fa7973afae8b4e051aa087bd217bd0fe0a2afe1a2d5.
- Root가240raw의byte/EOS/UTF8/exact/의미동치/headargmax/head정답/동시정답시생성오류를CPU재구성했다. 0/128/256/384/512의5fullstate전체SHA·safeload·재귀digest·동결VP불변/cursor·populatedAdamW를검증했다. 원래부모와cache/core불변. Root추가모델호출0.
- V3CPU7검사PASS,별도원문runtime API의CPU3검사PASS(재현성/상한전실패/가중치load). Runtime파일은research/text_generation_runtime_v3.py이며실제새입력CUDA사용은별도ownedreadonly계약으로한다. 이것을추가GPU평가완료라고하지않는다.

다음연구는질문대상읽기와semanticfactor→문장충실도를분리해정렬하는것이다. 기존V3목적미달때문옛512회나900초를더쓰지않는다. 새CPU자료/계약에서대상span의근거와질문역전짝을보완하고,reader오답factor와gold문장사이의학습불일치를줄이는phase분리를검토한다. 모델이없는것처럼다시구현하거나동일baseline/광범위probe를무한선행조건으로반복하지않는다. 이후자유어휘·자연문장·긴문맥/생성corpus로확대해야하며현재범위는A–H짧은한·영비교/부정에한정된다.
