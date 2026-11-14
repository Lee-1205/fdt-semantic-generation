# V4 결과 — 기존 질문 형식의 결합 개선, 새 표현 일반화 퇴행

Root가에이전트없이새두phase학습과독립OS감독·원시검증을완료했다. Reader512/512와realizer512/512,global1024attempt/1024update/AMPskip0다. 원래V1/V2/V3·QueryCount9/128은재개하지않았다. 이번새900초/2800call예산도닫혔고미사용분은재사용하지않는다.

## 관측 개발 결과

| 패널 | V3→V4 문장 exact | V4 질문주어/목적어 ID | V4 질문 span | 긍정문장 exact | 정상UTF8/EOS |
|---|---:|---:|---:|---:|---:|
| 기존train probe16 | 6/16→9/16 | 100/100% | 100/100% | 4/11 | 16/16 |
| 미관측pair+language32 | 12/32→22/32(68.75%) | 100/100% | 100/100% | 12/22(54.55%) | 32/32 |
| 미학습wording32 | 13/32→11/32(34.375%) | 43.75/40.625% | 15.625/28.125% | 1/22 | 32/32 |

같은질문형식의미관측개체쌍에서주어/목적어읽기를개선했고,‘불명’상수기준선10/32를넘어긍정문장12개도맞았다. 그러나새wording은**40.625→34.375%로6.25pp하락**했다. 한국어·영어형식/EOS는유지됐지만일반화성공이나모든조건개선으로보고하지않는다. 두dev각75%전체exact/60%긍정exact목표는여전히FAIL다.

미관측pair의관계head는71.875%,새wording은46.875%이며언어head는모두100%였다. 질문대상을맞춰도관계판단과문장실현문제가남는다. 미관측pair에서세semantichead가모두맞은23행중생성의미오류3개,전체생성32개중예측head의미와일치27개다. 일부는head가틀려도VP조건때문정답문장을내거나그반대가나타난다. ‘맞는factor이면항상충실한문장’이라는계약을완전히입증한것은아니다.

새wording의`Compare G and E`에서두pointer가premise의같은개체위치를고르거나,`Compare`의대문자C를고르는사례를보존했다. 학습한질문표현/byte위치에대한과의존과표현coverage공백의증거이며,absoluteposition하나만의단독원인증명은아니다. 이패널은이미여러모델에서관측한개발자료다.

Zero-VPlatent통제exact는5/16·10/32·7/32,normal은9/16·22/32·11/32다. Head예측은동일하다. Latent에대한의존은관측되지만단일학습·작은상관표본이고,다른입력에독립적인diffusion추론기여를증명하지않는다. V4는frozenVP+learnedspanreader+causaldecoder의hybrid110,376,455parameters이며pure diffusion모델이나원래typed102M전체와혼동하지않는다.

## 실제 실행과 복구 보존

- ONEnativechain30008→23080→27408→43232. Owner생성08:27:55.304/.329KST,worker08:27:59.591/.606KST의parent/command/생존을root가직접관측했다. Worker/ownerexit0/COMPLETE,종료후모든owned/proxy부재·freshPython0확인했다.
- Worker52.192초,reader808+decoder1212=2020실제호출이며VPencode/denoise0이다. Root가저장생성의각batch최종EOS위치로AR호출수까지독립재구성했다. 모델메서드내부레이어수를top-levelcall로중복세지않았다.
- Reader4096cachedrows와realizer4096선택행/target80632UTF8bytes,transitionreader2048cachedrows다. Teacher2486행=초기teacher2048+fallback438,matchedpredicted1610행이었다. Gold조건은훈련전용이며평가는80행전체predictedfactor만썼다. 새언어encoder학습/새LMtokencorpus노출로세지않는다.
- Reader0/256/512와realizerglobal512/768/1024의6fullstate를보존했다. Reader512와realizer512사이의model/ReaderAdam/scaler/모든저장RNG/orders/history/cursors/exposure가정확히같고새DecoderAdam만비어있음을확인했다. Scaler는1024를유지해tracker0→1024로이어졌다. Readerphase의decoder,realizerphase의reader,전체VP/factorvalues가불변임을실제checkpointtensor로검증했다. 실제CUDA복구재실행을한것은아니다.
- Root가240raw생성+80headcheck행의byte/EOS/UTF8/exact/의미동치/headargmax/골드spanoffset/오류를CPU검증하고6fullpayload의전체SHA·safeload·재귀digest·두optimizer/cursor/노출을검증했다. Root추가모델호출0이다.
- 한계:rawpointerlogits는저장되지않았다. Span정확도는저장predictedindex와독립재구성한goldbyteoffset으로계산했으며pointerargmax자체를rawlogits에서재계산했다고하지않는다. 다음새worker에는fullpointerlogits/후보mask를보존한다. 이이유로같은평가를재실행하지않는다.

PlanSHAc93c8e6d4d7da89f176af56a2229726d2d9dbe43605d66df1afe44aa5621494a. 최종592,230,431B SHA5abf9351ddd12509e3622783dc95373f0ac1aeb1fef1a1feb5bb241c44a608df, algorithmSHA3a0017713bfa57a062a4f245d2f25786fbfefdd080186e6dc65045a55bd685d4. resultSHA2566da19ba2c14399533bc097eef63c69cff621f83e430ae1fb810459d32c2e8. 증거는run/root_result.json과scripts/review_text_span_v4_20261028_cpu.py다. V3부모/기존cache/core/E2final/모든과거실패와fullAdamRNG는보존했다.

다음은현재학습형식의위치결합을모든언어능력으로확대하지않고,질문표현·어순·위치변화와문장실현의일관성을검증하는새자료계약이다. 기존devwording의정답행을훈련에편입하거나같은512회를자동연장하지않는다. 동시에A–H짧은합성자료만반복하는한계를끝내기위해,별도자연언어생성corpus의출처/권리/문서단위분할/실제byte·token노출을CPU준비하는것을다음범위로삼는다. 새로운GPU계획/실행은아직없다.
