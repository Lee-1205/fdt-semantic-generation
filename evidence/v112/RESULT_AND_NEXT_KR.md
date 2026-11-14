# V112 조건 구별 개선 — 실제 결과와 다음 수정

**VERIFIED** ROOT 직접7focusedCPU검사와 실제388call 사전검사 후 ONE CPU 학습268attempt/268update/skip0을 완료했다.134TRAIN×16=2144행/37088target UTF8byte, thought40128+readout230=40358calls,642.193초다. 전체E/D/기억형성123497080parameter와 모든buffer는 고정, K_E NULL이었다. 원래V111θ만warm/freshAdam으로 이전optimizertrajectory를 재개하지 않았다. 실제4native의생성·parent·command·생존과worker-ownerexit0/allgonePython0를 먼저 확인했다. C종료106.211GiB, 새D출력0/삭제압축이동0. 타이머삭제 유지.

같은seed/조건의 기존230출력을 기준으로 재사용했고 실제2출력bitwise검증했다. 고정D의 oracle27종은 반복호출하지 않았다. 새40358/41000call·1200초는 CLOSED, 남은642call·시간은 재사용하지 않는다.

**OBSERVED** 전체 TRAIN bodyMSE1.093885→.722810, NLL2.707188→1.571820, byte정확도59.889→76.842%, 전체문자열1→9/134, 정답prefix4→22/134. 전부UNKNOWN성공으로 일반관계 이해라고 하지 않는다. 긍정80행 subject정확5→9/object6→29/관계word정확0→3이다. 자동 완전관계문법 점수는0/80이나 이것을 의미0으로 바꾸지 않는다.

실제 읽은 `H is largeer hhn B.r`는 H>B라는 core를 높은 확신으로 복원해 읽을 수 있었다. 최소1TRAIN 사례의 의미 core 회복으로 기록하되 정상언어출력·전체문자열 성공이 아니고 전체80행의인증 의미정확도도 아니다. `C is l aler hil F...` 등의관계철자가모호한출력은 따로보존했다. ROOT가전체128긍정출력의앞110자를읽었고 긴noise tail전부를수동판독했다고주장하지않는다.

32긍정역전쌍 동일출력9→2, first4slot 차이RMS/표적차이 median .028507→.476279(정확도 아님), target방향cosine평균.146994→.668333, target방향투영비율.004608→.310958, body차이MSE2.481724→1.479526이었다. 공통잡음+차이손실과 추가업데이트를 함께 적용했으므로 각각의인과기여를분리한성적은 아니다. 기존보고의lower-middle median과여기의interpolated median값을혼동하지않는다.

DEVpair NLL2.985546→2.487229개선/표현4.439446→4.695938악화/위치6.119875→6.957152악화다. 관계word정확0/48, 적어도확인한실제core에서명확한긍정정답은없었다. 훈련자료선택·관측DEV다중사용·짧은A-H/고정답27개제약이있고표현일반화완료가아니다.

**VERIFIED 저장 검증** state0/134/268의SHA/크기/weights-onlysafeload/recursivehash, 전부V111θwarm와freshAdam,64θtensor변경, codec전체digest불변, Adam268/scaler1024tracker268, 모든134행16회와1216독립noise draw의정확한RNG재생을 확인했다. 실제loss-onlypacket/460raw의UTF8/EOS/NLL/MSE/모든정답byte위치full258logit을독립재계산했다. 긴생성tail전체258는미저장이다. 이사후검토model/optimizer호출0이다. 최종checkpoint는학습완료후/최종평가전state이며평가참조는별도assessment에있다. CUDA복구실행증거는아니다.

**INFERRED** 조건에따른표적방향변화는개선됐다. 아직정상적으로언어화되는관계정답에는도달하지못했고, 전부의미생성실패또는전부출력기문제중하나로단정하지않는다. MSE는512좌표의평균오차를동일하게보며,고정D가필요로하는개체·관계결정경계를직접최적화하지않는다. 혼합된 `lmaleer`/깨진개체가남는것은이불일치가설의조사대상이다. 출력방향반응이증가했다고추상의미추론을입증한것도아니다.

**PROPOSED 다음 분기** E/D와모든기억추출경로를그대로두고, 실제target-free생성Z를고정D에넣어정답content-byte CE를θ에만역전파한다. body/tail/VP/실제짝대비는보존한다. D자체학습·AR교체·새Exactcopy·target주입·길이crop은하지않는다. EOSloss등다른여러보조는추가하지않는다. 실제frozenDgradient→θ가연결되고E/Dparameter·buffer불변인지전체크기preflight로확인한뒤별도새유한계획만실행한다. 이문서시점에는그새학습은미실행이다.

증거: root_saved_review.json / root_rows460.json / root_pair_response.json / root_manual_semantic_review.json / ROOT_terminal_native.json. 구현·수학·문헌범위는 research/artifacts/answer_pair_contrast_20261114_v112_C_cpu_v1/REPAIR_CONTRACT_KR.md. 실제안전한조회: `Get-Content -LiteralPath 'runs/answer_pair_contrast_20261114_v112/root_result.json' -Raw` (프로젝트루트에서). 기존봉인run 재실행금지.
