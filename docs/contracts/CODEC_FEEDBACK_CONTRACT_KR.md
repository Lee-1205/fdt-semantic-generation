# V113 고정 D content 피드백 계약

VERIFIED: V112는 실제268학습 후 TRAIN bodyMSE1.0939→.7228/byte59.89→76.84%, 역전짝 target방향cosine.1470→.6683으로 구별 반응을 개선했다. 자동관계문법0은 의미0이 아니며 ROOT가 H>B를 읽을 수 있는 깨진 문장1개 이상을 확인했다. 그러나 대다수 개체·관계 손상과 표현/위치 DEV NLL 악화가 남았다. loss·잠재거리 개선만으로 의미성공을 선언하지 않는다.

INFERRED: 좌표 MSE는 고정D가 필요한 content 결정경계를 직접 최적화하지 않는다. 따라서 같은 의미표적/역전짝/모든입출력메모리/16-step 궤적을 유지하며 고정D가 읽는 content 확률로 θ만 추가 감독하는 한 가지 변경을 검증한다. D 자체가 나쁘다는 단독 원인 증명이 아니다.

새 L = 기존 V112 endpoint body + .05tail + .1VP + .25실제짝차이 + .5 content-byte CE. 실제 target-free Z를 먼저 계산하고 고정D(Z) logits에만 정답byte label로 CE를 계산한다. 정답길이는 훈련 loss의 byte/EOS/PAD 구분에만 쓰고 추론 종료·crop에 주지 않는다. EOSaux0/새AR0/Exactcopy0/teacherlatent주입0. 고정D에서 θ로 gradient는 지나가지만 E/D 가중치·buffer·기억추출기는 전부 동결이다. 추론은 여전히 source+메모리+독립epsilon→R16→확정의미→D다.

VERIFIED 사전검사: 실제fullsize424calls=thought403+D21, fixture+재생4update(본학습과별도), 학습fixture D18/gradient-only D1/기존TRAIN-DEV parity D2. 실제 CE→θ gradient norm6.230322(한행 진단; 전체최적계수근거아님), codec불변/정확MID·READY6row 복구/기존2출력bitwise/CPU PASS. tinyCPU3PASS는 실제fullsize와 구분한다.

계획: V112 finalθ ONLYwarm/freshAdam-scaler, 동일134×16/2144행/37088bytes/268attempt, batch8·원자단위2행/1216noise draws. LR1e-4/decay.01/clip1/동일cosine. 기존230 final출력을baseline재사용 forward0, 새final134TRAIN+96DEV230개를동일ε·decoder조건으로평가한다. 학습thought36448+D2144, 평가thought3680+D230, 총42502/cap43000. 독립OSowner1200/cut1080/reserve120, C전체1GiB/실행하한50+owner54/시작101GiB+64MiB로종료100GiB보호. D신규출력0. 이전run예산재사용0/자동재시도0/원본수정삭제압축이동0. fullstate0/134/268+STOP에전체Adam/scaler/RNG/order/cursor/gradient/정확목적schema 저장.

PROPOSED 판단: 실제관계·개체·모순, 사람이읽은의미, UNKNOWN기준, 역전쌍과NLL/byte/EOS를구분한다. 성적이좋아져도 CE만의인과성은 추가학습량·freshAdam 영향과분리되지않았음을명시한다. 이번oneprofile을임의연장하지않고결과에따라다음최소문제를선택한다. 현재이문서는봉인전계약이며실제학습성공이아니다.
