# V112 조건 구별 개선 계약 — 실행 전 판단

## VERIFIED

V111 코드와 저장 이력을 먼저 확인했다. 주16-step rollout에 gradient는 연결되고 target은 E(Y)였다. 출력 rank512 및 E/D 고정도 이미 확인됐다. 전체TRAIN 긍정0/80, DEV0/48, 같은 잡음의32긍정 역전쌍에서 출력 잠재 차이가 표적 차이보다 작았고9쌍 출력 동일이었다. 평균 TRAIN 의미보다 body MSE도 컸다. 기본 적합과 조건 구별을 함께 조사해야 하며 완전 문자열 불일치만의 문제는 아니다.

현재 오류를 target mapping 오류나 디퓨전 전체의 불가능으로 단정하지 않는다. 기존 목적은 행별 endpoint MSE+.05tail+.1VP로서, 조건에 반응하지 않고 평균 근처에 머무는 상태를 별도로 강조하지 않았다. 이는 버그 확정이 아니라 최적화상 약점 가설이다.

## 수식과 수정

실제 역전짝에 대해 오차를 e_a=Zhat_a−Zy_a, e_b=Zhat_b−Zy_b라 한다. 같은 body mask에서 기존 두 행 평균 body MSE는

`MSE((e_a+e_b)/2) + .25*MSE(e_a−e_b)`

이다. 새 항 `.25*MSE((Zhat_a−Zhat_b)−(Zy_a−Zy_b))`를 더하면 평균 성분 계수는 그대로이고 차이 성분 계수만 .25→.5다. tail 및 VP 보조 항은 그대로다. 이 계수는 수식상 구별 오차의 비중을2배로 정한 첫 후보이며 최적값이라는 증거는 없다.

실제58짝에만 같은 독립 Gaussian ε를 준다. 각 행의 주변 잡음 분포는 유지하지만 두 행의 결합 분포는 바뀐다. 공통 잡음과 차이 손실을 함께 적용하는 복합 변경이므로 각각의 기여를 분리했다고 주장하지 않는다. 서로 UNKNOWN인26짝의 표적 차이는0이다. 임의로 함께 스케줄된18singleton에는 차이 손실을 주지 않고 서로 독립 잡음을 준다. 정답의미는 loss에만 있고 추론 API에는 여전히 source+메모리+ε만 있다.

역전 답문은 같은 canonical 사실을 다른 주어·방향으로 표현할 수 있다. 이것은 반대 사실 생성 목표가 아니라 해당 질문에 맞는 답변 코드와 표현 구성의 목표다. 잠재거리 개선만으로 의미 변환 성공을 선언하지 않는다.

## 유한 실행

- V111 최종 θ만 warm, fresh AdamW/scaler, hidden520/9576426θ. E/D/기존 K_P·K_C 형성 및 모든 buffer 고정, K_E NULL. 구조 변경0, 훈련 E0/D0/언어loss0.
- 기존134TRAIN/31facts/27answers, 새자료0/DEV편입0. 동일16finite orders/2144행/37088UTF8bytes, batch8/원자단위2행,268attempt. LR1e-4/.01decay/clip1/동일cosine.
- 독립 잡음 draw1216회. 두 행의 graph를 모두 backward한 뒤에만 cursor2행을 소비한다. 단위 중 예외면 이전 gradient 복구/시도한2행 소비/ABORTED/자동재실행0. 살아 있는 graph를 checkpoint했다고 주장하지 않는다.
- 실제 populated Adam/scaler/ALL RNG/중간 gradient가 있는 원자단위 후 MID 및 마지막6행 READY에서 다음 update를 전체크기 CPU로 검증했다.388calls/fixture와재생4update는 본 연구268과 별개다. 실제 기존 TRAIN/DEV 각1개 출력을 재생해 저장 baseline과 bitwise 동일 확인,27oracle baseline은 재호출하지 않는다.
- 원래134TRAIN 공통ε 평가와96DEV 최종 출력230개를 baseline으로 재사용한다. baseline 새 모델호출0. 새 최종230개는 같은 row/ε/decoding으로 생성한다.36448학습thought+3680평가thought+230D=40358calls, cap41000. wall1200/cut1080/reserve120, C 전체1GiB. 이전 남은 예산 재사용0.
- 저장C만, 실행50GiB 하한/owner54GiB 조기정리/시작101GiB+64MiB 여유로 종료100GiB 보호. 원본 삭제·이동·압축0. state0/134/268과 STOP현재 fullstate, 삭제·회전0. 독립 OSowner의 postPopen cleanup은 V111 검증 코드와 동일.

## 평가·후속 중단 기준

전체134TRAIN(긍정80/unknown54),96DEV, 실제32긍정/26UNKNOWN짝을 baseline과 비교한다. 의미 관계·개체·부정/모순, NLL/body/tail, byte 정확도, prefix/whole/EOS/UTF8를 구분한다. UNKNOWN substring을 확정 의미 정확도로 세지 않는다. ROOT가 실제 긍정 출력을 읽는다. 손실·차이만 줄고 관계가 계속 틀리면 평균 적합/조건 attention/표적 구조를 해당 실패에 한정해 조사한다. 결과 없이 무한 SGD로 연장하지 않는다.

## 관련 원문 검색의 범위

[Conditioning Matters, NeurIPS2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/ad6363efb7af02f7db13d087c7e649bd-Abstract-Conference.html)는 조건 구별이 어려울 때 주변 분포 모델링으로 퇴화하는 문제를 로봇 diffusion policy에서 연구했다. FDT의 원인이 같다는 증거가 아니며 해당 source-distribution 변경을 그대로 도입하지 않았다.

[Contrastive diffusion training, NeurIPS2024](https://papers.nips.cc/paper_files/paper/2024/hash/38d67d1df644cf2efe9ebd5521741dc5-Abstract-Conference.html)는 noise-level 구별과 OOD denoising을 다뤘다. 이번 endpoint pair-difference 식과 같은 방법이 아니다. 이번 수식은 자체 전개·gradient 검사로 검증하며 선행연구 성적을 승계하지 않는다.

아직 이 문서는 실제268학습 또는 성능개선을 의미하지 않는다. 실행 증거와 결과는 별도 디렉터리에 기록한다.
