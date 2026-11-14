# 검증된 의미만 사용한 디노이저 분리 학습 결과

2026-11-14. ROOT 직접 수행, 에이전트0. 타이머는 삭제 상태를 유지했다.

**현재 자료 전체480개의 언어→의미→언어 검증을 실제 수행했고, 엄격히 일치한 TRAIN134개만 분리해 디노이저를 학습했다. 기존 입력·세 메모리·결합·출력 가중치는 바꾸지 않았다.** 실험 과정에서 기존 디노이저의 잡음 제거를 제한하는 구체적인 출력 rank 문제를 발견해 수정했고, 이어 언어 출력기 없이 의미 손실만 쓰는 학습을 완료했다. 의미 복원은 아직 성공하지 못했으며, 이 결과는 기존 의미 공간 전체가 실패했다는 판정과 다르다.

## 1. 실제 데이터 선별

대상은 현재 사고 데이터의 TRAIN384+DEV96이다. OpenWebText 전체 corpus를 새로 전수 검사했다는 뜻은 아니다. 이전 V95의 실제 E(X)480개와 K_P/K_C/NULL K_E source-only snapshot을 재사용했다. 고정 V88 출력기로 D(E(X))480회를 새로 계산했으며, source encoder/optimizer/디노이저 호출은0이다.

선별 조건은 **원문 UTF8 byte 전체 동일 + 정상 UTF8 + 신경 출력의 EOS**다. 뒤에 잡음이 붙는 prefix 일치나 whitespace 정규화 일치를 통과로 올리지 않았다.

| 자료 | 전체 | 엄격 왕복 일치 | 학습 편입 |
|---|---:|---:|---:|
|TRAIN|384|134|134|
|dev_pair|32|28|0|
|dev_surface|32|29|0|
|dev_position|32|0|0|

TRAIN의 정규화 일치는148개, 원문 prefix 일치는160개였지만 선별에는 쓰지 않았다. 선택134개는56–77byte 문장이다. 기존 관계 label 분포는 unknown54/less42/greater38, 표현 family는0/1/2/3 각각23/31/50/30이다. 이번 identity 학습에서 이 relation label은 NN feature나 target이 아니다. target은 원문의 E(X) 자체다.

선별은 codec이 잘 처리하는 표면·길이로 편향된다. 선택134개를 전체 언어 분포로 일반화하지 않는다. dev_position의0/32에도 `xxxx…` 패딩 손상뿐 아니라 `A→1`, `H→9`, 질문의 개체 글자 손상이 있었다. 반대로 끝에 있는 사실 문장이 그대로인 사례도 있었다. 따라서 이0은 **전체 문자열 일치0**이며 모든 문장의 의미 손실100%가 아니다.

V108 actual480D/10.172worker초/CPU4threads/CUDAfalse, 새75초 작업+30초 reserve/8MiB 범위에서 완료했다. ROOT가 owner22168→17536,worker32936→8688의 생성·parent·command·생존을 확인했고,08:56:51 KST allowned/proxy gone+Python0을 먼저 저장한 후 분석했다. 저장480개 trace의 SHA/독립 UTF8 decode/엄격 gate와 사전에 고른8개의 전체258-class logit을 검증했다. 나머지472개는9개 UTF8 결정·argmax·gold log-probability를 저장했으며, 전체258 점수를 모두 저장한 것은 아니다.

## 2. 무엇을 학습했는가

이번 통제는 **원문의 의미→원문의 의미 복원(identity)**이다. 입력 의미로 조건화한 독립 잡음에서 원래 의미를 복원한다. 적은 noise가 섞인 clean state만 청소하는 시험보다 어렵고, 관계를 계산해 다른 답 의미를 만드는 시험보다 단순하다.

```text
이미 검증된 원문 X → 고정 E/메모리 → Zx
                                      │ 조건만 제공
                                      ▼
                             독립 잡음 → Rθ×16
                                      │
                                 예측 원래 의미
                                      │ 평가 때만
                                  고정 D → 복원문
```

source 의미와 clean target이 같은 것은 이 identity 통제의 의도다. 다른 정답의 의미·역할·길이를 inference에 주는 답변 과제가 아니다. source hash/ID는 검증 metadata이며 NN이 읽는 id embedding을 만들지 않았다. K_P/K_C는 readonly이고 K_E는 invalid NULL/content0/copy0이다. 원문 문자열·token H·template·Exact 복사로 출력을 조립하지 않았다.

## 3. V109 첫 분리 통제: 사고 경로 병목을 더 좁혔다

V109는 기존3,224,322파라미터 사고 구조를 fresh initialization/fresh Adam으로 시작했다.134개 TRAIN을8회씩,1,072행/67,832target UTF8byte로 학습했다.134attempt134update/AMPskip0,21,808thought+1,296D=23,104call,241.398worker초로 완료했다.

입력·메모리·D의 가중치는 고정했으나, 이 첫 통제에는 **D를 통과하는 언어 보조 loss**가 θ로 전달됐다. 따라서 ‘의미만’을 더 엄격히 분리한 후속 V110에서는 D를 학습 graph에서도 제외했다.

같은16개 TRAIN probe는 D(E(X))에서 모두 정확히 복원된 문장이다. 그러나 Rθ를 끼운 뒤에는 NLL12.961→5.385로 개선됐어도 정상 복원은 확인하지 못했다. DEV pair/surface/position NLL도11.417→6.208,12.326→6.020,16.210→7.704로 줄었지만 실제 문장은 깨졌다. ROOT가112개 final 원문과 출력을 모두 읽었다. 전체 일치0만으로 의미0이라고 판정한 것이 아니다.

직접 입력 의미를 D에 줄 때는 원문을 복원할 정보가 있었으므로, 이 선택 문장들의 실패를 E가 정보를 이미 잃었다는 설명만으로 돌릴 수 없다. **새 Rθ의 상태 변환이 병목이라는 근거가 강화됐다.**

## 4. 저장 가중치에서 확인한 출력 rank 제약

기존 디노이저의 마지막 출력은 다음과 같았다.

`vθ = W h + b`, `W:512×256`

즉 어떤 input/time/memory를 주어도 velocity는 고정 affine 부분공간 안에 있다. W의 column space에 직교하는 방향이 최소256개 존재한다. self/source attention이 아무리 달라져도 마지막 W 뒤에서 이 방향으로 새 velocity를 만들 수 없다.

현재 VP/DDIM sampler를 centre/scale 기준 residual r로 쓰면:

`r_next = cosδ r − sinδ vθ`, `δ = π/(2N)`

`QᵀW=0`인 직교 방향에 투영하면:

`Qᵀr_next = cosδ Qᵀr − sinδ Qᵀb`

따라서 N회 뒤의 초기 독립 Gaussian noise 계수는 `cos(π/(2N))^N`으로 남는다.

| step 수 | 해당 방향의 초기 noise 표준편차 잔존 계수 |
|---:|---:|
|1|0|
|4|0.728553|
|16|0.925676|

16step이면 그 방향의 분산은 약85.7% 남는다. 물리적 scale에 최소0.5가 적용되므로 이를 무조건0으로 줄이는 우회도 막혀 있다. 고정된 입력에 대한 평균 squared error에는 이로 인한 비영(非零) noise 하한이 존재한다. 이 하한만으로 모든 관측 오류나 의미 부재가 설명된다는 주장은 아니다. D가 그 방향에 불변이라면 출력에는 영향이 작을 수도 있지만, 현재 그런 불변성이 입증되지 않았다.

이것은 단순 추정에 그치지 않았다. 저장 V109의 W SVD rank256을 확인하고,16개 실제 final sample에서 source-dependent scale과 noise seed를 복원해 **slot 간 차이의 Q 투영**을 계산했다. centre와 공통 bias를 차분으로 없애면 위 식과 최대 relative L2 **3.2777×10⁻⁷** 안에서 맞았다. 모델 E/R/D 전체를 재실행하지 않았고, 저장 parameter의 정규화 affine 계산16회와 SVD만 수행했다.

이는 ‘더 오래 학습하면 저절로 모든 noise를 지울 것’이라는 가정에 실제 구조적 장애가 있었음을 뜻한다. 낮은 noise의 정답 조건 한 번 복원과 실제16step의 간격,1step unknown 성공이 후속 step에서 깨진 현상과도 일관된다. 유일한 원인이라고 단정하지는 않는다.

## 5. V110 수정: full-span velocity와 순수 의미 학습

입력·D·메모리는 그대로 두고 **새 thought 내부 hidden을256→520**으로 변경했다. 왜 단순히512가 아닌520인가 하면 마지막 LayerNorm이 평균0 조건을 주므로, 그 뒤에서도512차원 affine span을 확보할 여유를 두기 위해서다. heads8 조건에도 맞는다.

- 새 thought9,576,426파라미터. 기존 codec123,497,080파라미터와 buffer는 전부 고정.
- source/state projection rank512, LayerNorm 이후 유효 velocity map rank512를 실제 전체 크기에서 확인.
- 훈련 loss는 **최종 의미 body MSE + .05 tail MSE + .1 VP 보조 loss**만 사용.
- 훈련 중 E0/D0, 언어 loss 없음. log의 byte_CE/EOS_CE=0은 공통 logger 자리표시자이며 언어 성능이0loss가 됐다는 측정이 아니다.
- 고정 D는 baseline/final의 평가에만 호출.
- 동일134 TRAIN을4회씩536행,67attempt67update/skip0,33,916target UTF8byte.
- actual12,696thought+224평가D=12,920calls/204.478worker초. 새13,200call/600wall480cut120reserve/D640MiB 범위로 완료.

새 폭과 loss,epoch수,seed를 함께 바꾼 통제이므로 V109와의 성능 차이를 rank 수정 하나의 효과로 분리해 주장하지 않는다. rank 제약 자체의 제거는 가중치 행렬 계산으로 독립 확인했다.

CPU tiny4계약 검사와 실제 전체 크기 CPU99thought+2D/E0 preflight를 구분해 보존했다. 실제 preflight는 학습 D0,16step train/eval 최대차3.34×10⁻⁶,READY의 populated Adam/scaler/모든 RNG/gradient 정확 복구,선택 원문의 padded target 복원,기존 전체 codec 불변을 확인했다. 실제 CUDA restart 검사가 아니다.

## 6. V110 결과: 의미 오차는 줄지만 아직 기본 복원도 덜 학습됐다

| 동일 패널 | 초기 NLL | 최종 NLL | 초기 valid 의미 RMSE | 최종 valid 의미 RMSE |
|---|---:|---:|---:|---:|
|검증 TRAIN probe16|18.7246|12.6142|1.7457|1.4098|
|DEV pair32|18.0421|13.0073|1.7733|1.4445|
|DEV surface32|19.8748|13.4910|1.7513|1.4467|
|DEV position32|27.7015|14.8254|2.0526|1.8191|

훈련 처음8group→마지막8group의 실제16step body MSE는2.58693→2.01410,tail MSE는1.35851→0.91680이었다. 개선은 있었지만 깨끗한 상태에 도달한 수치는 아니다. 마지막112개 출력은 대부분 빈문자·공백·몇 글자이고,긴 경우도 깨진 문자열이다. ROOT가 모두 읽었으며 원래 명제를 명확히 복원한 결과는 확인하지 못했다. EOS112/112 중111개는 기대 원문보다 일찍 끝났다.

중요한 추가 sanity calculation도 했다. 선택 TRAIN134개의 target만으로 per-slot constant mean을 맞추고, **똑같은 TRAIN probe16**에서 MSE를 계산했다. constant mean은1.06781인데 새 디노이저는1.98898이었다. 이 평균 벡터는 개별 문장을 재현하는 해결책도 언어 생성 성공도 아니다. 다만 모델이 아직 **입력별 상세 복원 이전의 낮은 수준 오차 감소조차 충분히 학습하지 못했다**는 근거다.

tail target은 모든 자료에서 동일한 frozen terminal 벡터다. constant baseline의 tail MSE는 수치상 약10⁻¹⁵인데, 디노이저의 마지막 훈련 tail MSE는0.9168이었다. 이 부분을 맞추는 데 추상적 관계 이해나 의미 정답의 철학적 정의가 필요한 것은 아니다. 남은 noise와 denoising fit 문제를 먼저 다뤄야 할 이유다.

따라서 현재 판단은 다음과 같다.

1. **선택134개의 정보가 E(X)에 없다는 주장은 맞지 않는다.** 고정 D에서 실제 정확히 복원했다.
2. **기존 디노이저에는 학습량과 별개의 특정 rank 제약이 있었다.** 이를 찾아 제거했다.
3. **수정된 디노이저는 학습 중이지만 아직 미숙하다.** 현재67update 결과만으로 새 구조의 충분한 수렴이나 실패를 선언할 수 없다.
4. **codec의 추상 의미성은 별도 미인증이다.** 이번 identity 통제가 이를 증명하지도 부정하지도 않는다. 지금은 이것을 모든 사고 실패의 원인으로 먼저 가져오지 않는다.

후속은 검증된134개 의미와 고정 codec 경계를 유지하면서 순수 의미 denoising을 충분히 fit시키는 쪽이 우선이다. ‘epoch를 무조건 늘린다’ 대신 실제 trajectory의 body/tail 오차가 같은 TRAIN mean sanity보다 낮아지는지,noise/t 조건을 바꾸어도 원래 상태를 읽을 수 있는지를 중심으로 판단할 수 있다. 이 report를 근거로 자동 추가 SGD를 시작하지 않았다. 모든 기존/이번 trial 예산은 닫혔고, 미사용 call/time을 이어 쓰지 않는다.

## 7. 보존·위치·종료 확인

- 선별: `fdt_local_102m_20261026/research/artifacts/roundtrip_identity_screen_20261114_v108_cpu_v2/`.
- 첫 identity 통제: `D:/Codex/FDT/runs/identity_thought_20261114_v109/`.
- 순수 의미 통제: `D:/Codex/FDT/runs/latent_identity_20261114_v110/`.
- C: 프로젝트의 `runs/*_LINK.json`에 D: 복구 경로를 명시했다. D: checkpoint를 지워도 된다는 뜻이 아니다.
- V110최종 state_0067.pt115,192,893byte SHA `b5e07126088d1864c9c2e787dca2b3a719b89ae889ad6001e6b7370b3668b6f9`.
- V109최종 state_0134.pt39,017,917byte SHA `bdd994257c90d237fca709f849b4f2bcbd6c04bb00e11ff149319cf9b45d80ee`.

V109의 owner10444→21932/worker10128→30600, V110의 owner21096→37364/worker7004→12776은 ROOT가 실제 생존·생성·parent·명령을 관측했다. V110은09:38:37 KST에 all4owned/proxy gone+freshPython0을 먼저 확인한 뒤 분석했다. CPU preflight 중 빠르게 끝나 ROOT의 동시4PID 관측을 못 한 경우는 owner CIM의 검증 기록과 종료 확인을 구분한다.

세 checkpoint 각각 SHA/weights_only safeload/recursive digest,전체 고정 codec digest,Adam step/scaler tracker,선택 TRAIN의 유한4회/8회 노출,cursor/history,모든 실제 train noise RNG draw를 저장 파일만으로 독립 재구성했다.112전후=224개 raw의 선택·정답 점수·저장 sample258도 각각 검증했다. 전체 위치의258개 점수를 모두 저장했다는 주장은 하지 않는다. 상세 근거는 각 run의 root_saved_review.json/root_result.json 및 V109의 root_velocity_rank_algebra.json, V110의 root_matched_mean_sanity.json이다.

작업 중 C:100GiB 보호와 시작8GiB headroom을 지켰다. 새 큰 파일은 D:에 저장했다. 종료된 feature/checkpoint/JSON의 NTFS 투명 압축으로 이번 합계1,751,150,330byte의 물리 공간을 확보했으며,가중치·Adam·RNG·원문·실패의 **논리 내용 SHA는 그대로**다. 이미 압축돼 있던 CUDA DLL에서는 추가 회수0이다. 삭제·게임 변경·외부 게시·에이전트·E2최종holdout/registry 접근은0이다. 타이머를 다시 만들지 않았다.
