# FDT 현 시점 종합 분석 및 작업 종료 기록

2026-11-14, Asia/Seoul. 최신 사람 지시: **“정리하고 타이머 삭제, 현 시점 완전분석.”**

추가 학습·추론을 시작하지 않았다. `fdt` 30분 재개 타이머는 앱 도구에서 `deleteStatus: deleted`를 확인했다. 중단 지시를 받은 시점의 Python/학습·추론 프로세스는 0개였다. 이미 종료한 run의 상태·예산·가중치는 그대로 보존하며, 새 명령 전까지 자동 재개하지 않는다. 이번 작업은 저장된 결과의 계산·코드 검토·보고서 정리다. 새로운 NN forward, optimizer update, GPU 계산은 모두 0이다.

## 1. 현재 수준에 대한 판단

**언어→퍼지 메모리 표현→언어 복원 경로는 실제로 작동하며, 제한된 영어 자료와 작성된 장문에서 상당한 정보 보존을 보였다. 그러나 모든 문장의 의미를 무손실로 추출·복원하는 장치가 완성됐다는 근거는 없다. 별도로 붙인 새 사고 디퓨전은 실제 학습을 시작했으나, 정답 없이 생성한 최종 의미로 올바른 관계 답을 안정적으로 표현하는 수준에는 도달하지 못했다.**

현재의 나쁜 답변을 곧바로 원래 언어 codec 전체의 붕괴, 퍼지 메모리의 무효, 디퓨전 원리의 실패로 해석하는 것은 과장이다. 반대로 “의미는 이미 완벽하고 출력기만 잘못됐다”고 단정할 수도 없다. 두 경로의 증거가 다르다.

| 구분 | 확인된 성과 | 아직 확인되지 않은 것 |
|---|---|---|
| 언어→표현→언어 codec | 원문/Exact 복사 없이 Commit만 받아 영어를 복원; 작성 장문 약524KB에서 잔여49byte 오류 | 임의 영어·모든 값·관계·부정의 무손실 보존, 표현이 달라도 같은 추상 의미라는 보장 |
| 기본/입력별 퍼지 메모리 | 실제 읽기와 입력별 snapshot, 순서 결합 경로 존재 | 각 메모리가 독립적으로 올바른 사실 추론에 기여한다는 충분한 통제 결과 |
| Exact | 현재 요구대로 입력 내용 저장·복사0, NULL 정책 유지 | 미래의 명시적 정확 복사 요청 처리 |
| 새 사고 디퓨전 | 입력 의미로 잡음 분포를 조건화하고 매 단계 메모리를 읽는 경로, 사고 가중치만 실제960회 업데이트 | 새 입력에서 올바른 답변 의미를 만들고 언어화하는 안정적 능력 |
| V107 후속 | 실제 추론 경로를 역전파하는 CPU 후보와 tiny 계약 검사4개 통과 기록 | 전체 크기16단계 학습·실제 worker/owner/plan/seal/launch·성능 향상 |

## 2. 어떤 모델을 현재 모델이라고 부르는가

**고정 입출력 기준은 V88**, 최신 학습된 별도 사고 가중치는 **V105**다. V91 codec은 새 장문 오류 때문에 기준에서 제외했고, 모든 원본·복구본은 보존했다. V107은 학습된 모델이 아니다.

| 구성 | 실제 상태 |
|---|---|
| 전체 보존 codec | 123,497,080파라미터와 모든 buffer. 사고 학습 중 전부 동결 |
| 새 thought | 3,224,322파라미터, 2개 attention/memory block, hidden256, 의미 차원512 |
| 보존 파라미터 합계 | 126,721,402. 상속된 미호출 AR/typed branch까지 포함하므로 전부 활성 추론량이라고 부르지 않음 |
| 사고 학습량 | V102 192update + V105 768update = 누적960update. V105는 warm weights만 이어받고 fresh Adam/scaler를 썼으므로 하나의960step Adam trajectory는 아님 |
| 실제 사고 학습 시간 | V102 약68.605초 + V105 약158.500초 = 약227.106 CPU worker초. 준비·검증·감독 시간과 구분 |
| 실제 사고 자료 | TRAIN384상관 view, 192역전짝, 32fact key, 19pair group; 반복 총7,680행 선택/135,680 target UTF8byte |

20회 유한 epoch의 같은384행을7,680개 독립 문제나 새로운 자연문서 학습으로 세지 않는다. TRAIN 답변 문자열은27개, TRAIN+DEV 합쳐31개다. 따라서 현재 결과만으로 파라미터 수 부족이나 학습량 부족을 유일한 원인으로 결정할 수는 없지만, 사고 모델이 아직 매우 초기의 좁은 학습 단계라는 사용자의 지적은 맞다.

선택/복구 경로는 다음과 같다. 이번에 파일 SHA를 다시 확인했다.

- 고정 codec delta/fullstate: `fdt_local_102m_20261026/runs/critical_repair_20261113_v88/model/runtime.pt`, SHA `ebe939a6c14b205bd92a6cad772b33cf2ee462473bb77daadcc6ba48926c84d7`.
- 필수 standalone base: `runs/context_verbalizer_20261112_v74/model/checkpoints/state_0640.pt`, SHA `734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7`. V88 파일 하나만으로 이 부모 의존성이 사라지는 것은 아니다.
- 최신 사고 fullstate: `runs/grid_tail_thought_20261114_v105/checkpoints/state_0768.pt`, 39,500,029byte, SHA `13142e6e721743f381d69cd7dfcaedeef069e6ed4f82b3526f9e7bf0083b3a75`.
- 고정 codec model/buffer 내용 digest: `42a2032db3137ffb9f876585cf26979854d2555e19639fa5c2f1e8d61201b96a`. 파일 SHA와 재귀 model digest, 이름·dtype·shape를 포함하는 경계 digest는 서로 다른 정의다.

이전에 검증한 전체 Adam/scaler/RNG/history/cursor/order를 보존했다. 이번 정리를 이유로 최신 checkpoint만 남기거나 이전 실패를 삭제하지 않았다.

## 3. 현재 실제 연산 구조

```text
영어 X
  → 고정 9층 language encoder
  → token별 fuzzy membership/value
  → K_P 기본 퍼지 + K_C long/recent + K_E NULL
  → global2 + 16byte마다 ordered local state
  → 순서 fuzzy-role 결합을 끝낸 입력 Commit Zx
       │
       ├── 고정 codec 복원: D(Zx) → 복원문
       │
       └── 새 사고:
            Zx로 centre/scale 조건화 + 독립 잡음
              → Rθ:16단계, 매 단계 Zx와 K_P/K_C/K_E 읽기
              → 최종 답변 의미 Commit 확정
              → 고정 D → 답변문
```

현재 D는 **local512 + global2×512 → LayerNorm → MLP → 각16위치의258 Byte/EOS/PAD 점수**를 내는 신경 출력기다. 과거 DIANA 계열 AR decoder는 보존되지만 이 경로에서 호출되지 않는다. 자기 출력 byte를 다음 token 입력으로 재투입하는 AR feedback도 없다. 따라서 V7의 원문-context 우회와 AR 반복 prior 진단은 당시 분기의 근거이며, 지금 같은 문제가 그대로 남았다고 옮겨 말할 수 없다.

순서 결합은 이미 추출 Commit 안에서 끝난다. 출력기는 원문 문자열, token H, 정답 prefix, 정답 역할·span·길이를 받지 않는다. UTF8 FSM은 신경 점수에서 유효 UTF8 byte를 고르고 EOS를 만나면 멈춘다. 문장의 정답 의미를 해석하거나 정답 길이로 자르는 장치는 아니다. **UTF8가 정상이라는 점수와 의미가 맞다는 점수를 분리해야 한다.**

새 사고의 학습에서도 전체 codec 입력·메모리·binding·출력·미사용 상속 module·모든 buffer가 고정됐다. 출력기 연산을 통과하는 gradient는 새 thought로 전달하지만 출력기 parameter는 업데이트하지 않는다. 실제 V102/V105 전체 경계 digest 검증이 이를 뒷받침한다. 단순히 optimizer 이름만 선언한 준비 기록과 구분한다.

## 4. 세 메모리는 정확히 어디까지 동작하는가

| 메모리 | 현재 구현/검사 | 해석의 한계 |
|---|---|---|
| K_P 기본 퍼지 | learned canonical anchors와 prior를 μ/정규화ρ/confidence로 조회. 새 thought도 매 단계 자기 query로 읽음 | 인증된 사실 writer/장기 사실 통합은 없음. learned prior를 참인 지식 DB라고 부르면 안 됨 |
| K_C 입력별 퍼지 | 입력별 long/recent N/D 누적과 gate, global+ordered local state, source별 snapshot. 사고 중 내용은 readonly | 같은 문서 내용 보존과 새로운 관계 추론은 다른 능력. 문서 간 지속 기억 통합은 미완성 |
| K_E Exact | invalid NULL만 있고 입력 의존 key/value/span0, 내용0·copy0 | 현재 정상적으로 **사용을 금지한 상태**다. “세 메모리 모두가 유효 내용으로 작동한다”는 표현은 틀림 |

옛 codec에서 NULL value0이 bias가 있는 투영을 거치면 입력과 무관한 상수 offset은 남을 수 있다. 이것을 원문이 Exact로 운송되는 것과 혼동하지 않는다. 새 thought의 Exact 조회는 모든 유효 항목이 꺼져 있어 실제 content read가0이다. 내부 명시적 copy intent의 선택·permit·회수 정책은 아직 구현·학습하지 않았다.

V105의 같은8개 입력/잡음 대조에서 K_C만 제거하면 NLL이 평균0.504 낮아지고3개 출력이 바뀌었으며, K_P만 제거하면0.025 낮아지고 출력 변화는0개였다. 이것은 메모리 기여가 유익하다는 증거가 아니다. 이미 입력 Zx에 정상 메모리의 효과가 포함되고 출력 자체도 나쁘므로, 이 작은 제거 대조로 메모리 전체가 무의미하다고 결론내릴 수도 없다. 현재 확인된 것은 **실제 경로·동결·입력 내용 분리**이며, **유용한 의미 인과성**은 미확인이다.

## 5. 높은 복원이 곧 추상 의미 공간을 증명하는가

완전 복원은 표현에 필요한 정보가 남았다는 강한 근거다. 그러나 임의의 가역 부호화도 복원할 수 있으므로, **복원 loss만으로 표현이 추상적인 의미 공간임을 식별할 수는 없다.** 단어 순서와 표면 문자를 다른 신경 부호로 보존한 경우도 포함한다.

현재 입력16byte마다 local512차원 state가 하나 증가한다. FP32로 저장하면 local 벡터만2,048byte이므로 원문16byte의128배다. 이 계산은 저장 형식의 수치적 크기 비교이며 의미 정보량이나 실제 압축 파일 크기와 같지 않다. 의미 공간이 작게 압축된 고정 벡터라는 설명에는 맞지 않는다. 원문/Exact 복사 shortcut을 없앴다는 것은 지켜졌지만, **문자·순서를 풍부하게 보존하는 길이 비례 신경 codec**이라는 성격도 함께 인정해야 한다.

의미성의 증거는 같은 뜻의 다른 표현에서 같은 사실·관계를 읽을 수 있는지, 주어/목적어 교환·부정·수량 변경 같은 작은 의미 변화는 구별하는지다. 표현 벡터 전체가 같아야만 한다는 기준도 적절하지 않다. 내용과 표현 방식이 함께 들어 있을 수 있기 때문이다.

비지도 disentanglement의 식별에는 모델·자료의 inductive bias가 필요하다는 원 연구도 이 구분을 뒷받침한다. 다만 그 정리를 현재 FDT가 불가능하거나 실패했다는 증명으로 적용하지 않는다. [Locatello 등, 원 논문](https://proceedings.mlr.press/v97/locatello19a.html).

옛 typed/given-role/fuzzy-closure 보조 분기의 좋았던 관계 예측 성능은 유효한 과거 성과다. 현재 raw-text Commit 기반 새 thought의 정확도와 입력 조건이 다르므로 그 수치를 그대로 이식하거나, 현재 낮은 답변율로 과거 성과를 소급 취소하지 않는다.

## 6. 장문과 숫자 오류를 과장하지도 숨기지도 않는다

고정 V88의 저장 결과상 262,042+262,029byte 두 장문의 합계524,071byte 중49byte 오류가 남는다. byte 오류율은 약**0.00935%**, 일치율은 약**99.99065%**다. 전체 문장 일치0을 이유로 이 정보 보존을 무가치하다고 보는 것은 잘못이다. 그렇지만 숫자·개체·동사 하나가 바뀌면 짧은 문자 차이로 중요한 사실이 달라질 수 있다.

별도64개 숫자 스트레스 자료에서는 명백히 맞는52개, 명백한 값 오류4개(6.25%), 비표준 접두사로 미인증8개가 남는다. 이를 자연 영어 전체의6.25% 의미 오류율이라고 일반화하지 않는다. 그 목적의 스트레스 패널에서 아직 값 안정성이 불완전하다는 근거다.

V91의 퇴행은 장문49→52byte, 증가3byte로 전체의 약0.000572%였다. 대규모 붕괴로 부를 크기는 아니다. 그러나 같은 Commit·같은 frozen prefix에서 마지막 분류기 두 tensor만 바꿔 세 오류가 재현됐고 정답/오답 margin이 실제 음수로 바뀌었으므로, 학습 간섭에 의한 출력 오류라는 원인은 강하게 지지된다. TRAIN bank119,769개 top1과152개 floor를 보호해도 미관측 장문 feature를 모두 보장하지는 않았다. 이 때문에 V91을 제외하고 V88 전체를 선택했다. 평가 정답 class row만 골라 원상복구하는 방식은 실제 모델에 적용하지 않았다.

이전 gate 반복도 구분한다. V88에서는 실제 Adam 방향과 유한 보호 제약의 충돌이 주된 정체였고, V91 보정 거부에는 FP32 여유보다 작은 padding 문제가 관측됐다. 하지만 수락 후의 새 장문 오류3개는 **검사 범위 밖 feature에서 공유 분류기가 움직인 현상**이다. padding만 키워 수락을 늘리면 이 원인이 해결되는 것은 아니다.

장문 codec은 9개48byte causal-window 층의 직접 receptive field가 이전423byte이고, 별도의 fuzzy 누적/global summary가 더해진다. streaming K/V는 같은 계산의 장문 실행이다. 262KB 전체를 dense Transformer attention으로 동시에 이해했다는 결과는 아니다. 반복 작성 문법과 이미 관측한 자료라는 한계도 있다. 새 thought의 소스 slot 상한은262,144byte를 수용하는 geometry지만 **현재 사고 학습 입력은 최대120byte, DEV는 최대127byte뿐**이다. 장문 복원 성공을 장문 사고 성공으로 부르면 안 된다.

## 7. V102→V105에서 실제로 무엇이 좋아지고 나빠졌나

같은96개 관측 개발 패널, 같은 고정 codec·추론 operator·입력별 잡음 정책을 비교했다. V105의 baseline96은 V102 저장 결과를 재사용했고 warm1행의 실제 동등성을 사전 확인했다. 새로운96 baseline NN 계산을 반복하지 않았다. NLL은 정답 byte/EOS의 저장 점수이며 현재 MLP가 정답 prefix를 입력받는 teacher-forced 생성이라는 뜻은 아니다.

| 패널(각32개) | 무작위 thought NLL | V102 192update | V105 추가768update | V105−V102 |
|---|---:|---:|---:|---:|
| 미학습 pair group |15.4054|5.8358|5.0509|−0.7848 개선|
| 표현 변경 |14.3513|6.7169|7.6872|+0.9703 악화|
| 위치/앞뒤문맥 변경 |14.7665|5.8829|6.6891|+0.8061 악화|
| 단순 평균 |14.8411|6.1452|6.4757|+0.3305 악화|

V102가 처음부터 아무것도 배우지 않았던 것은 아니다. 초기 NLL은 크게 내려갔다. V105에서도 target byte argmax 일치율은 pair13.07→27.95%, surface9.63→19.57%, position11.49→20.06%로 올랐다. 그럼에도 뒤 두 패널 NLL과 실제 문장은 나빠졌다. 더 많은 위치에서 맞춰도 틀린 위치에서 정답 확률이 크게 낮아질 수 있고, 전체 문장은 그 몇 위치와 종료에 민감하기 때문이다.

ROOT가 실제 질문과 V105의96개 출력 전부를 다시 읽고, 저장 trace를 모델 재실행 없이 독립 재구성했다.

- 정해진 답문 전체 일치는0/96이다. 이것만으로 의미0/96이라고 선언하지 않는다.
- `"\nsTiy unknown."` 한 건은 prefix가 망가졌지만 ‘알 수 없다’는 응답 단서가 분명하다.
- `Ittis unkn wns`, `It is mtknown.`, `I  0s Anknown.` 등8건은 불확실성 표현을 의도했을 가능성이 있으나 깨진 철자 때문에 인증된 정상 답으로 올리지 않았다. 이는 사후 인간 판독의 모호성 범주이며 새 의미 정확도 점수가 아니다.
- 긍정 관계48행에서는 주어·목적어·방향이 명확히 맞는 답을 확인하지 못했다. 질문이 ‘B를 A와 비교’이고 사실이 A>B인 곳의 단독 `>`는 그 순서의 비교 부호라면 반대다. 한 글자 응답을 무조건 문자열 실패로만 세지는 않았다.
- EOS는96/96에 존재하지만72개는 기대 내용보다 이르고17개는 길이만 같으며7개는 늦다. 종료 토큰 존재가 정상 문장 종료 성공은 아니다.

각 DEV의 unknown은16/32다. unknown만 항상 답해도50%가 되는 패널에서 드문 unknown 출력이나 두 건의 unknown 성공을 관계 추론 완성으로 볼 수 없다. surface/position은 일부 TRAIN facts를 의도적으로 공유하고, 반복해 관측한 개발 자료다. 새로운 독립 blind benchmark96개가 아니다.

전체96개의 원문·기대 답·실제 답·판독은 별첨 `FDT_CURRENT_COMPLETE_ANALYSIS_20261114_evidence/V105_all96_source_output_review.md`와 JSON에 보존했다. 어휘 유사도나 edit similarity를 사실·관계 정확도의 대용으로 쓰지 않았다.

## 8. 현재 사고 병목: 확인된 사실과 원인가설

### 8.1 정답을 섞은 학습 상태와 실제 추론 상태의 차이 — 가장 강한 근거

V102/V105의 기존 loss는 정답 의미 E(Y)에 잡음을 넣은 상태에서 한 번 예측한 clean 의미에 언어 손실을 준다. 실제 추론은 정답 없이 초기 잡음에서16번 자기 출력을 이어간다. **각 단계의 시간값을 학습했다는 것과 자기 예측으로 도달하는 상태 분포를 학습했다는 것은 다르다.**

V103 TRAIN4 진단에서 낮은 noise t=.05의 byte CE는0.00646, t=.5는0.13493, t=.95는1.95698인데 실제16단계 target-free는5.54082였다. 정답 정보가 남아 있는 noisy input을 잘 정리하는 능력과, 입력 조건만으로 답 의미를 새로 만드는 능력 사이에 큰 간격이 있었다. 중간 noise에서 나온 정답은 실제 inference 성공이 아니다.

이후 V104/V105에서384TRAIN 각각을 실제16개 sampler 시간값에 한 번씩 배치하고, 손실에서 빠졌던30/31개 tail slot에도 작은 별도 손실을 추가했다. 전체6144행과 모든 시간 coverage는 저장 RNG/order로 검증됐다. 그런데 실제 답문이 여전히 깨졌다. 따라서 “높은 noise가 부족했다” 또는 “tail loss가 없었다”만으로 충분한 설명이나 해결이 되지는 않았다.

### 8.2 같은 입력·잡음에서 후속 단계가 출력을 망가뜨림 — V106 좁은 통제

V106은 TRAIN4개에만 같은 입력·잡음·가중치·출력기를 사용하고 sample 단계수를1/4/16으로 바꿨다. 84thought+12D=96calls, optimizer0/GPU0, 약8.609초로 완료하고 모든 프로세스 종료 후 분석했다. 이번에는12개 저장 trace의 출력·CE·저장된 full258 sample을 독립 계산으로 다시 확인했다.

| 기대 답 |1단계|4단계|16단계|
|---|---|---|---|
| C is larger than F. |C is sarglr thhnn A.|2 3i|2 3i|
| F is smaller than C. |A i|공백 두 개+a|공백 두 개+as|
| It is unknown. |정확|Itwisqunknowi.|Itwisqunkiowi.|
| It is unknown. |정확|ItKa|ItKa|

unknown 두 건은1단계에서 고정 D가 올바르게 읽는 상태를 만들었는데16단계 구성에서는 깨졌다. 후속 trajectory가 출력 가능 상태를 보존하지 못한다는 좁고 구체적인 근거다. 반면 긍정 관계 두 건은1단계에서도 틀렸다. 따라서 후속 단계만이 유일한 문제라고 할 수 없다. 단계수를 바꾸면 τ 경로 자체도 바뀌며, 이4개는 사후 선택한 훈련 사례다. 전역적으로1단계가 최적이라는 결론이나 독립 관계 이해 증명이 아니다.

### 8.3 출력 경계와 의미 표적의 종료 불일치

V95에서 실제 E(Y)를 직접 준31개 정답의 내용599byte는 prefix에서 모두 읽혔지만, 부분 patch의 EOS가 늦어 전체 문자열은31개 모두 끝이 깨졌다. 최종 target vector와 바람직한 종료 출력이 완전히 일치하지 않는 것이다. 이를 latent MSE만으로 학습하면 알려진 종료 한계를 물려받을 수 있다. 동결 D를 통과하는 byte/EOS 보조 손실은 이 문제를 겨냥한다.

하지만 **D가 유효한 E(Y)를 읽는 능력**과 **새 Rθ가 만든 off-manifold 상태를 안정적으로 읽는 능력**은 다르다. 현재 garbled output만 보고 내부에 올바른 사실이 있는지 없는지는 분리 판정할 수 없다. 독립 사실 판독/통제 결과가 부족하다. “출력기만 고치면 된다” 또는 “의미가 완전히 없다” 둘 다 근거보다 강하다.

### 8.4 VP 수식 오류나 반복 벌점 부족으로 단정하지 않는다

코드의 정규화 residual r에 대해 `r_t = α r_0 + σ ε`, `v = α ε − σ r_0`이고, 역변환 `r_0 = α r_t − σ v`, `ε = σ r_t + α v`는 α²+σ²=1에서 서로 맞는다. 다음 시각의 `α_next r_0 + σ_next ε`로 진행하는 기본 대수에 명백한 부호 뒤집힘은 발견하지 못했다. 이 수식 점검은 학습/수치/메모리 구현 전체의 무결성 증명이 아니다.

DDIM은 같은 계열의 학습 목적에서 다른 샘플링 경로를 구성하며, v-parameterization과 적은 step의 안정성은 별도로 연구돼 왔다. 이것이 FDT에서16단계 또는 V107 계수의 최적성을 보증하지는 않는다. [DDIM 원 논문](https://arxiv.org/abs/2010.02502), [Progressive Distillation 원 논문](https://arxiv.org/abs/2202.00512).

현재 대다수 출력은 긴 반복보다 조기 종료·글자/개체/방향 손상이다. AR 반복 feedback도 없는 현재 경로에서 강한 반복 벌점을 먼저 높이는 조치는 원인과 맞지 않는다. 과거 β=10의 국소 logit-gradient 계산을 현재 최적 계수로 가져오지 않는다.

## 9. V107은 어떤 수정이며 어디까지 준비됐나

`research/runtime_aligned_thought_v107.py`는 **정답 없이 돌아가는 실제16단계 trajectory 전체**를 학습 graph에 둔다. 입력 의미/readonly 메모리/독립 잡음만으로 끝까지 계산한 뒤, 최종 의미에 대해 frozen D의 byte/EOS 손실과 body/tail 의미 손실을 준다. 작은 noisy-target VP 보조 손실은 별도 branch로 유지한다.

```text
학습 main: Zx + 독립noise → Rθ×16 → 최종Z → 고정D → 언어/종료 loss
                                       └──────── 의미 target loss
학습 보조: E(Y)에 noise → Rθ 한 번 → 작은 VP loss
추론:      Zx + 독립noise → Rθ×16 → FINAL Commit 확정 → 고정D
```

주 경로는 target/mask/답 길이를 인수로 받지 않는다. 추론 시 FINAL Commit을 detach하는 경계는 유지하되 학습 때는 너무 일찍 detach해 gradient를 끊지 않는다. 기존 codec 전체 동결 원칙도 유지한다.

당시 tiny CPU4개 검사에서 target-free 인수 경계, 샘플러 수식 일치, 마지막 D 이후 새 thought로의 gradient, 기존 D 불변, READY Adam/scaler/RNG 복구와 input checkpoint 비변조, eval에서 target loss 거부를 확인했다. **tiny fixture는4step이며 실제 전체 모델16step의 worker/preflight/gradient 안정성 검사가 아니다.** 이번 중단 정리에서는 이 시험을 재실행하지 않았다.

실제 새 runner/owner, 전체 크기 preflight, 실행 plan·상한·seal·launch는 없다. V105 성능을 V107의 성능으로 보고하지 않는다. 이 수정은 현재 가장 직접적인 후속 가설이지만 성공이 입증된 해결책은 아니다. 사람의 중단 지시에 따라 여기서 멈춘다.

## 10. 다음 명령이 온다면 우선할 일

다음 목록은 자동 재개 지시나 실행 승인 대기가 아니라, 현재 분석에서 남긴 우선순위다.

1. **기존 codec 동결을 유지하고 V107 실제 trajectory 학습을 좁게 통합**한다. 목표는 ‘더 오래’가 아니라 noisy-gold 한 번과 실제16단계의 학습 불일치를 줄이는 것이다. 초기에는 같은 TRAIN에서 정방향/역방향/unknown과 두 표현 조건을 함께 확인한다.
2. 새 thought의 **사실 형성과 언어화 충실도**를 나눈다. 질문 대상·방향·부정·수량이 맞는지와 최종 문장의 사실을 분리해 기록한다. 낮은 vector MSE 또는 비슷한 철자만으로 의미가 맞다고 세지 않는다. Gold 의미를 평가 condition에 넣지 않는다.
3. 독립 관계가 읽히기 전에는 장문 출력 용량부터 크게 늘리지 않는다. 기본 단위가 안정된 뒤 입력 길이/질문 위치/장거리 의존을 한 축씩 확대한다. 기존 장문 codec 평가를 계속 반복하는 것으로 새 thought 장문 능력을 대신하지 않는다.
4. 추상 의미 codec 검증은 영어의 미학습 동치 표현과 최소 의미 변경을 중심으로 한다. 중요 값·관계 보존을 우선하고 표면 전체 일치를 유일한 목표로 하지 않는다. 지속적인 canonical facts 작성과 Exact의 내부 명시적 복사는 별도 설계로 둔다.
5. 새로운 학습은 새 목적·유한 노출/시간/call/저장 상한·정확한 fullstate·전체 동결·owned OS 감독을 갖춘 별도 trial이어야 한다. 닫힌 V102/V105 예산이나960step의 미사용 목표를 이어 달리지 않는다. 새 승인 질문 없이 진행할 수 있었던 기존 자율 범위와, **현재의 명시적 중단 지시**를 혼동하지 않는다.

## 11. 종료·저장·증거

V105는768/768 update, AMPskip0, 14,192/14,400calls, worker158.500초로 COMPLETE다. native owner30708→37028, worker33884→30172의 생성·부모·명령·생존을 ROOT가 당시 확인했고, 종료 후 **01:29:47 KST에 allowned/proxy gone+Python0**을 분석보다 먼저 확인했다. V106도 worker/owner exit0이며 **01:40:55 KST Python0** 기록이 있다. V107 actual run은 존재하지 않는다. 최종 CURRENT에 남았던 ‘V106 active’ 문구는 과거 스냅샷으로 보존하고 새 첫 CURRENT에서 종료 상태로 바로잡는다.

이번 종료 분석은 saved raw96+12개의 UTF8 선택·argmax·정답 log-probability·저장된 full258 sample과 checkpoint 파일 SHA를 다시 계산했다. 모든 위치의258개 logit 전체를 저장/재계산했다는 주장은 하지 않는다. 기존 actual run의 전체 optimizer/scaler/RNG와 full-frozen 검증은 해당 root review를 근거로 구분해 인용했다. 모델을 다시 생성하거나 학습하지 않았다.

C: 여유는 분석 저장 직전 약**106.53GiB**,100GiB 보호선 이상이다. 이는 전역 가용 공간이므로 앞으로도 고정값은 아니다. 이번 정리에서 파일·가중치·실패·게임 삭제 또는 압축0이다. 이전167개 CLOSED raw의 NTFS 무손실 압축으로 회수한 약2.97GiB는 과거 작업이며 이번 회수량으로 다시 세지 않는다.

주요 근거:

- `runs/grid_tail_thought_20261114_v105/{result.json,root_saved_review.json,root_result.json,assessment.json,ROOT_terminal_native.json}`.
- `research/artifacts/thought_rollout_20261114_v106_cpu_v1/{ROOT_terminal_native.json,cpu_preflight_owned/final_status.json,cpu_preflight_owned/runtime/result.json,raw.pt}`. raw.pt는 runtime 아래 있다.
- `outputs/FDT_CURRENT_COMPLETE_ANALYSIS_20261114_evidence/{analysis_summary.json,checkpoint_identities.json,V105_all96_source_output_review.json,V106_independent_saved_review.json,evidence_SHA.json}`.
- 코드 `research/{fuzzy_roundtrip_v64.py,ordered_fuzzy_readout_v55.py,commit_patch_readout_v54.py,memory_thought_diffusion_v94.py,thought_grid_tail_v104.py,runtime_aligned_thought_v107.py}`.
- 이전 보고서 `FDT_LANGUAGE_MEANING_THOUGHT_BOUNDARY_REVIEW_20261113_KR.md`, `FDT_V91_ERROR_INCREASE_CAUSE_20261113_KR.md`, `FDT_MEMORY_THOUGHT_DIFFUSION_PREP_V94_20261113_KR.md`, `FDT_FIRST_THOUGHT_V102_AND_NOISE_GAP_V103_20261114_KR.md`.

**작업 상태는 사람 지시에 따른 중단·정리 완료, 타이머 삭제, 추가 명령 대기다. 연구 목표를 완수했다고 표시하지 않는다.**
