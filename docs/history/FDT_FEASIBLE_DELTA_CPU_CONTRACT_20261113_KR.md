# 거부되는 업데이트의 방향 보정 — V89 CPU 계약

2026-11-13. V88은 같은 방향의 update를 줄여도 취약 critical margin 제약 때문에 멈췄다. 이를 더 반복하거나 기준을 낮추는 대신, **실제 Adam이 제안한 파라미터 변화의 방향을 제한적으로 보정하는 후보**를 작성했다. 새 연구 모델 학습이나 실제 거부 proposal의 복원 결과는 아니다.

먼저 V88 결과와 V80의 actual-delta transaction 코드를 다시 읽었다. [GEM 원 논문](https://arxiv.org/abs/1706.08840)은 과거 경험 손실의 gradient 제약을 사용하는 연구이고, [Adam 원 논문](https://arxiv.org/abs/1412.6980)은 moment와 적응적 scale로 update를 만든다. 따라서 raw gradient를 수정하는 것과 최종 Adam 파라미터 변화를 수정하는 것을 혼동하지 않는다. 이번 후보의 유한 반복·상한이 논문의 최적 설정이라는 주장은 하지 않는다.

## 계산과 구현

고정 prefix feature가 `h`이면 특정 byte role의 정답 g와 경쟁자 j의 차이는 다음과 같다.

`gap = (W_g − W_j) · h + b_g − b_j`

따라서 원래 reference의 `gap ≥ floor` 조건은 마지막 Linear 파라미터에 대한 선형 부등식이다. 한 부등식이 d만큼 모자라면 그 부등식에 대한 최소 Euclidean 보정은 두 class row와 bias에 다음 양을 반대로 더하는 것이다.

`τ = d / [2 (||h||² + 1)]`

`W_g += τh, W_j -= τh, b_g += τ, b_j -= τ`

`research/feasible_readout_delta_v89.py`는 실제 proposal을 입력받는 이러한 보정 구성 요소다. 최대2회 sweep, 최대16개 활성 TRAIN 제약, 보정 크기≤원래 proposal 크기, 고정 수치 여유1e-6의 후보 계약이다. 여러 부등식 전체의 최단거리 해를 구하는 최적 solver로 부르지 않는다. 이 수치들을 실제 GPU plan에 봉인한 상태도 아니다.

원래 Adam proposal과 양의 방향 성분이 남아야 하며, 원래 reference floor를 바꾸지 않는다. 활성 제약만 맞았다고 수락하지 않고 **전체 유한 TRAIN reference 판정**을 반드시 통과해야 한다. 보정 상한 초과, 유한 sweep 미수렴, 전체 판정 거부 시 유효 candidate를 반환하지 않는다. callback의 candidate 변조나 bool만 반환하는 가짜 성공도 거부한다. 원본 warm/proposal tensor는 수정하지 않는다.

CPU 해석 예에서는 기존정보 A 보존에 scalar α≤0.25, 새정보 B 개선에는 α>0.5가 필요해 같은 방향을 축소하는 것만으로는 불가능했다. 방향을 보정한 결과는 A gap0.050001로 하한0.05를 유지하면서 B gap0.1750005가 됐다. **이는 합성 선형 예의 계산이지 FDT 미학습 문장 개선 증거가 아니다.** 다른 문맥을 해치는 보정은 전체 reference 검사에서 실제로 거부되는 사례도 검사했다.

## 원래 anchor 유지

`research/immutable_reference_anchor_v89.py`는 warmstart와 reference 재초기화를 구분한다. 원래 margin1.0의 보호 하한0.5가 있는 상황에서 현재 margin이0.51이라고 다음 실행의 하한을0.255로 다시 낮추면 여러 실행에 걸쳐 기준이 약해진다. 이러한 target/reference 파일 교체를 거부하도록 했다.

실제 V80 source와 V88 checkpoint의 직렬화 구조를 확인했다. 같은 standalone parent와 model key 집합, 7개 override 중 동결된5개 prefix tensor의 일치, frozen embedding, core/capacity/FP32 정책의 일치를 확인했다. 기존 V88 독립 검토에서도 마지막2개 외 tensor가 같음을 검증한 기록과 연결했다. V88에서 저장한 bank는 초기 V80 FP32 reference이므로 그 feature/target 파일과 **152개 원래 하한을 그대로 가리키는 anchor**를 만들었다. 새로운 teacher 값으로 하한을 재계산하지 않았다.

표현에 관련된105개 기존 operator 파일 SHA를 V88의 봉인된 source 목록에서 고정했다. 이후 입력·prefix·core·정밀도·operator가 달라지면 이 anchor를 그대로 쓸 수 없다. 이 계약은 metadata만 보고 실제 계산을 자동 증명하는 장치가 아니다. 다음 실제 worker가 전 tensor·코드·표현 경로를 검증해야 한다. 이 bank는 훈련 제약용 자료이며 inference의 K_P/K_C/K_E에 네 번째 메모리를 추가한 것이 아니다.

## 확인 범위와 남은 작업

새 CPU 검사8개 PASS: 방향 변경이 필요한 해석 예, 단일 제약 보정식, 전체 reference가 active-only 수정을 거부하는 경우, 보정 상한, 유한 반복 미수렴, 실제 byte-head shape, callback 변조, anchor 재초기화·표현 변경 거부를 확인했다. ROOT의 별도 CPU 검토와 실제 checkpoint/reference identity 확인도 통과했다. 실제 학습된 모델 forward0/parameter update0/GPU0, CUDA 초기화false다.

V88의 거부 candidate weights/logits/위치 ID가 저장되지 않았으므로 이 후보가 실제92회 거부를 해결했다고 말할 수 없다. 같은 V88을 로그 때문에 다시 돌리지 않는다. 다음 실행에서 이미 계산하는 제약 검사 결과에 source/byte/rival/하한/위반량을 기록하면 된다.

다음은 실제 Adam/scaler transaction에 보정을 연결하고, proposal moment 처리·accept/reject rollback·모든 RNG/cursor/노출·STOP 복구와 고정 비용을 검증하는 작업이다. 유한 전체 reference는 계속 최종 판정을 맡는다. 이후 별도 worker/plan/CPU preflight/독립검토/owned OS 감독을 갖춘 원인 대응 실행만 가능하다. 원래 V80 anchor·과거 예산·시계를 초기화하거나 gate를 자동 완화하지 않는다.

디노이징은0이며 새로운 학습을 실행하지 않았다. 사고 학습을 진행할 때는 입력·기본/문장/Exact 메모리·결합·출력 codec 전체를 고정하는 기존 원칙을 유지한다.

증거: `research/artifacts/feasible_readout_delta_20261113_v89_cpu_v1/root_cpu_review.json`, `original_V80_FP32_reference_anchor.json`, `operator_files.json`, `evidence_SHA.json`.
