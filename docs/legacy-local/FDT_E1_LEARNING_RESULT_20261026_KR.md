# 102M 첫 학습 결과와 다음 보강 — 2026-10-26

**기본 관계 처리의 학습 가능성은 확인됐지만, 새로운 문구 표현과 diffusion의 추가 이득은 아직 해결되지 않았다.** 실제102,237,197 파라미터로 성공1,440회/시도1,440회(grounding480·semantic320·diffusion640)를 완료했다. AMP skip0, nonfinite gradient0이며22:16 KST에 최종 평가까지 마쳤다. 관련 감독·worker 프로세스는 모두 종료됐다.

중간110회에서 Windows 진행 JSON 교체 오류가 있었으나 full model/Adam/scaler/모든 RNG를 같은 수치 상태로 복구했다. 다음111회 배치도 저장된 sampler와 일치했다. 최초110회를 다시 학습하지 않았고 원래22:43:46 종료 상한을 유지했다. 이식·운영 migration과 실패 원문은 별도 보존했다.

## 저장과 실제 학습 과정

최종 marker는 `runs/e1_masked_learning_20261026_recovery_v2/model/complete_001336.json`, payload는636,785,153B, SHA256은 `d57d06ec7cc1ebb4f5ccbf365673554efc8a705043d492dd802fff1b6c7c8edb`다. 현재 phase의 populated Adam·AMP·RNG와 전체 모델이 있고 이전 phase의 상태도 tagged checkpoint에 보존됐다. 로컬 hash 검증과 재로드를 수행했으며 Drive 백업을 주장하지 않는다.

| Phase | 성공 시도 | 첫/마지막 batch loss | 최대 clip 전 norm |
|---|---:|---:|---:|
| grounding | 480 | 1.38482 / 0.01284 | 0.7614 |
| semantic | 320 | 0.06406 / 0.05402 | 0.3303 |
| diffusion | 640 | 2.25294 / 0.18499 | 6.5374 |

서로 다른 batch의 loss이므로 위 두 점만으로 일반화를 판단하지 않는다. 최대 allocated VRAM2,671,343,104B(약2.49GiB), reserved2,986,344,448B(약2.78GiB)였다. 실제 입력 노출769,695 UTF-8 bytes는 tokenizer token 수가 아니다. Query 정답 노출은 `[33279,9552,9321,60812]`, grounding 노출은 `[10605,10589,6487,6323]`로 별도 집계됐다. 기존 데이터의 클래스 편중은 이 학습에서 제거하지 않았다.

Step wall time 합계330.57초, 매 시도 full 저장 시간 합계3,438.04초였다. 이 두 시간의 합 중 약91.2%가 저장에 쓰였다. GPU kernel 시간만을 측정한 수치는 아니다. 안전한 지속 저장은 확인됐지만 처리량 병목은 I/O였다. 처음 요청한 매 시도 저장 정책을 그대로 지켰고 다음 후속 실험에서도 조용히 주기를 늘리지 않는다.

## 성능: 같은 모델, 검증기 이전 출력

| 평가 패널 | 고유 그래프 | Grounding | Initializer accuracy/F1 | Diffusion accuracy/F1 |
|---|---:|---:|---:|---:|
| 새 typed 그래프 | 16 | 100% | 100% / 1.000 | 98.61% / 0.979 |
| 원래 heldout 4문구 | 16 | 100% | 100% / 1.000 | 98.58% / 0.981 |
| 큰 그래프(7~8개체) | 16 | 100% | 100% / 1.000 | 96.09% / 0.888 |

각 패널은 noise seed17/29/43을 사용했다. Query 집계720/636/768에는 seed 반복이 포함되며 독립 표본 수가 아니다. 별도96그래프 개발 평가에서도 initializer100%, diffusion98.72%였다. Raw 첫4그래프씩 총12사례를 원문 전제·query·label과 대조했다. 큰 그래프의 직접 관계도 diffusion에서 일부 뒤집혔다.

이 모델은 관계 문구를 byte로 학습하지만 subject/object 역할은 구조화 입력으로 주어진다. 학습된 문구 grade 뒤에 고정 max-product 관계 폐쇄가 있다. Initializer100%를 전부 자발적으로 학습한 다단계 추론, 범용 NER나 언어모델 완성으로 해석하지 않는다. 로컬 matched baseline 학습도 없으므로 mask 수정만의 단독 이득은 확정하지 않았다.

## 작은 후속 진단으로 확인한 두 약점

별도 고정 읽기 전용 계획 `runs/e1_posttrain_proof_20261026_v1`을 실행했다. Optimizer0, 논리 forward548회,11.25초이며 원래4-step 큰 그래프 confusion을 정확히 재현했다.

첫째, 큰 그래프의 DDIM4/8/16단계 accuracy는96.09/95.96/96.09%, macro F1은0.888/0.885/0.888이었다. 이 작은 개발 패널에서는 단계를 늘리는 것으로 오류가 해결되지 않았다. 단순 적분 단계 부족보다 학습된 denoiser의 의미 보존·조건 사용을 후속 검토할 근거다. 원인을 수학적으로 확정하거나 diffusion을 폐기한 것은 아니다.

둘째, 같은 typed 그래프에 표현만 교체하면 grounding이100%에서37.5%, 관계 initializer가100%에서25~37.5%로 떨어졌다. Raw에서 `value exceeds`를 LESS로, `value does not exceed`를 NOT_LESS로 분류한 오류가 반복됐다. 새로운 영어·한국어 어휘 표현을 제대로 배우지 못한 문제가 직접 보인다.

교차 패널 파일의 `seen_graph`는 **E2 후보의 train fold**라는 이름이지 E1 모델이 실제로 학습한 그래프라는 뜻이 아니다. 해당 그래프는 negative facts4개인데 E1 원래 생성기는 최대3개이므로 두 graph fold 모두 E1에는 새 그래프다. 표현을 바꾼 짝끼리는 그래프·query·정답이 같아 표현에 따른 차이는 볼 수 있지만 표본은 서로 독립이 아니다. E2 final에는 모델을 실행하지 않았다.

따라서 원래 heldout 4문구100%를 넓은 표현 계열 일반화로 주장할 수 없다. 현재 가장 실용적인 다음 보강은 어휘·부정 범위 다양성이다. 이미 완벽한 initializer보다 diffusion의 추가 정확도 이득이 있다는 주장도 보류한다.

## 다음으로 준비한 한정된 개선

새 `research/artifacts/e2_lexical_20261026_v2`는 train512/dev64, 독립 typed 그래프576개다. Train8개 표현 family(영어6·한국어2), 새 dev2개 family를 사용하고 각 grounding 종류를25%로 맞췄다. CPU 독립 경로 oracle576건·전체 collate·문구48-byte 한도·분리 검사를 통과했다. 영어75%는 행 비율이며 token·byte·gradient 비율이 아니다.

이미 측정한 `en_exceed`/`ko_numeric` 개발 family를 이번 새 train으로 **명시적으로 승격**했다. 이후 그 두 family의 성능은 미학습 성능으로 보고하지 않는다. 별도의 새 dev를 만들었고 원래 final 문항은 모델 평가·훈련에 사용하지 않는다. 여전히 작은 합성 자료이며 이상적인 자연어 corpus는 아니다.

다음 worker는 같은 모델에서 언어 인코더·phrase router·ground head만 작은 학습률로 보강하고 앵커·prior와 관계/refiner는 동결하도록 준비 중이다. 모델 weights-only warm start, 새 Adam/RNG, 최대128회 성공/140회 시도/1,800초의 **별도 새 계획**이며 아직 학습 실행 증거는 없다. 기존 grounding/관계 retention과 앵커 불변성을 계속 검사한다. 옛1,440회 실행과 deadline을 다시 열지 않는다.
