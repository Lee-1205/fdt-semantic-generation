# FDT 102M 표현 보강 결과와 개체 번호 coverage — 2026-10-27

E2 표현 보강은 128회 성공/128회 시도로 완료됐다. 이전에 틀리던 표현을 학습한 뒤 인식이 좋아졌지만, 새로운 표현 전반의 일반화 향상이 입증된 것은 아니다. 다음 개선은 학습에서 빠졌던 개체 번호 6·7을 diffusion 학습에 포함하는 제한된 실험이다.

## 실제 종료와 복구

- 실행: `runs/e2_lexical_grounding_20261026_v1`, 계획 SHA `4a15fd99745e22cc165c1c5ac63ecdebfa200f7d686b68edc8a0e6bb0a74dcac`.
- 10월 26일 22:49:38 KST 시작, 22:56:30 KST 감독 종료/exit0. Worker·redirector·supervisor 종료를 확인했고, 10월 27일 재개 시 native Python 프로세스도 없었다. 계획/위임을 실행 증거로 사용하지 않았다.
- E1의 1,440회 가중치만 가져온 새 단계다. 새 Adam/scaler/RNG seed20과 단계 counter0에서 시작했으며 E1 optimizer 연속 재개라고 하지 않는다.
- 최종 `model/complete_000129.json`, payload 921,464,853B, SHA `938b15e963ad4f9a120aed333653e02f588594c4235482960cc4164bb1d1bac4`. Full model·Adam/scaler·RNG·counter를 보존했다.
- language/phrase_router/ground_head만 변경됐다. 사후 tensor 비교에서 나머지 모든 tensor가 E1과 같았다. 원래 core source SHA `eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352`는 유지된다.

## 같은 입력에서 전후 비교

`runs/e2_paired_comparison_20261026_v1/summary.json`은 optimizer0, 264 logical forwards, 9.234초의 완료된 비교다. 부모의 새 dev64 baseline은 보존된 E1 모델로 사후 측정했다.

| 개발 패널 | E1 → E2 grounding | E1 → E2 초기 관계 판정 |
|---|---:|---:|
| 새 dev64 | 100% → 100% | 100% → 100% |
| 학습 표현 8종 32문항 | 65.625% → 100% | 59.375% → 100% |
| 이전 개발 표현을 학습에 편입한 16문항 | 37.5% → 100% | 25% → 100% |

이전 dev의 `en_exceed`, `ko_numeric`는 E2 학습에 명시적으로 편입됐다. 마지막 행은 학습한 표현을 새로운 그래프에 적용한 결과이며, 미관측 표현 일반화로 보고하지 않는다. 새 dev64는 부모부터 이미 100%여서 향상 근거가 없다. 최종 holdout은 모델 평가·학습에 사용하지 않았다.

원래 새 그래프 16개/3 noise seeds의 DDIM4는 98.6111% → 98.4722%로 720개 seed 반복 query 중 오류가 1개 늘었다. 큰 그래프 16개는 96.09375% 그대로다. seed 반복은 독립 그래프 표본이 아니며, 바뀐 encoder와 동결 refiner를 자동으로 승격하지 않았다.

## 개체 번호 누락을 확인한 근거

원래 E1 생성기는 n=4~6이므로 ID0~5만 학습했다. E2도 n=4~5이며 관계/refiner는 동결돼 ID6·7 노출을 보완하지 않았다. 큰 그래프 원문에서 ID6·7이 포함된 query는 288개 중 오류26개, ID0~5만 있는 query는 480개 중 오류4개였다. 이는 반복 noise가 포함된 관측치다.

`runs/entity_role_probe_20261026_v1/summary.json`은 같은 16개 그래프·문구·query 순서·noise를 유지하고 ID만 교환했다. 독립 CPU oracle로 정답/hops 불변을 확인했다. Optimizer0/156 logical forwards/5.469초다.

| ID 처리 | 초기 관계 판정 | DDIM4 정확도 |
|---|---:|---:|
| 원래 ID | 100% | 98.4722% |
| 학습된 ID0↔1 | 100% | 98.0556% |
| 미학습 ID0↔6·1↔7 | 100% | 90.2778% |

Grounding은 세 경우 모두 100%다. 이 비교는 관계 정제기의 개체 번호 취약성을 보여주며 큰 그래프 오류 전체의 유일 원인이라고 단정하지 않는다. DDIM4/8/16 단순 증가로 해결되지 않았던 이전 결과도 보존한다.

## 다음 한정 실험의 자료와 경계

`research/artifacts/role8_coverage_20261026_v1`은 CPU 6검사를 통과했다. Train은 E1에서 본 64개 추상 그래프에 8회 순환 재명명을 적용한 512행, dev는 관측된 olddev96의 16개 추상 그래프에 같은 변환을 적용한 128행이다. 512개의 독립 그래프라고 부르지 않는다. 동일 추상 그래프의 모든 변형을 같은 분할에 두고, class/hops/문구와 다섯 neural input key를 유지했다. 전체 후보에서 각 ID의 fact·query source/object 노출은 정확히 균형이지만 실제 optimizer 샘플 노출은 별도로 계측해야 한다.

Manifest SHA `3de8be1a14e959edb422e34b744437031dac4d9f733eba4add054c0fcf3b1e68`, train SHA `00ac972def12ca6d2530400e85dcbc97e7ed733f112593275fa6df1107a02128`, dev SHA `e338e2ed53fde683d790aab895a2f7a74d5b0b1f95e5d020ce077840f2321076`.

별도 새 계획으로 E2 모델 가중치에서 시작해 phase2만 최대128회 학습한다. Backbone/grounding/initializer와 원래 손실·구조를 고정하고, 실제 ID별 노출·기존 정확도 보존·paired ID 성능을 확인한다. 코드/CPU계약/출력/계획SHA/시간·시도 상한/독립 감독/자원 확인이 끝나기 전 실행하지 않는다. 과거 terminal/deadline은 다시 열지 않는다.

C: 20GiB와 임시 checkpoint 여유를 보호한다. 원본·다른 run·실패·모든 필요한 가중치/raw 증거를 보존한다. 이 연구는 주어진 개체/typed record를 다루는 제한된 adapter이며 범용 언어모델·자연어 개체 추출·vNext 전체 완성을 뜻하지 않는다.
