# E2 표현 계열·노출 균형 후보 — 2026-10-26

추가 GPU 학습 없이 데이터 후보와 CPU 계약 검사를 준비했다. 기존 `src/fdt_local/`과 smoke 체크포인트는 변경하지 않았다. 이것은 학습된 모델의 성능 개선이나 이상적인 자연어 데이터셋의 증거가 아니다.

| 분할 | 고유 typed 그래프 | 유효 query | query 클래스별 수 | grounding 종류별 수 | 입력 UTF-8 bytes |
|---|---:|---:|---|---|---:|
| train | 128 | 128 | 32 / 32 / 32 / 32 | 256 / 256 / 256 / 256 | 13,824 |
| dev | 32 | 32 | 8 / 8 / 8 / 8 | 64 / 64 / 64 / 64 | 5,056 |
| final | 32 | 32 | 8 / 8 / 8 / 8 | 64 / 64 / 64 / 64 | 7,296 |

Query 순서는 UNRESOLVED / GREATER / LESS / CONFLICT, grounding 순서는 GREATER / LESS / NOT_GREATER / NOT_LESS다. 두 집계의 분모는 각각 유효 query와 유효 fact 수다. 각 분할에서 각각 25%이며 loss weight나 실제 학습 노출량을 뜻하지 않는다. 향후 sampler의 행 선택에 따라 실현 노출은 달라질 수 있다.

행마다 전제 8개(종류별 2개), query 1개다. 원래 oracle를 유지한 채 원하는 query 클래스를 선택하고 공급이 부족하면 고정 후보 상한에서 실패한다. 실제 생성 후보는 train 536 / dev 108 / final 89개였고 분할당 상한은 10,000개다. 정답을 바꾸거나 모순 규칙을 완화해 수를 맞추지 않았다.

## 분리와 검사

각 분할은 영어·한국어 표현 family 각각 하나를 가진다. 총 6 family / 24개 typed 문구이며 literal 중복도 없다. family 내에서도 네 query 클래스가 같은 수로 배정된다. 전체 문장 이해나 개체 binding 추출을 추가한 것은 아니다. entity 역할은 기존과 같이 구조화 입력으로 제공한다.

4~5개 개체의 모든 순열을 확인하는 canonical key로 개체 이름만 바뀐 동일 typed 그래프까지 분리했다. fact 순서·중복은 key에 영향을 주지 않는다. 다만 GREATER와 방향을 뒤집은 LESS처럼 서로 다른 typed 인코딩의 논리적 동치 전체를 제거한 것은 아니다.

관련 CPU 검사 5개가 통과했다(4.96초). 별도 `prepare_e2_review.py`는 저장 파일의 SHA를 대조하고 전체 192행과 교차 패널 64행의 정답 256개를 경로 탐색으로 확인했다. production oracle의 Floyd 폐쇄와 별도 구현이다. 모든 행을 CPU collate하여 실제 mask·target·UTF-8 byte 집계와 neural 입력의 다섯 키도 확인했다. 모델 forward나 optimizer는 호출하지 않았다.

개발용 교차 패널은 각 16개다. 같은 그래프·query·정답을 유지하고 표현만 바꾼 쌍을 사용한다.

| 패널 | 그래프 | 표현 family |
|---|---|---|
| seen_graph_seen_family | train 일부 | train |
| seen_graph_dev_family | 같은 train 일부 | dev |
| dev_graph_seen_family | dev 일부 | train |
| dev_graph_dev_family | 같은 dev 일부 | dev |

교차 패널의 재사용은 통제 비교를 위한 의도적 재사용이다. 64개의 추가 독립 표본이나 학습 행으로 세지 않는다. final은 모델 평가·모델 선택에 사용하지 않았으며 CPU label/분리 검증만 했다.

## 저장 위치와 재개 경계

- 생성기: `research/e2_data_v1.py`
- 후보: `research/artifacts/e2_candidate_20261026_v1/`
- 교차 패널·별도 검사: `research/artifacts/e2_review_20261026_v1/`
- 후보 manifest SHA: `a66b8e5b3a75559678b4258522feb247ce596a4a6e8b496e97fcec1ca1dff42c`
- 보존된 runtime source SHA: `eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352`

후보는 고정 baseline runner에 연결하지 않았다. E1 mask 수정 대조 실험에는 기존 데이터와 손실을 유지하고 E2는 별도 계약과 새 run으로 연결해야 두 변경을 구분할 수 있다. 기존 smoke 5회 예산은 소진됐으며 재설정하지 않았다. 인계 프롬프트가 요구한 사용자의 **‘학습 시작’** 지시 전에는 추가 GPU 학습을 실행하지 않는다. 이는 하드웨어 사용에 대한 재승인 요청이 아니라 이번 인계의 실행 단계 경계다.

후보의 한계는 전제 8개 고정, 행당 query 1개, 작은 합성 그래프, 24개 문구, 시간·scope·문서·자유 개체명 미지원이다. 넓은 언어 일반화에 충분한 자료가 아니다. 기존 oracle의 global CONFLICT 정책도 그대로다. E2 학습 전에는 이 행 단위 계약을 실제 sampler의 누적·window 노출 기록으로 연결해야 한다.
