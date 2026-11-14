# FDT content 학습 결과 — 2026-09-30

## 판단

**학습과 기존 성능 보존은 완료, 실용 QA/응답 개선은 아직 실패.** 127회 추가 업데이트와 이전 1회를 합쳐 128회, 46,154 supervised tokens를 사용했다. 새 실행 921.52초, 2,340 forwards, 생성 1,399 tokens. 감독/worker 모두 종료했다. 모델 완성이나 자동 승격으로 처리하지 않는다.

| 관측 개발 패널 | 부모 → 최종 |
|---|---|
| 원래 alpha.20 fixed52 row NLL 대비 | 최종 +0.3114%, +1% 기준 PASS |
| 원래 alpha.20 fixed52 token NLL 대비 | 최종 -0.0514%, +1% 기준 PASS |
| 새 정상 Exact 전체 사슬 | 4/4 → 4/4 |
| 이전 정상 Exact 전체 사슬 | 3/4 → 3/4 |
| Exact negative 오승인 | 2/17 → 1/17 |
| 새 negative의 실제 Base 응답 정답 | 0/12 → 0/12, 공백 지속 |
| QA strict exact | seen 0/8, held 0/8, 이전 dev 0/4 유지 |
| QA 의미 검토 | seen의 `RIBA Stirling` 1/8은 상 식별로 인정, 전후 동일; held/dev 개선 없음 |
| 전체 63개 공백/EOS | 공백 32→33, EOS 28→26 |

두 자연어 출력은 주제를 이어가지만 반복/인물명·상황 이탈이 남고 64-token 상한에서 끝났다. EOS 수 감소를 독립 EOS 결함으로 단정하지 않는다. 오승인 하나 감소는 올바른 거절로 바뀐 것이며 뒤의 공백 답변은 여전히 실패다. 알려진 비복사 오승인 하나를 보존한다.

## 직접 확인한 증거

`execution/root_blind_judgment.json`을 sample→model key를 열기 전에 기록했다. root가 과거 원문을 이미 알고 있으므로 독립 blind 평가라고 부르지 않는다. 126 raw 출력/63 쌍을 직접 검토했고 `root_result.json`이 실제 사슬과 누적 CE를 집계한다.

승인된 정상 7개는 source 선택, first token, hard-copy cursor commit, 모든 cursor continuation, whole string 및 EOS가 통과했다. source 선택은 whole-identifier literal HYBRID이며 learned neural retrieval을 입증한 것이 아니다. 거절은 actual Base logits로 이어진다.

추가 127 step의 weighted token mass: QA 18,738, 설명 문단 41,187, 등록 응답 합계 1,483. 문단 약67%, QA 약31%, 응답 약2.4%다. 이것은 CE/gradient 비율이 아니다. `root_result.json`은 실제 content/EOS CE 합과 effective scalar도 별도 보존한다. 정답/응답 개선을 위한 학습 신호가 작다는 진단은 유력하지만 독립 인과 검증은 아니다.

## 보존 및 다음 단계

- 최종 가중치 SHA: `d8a1057b876855155af1f40f9e21706d7b9940def899aa9e1f6b2d2154c49dae`.
- 전체 model+Adam/RNG/cursor 복구 SHA: `5b75924568585f13a0f45e9f84f7957476ea799bf77dc2a5cc34b4089caf0afd`.
- 모든 실패 원문/체크포인트/기준을 유지한다. fixed52 성공은 V20 역사적 실패나 이전 Base97 실패를 없애지 않는다.
- 다음은 현재 보존된 가중치/Adam에서 별도 정답 중심 학습: 기존 train 후보의 아직 안 쓴 QA를 먼저 사용하고 QA·거절 후 응답의 토큰 가중치를 높인다. Exact가 담당하는 copy를 Base에 반복시키는 비중을 줄인다. 영어 문단과 동결 교사 replay를 남긴다.
- 새 최대512 updates/3600초 계약과 실제 토큰 수, 원래 +1% 보존 기준, 64/128/256/512 체크, full recovery 및 독립 감독을 동결한 후 한 번 실행한다. 기존 실행의 deadline을 연장하지 않는다. 데이터·가중치·LR 동시 변경의 독립 원인을 주장하지 않는다.
- QA1753/26기사 코퍼스는 좁고 나머지 annotation 전부를 semantic VERIFIED로 선언하지 않는다. 본 단계는 유용한 QA/응답 입증을 목표로 하며 범용 Base 완성이 아니다.
