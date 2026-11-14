# 다음 실험: 준비와 실행 구분

| 단계 | 준비된항목 | 사전통제/통과기준 | 실제상태 |
|---|---|---|---|
| E0로컬동치·복구 | baselineconfig/CLI/원본archive/fullcheckpoint/복구proof | originaltinyforward동치,CUDA102Mfinitegradient,5attempt/600s이내,독립nextupdate동일 | 통과. 성능향상실험아님 |
| E1mask | masked102M/원래data·loss/1440updates/fullstate복구 | 원래ground.9/semantic.8gate,저장·수치안전 | 1440/1440 COMPLETE. Initializer100%,diffusion96.09~98.61%; matchedbaseline미실행이므로mask단독이득미확정 |
| E2표현·부정/균형 | v1후보/교차패널 및 lexicalv2 train512/dev64 | balancedground/family/typedgraph분리; olddev승격명시; final보존 | 128/128 COMPLETE. 학습편입 표현37.5→100%; 새dev64는부모부터100→100으로일반화향상미입증 |
| 개체번호 보강 | Role8 seen64base×8변형/phase2 128회 | 동결subset/기존성능보존, 동일seed ID교체 +5%p목표 | 128/128 COMPLETE. ID교체90.83→92.08%(+1.25%p),목표미달; 큰그래프96.09→97.27%,dev128소폭하락. 새고유8개체/균형자료 CPU준비 |
| 혼합 질문 QueryMix | 기존1024/64facts유지·16비중복queries/phase2 256회 | same-seed17 +3pp·94% qualification 및 기존성능보존 | 256/256 COMPLETE. Mixed89.55→92.38%(+2.83pp),두성능목표FAIL/보존PASS. 동일replay연장금지; paired query-context readonly CPU준비 |
| 질문 간섭 통제 | fixed8 observedgraphs·matchednoise/full-reverse-single | 원본 예측 일치·정답 입력 배제·optimizer0 | COMPLETE26actualcalls. Full/reverse117/128, single/init128/128. 원인 단정 없이 query-local CPU후보 준비 |
| Query-bound 정렬 | Full encoder/관계/selfattention 보존·crossattention 자기slot결합 | 같은weights·seed17/보존·0SGD규칙 | ZERO-SGD COMPLETE. Mixed92.38→99.90/번호교체100%,uniform8은99.80→98.83% 하락 보존. 신규graph/query수 전이 CPU자료준비 |
| E3독립이득 | role/no-role,fuzzy대체,initializer/diffusion/recurrent통제표 | sameparent또는sameinitialization/optimizer/data/실현노출/compute;graphclusterCI+seed분산 | 미실행 |
| E4손실 | 현재loss보존·MSE/Huber/negative-margin후보 | 한요소씩변경,clip전norm/loss분모/scale기록,softmaxCE와margin분리 | 60/15/25총1000회도미실행가설 |
| E5vNext | source명세와현재adapter대응표 | oracleOutcomeCodec정보보존→Commit/Exact権限→refiner→전체integration순서 | 전체구현미완료 |

장기실험예산은사용자'학습 시작'지시후새run/새고정time-attemptcap으로정한다. 기존smoke5회는모두소진됐으며추가미니실험을그예산에몰래붙이지않는다. 완성LM·범용추론·현재미학습문구성능향상은아직주장하지않는다. 원본480/320/640계획과과거2240업데이트보고를새run과혼합하지않는다.

사용자는 이후 실제 학습 시작과 지속 검사·이상 수정까지 명시 승인했다. E1은 완료됐고, 다음 별도 한정 계획은 이 승인 범위에서 준비한다. 읽기 전용 DDIM4/8/16 진단에서는 큰 그래프 개선이 없었다. 상세는 `FDT_E1_LEARNING_RESULT_20261026_KR.md`.

E2 후보의 실제 수치·교차 패널·한계는 `FDT_E2_DATA_READINESS_20261026_KR.md`에 기록했다. E1에서는 기존 데이터를 유지하고 E2와 동시에 바꾸지 않는다. 후속 후보를 매 heartbeat마다 재생성하거나 final을 모델 선택에 사용하지 않는다.
