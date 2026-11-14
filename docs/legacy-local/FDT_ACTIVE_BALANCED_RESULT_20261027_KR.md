# 균형 그래프 학습 결과와 중단 이유 — 2026-10-27

이번 단계는 128회 성공/128회 시도에서 사전 중단 조건에 따라 종료됐다. 혼합 질문 정확도는88.57→89.55%로 개선됐으나, 고정94% 하한에 두 번 미달했다. 기존 성능 저하나 수치 발산 때문에 멈춘 것은 아니다. 이 단계의 STOPPED 판정과 실패 기록을 그대로 보존한다.

## 실제 실행과 복구

권위 실행은 `runs/active_balanced_diffusion_20261027_runtime_v3`, plan SHA `550858de33fa1056a3b166a478095de9dafc010b3ec4017d92d0c176519b370a`다. Worker 08:27:12 KST 시작, 08:32:23 종료, supervisor08:32:25 STOPPED/exit0/launch1이다. Root08:44 native 조회에서 네 owned/proxy 프로세스가 모두 없었다. 원래08:47:57.644 절대 종료는 연장하지 않았다.

앞선 두 operational 실패도 보존했다. V1은 supervisor 로그를 worker 예약 경로에 만들어 Popen 전에 거절됐고, launch_v2는 fresh enabled GradScaler state가 비어 있어야 한다는 잘못된 검사 때문에 초기화에서0update로 끝났다. V3는 실제 CPU GradScaler 기본값을 검증하는 회귀검사를 포함하고, worker/supervisor를 새 분기로 기록했다. 두 실패는 optimizer0이며, 이전 모델 가중치·데이터·손실을 바꾸거나 old terminal을 다시 열지 않았다.

실제 v3는 Role8 final 모델 가중치만 가져온 fresh Adam/scaler/allRNG seed22 단계다. 최종 marker `model/complete_000131.json`, payload637,459,201B, SHA `cf85d259de333b25a05f52e5ac0ad3711e02154bdf176d58adf028a07ddcaa5f`. Root 전체 hash·CPU safe-load·populated Adam/scaler·모든 RNG/history/노출을 확인했다. 모든 동결 tensor는 부모와 같고 AMP skip0이다. Step64/128 fullstate milestone도 보존했다.

## 동일 noise17의 개발 결과

| 패널 | 부모 → step128 정확도 | 부모 → step128 macro-F1 |
|---|---:|---:|
| 같은 정답 query8개를 선택한 새 dev64 | 99.609→100% | .99609→1.00000 |
| 같은 사실의 무편향16query mixed64 | 88.574→89.551% | .87971→.88968 |
| 기존 Role8 dev128 | 95.121→95.779% | .88182→.89267 |
| 기존 일반 그래프16 | 99.167→99.583% | .98603→.99214 |
| 기존 큰 그래프16 | 97.266→97.266% | .91496→.91924 |
| ID0↔6·1↔7 paired | 92.083→92.083% | .87498→.86786 |

모든 grounding/initializer는100%다. 이들은 관측 개발 패널이며 mixed64는 새64개 독립 그래프가 아니라 동일 사실의 query view다. 중단 이후3-noise 최종 평가는 실행하지 않았다. Step256에 대한 원래+3%p 목표도 완료 판정하지 않는다. 특히 uniform dev의 부모가 이미99.609%여서 +3%p는 ceiling 때문에 불가능했다.

Mixed64는 step64에서89.258%, step128에서89.551%였고 두 번 모두 개선됐다. 원래 guard `accuracy < max(.94, baseline-.02)`가94%를 요구해 두 관측이 실패했다. Baseline-relative 정확도·macro-F1 기준으로 보면 모든 패널이 허용 범위다. 역사적 reason `OLD_RETENTION_FAILED_TWICE`를 바꾸지 않되, 정확한 해석은 **혼합 질문 절대 qualification 하한 미달**이다. 이를 과거 통과로 소급 처리하지 않는다.

## 데이터 실효성과 원문

이번 후보의 train1,024개 서로 다른 추상 그래프 중 실제 sampler가 사용한 고유 그래프는636개다. 1,024행 노출, query class 노출은2,024/2,016/2,160/1,992로 균형에 가까웠다. 저장된 batch indices를 통해 ID·class·고유 graph 카운터를 정확히 재구성했다. 입력105,140 UTF8 bytes, 저장·평가를 제외한 step 시간 합계23.08초, preclip norm 최대0.60이었다.

원문에서 mixed64의1,024query 중107개가 틀렸다. 방향 반전47개, 없는 관계 생성30개, 실제 관계 누락30개였다. Initializer가 맞던 query를 diffusion이 틀리는 사례가 남는다. 같은 정답8query 패널100%와 무편향16query 패널89.55%의 차이는 query 구성에 대한 취약성을 보여주지만, query 수·위치·noise/slot·문맥 변화가 섞여 있어 한 원인의 인과 효과라고 단정하지 않는다. 고정 fuzzy closure·역할·basis도 기여하므로 정제기의 독립적 추론 이득이나 범용 LM 완성으로 포장하지 않는다.

## 다음 한정 작업

기존 v3의 남은128회를 그대로 재시작하지 않는다. 같은 train1,024/dev64의 사실·graph family·split을 유지한 별도 query-mix 후보를 CPU로 준비한다. Train은 seed33017, dev는 기존 mixed 패널과 같은 seed22017로16개 query를 무편향 선택하며, 정답이 uniform하다는 행 단위 가정을 제거한다. 원래 전역 CONFLICT 규칙은 유지한다. 실제 label/hops·입력5key·원문 byte·노출을 검증하고, dev는 이미 관측한 패널이라고 명시한다. 새로운 graph 수가 늘어난 것은 아니다.

후속 계획에서는 **보존 기준**(자기 부모 baseline 대비 허용 하락)과 **성능 목표**(혼합 질문94%)를 별도로 사전 등록한다. 이는 과거 실패를 변경하는 것이 아니다. 같은 자료·하한을 바꿔 반복 통과시키지 않고, 실제 혼합 query 학습을 한 번의 새 고정 한도 curriculum으로 검증한다. 새 계약/코드/출력/계획SHA/고정 시간·노출 상한/중단 기준/독립 감독/자원 확인 전에는 optimizer를 시작하지 않는다. 원래 source/archive/core/Phase2/실패/raw/이전 가중치·Adam/RNG/E2 final을 보존한다.

기계 증거는 `runs/active_balanced_diffusion_20261027_runtime_v3/root_result.json`, 원문과 milestone은 해당 run의 `baseline/`, `diagnostics/`, `model/`에 있다.
