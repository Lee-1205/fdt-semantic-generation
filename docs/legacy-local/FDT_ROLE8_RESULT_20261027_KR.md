# FDT 102M 개체 번호 보강 결과 — 2026-10-27

학습은 128/128회 성공하고 모든 저장·보존 검사를 통과했다. 개체 번호 교체 정확도는 같은 noise에서 90.83→92.08%로 소폭 개선됐지만, 사전 목표인 5%p 개선에는 미달했다. 단순히 8개 ID를 노출하는 것만으로 정제기의 취약성을 해결했다고 볼 수 없다.

## 실행과 보존

`runs/role8_diffusion_20261027_v1`은 07:40:01 KST 시작, worker 07:45:07/감독 07:45:12 종료, COMPLETE/exit0/launch1이다. Root의 07:56 native 조회에서도 네 owned/proxy 프로세스가 모두 없었다. 고정 30분·140시도·4,096 logical forward 상한 안에서 128성공/128시도/AMP skip0, 실제 logical forward2,184회를 사용했다. 원래 deadline이나 terminal을 다시 열지 않았다.

계획 SHA `5a2ef7cd5fe6fcf78c217d9c4ad4c8c13989844cdbb559e78a07a942912331a8`. E2 모델 가중치만 가져온 새 Adam/scaler/RNG seed21 단계이며, E2의128회/E1의1,440회와 별도 counter다. 원래 phase2 손실과 구조를 유지하고 해당 whitelist만 학습했다.

최종 `model/complete_000130.json` / payload636,931,329B, SHA `0b5066444b9fdf0338219fba167edca8c2d211dbdda0bc7ea202bcc8b2784829`. Root CPU safe-load/전체 파일 hash 재검증을 통과했다. Model·populated Adam/scaler·sampler/noise/Torch CPU·CUDA/Python/NumPy RNG·실현 노출·history가 모두 들어 있다. E2와 tensor를 비교했을 때 동결 subset 변경0, phase2 변경103개였다. 검토 자체의 optimizer/forward는0이다.

## 사전에 정한 동일 noise17 비교

| 관측 개발 패널 | 전 → 후 정확도 | 전 → 후 macro-F1 |
|---|---:|---:|
| 16개 추상 그래프의 8회 재명명 dev128 | 95.340→95.121% | .88537→.88182 |
| 기존 일반 그래프16 | 98.750→99.167% | .97821→.98603 |
| 기존 큰 그래프16 | 96.094→97.266% | .88784→.91496 |
| 동일 그래프에서 ID0↔6·1↔7 | 90.833→92.083% | .85999→.87498 |

모든 panel의 grounding/initializer는100%다. Step32/64/96/128의 기존 성능 보존 fail streak는0이며 사후 기준을 완화하지 않았다. ID 교체 +1.25%p는 사전 목표 +5%p에 미달했다. Dev128은 같은 seed에서 소폭 악화했으므로 전체 개선으로 요약하지 않는다.

최종 noise17/29/43에서 일반98.889%, 큰 그래프96.875%, ID 교체91.806%, dev12895.340%였다. 같은 checkpoint·입력·noise의 기존 부모 3seed proof와 비교하면 일반98.472→98.889%, 큰 그래프96.094→96.875%, ID 교체90.278→91.806%다. 이때 seed들은 같은 그래프 내 반복이고, 서로 다른 seed 수의 평균을 단일 seed baseline과 직접 비교해 개선을 만들지 않는다. E2 최종 holdout은 사용하지 않았다.

## 실제 노출과 남은 오류

후보 train512는 E1에서 본 추상 그래프64개를8회씩 재명명한 상관된 자료다. 모델이1,024개 행을 샘플링했으며 고유 그래프1,024개를 학습한 것이 아니다. 실제 query label 노출은 UNRESOLVED4,032/GREATER910/LESS842/CONFLICT8,472였다. GREATER·LESS를 합쳐 약12.3%다. 기존 inverse class weighting은 유지했으나 weight mass를 실제 CE/gradient 기여로 부르지 않는다.

각 ID의 실제 fact source/object 노출은617~713회, query source/object는1,719~1,844회였다. 저장된 batch indices로 재구성한 결과가 payload counter와 정확히 일치했다. 입력68,347 UTF8 bytes, 학습 step 시간 합계23.46초, preclip gradient norm 최대5.17이었다. 이 step 시간에는 입력 준비와 동기화도 들어가므로 GPU kernel만의 시간은 아니다. 전체 wall 약311초에는 매회 fullstate 검증 저장과 평가가 포함된다. 수치 폭주·gradient 소유 오류·AMP skip는 없었다.

Root는 최종 panel별 raw를 읽어 오류 유형을 집계하고 각4개 사례를 보존했다. Dev128의255개 오류 중 GREATER→LESS82/LESS→GREATER58, ID 교체의59개 오류에도 방향 반전·근거 없는 관계·관계 누락이 남았다. Initializer가 맞던 답을 diffusion이 틀리는 사례다. CONFLICT의 F1은1.0으로, 전체 정확도만으로 방향 판정 문제가 가려질 수 있다. 주어진 개체 역할·고정 fuzzy closure·고정 target basis도 결과에 기여하므로 순수 neural 추론이나 diffusion의 독립적 이득이라고 주장하지 않는다.

## 다음 한정 작업

같은64개 replay를 단순 연장하지 않는다. 별도 CPU 후보에서 실제8개 ID가 모두 premise에 참여하는 새로운 추상 그래프와 네 query class의 균형을 확대한다. 기존 관측 패널과 그래프 family를 분리하고, oracle/독립 path/입력 whitelist/노출·중복·분리의 필수 계약만 검사한다. 이 변경은 data coverage·분포를 함께 바꾸므로 단독 인과효과라고 하지 않는다.

현재 run은 terminal이며 재시작하지 않는다. 다음 optimizer는 새 데이터·코드·계획SHA·출력·고정 시간/노출 상한·중단 기준·독립 감독·실제 자원을 준비한 별도 실행으로만 진행한다. 유용한 최종 가중치와 Adam/RNG/실패·원문을 보존한다. C: 관측 여유117,079,146,496B로20GiB 하한 이상이며 삭제는 필요하지 않다. Windows 임시 sleep 방지 요청은 worker 종료 시 해제됐다.

기계 판정은 `runs/role8_diffusion_20261027_v1/root_result.json`, 원문은 `baseline/`, `diagnostics/`, `final_evaluation/`에 있다. 이 연구 결과로 범용 LM·NER·vNext 전체 완성을 선언하지 않는다.
