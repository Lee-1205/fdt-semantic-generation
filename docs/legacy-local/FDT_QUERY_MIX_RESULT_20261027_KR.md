# QueryMix 학습 결과 — 2026-10-27

256/256 성공, AMP skip0, worker와 독립 감독이 모두 exit0으로 완료됐다. 최종 저장은 09:17:38 KST, 감독 종료는09:17:43 KST이며 root가12:29 이후 native Windows 프로세스 전체에서 owned4개 종료를 확인했다. 저비용 agent는 사용량 제한으로 중도 보고가 멈췄지만 독립 OS 감독과 worker는 완료했다. 현재 학습 중이 아니다.

Root는 최종638,635,329B fullstate 전체 hash·CPU safe-load·populated Adam·scaler·모든 RNG·history·동결 tensor 불변·실제 노출을 확인했다. 최종SHA `a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36`. 평가가 가리키는 milestone256/최종평가 전 payloadSHA는 `5ea1052b610c07bb61189fbad7d352bf27df361cdaa163ce8d9913f351e76cb8`이며 최종평가 후 payload와 model tensor digest가 정확히 같다. Metadata가 달라 fullstate SHA가 다른 것을 모델 차이로 해석하지 않는다. 원래 모든 milestone64/128/192/256과 부모/과거 실패를 보존했다.

| 관측 개발 패널 | 학습 전 seed17 | 256회 뒤 seed17 | 최종3noise 평균 |
|---|---:|---:|---:|
| primary_querymix_dev64 | 89.551% | 92.383% | 92.415% |
| uniform_dev64 | 100.000% | 99.805% | 99.935% |
| old_role_dev128 | 95.779% | 96.272% | 96.637% |
| old_normal16 | 99.583% | 99.583% | 99.444% |
| old_large16 | 97.266% | 97.656% | 97.526% |
| paired_new_ids16 | 92.083% | 92.083% | 92.778% |

Mixed primary는89.5508→92.3828%(+2.8320%p), macro-F1 .88968→.91870이다. +3%p 개발목표와94% qualification 모두 FAIL로 보존한다. 완료는 예정된 학습 횟수 완료를 뜻하며 능력 목표 달성이 아니다. 각64/128/192/256의 보존 검사는 모두 PASS다. Uniform64는 같은 사실의 상관된 view이며 fresh blind/final이 아니다. 최종3noise 평균을 seed17 baseline과 섞어 이득을 계산하지 않는다. E2 최종 holdout에는 접근하지 않았다.

같은1024개 primary query의 오류는107→78개다. 45개를 고쳤고16개가 새로 틀려 순29개 감소했다. 최종78개 중 방향 반전41, 근거 없는 양성21, 관계 누락16개다. Grounding과 initializer는 모든 패널100%였으므로 refiner의 출력 손상이 남는다. Typed role 입력, 고정 fuzzy closure와 target basis의 기여가 크며 순수 neural 추론·범용 LM·NER·vNext 전체 완성의 증거가 아니다.

실제2,048행 노출/883개 고유 train graph/32,768query, class별9,973/7,660/7,439/7,696, premise209,010 UTF8bytes였다. 전체1024개 train graph를 모두 학습한 것이 아니며 숫자 query ID를 LM token으로 더하지 않는다. 전체3912 logical forwards, worker681.793초/학습 step 계산46.092초, 최고 clip전 gradient norm1.696, 추가 root검토 optimizer0/forward0이다.

다음 한정 작업은 별도 fixed-budget readonly query-context 검사다. 사전 고정한 관측 dev 첫8graph에 같은 초기 noise를 사용하고 full16·정렬을 되돌린 reverse16·같은 전체 encoder 결과를 slice한 single-query를 비교한다. Single은 self-attention/context/aggregation이 함께 바뀌는 조작이므로 단일 원인 확증으로 부르지 않는다. Gold는 점수 계산에만 쓰고 모델 입력에는 넣지 않는다. 기존 core·봉인 run은 수정하지 않는다. 이 QueryMix replay optimizer를 재연장하거나 terminal/deadline을 다시 열지 않는다.

근거: run/root_result.json, final_status.json, supervisor/final.json, evaluation_summary.json, 각baseline/diagnostics/final_evaluation raw.
