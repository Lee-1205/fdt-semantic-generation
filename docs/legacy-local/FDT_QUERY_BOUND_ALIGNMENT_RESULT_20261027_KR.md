# 디퓨전 query 결합 정렬 결과 — 2026-10-27

최신 사용자 지시에 따라 디퓨전 자체의 한계를 단정하지 않고 query·개체방향·초기상태의 결합을 먼저 수정했다. Full encoder/initializer/self-attention/관계 집계/모든 가중치는 그대로 두고, cross-attention에서 query i가 자기 recalled i와 initializer i 및 모든 유효 facts/null을 읽도록 mask를 결합했다. Gold는 입력하지 않았다.

처음 v1 실행은 빈 placeholder 데이터 manifest의0분모 오류로 초기화 중 FAILED, optimizer0/attempt0/forward0으로 종료됐다. Root가 모든 owned 종료를 확인하고 source·로그·실패를 보존했다. 별도 operational v2는 실제 pinned train/dev를 초기 manifest에 전달하는 부분과 경로만 고쳤다. 독립 diff review/19CPU검사/실제 preflight/validate를 통과했다. 원래 절대 종료13:41:39.809KST와 cutoff13:39:39.809를 유지했으며 시간을 초기화하지 않았다.

v2의 baseline이 사전94% 기준과 모든 보존 기준을 이미 만족해 BASELINE_ALIGNMENT_MET_NO_OPTIMIZER로 종료했다. 따라서 Adam/scaler를 만들지 않았고 optimizer 업데이트0/attempt0/훈련노출0이다. 총1368 실제 encode+denoise calls(baseline380+최종988), baseline6.485초/최종3noise13.047초이며 worker13:30:40/감독13:30:44KST 종료, 모든4개 owned/proxy 종료 확인. 예정64회 중 남은 횟수는 이 terminal을 다시 열어 사용하지 않는다.

| 관측 개발 패널 | 기존 full seed17 | 정렬 mask seed17 | 변화 | 정렬 후3noise 평균 |
|---|---:|---:|---:|---:|
|primary_querymix_dev64|92.383%|99.902%|+7.520%p|99.935%|
|uniform_dev64|99.805%|98.828%|-0.977%p|99.089%|
|old_role_dev128|96.272%|99.836%|+3.564%p|99.799%|
|old_normal16|99.583%|100.000%|+0.417%p|100.000%|
|old_large16|97.656%|100.000%|+2.344%p|100.000%|
|paired_new_ids16|92.083%|100.000%|+7.917%p|100.000%|

Primary는78개 과거 오류를 모두 고쳤지만 새 UNRESOLVED→LESS 오류1개가 생겨1023/1024다. 동일 query[7,6]의 과소근거 관계 생성은 seed17과43에서 재현돼 최종3noise에서도2/3072 오류로 보존했다. 방향 반전은 이 primary 관측에서 사라졌다. 일반/큰graph/번호교체 패널은100%, oldrole128도99.835%로 개선됐다.

중요한 한계는 uniform8query 패널이다. 99.8047→98.8281%로0.9766%p 하락했고, 기존1개 오류가 남은 채 새5개가 추가됐다. 최종3noise는14/1536 오류다. 기존2%p 보존 허용폭은 통과했지만 하락 자체를 성공으로 바꾸지 않는다. Grounding/initializer는 여전히100%이고 UNRESOLVED/LESS 구분의 잔여 오류가 있다. 완전히 정렬됐거나 모든 query 구성에서 좋아졌다고 선언하지 않는다.

이득은 학습이 아니라 같은 가중치의 연결 mask를 바꾼 구조적 차이다. 최종17/29/43 평균을 기존seed17과 섞지 않았다. 관측 개발 패널이며 같은 facts의 여러 view와noise가 상관돼 있고 고정 fuzzy closure/given roles/basis 도움을 포함한다. 미지 문서·언어/범용LM/NER/vNext전체/독립 neural 추론 개선으로 확대하지 않는다. E2 최종 holdout은 접근하지 않았다.

Root는409,065,431B weights-only artifact 전체SHA `80b9fa7b1c649c70fdff06b229c6b5a9fc7b16d70bb0fe4f0a818f9450d640db`와CPU safe-load, 원래 parent의 모든 model tensor가 정확히 같음을 확인했다. Model digest `b58f9724e69a96bf8bf6c4710b0fa3021be63a7d10a38dfae414a32d3470589e`. 새 optimizer trajectory가 없으므로 새 full Adam/RNG checkpoint도 만들지 않았다. 기존 fullstate parent `a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36`와 그 optimizer/RNG는 그대로 보존한다. 새 variant는 helperSHA와contract를 포함해 식별하며 원래 경로를 자동 교체하지 않는다.

다음 한정 과제는 CPU 자료 준비다. 기존 QueryMix1088개 및 명시적으로 기록한 관측112개 canonical graph와 겹치지 않는16개 신규8-active-entity graph를 만들고, 각 graph의 같은16query에서 prefix1/4/8/16 view를 만든다. 64view이지만16개 graph이며 독립표본64개가 아니다. 원래 oracle/독립BFS/hops/입력5key·mask·query-prefix·고유성·실제bytes를 검사한다. 모델 오답으로 선택하지 않는다. 이는 query수 변화와 미관측 graph에서 결합 정렬이 유지되는지 보기 위한 후보이며 추가 GPU/optimizer는 아직 준비·실행하지 않는다. 광범위평가를 반복하거나 동일 replay 학습을 늘리지 않는다.

근거: runs/query_bound_alignment_20261027_runtime_v2/root_result.json, baseline/final_evaluation raw, weights_only_baseline_verification.json, supervisor/final.json. 실패한 v1의 로그·원문도 보존했다.
