# 실제8개체·균형 query 후보 — 2026-10-27

이 자료는 완료된 Role8의 제한된 개선을 바탕으로 만든 후속 CPU 후보다. 아직 모델에 사용하거나 새 optimizer를 실행하지 않았다. 같은64개 추상 그래프 replay의 연장이 아니라, 주어진 typed relation 과제에서 새로운 추상 그래프와 GREATER/LESS 신호를 늘린다.

## 실제 후보와 검증

권위 자료는 `research/artifacts/role8_active_balanced_20261027_v2/`다. Train1,024개/dev64개 서로 다른 추상 그래프이며, 모든 그래프에서8개 ID 모두 premise에 참여한다. 각 그래프는 실제 oracle의 같은 정답 class를 갖는 서로 다른 query8개를 선택한다. Train은 class별256그래프/2,048query, dev는 class별16그래프/128query다. 무작위 정답을 붙이거나 정답 class를 neural input에 넣지 않는다.

| 항목 | Train | Dev |
|---|---:|---:|
| 서로 다른 추상 그래프 | 1,024 | 64 |
| 전체 query | 8,192 | 512 |
| 저장 premise | 8,918 | 570 |
| 실제 collate 입력 UTF8 bytes | 105,337 | 6,601 |
| 생성 후보 검사 수 | 4,645 | 2,359 |

각 분할 최대50,000후보 상한 안에서 quota를 충족했다. Exact typed-graph key의 hash bucket이 전체 재명명 family의 train/dev/test 분할을 결정한다. Train/dev 추상·물리 중복0이며 이전 관측 일반16/큰16/Role8 source80의 합집합112개 추상 그래프를 제외했다. 같은 의미의 모든 과거 자료와 다르다는 보장은 하지 않는다.

Canonical key는 방향·관계종류별 color refinement 후 같은 color cell 내부 순열을 정확히 탐색한다. 최대8!=40,320회/그래프, 실제 관측 최대8회였다. CPU fixture는 ID 재명명 불변성 및 소수 graph 쌍의 전수 isomorphism 판정과 비교했다. 이는 통계적 표본 독립성의 증명이 아니다.

Root가 저장된1,088개 그래프에서 원래 Floyd-Warshall oracle와 별도 adjacency-list BFS의 label·최단 hops 일치, 모든 canonical key,112개 제외 집합, hash 분할, 클래스 quota, 실제 collate의5개 입력 key/byte 수를 독립 재구성했다. 관련 CPU7검사와 `root_cpu_review.json`이 PASS다. 모델 forward/optimizer는0, E2 최종 holdout 접근0이며 core source는 `eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352` 그대로다.

## 파일 신원

- Manifest **파일 전체** SHA: `2e67a3661f71545900934f0cd3717af58046848da3572f5ec884ca1230b0967e`.
- Builder SHA: `5433a8d9887c2fc3630bc2b8b8bb48852c2d7cc106701841776b45c71537f0a2`.
- Train SHA: `62b6326b251e0aeeca8a6d7eb19c513edd24f4beb3ef8a15c2276d9e97bd4628`.
- Dev SHA: `fdb790b99b7138dc5fe15cf019b9985c98acc1cd16a3ba7176a5ed9c9d57a657`.

Manifest의 `manifest_payload_sha256_without_self`는 self 필드를 제외한 payload hash로 파일 hash와 구분한다. V1은 원본 candidate와 code snapshot을 보존했다. V1의 보조 oracle가 원래와 같은 Floyd 계열이던 점과 독립성 표현/hash 명칭을 V2에서 고쳤다. BFS 수정 후 train/dev 내용과 hash는 V1과 동일했다.

## 후속 실행 준비와 한계

다음 별도 단계는 보존된 Role8 final `0b506644…b2784829` 가중치만 가져와 fresh Adam/scaler/RNG seed22, counter0으로 phase2를 최대256회 학습하는 계획이다. 원래 모델·whitelist·손실·LR3e-5를 유지하고 고정256success/280attempt/1,800초 reserve180/6,144 logical forwards, 최종평가150초, 매시도 fullstate 검증 저장을 준비한다. 이는 아직 봉인되거나 실행된 계획이 아니다.

Class 균형을 위해 graph별 query 정답이 동일하도록 선택한 편향이 있다. 후속 평가에는 동일한 새 dev64 사실에서16query를 고정 무작위 선택한 mixed-query view를 포함해 query 무시 shortcut을 점검한다. 이는 추가64개 독립 그래프가 아니다. 기존 mixed-query 일반·큰 그래프, ID 교체, 이전 Role8 dev도 보존한다. Same-seed baseline 대비 old/mixed 성능 저하·frozen 변화·수치/gradient 오류·저장/진행 정체·disk/deadline 시 중단한다.

새 worker·config·독립 supervisor·필수 CPU/실제 preflight·계획SHA·출력·자원·고정 상한이 모두 준비된 뒤에만 ONE 새 실행을 진행한다. 과거 terminal/deadline은 다시 열지 않는다. 자료 분포·노출량·학습 schedule이 함께 바뀌므로 한 요소의 인과 이득이나 범용 언어모델 완성으로 해석하지 않는다.
