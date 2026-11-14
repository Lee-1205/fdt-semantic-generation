# 디퓨전 초기 잡음·잠재 상태 정렬 추적 — 2026-10-27

관측된4질문 누락을 기준으로 graph batch8–11의 Q4/8/16을 고정하고, 원래full과 query 결합 mask의 DDIM 상태를 수동 변경 없이 기록했다. 추가 학습·새 잡음·구조 ablation은 없었다. 같은 부모 가중치와 이전에 저장한 잡음을 썼다. 이 batch는 오류 관측 뒤 고른 진단 사례이며 새로운 성능 benchmark가 아니다.

Root가 모든112개 활성 질문의 초기/4단계 clean projection을 읽었다. 저장된 이전 실행의 최종 예측224개(두 모델)는 모두 같았다. Batched readout 모양이 달라 최대logit차이는1.43e-6이며 bit-exact라고 하지는 않는다. 저장 tensor의 DDIM 식과 고정 basis readout을 CPU에서 독립 재구성했고, 모든 noise/mask/step연결·finite 검사를 통과했다.

| query[4,6]의 구성 | initializer | clean@t1 | clean@t.75 | clean@t.5 | 최종clean@t.25 |
|---|---|---|---|---|---|
|Q4 원래full|GREATER|LESS|UNRESOLVED|UNRESOLVED|UNRESOLVED|
|Q4 결합mask|GREATER|UNRESOLVED|UNRESOLVED|UNRESOLVED|UNRESOLVED|
|Q8 원래full|GREATER|LESS|LESS|GREATER|GREATER|
|Q8 결합mask|GREATER|GREATER|GREATER|GREATER|GREATER|
|Q16 원래full/결합mask|GREATER|GREATER|GREATER|GREATER|GREATER|

남은 Q4 누락은 마지막 한 단계의 미세한 오차에서 시작하지 않았다. 결합 mask의 첫 순수잡음 clean 추정부터 GREATER를 잃었고, GREATER 대 나머지 최고값 margin은 초기+3.6487에서−0.1722/−0.5748/−1.3023/−1.2505로 변했다. 같은 질문·사실·slot잡음의 Q8/16은 유지된다. 다만 양의잡음 시점의 clean readout은 내부 진단용 추정값이고 일반 답변 성능으로 해석하지 않는다. 이 증거만으로 self-attention/관계 집계/잡음 분포 중 하나를 원인으로 확정하지 않는다.

실용적으로 보완할 공백도 확인했다. 저장된 Role8 train512는12/16질문 각각256행, active-balanced train1024는전부8질문, QueryMix train1024는전부16질문이다. 원래E1 생성기도 n4–6의 nonself query를16개로 자르므로12/16질문만 만든다. 이 계약들에는4질문 정제 학습 문맥이 없다. 질문 수 분포의 공백이 관련 가설이며 원인 증명이나 성능 개선은 아직 아니다.

다음 단계는 기존train/dev graph의 같은facts를 유지하면서 질문 수1/4/8/16을 명시적으로 학습할 수 있는 CPU 자료 준비다. 기존 graph와 query prefix를 재사용하며 새로운 독립 graph라고 부르지 않는다. 현재transfer16개는train에 넣지 않는다. 원래full/결합mask/과거가중치는 보존하고, 새 학습은 별도 data/code/planSHA/output·고정 상한·중단 기준·owned 감독을 검토한 뒤만 한다. 지금 추가GPU/optimizer를 실행하지 않는다. 같은 오류의 추적이나 broad baseline을 반복 선행조건으로 삼지 않는다.

실행은 ONE/COMPLETE/worker·proxy exit0/모든owned40764→32668→30948→15704 종료, owner14.650초, encode6+denoise24=30calls/별도readout6이다. Trace17,534,983B는128MiB 안이며 두 model/buffer digest와 global torchRNG는 불변이다. Optimizer0/checkpoint0/E2final접근0. 원래90초/40call terminal은 닫혀 남은 시간을 재사용하지 않는다.

PlanSHA 3983d42007e76d5f0b162a85d73f1d09f7fe5df19c5d32ce324b084462debc04
ResultSHA d90f96d80f74af3da0e273f30cfc66b3459040032ec5264fab33dd858fe60829
TraceSHA 145de1dc65ba323d1b76f1a90b0e06fef76e0af543dbf91efc713502e923c6d6

근거: runs/alignment_trajectory_20261027_v1/root_result.json, diagnostic_result.json, trajectory_trace.pt, code_snapshot/root_native_observation.json, supervisor/final.json.
