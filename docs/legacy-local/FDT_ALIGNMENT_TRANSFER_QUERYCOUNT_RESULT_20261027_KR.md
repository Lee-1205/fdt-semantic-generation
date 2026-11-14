# 디퓨전 결합 정렬의 새 그래프·질문 수 전이 검사 — 2026-10-27

추가 학습 없이 같은 부모 가중치를 원래 full과 query 결합 mask에 각각 적용했다. 명시적으로 제외한1200개와 겹치지 않는 새 관계 graph16개를 고정해, 같은16개 질문의 prefix1/4/8/16을 비교했다. 총64view는16개 graph의 상관된 관측이며 독립표본64개가 아니다. 알려진 문구와 주어진 개체 역할을 사용했고 E2 최종 holdout은 접근하지 않았다.

| graph당 질문 수 | 활성 질문 위치 | 원래 full | 결합 mask | initializer |
|---|---:|---:|---:|---:|
|1|16|16/16 (100.000%)|16/16 (100.000%)|100%|
|4|64|63/64 (98.438%)|63/64 (98.438%)|100%|
|8|128|123/128 (96.094%)|128/128 (100.000%)|100%|
|16|256|237/256 (92.578%)|256/256 (100.000%)|100%|

8질문에서는 기존5개,16질문에서는 기존19개 오류가 모두 교정됐다. 이 관측에서 새 mask가 추가한 오류는 없었다. 같은 seed17과 batch별 원래16slot float32 잡음을 두 모델·모든prefix에서 그대로 재사용했고 root가 저장 잡음과 모든64행의 예측·정답·argmax를 독립 재구성했다.

잔여 오류는4질문에서 graph index8의 query[4,6]이다. 정답 GREATER를 두 정제기가 UNRESOLVED로 지웠다. 원문에는6<4라는 직접 사실이 있고 initializer는 옳았다. 같은 질문·동일slot잡음은8/16질문에서 맞는다. 따라서 기존16질문 개선이 새graph에서도 유지된다는 작은 증거는 얻었지만 질문 수에 따른 정렬이 완전히 해결된 것은 아니다. 앞선 uniform8의0.9766%p 하락도 그대로 보존한다.

보존한 원래 encoder/initializer/query self-attention/관계 집계의 영향과 time/noise 잠재 정렬은 아직 분리되지 않았다. 관측상 차이를 학습 이득이나 독립 neural 추론으로 부르지 않는다. 고정 fuzzy closure/given roles/basis, 네 sampling stratum의 선택 편향을 포함하며 모든 과거 자료에 대한 신규성·범용LM 향상·자동 승격을 주장하지 않는다.

실행은 ONE invocation/optimizer0/checkpoint0, 실제 encode+denoise160calls(별도readout64), owner16.261초로120초/200call 상한 안에서 COMPLETE였다. Proxy/worker exit0, root가 실제 native chain38708→14016→22492→14360의 생성·parent·command를 관측했고 종료 후 네 프로세스 모두 없음을 확인했다. 최종 두 model/buffer digest는 부모와 정확히 같다. 물리적 premise1671UTF8bytes, 두 모델×네 view 반복 encoder 노출13368bytes이며 LM token 수가 아니다.

다음은 남은4질문 누락의 time/noise·잠재 상태를 좁게 추적하는 CPU 준비다. 고정된 graph batch8–11의Q4/8/16과 저장한 동일잡음을 쓰는 passive trajectory 기록을 준비하되, 아직 새GPU·optimizer·봉인·launch를 승인하지 않는다. 오류를 관측한 뒤 고른 진단 사례임을 명시하고 새 성능 benchmark로 포장하지 않는다. 원래 full과 mask를 유지하며 neutral-init 대체 실험은 계속 보류한다.

근거: runs/alignment_transfer_querycount_20261027_v1/root_result.json 및 diagnostic_result.json, paired_seed17_noise.pt, supervisor/final.json.

PlanSHA 732efc57fb07b5524604762d94748601c1b259b582241f11112614179e589f63
ResultSHA 4ca65f7ce7384a40304160468b041fb5ace619af310c3e65181f28134c624a4b
NoiseSHA 254688fce9a8471e8e2251f26fbed5031f8b0af7310616766d1099a14f56b55c
