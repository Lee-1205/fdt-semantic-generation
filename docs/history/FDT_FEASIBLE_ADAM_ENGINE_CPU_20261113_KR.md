# V90 — 실제 Adam 변화의 방향 보정과 원래 reference를 연결한 CPU 엔진

2026-11-13. V89의 선형 보정식을 실제 AdamW·enabled GradScaler·유한 source cursor에 연결했다. **새 연구 모델 학습이나 문장 복원 개선 결과는 아직 아니다.** 이전 V88의108attempt/16수락/92거부와 원문·결과·가중치·Adam/RNG를 보존했다.

## 바꾼 부분

`research/feasible_adam_transaction_v90.py`는 누적 gradient를 한 번 unscale/clip한 뒤 실제 AdamW가 제안한 두 tensor의 변화를 얻는다. 이 변화에는 momentum, adaptive scale, weight decay가 포함된다. 원래 proposal이 전체 TRAIN reference를 통과하면 그대로 수락한다. 거부되면 이미 계산한 위반 위치를 이용해 한 번의 제한된 방향 보정을 시도한다. 같은 방향을 여러 scalar로 줄이는 V88 방식의 단순 재시작이 아니다.

V89 계약의 최대2회 sweep·16개 활성 제약·원래 proposal 이하의 보정 크기·수치 여유1e-6을 유지한다. 전체 위반 위치가17개 이상이면 보정용으로16개만 골라 성공 처리하지 않는다. 보정을 거부하고 전체 개수와 앞16개 진단만 남긴다. 활성 제약을 만족시킨 뒤에도 **전체 유한 TRAIN bank를 다시 통과해야** 수락한다. 한 group의 reference projection은 최대2회이며, 재귀 보정·추가 optimizer step·상한 확대는 없다.

수락된 보정은 원래 Adam proposal의 moment를 유지한다. 따라서 보정된 weight와 proposal moment를 사용하는 명시적인 projected Adam 알고리즘이며, 변경 없는 일반 AdamW라고 부르지 않는다. 전체 거부 시 weight/Adam을 복구하고 유한 attempt의 scaler 진행은 소비한다. 예외·STOP에서는 weight/Adam/scaler까지 복구하며, 이미 소비한 source/cursor/RNG/call은 되감지 않는다. gradient도 정리한다.

`research/diagnostic_reference_v90.py`는 같은 full projection의 결과에서 source ID, 원문 byte offset, gold/rival, gap, 원래 floor와 위반 종류를 기록한다. 진단을 위한 추가 모델 forward는 없다. 보정 후 다른 경쟁 class나 문맥에서 새 위반이 생기면 두 번째 전체 판정에서 거부된다. 보정이 전체 교집합의 최적해이거나 실제 V88의92회 거부를 해결했다는 주장은 하지 않는다.

## Warm head와 원래 teacher를 분리

`scripts/feasible_critical_engine_v90.py`는 새로운 warm head와 **원래의 고정된 TRAIN teacher/reference**를 서로 다른 identity로 취급한다. 원래 feature·target·floor 파일과 SHA를 그대로 검증하고, 감소한 warm margin에서 새 하한을 만들지 않는다. 실제 V80 원래 anchor의152개 취약 critical floor,119,769개 top1 보호 위치,7,623개 patch를 ROOT가 다시 읽어 원본 SHA와 일치를 확인했다.

입력·memory binding·readout prefix가 같은 경우 기존 teacher target을 훈련용 static 자료로 재사용할 수 있게 했다. 이 재사용을 새 teacher encoder forward나 새 문서 노출로 세지 않는다. 다섯 gradient micro에서는 여전히 실제 frozen encoder를 실행한다. teacher target, span, gold 길이·접두사, source별 reference를 NN inference feature로 넣지 않는다. 이 TRAIN bank는 네 번째 inference memory가 아니다.

현재 엔진은 operator signature를 검증하는 계약을 받는다. 실제 worker는 전체 warm tensor와 원래 표현 operator의 SHA를 직접 확인해야 한다. metadata가 같다는 이유만으로 실제 표현 계산을 자동 증명하지 않는다. K_P/K_C 경로와 NULL인 K_E, copy0, denoise0 원칙도 유지한다.

## 확인한 범위

- 새 transaction/reference CPU 검사11개 PASS. 실제 CPU AdamW와 enabled scaler로 보정 수락·다른 문맥 손상 시 전체 거부·17개 위반 상한·weight/Adam/scaler 변조·보정 판정 뒤 STOP·AMP skip을 확인했다.
- tiny synthetic encoder와 실제512 hidden/마지막 Linear2의2,117,664 parameter shape를 연결한 엔진 검사가2.531초에 통과했다. 7개 저장 override와2개 optimizer tensor를 구분하고, populate된 Adam·scaler·Torch/Python/NumPy RNG의 다음 group 일치를 확인했다. 약25.65MB의 fullstate를 메모리에서 직렬화/safeload/재구성했다.
- encoder 전, encoder 후/readout 전, 한 micro 후, 실제 Adam proposal 후의 STOP 상태를 정확하게 저장·복구하고 자동 group 재생을 거부했다. 부분 encoder 노출과 완료된 backward micro를 별도로 기록한다. 거부와 AMP skip도 source/attempt를 소비한다.
- ROOT가 실제 기존 anchor의 파일·floor·105개 표현 operator SHA와 CPU 증거를 독립 확인했다. Adam의 첫 proposal을 weight decay·epsilon까지 별도 계산한 선형 예에서 A gap−0.100199997을0.050001로 보호하면서 B gap0.175300496을 얻었다. 이것은 합성 계산이며 새 영어 문장 성능이 아니다.

실제 학습된 FDT forward0/update0/GPU0, CUDA 초기화false다. CPU fixture의 실제 optimizer 작동을 연구 가중치 업데이트로 세지 않는다. 실제 CUDA 재시작, 전체123M 모델의 새 preflight, 실행 가능한 새 worker/owner/plan/seal은 아직 없다. V88의 거부 candidate weight/logit/ID는 저장되지 않았으므로 같은 실행을 로그 때문에 재평가하지 않는다.

## 다음 한정 실행의 범위

다음은 V88 가중치에서 fresh2Adam으로 시작하고 원래 V80 anchor와 teacher를 재사용하는 별도 worker다. 기존 V88 final194행을 warm source와 identity가 일치하는 저장 baseline으로 검증해 사용할 수 있으며, 새 baseline NN 평가라고 세지 않는다. 최종194행은 새 모델의 출력으로 다시 측정하되 이미 관측한 개발/장문 패널이라는 한계를 유지한다. 장문 입력 재인코딩 없이 고정 Commit을 사용하는 출력 비교다.

CPU 비용 초안은 새96attempt/최대480 live TRAIN 행, head encode480+readout480, 원래 reference 확인·terminal 포함최대194, final cached readout194로 최대1,348call이다. 1,400call/600초 owner/480초 model cutoff/120초 reserve/STOP·임시 저장 포함144MiB를 실제 코드 검토 대상으로 제안했다. **아직 실행 plan이나 봉인된 예산은 아니다.** 기존 teacher1024행 재사용은 정적 자료 이용이며, 이전 실행의 미사용 call/time/update 재사용이 아니다. 32회 연속 미수락 중단 기준과 원래 floor를 자동 완화하지 않는다.

C:100GiB 보호를 유지한다. 실제 시작에는100GiB+8GiB headroom+새 전체 output 여유가 필요하다. 현재 약108GiB만으로 위 초안의 시작 여유가 충족된다고 가정하지 않는다. 원본·유용 가중치·Adam/RNG·실패·게임을 삭제하지 않았다.

V88의192개 짧은 출력 불변, 장문52→49 byte 오류/4교정·1새 `drawer→draber`와 남은 의미 오류는 그대로다. 언어→의미→언어의 완성이나 성능 향상을 선언하지 않는다. 확인된 출력 update 문제부터 해결하며, 향후 사고 diffusion에서는 입력·memory·binding·출력 codec 전체와 buffer를 고정하고 별도 사고 가중치만 학습한다.

증거: `research/artifacts/feasible_critical_engine_20261113_v90_cpu_v1/root_engine_cpu_review.json`, `root_independent_review.json`, `evidence_SHA.json`, `root_independent_evidence_SHA.json`.
