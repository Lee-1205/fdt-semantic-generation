# FP32·핵심 위치 보존 학습 엔진 — CPU 결합 검증

2026-11-13. V85의 reference bank를 실제 teacher/gradient/Adam/scaler/복구 흐름에 연결하는 엔진을 작성했다. **현재 연구 모델을 추가 학습한 결과는 아니다.** 실제 연구 가중치는 V80 그대로이며 V75도 보존했다.

추가한 코드는 다음 역할을 맡는다.

- `research/fp32_ordered_readout_v86.py`: 기존 fuzzy 역할 결합, 파라미터 이름과 값을 유지하면서 출력 계산을 고정 FP32 경로로 연결한다. 입력 추출은 기존 경로를 유지한다. 원문·정답·정답 길이·Exact 복사를 출력기에 주지 않는다.
- `scripts/critical_bank_engine_v86.py`: 유한 teacher pass에서 실제 FP32 prefix를 잡아 별도 loss-only teacher packet과 bank를 만든다. 5개 source micro의 gradient를 같은 파라미터에서 합친 뒤 한 번 unscale/clip/Adam proposal을 수행한다. 최종 Linear 2개 tensor만 학습 대상으로 두고, 실제 update의 TRAIN 보존 판정이 실패하면 정해진 damping 후보만 확인하고 weights/Adam을 되돌린다. reject와 AMP skip도 attempt/cursor/노출을 소비한다.
- `research/critical_reference_device_v86.py`: CPU 계약과 동일한 reference 판정을 명시적인 device tensor 연산으로 구현한다. 정밀도를 바꿔서 통과시키거나 정답을 NN feature로 쓰지 않는다.
- `research/critical_bank_storage_v87.py`, `scripts/compact_critical_engine_v87.py`: 완성된 bank는 feature와 loss-only target 파일을 분리해 한 번 저장한다. 이후 fullstate에는 해당 파일의 SHA 참조를 넣고, partial teacher 상태는 생성 중인 bank를 그대로 담는다. source range와 patch 순서, calibration, 모든 tensor를 재구성한 digest가 같아야 한다. 같은 bank를 체크포인트마다 중복 저장하지 않기 위한 구성이다.

CPU 결합 검증에서는 작은 언어 인코더와 **hidden 512 / 마지막 Linear 2,117,664 parameter**를 사용했다. 실제 123M 연구 인코더를 실행한 검사가 아니다. CPU fixture에 실제 Adam과 enabled GradScaler를 사용했으며 다음을 확인했다.

1. teacher 중간 상태와 reference 초기화 경계의 정확 복구.
2. 7개 readout tensor 저장과 실제 optimizer 소유 2개 tensor의 구분. 동결된 prefix 5개를 빼서 저장하려는 경우 거부.
3. populated Adam, scaler, Torch/Python/NumPy RNG를 복구한 뒤 다음 그룹 결과가 연속 실행과 재귀 digest까지 동일함.
4. AMP skip, 네 damping 후보 모두 reject되는 경우의 weights/Adam rollback과 scaler/cursor/노출 소비.
5. teacher 요청, teacher packet 저장 직후, head 한 micro 직후, 실제 proposal 중 STOP 상태의 정확 복구와 자동 재실행 금지.
6. CPU reference callback과 V85 판정의 일치, 별도 feature/target 파일을 참조하는 fullstate의 정확 재구성.

두 성공한 CPU fixture는 각각 약 2.49초와 2.64초였다. CUDA는 초기화하지 않았다. CPU fixture의 업데이트는 테스트용 작은 모델 업데이트이며 연구 모델의 SGD나 성능 향상으로 세지 않는다.

첫 CPU 실행은 기능 검증 assert들을 마친 뒤 보고서 JSON을 쓰는 단계에서 `learning_io_v2` import 누락으로 실패했다. 원래 코드·파일·실패 기록을 보존하고 import를 고친 뒤 revision2에서 전체 계약을 다시 통과했다. 중복 테스트 가중치 파일을 늘리지 않도록 두 번째 검사부터는 fullstate를 메모리에서 safeload·재구성했다. 첫 physical fixture와 populated runtime도 삭제하지 않았다. 실제 연구 가중치와 GPU 작업에는 영향이 없다.

**아직 실제 연구용 worker/plan/seal/OS owner와 실제 full-size source CPU preflight가 완성된 상태는 아니다.** 실제 source에 대한 전 tensor 복원, 전체 실행 비용·노출·출력·checkpoint/STOP 임시 여유와 고정 종료 조건을 연결해야 한다. CPU fixture PASS를 이 단계의 완료로 부르지 않는다. GPU device에서의 수치 재현도 아직 이 검사로 입증하지 않았다.

다음 한정 실행에서는 같은 TRAIN broad top1과 전체 TRAIN의 취약 critical 위치를 같은 FP32 teacher/prefix/runtime 기준으로 구성한다. 개발 문장·정답을 bank나 학습 데이터에 넣지 않는다. 연속적으로 update가 전혀 수락되지 않는 경우에는 고정된 짧은 관측 창에서 멈춰 제약·학습 방향을 분석해야 하며, gate를 몰래 낮추거나 동일 작업을 자동 반복하지 않는다. 강한 일반 byte margin gate만 추가하는 방식은 V83 진단에 맞지 않는다.

현재 bank 저장 비용의 과거 추정은 7,666 patch, FP32 hidden 약 15.7MB다. 전체 실행 출력은 이것보다 크므로 실제 checkpoint/teacher/평가 기록과 STOP 임시 여유를 합쳐 새 상한을 정해야 한다. C: 100GiB 보호, 시작 시 8GiB headroom, 모든 기존 원본·가중치·Adam/RNG·raw·실패·게임 보존을 유지한다. 디노이징은 0이며, 미래 사고 학습은 입력·메모리·결합·출력 전체 codec을 동결한다.

증거: `research/artifacts/critical_bank_engine_20261113_v86_cpu_v2/root_engine_cpu_review.json`, `research/artifacts/compact_critical_engine_20261113_v87_cpu_v1/root_engine_cpu_review.json` 및 각 `evidence_SHA.json`. 최초 출판 오류는 V86 CPU revision1의 `CPU_revision1_publication_incident.json`에 보존했다.
