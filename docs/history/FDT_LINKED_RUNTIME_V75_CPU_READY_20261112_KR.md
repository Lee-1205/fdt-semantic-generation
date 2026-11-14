# 부모 참조형 복구 저장 CPU 준비

V74의 가중치·Adam/scaler/RNG를 유지하면서 후속 학습의 저장 부담을 줄이는 유틸리티를 ROOT가 구현했다. **실제 V74 최종 fullstate518,012,241바이트를35,539,462바이트의 부모 참조형 파일로 표현하고,전체 재귀 digest가 같은 상태로 정확히 재구성**했다. 저장 표현은 약93.14% 작지만 필수 V73 부모와 기존 V74 standalone을 모두 보존했다. 기존 공간을 그만큼 회수한 것이 아니며,이 파일 하나만으로 복원할 수도 없다.

`research/linked_runtime_v75.py`는 active NN tensor만 따로 저장하고 고정 NN tensor는 SHA가 고정된 standalone 부모의 `model`에서 가져온다. optimizer/scaler/모든 저장RNG/order/cursor/history/exposure/calls/cache/teacher/phase/inflight 등 나머지 runtime은 전부 보존한다. 고정 tensor가 부모와 다르거나 dtype/shape/소유권이 맞지 않으면 저장·복원을 거부한다. 부모 파일 SHA/모델 digest/고정 부분 digest/전체 재구성 digest를 확인하며,부모 참조를 재귀적으로 따라가지는 않는다. 다른 파일 덮어쓰기·실패한 partial 자동 재시도는 하지 않는다. 이 유틸리티는 checkpoint 복구용이며 inference의 Exact/text 검색과 무관하다.

실제 V74의7개 readout tensor/2,908,192파라미터,7개 populatedAdam(step640),enabledscaler1024 tracker640,torch/CUDA/Python/NumPy 저장RNG,전체 모델과 모든 runtime 항목의 정확 동일성을 확인했다. 원래 V73/V74 파일의 SHA도 전후 동일하다. 새 NN forward/실제 optimizer step은0/CUDA 초기화false다. 추가 CPU 작업은6.296초였고,이것을 실제 CUDA 재시작이나 새 모델 학습 성공으로 세지 않는다.

focused CPU5검사도 통과했다. populatedAdam/enabledscaler/RNG를 복구한 다음 tiny 산술 update가 동일함을 확인했고,부모 손상·부재·허용 경로 밖 참조,고정 tensor 변경/active dtype 오류,unknown active 항목/runtime 오염,중단 중 in-flight 상태의 보존을 검사했다. tiny 산술은 실제123M 모델 학습이 아니다. 새 V75 engine/owner/plan/whole-worker recovery/preflight/GPU는 아직 없다.

필수 부모는 `runs/fuzzy_extraction_20261112_v73/model/checkpoints/state_0960.pt`,SHA `81f4bb4bd41f6b3b89131a4aed7f5de6dc34635e1d5dffc8e42eeee22b738300`이다. 새 파일은 `research/artifacts/linked_runtime_20261112_v75_cpu/actual_V74_final_linked_runtime.pt`,SHA `10304478adcfb6fa5da8a80c2843d4bf188cdd43ea7f53a2c16e88f088346317`이며 재구성 fullstate digest는 원래와 같은 `b768fcd932a33a868ba6399275a64c6c8751c1483d4b146211dd104cbdd9f911`이다. 자세한 proof/codeSHA는 같은 폴더의 `root_CPU_review.json`과 `linked_manifest.json`에 있다.

CPU 검토 종료 시 C여유117,126,823,936바이트/약109.08GiB로100GiB 보호선을 유지했다. 다음은 이 저장 표현을 실제 finite training/STOP 복구에 연결하고,새 Commit cache를 계속 복제하지 않는 고정 상류 재인코딩/teacher loss-only 저장 후보의 비용·출력 cap을 계산하는 것이다. 언어 출력의 수량 교정과 장문 보존 간섭을 해결하는 후속 학습 목적은 별도이며,저장 유틸리티가 성능 개선을 뜻하지 않는다.
