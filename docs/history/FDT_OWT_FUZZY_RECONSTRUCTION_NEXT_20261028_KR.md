# 다음 작업: 같은 분리 구조의 추출·복원 학습 계속

최신 사람 지시는 OpenWebText 문장→기존 퍼지 앵커 기반 추출→추출 상태만으로 재복원이다. 추론/출력 의미 디노이징은 추후 별도로 학습한다. V12첫 사이클은2048업데이트 완료/예산CLOSED/allownedgone이며, 현재 모델 프로세스는 없다.

실제 성과는 held256 teacher byte 정확도8.62→43.15%, 교환34.45%/제거36.63%다. 정상 상태가 문장별 정보를 전달한다. 자유 복원은0/16exact, UTF8/EOS16/16이며 비문이 남는다. 따라서 추론 단계가 완성됐거나 의미 추출 검증이 모두 끝났다고 하지 않고 **추출·복원 학습을 이어간다.** 현재 작은181,203고유 sentence UTF8bytes/512문서 표본으로 전체 방식의 실패를 단정하지 않는다.

다음 ROOT는 다음 순서로 구체화한다.

1. 최신 source는 `runs/owt_fuzzy_reconstruction_20261028_v12/model/checkpoints/state_2048.pt`, SHA `1fa94def8635f40e5aac1f12b82176c39118cd1abde65f801323f38438d585da`다. 모든63populatedAdam/scaler2048 tracker48/모든 RNG/초기·milestone·실패·raw를 보존해 이어갈 명시적 importer를 만든다. 소스의 이미 닫힌 attempt/order/deadline을 다시 여는 방식은 사용하지 않는다. 새 phase/run의 finite order·노출·call budget을 별도로 선언하고 누적 이력을 ancestry에 유지한다.
2. 원래64dev문서/256문장을 그대로 관측 개발 set으로 유지한다. Corpus revision79d93d...1762c와 문서 hash split 규칙을 유지해 추가 훈련 문서를 확보한다. 먼저 저장 source_pages의 미사용 문서를 사용하고 필요한 경우 제한된 추가 다운로드만 한다. 기존 dev문서나 그 문장을 train에 넣지 않는다. 기존512train문서와 새 문서의 실제 수/bytes/반복 노출을 분리해 기록한다. 새 immutable artifact를 만든다.
3. 후보 범위는 train1024문서/4096문장으로 확대,4유한 epoch/최대2048새attempt 정도다. 구체적 자료가 확정된 뒤 실제 target bytes와 call 비용을 계산해 고정한다. 같은 모델·퍼지 알고리즘을 유지하고, 첫 사이클의 앵커 coverage30→15 집중과 복원·상태 의존성 변화를 함께 본다. 일단 임의의 큰 균등 사용 벌점·새 앵커 재초기화·추론 모듈 학습을 끼워 넣지 않는다.
4. Worker/owner의 실제 delta에 필요한 exact importer/fullstate/scaler·RNG/유한 order/STOP 계약만 CPU검사하고 실제 preflight·독립validate·새 SHA/clock/caps/resources를 확인한 뒤 한정 실행한다. 기존8개 검사나 broad GPU baseline을 무한 반복하지 않는다. 같은 구조의 held 초기값은 V12저장 결과를 참조할 수 있으나 문서 분할/추론함수/노이즈 조건이 같은지 명시한다. Source V12평가를 다시 돌리는 이유 없는 GPU trace는 금지다.
5. 결과는 teacher byte 점수·자유 문장 충실도·fuzzy state 교환/제거·실제 앵커 사용을 분리한다. 저장된 raw를 이용한 검증을 우선하고, 미달을 전체 모델 실패나 자동 추가 epoch의 이유로 바꾸지 않는다. 다음 학습은 별도 bounded 목적과 예산이다.

현재 후속 importer/worker/plan/seal/GPU는 **아직 없다.** Source V12의 미사용850call/시간은 닫혔다. 전체 모델111,485,711/학습24,150,914parameter와 원래 fuzzy Router/비배타적 mu/normalized rho 구분을 유지한다. 추론/디노이징 동결과 디코더의 원문 직접 접근 금지를 유지한다.

ROOT 직접/noagents, 하드웨어 재승인 질문 없음, C20GiB 상시 보호/임시 checkpoint+시작8GiBheadroom/GPU12GiB, owned OS감독/ANYpostPopen cleanup/terminal뒤모든종료확인, 원문·모든원래weights/AdamRNG·실패 보존 규율은 그대로다. E2final/registry/외부 게시/원본 삭제는 금지하고 타이머는30분/정확히 `재개`다.
