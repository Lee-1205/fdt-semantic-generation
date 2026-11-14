# V42 문장 복원 두 순회 계약

V41 최종 상태에서 문장→입력 의미→같은 문장을 복원하는 학습을 두 유한 순회 더 진행한다. 세 메모리와 Commit-only 디코더, 같은111732306개 파라미터/38556357개 학습 파라미터를 유지한다. 새 신경망·정답 길이·원문 우회·문장 통째 복사·디노이징은 없다. 현재 목표는 언어 인코더/디코더 안정화이며 완벽한 복원이나 전체 FDT 완성을 선언하지 않는다. K_P의 인증 Writer, K_C의 전체층 prefix 처리, K_E의 literal transport는 여전히 구현 범위 밖이다.

부모는 V41 FINAL `f242fc902bd65f331cde625378bf832cbb54c5e26ef4842cf6352ec64fb48887`, algorithm `b74eb3f6f7137c5967fe1d6a53d54e49ea3f97ca9fd10c28568cdc26ef6d201e`다. CPU 압축 동등성 실험에 사용했던 V41 학습 전 참조로 되돌리지 않는다. ALL91 populated Adam/model/scaler/모든 저장 RNG와 기존 평가·노출 이력을 보존하고, 같은 기본 LR에 새8192-step cosine만 적용한다. 기존 V41 및 모든 이전 실행의 예산은 닫혔다.

자료는 기존8192 TRAIN 문서/32768상관span 그대로다. 새11·12순회 seed41001/41002, batch8,8192attempt/65536행선택/5852956 target UTF8 bytes로 고정한다. AMP skip도 시도/행/노출을 소비하며 되감지 않는다. 관측 개발자료를 훈련에 넣지 않는다. V41 저장 평가를 baseline으로 사용하고 같은 초기 GPU 평가를 반복하지 않는다. 최종에는 기존 teacher256, TRAIN참조64와 held16의 BOS부터 같은 입력 복원, 세 READ 통제를 기록한다. 유사도와 의미 보존은 구분하고 실제 원문/복원문을 읽는다. 반복 관측 개발자료·TRAIN 참조의 암기·같은 문서의 여러span 한계를 유지한다.

새 고정 예산은2100초, modelcutoff1920초, shutdown reserve180초, actual call30000 상한/코드상최대27348다. 계획시간1639학습+120startup+60저장+60평가+180종료=2059초다. 시작C30GiB=20보호+1output+1임시+8여유/GPU12GiB, 출력1GiB/최대992MiB 구성은 유지한다. 큰 저장 직전23GiB 검사를 유지하고 저장 실패 시 후보를 보존한다. 추가 시도·상한 확대·옛 시간 재개는 없다.

초기 참조는 새 gzip helper를 사용한다. 실제 V41최종 import/export 메타데이터는33659437B로 기존 물리32MiB보다105005B 크지만 lossless gzip12917556B로 같은32MiB 한도 안이다. 비압축 CPU 직렬화/해제는 별도64MiB 상한이며, 모든 상태/부모pin/CRC/추가member/잘림/덮어쓰기 거부를 유지한다. 새파일 `state_0000_reference_gzip.pt`는 부모 필수 gzip이므로 새loader로만 읽는다. 갱신된 최종 또는 emergency는 일반 self-contained fullstate 하나만 저장한다. midpoint/삭제/rotation/두 번째 full-write는 없다. 초기 이후 최대8192시도가 미저장일 수 있다는 비용을 명시한다.

18focused CPU 검사 통과: 새 worker12/owned preflight6. 실제 V41 전체 상태를 CPU import하고 새로운 참조를 정확 roundtrip한 독립 OS 검사는56.406초에 자연종료했다. 실제 preflight33524←4132, owner11612←34352의 생성/parent/command와 종료를 확인했다. 별도 Windows CPU fixture11904←39284도 자연종료/allgone을 확인했다. 모델 크기의 forward/GPU update/CUDA 복구 재실행은 이 CPU검사에 포함되지 않는다. frozen 저장 모듈10검사와 구분한다.

새 CPU preflight는90초 worker hard clock+30초 cleanup reserve로 소유 OS 프로세스를 감독한다. preflight 결과가 없거나 불확실한 잔존 프로세스가 있으면 GPU launch를 막는다. status/STOP 쓰기 예외에도 검증된 Popen 소유 tree cleanup을 실행하며, 미확인은 UNCONFIRMED_EXIT로 남긴다. 실제 training 동안 ROOT는 OS 감독만 하며 에이전트를 쓰지 않는다. terminal 뒤 모든 owned/proxy 종료 확인부터 하고 결과를 읽는다. 외부 로그는 supervisor.stdout/stderr, worker 로그는 owner 전용 xb다.

CPU 증거는 `research/artifacts/three_memory_codec_20261030_v42_worker_cpu`다. 실행 계획/봉인/실제 GPU launch/완료는 각각 별도 증거로 기록한다. 원래 core/typed/모든 부모·실패·weightsAdamRNG/raw/사용자 원본은 보존한다. E2 final/registry 접근은0이다. 사용량이 줄면 최신 복구 위치와 다음 작업을 저장하며 계정 잔여량을 조회한 것으로 표현하지 않는다.
