# 초기 복구 참조 압축: CPU 저장 모듈 준비 완료

현재 학습 모델은 완료·검토된V41이다. 이번 작업은 다음 초기 참조의 메타데이터 크기를 줄이는 저장 모듈이며 새GPU학습·모델개선·V42실행준비 전체 완료가 아니다. V42 runner/학습용OSowner/실행plan/seal/launch는 아직 없다. V41 최종 가중치·ALL91Adam/scaler/RNG·모든 원본과 과거 결과를 보존했다.

새 `scripts/codec_initial_reference_v42_gzip.py`는 기존format39의 부모pin/공유 model·Adam·scaler·모든저장RNG/zero-work guard/전체digest 의미를 유지한다. 초기 참조의 Torch weights_only 메타데이터만 lossless gzip으로 감싼다. 기존 물리32MiB 상한은 유지하고, 비압축 CPU직렬화·해제에는 별도64MiB 상한을 둔다. CRC/잘림/추가 gzip member·trailing data/크기초과/틀린부모/업데이트 상태의 초기 참조화/덮어쓰기를 거부한다. 과거helper와완료계약을 수정하지 않았다.

새 파일은 `state_0000_reference_gzip.pt`다. gzip인 `.pt`이므로 일반 `torch.load`나구format39 loader로 읽으면 안 된다. 새helper의 `load_initial_reference`와 보존된부모를 사용한다. 새 `safe_memory_roundtrip`은 `(restored_state, sizes_dict)`를 반환하므로 후속worker는 compressed_bytes/serialized_bytes를 구분해 기록해야 한다. 갱신된 최종/중단 fullstate는 계속 일반 self-contained Torch 형식이다.

## 필요한 CPU 검증

- 저장계약10검사 PASS3.63초: 전체상태·RNG 일치/실제파일 roundtrip/부모 필수·덮어쓰기 거부/직렬화·해제·압축 크기상한/CRC·잘림·추가member·trailing 거부/잘못된부모·업데이트상태·변조 거부/기존tiny91populatedAdam·enabledCPUscaler 복구 뒤 정확한 다음update 일치다. Tiny검사는 실제모델 능력 또는 CUDA복구 재실행이 아니다.
- 실제 기존V41 초기 참조31583981B를 재봉인한 압축 파일은12174031B였다. 새Torch직렬화 메타데이터는31584109B다. 원래 `.pt`의 byte동일성을 주장하는 것이 아니라 원래·새 참조의4686690개 payload노드가 모두 동일함을 별도 읽기전용 검사에서 확인했다. 부모파일SHA도 대조했다. 저장된전체상태 algorithm은 기존초기와 같은 `99e73d7382ac17fc6e5e9d0ba516ef163fb1d9bfe07a3ac41d2b1961c4845e05`다.
- 읽기전용 검사는 부모 모델tensor를 다시 로드하거나 반복 전체hash를 만들지 않았다. 새고정90초/workercutoff75초/독립owned OS감독으로16.734656초에 완료됐고 실제CPU worker23012→17032의parent·command·identity·자연종료/allgone을 확인했다. owner Torchimport0/새model construction-forward0/CUDAinitializedfalse/모든GPU호출0이다.

**이 CPU 압축 참조는 V41 학습 전 초기 상태이며 V40부모를 사용한다. 다음 학습을 여기서 warmstart하면 V41 진행을 되돌리므로 사용하지 않는다.** 다음 학습의 부모는 계속 V41 최종 `f242fc902bd65f331cde625378bf832cbb54c5e26ef4842cf6352ec64fb48887`다. 실제V41최종→미래V42 engine import/export 크기와 전체worker 복구·preflight는 아직 검사하지 않았다. 다음세대 추정크기를 실측이라고 하지 않는다.

## 첫 대형검사의 시간 초과 기록

첫 CPU계획은120초였으나 동기적인 전체상태hash를 여러 차례 반복하고 단계 사이에서만 시간을 검사해181.797초에 TimeoutError로 자체 종료했다. 시간 계약은 FAIL로 보존한다. 모델학습·데이터손상 실패는 아니며 원래code/plan/incident/후보파일을 보존했다. 후보 저장·roundtrip 주장과 전체CPU시간계약 통과를 구분한다. 과거120초를 늘리거나 재사용하지 않았다.

ROOT가 종료를 확인했을 때 이미 관련CPU 프로세스는 없었고 강제종료한 수는0이다. 초기 root stop기록의 무조건적인ROOT_STOPPED 라벨은 실제조치를 과장하므로 원본을 보존하고 `cpu_ready/root_cpu_terminal_audit.json`에 정정했다.291.5초는 종료후ROOT확인시각까지의 경과이지 프로세스 실행시간이 아니다. 그 뒤 새 목적의 읽기전용payload 대조를 별도90초계획으로 수행해 위의16.7초 내에 완료했다. 다음 대형CPU preflight에도 단계검사만 의존하지 말고 owned OS hardtimeout/종료확인을 적용한다. 같은 모델·참조를 반복 fullhash하는 검증은 피한다.

## 증거와 재개 지점

동결 증거 `research/artifacts/initial_reference_gzip_20261030_v42_cpu_ready`의 status는 `V42_REFERENCE_STORAGE_HELPER_CPU_READY_NOT_WORKER_READY`다. ManifestSHA `ba5d34e26172dfd400bfb675ff34317400bbc9e7bf65d2b1bacc00ad0fb2f0c7`,helperSHA `a53534736fa2536e5963394bc9c2e8249fd6d6c60fa4e4bd7f4fc736690e4f19`다. 원래검사기록은`initial_reference_gzip_20261030_v42_cpu`, 별도읽기전용계획/결과/owned종료는`initial_reference_gzip_20261030_v42_readonly_cpu`에 보존했다. 이들 실행 예산은 모두 CLOSED다.

사용량이 줄면 정리·저장하라는 사람 지시에 따라 현재상태와다음작업을 저장한다. 계정잔여량을 직접 조회했다고 말하지 않는다. 명시적 중단은 아니므로 automation은ACTIVE30분/프롬프트`재개`다. 지금 추가GPU학습은 하지 않았다.

다음은 새한정V42 worker/plan/학습용OSowner에 새저장API·초기filename·marker/loader·단일final-or-emergency 저장계약을 연결하는 일이다. 실제V41최종의91Adam/scaler/ALLRNG import,새초기 참조의크기·전체복구·정확한nextupdate,실제plan-schema·CPUpreflight·독립ownedOSvalidate·새SHA/output·고정time/call/row/epoch/resource를 검토한뒤 별도11·12순회만 실행한다. 새fullsize검사를 이 CPUparity로 대체하지 않는다. 원래30GiB/GPU12GiB/20GiB보호·임시·8GiB여유,Commit-only/세메모리/no rawH-bypass/wholecopy0/denoise0/E2final-registry0/ROOTONLY-noagents는 유지한다. 이전가중치·원본삭제·oldbudget재개·실행중코드/상한변경·자동재시작은 없다.
