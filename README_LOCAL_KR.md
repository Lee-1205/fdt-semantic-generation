# FDT 102M 로컬 실행

이 프로젝트는 UTF-8 관계 문구를 처리하는 **102,237,197 parameter adapter**다. 범용 언어모델이나 vNext 전체 구현이 아니다. 원본은 `archive/`에 보존했고, 이번 로컬 검증은 새 seed에서 시작했다. 과거 2,240회 가중치를 복구한 결과가 아니다.

검증 환경은 Windows11 build26200, Python3.10.11, PyTorch2.10.0+cu128, CUDA12.8, RTX5060Ti16GB/compute12.0이다. 실행은 전용 `.venv`를 쓰며 Colab/Drive/Google 로그인 없이 동작한다. 연결된 Google 도구로 Colab GPU를 시작·지속 저장할 수 없어 사용자가 허용한 로컬 fallback으로 검증했다. 마우스 제어는 중단했다.

## 설치와 점검

PowerShell에서 이 프로젝트 폴더를 작업 위치로 연다. 기존 `.venv`는 설치·검증됐다. 새 설치에는 `setup_local.ps1`을 사용한다. 관리자 권한이나 execution policy 변경은 필요하지 않으며 개별 명령을 터미널에서 실행할 수도 있다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local inspect
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local preflight --config configs/local_baseline.yaml
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q --basetemp .pytest_new_check
```

공식 설치 근거: [PyTorch2.10 CUDA12.8 Windows wheel](https://pytorch.org/get-started/previous-versions/), [NVIDIA RTX5060Ti capability12.0](https://developer.nvidia.com/cuda/gpus). 최신 버전을 임의로 설치하기보다 실제 검증한 조합을 고정했다. `requirements-lock.txt`는 실제 환경이다.

## 이번에 실행한 제한 검증

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local smoke --config configs/smoke.yaml --max-attempts 5 --wall-seconds 600
```

위 명령은 **이미 실행됐으며 재실행하면 차단된다**. 전역 smoke 예산5/5회가 소진됐다. GPU window 19.562초, 모든 시도 성공/AMPskip0. 실제102M grounding2고유update + fresh-process 반복1, tiny semantic1/diffusion1이다. 작은 모델 두 phase를 실제102M의 semantic/diffusion 훈련 완료로 해석하지 않는다.

102M 첫 loss=1.37062311, clip전 norm=0.32418221. 다음 loss=1.41977310. 이는 다른 입력의 smoke 수치이며 성능 개선 추세가 아니다. peak allocated=1.360GiB/reserved=1.510GiB.

## 지금 실행해도 되는 검증된 다음 명령

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local verify-checkpoint --run-dir runs/e1_masked_learning_20261026_recovery_v2/model --fresh-process
```

완료된 E1의 전체 payload를 별도 CPU 프로세스에서 다시 읽고 모델 구조·marker·hash를 확인하며 실제 통과했다. 초기 smoke의 다음 optimizer update 복구는 `runs/smoke_102m/resume_proof.json`, E1 운영 복구의 model/Adam/RNG 동일성과 다음 batch는 recovery_v2의 import_verification/first_continued_update에 있다.

## 장기 학습 경계

사용자가 실제 **“학습 시작”**을 승인한 뒤 masked E1의 1,440회 학습과 평가를 완료했다. 결과·한계·후속 표현 보강은 [학습 결과](docs/FDT_E1_LEARNING_RESULT_20261026_KR.md)에 있다. 이후 별도 한정 실행도 지속 개선 승인 범위에서 준비하되 각 plan/출력/SHA/예산을 구분한다. 아래는 기본 CLI 인터페이스 예시이며 이번 E1 독립 감독의 실제 실행 명령과는 다르다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local train --config configs/local_baseline.yaml --run-dir runs/baseline_authorized_001 --max-attempts 1440 --wall-seconds 21600
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local resume --run-dir runs/baseline_authorized_001 --max-attempts 100 --wall-seconds 1800
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local status --run-dir runs/baseline_authorized_001
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local stop --run-dir runs/baseline_authorized_001
```

`stop`은 현재 kernel/update와 durable 저장이 끝난 경계에서 중단한다. SIGINT/SIGTERM도 cooperative stop을 요청한다. `STOP` 파일이 있으면 재개도 멈추므로 재개 지시 후 해당 run의 STOP 파일만 명시적으로 해제해야 한다. 전원 강제 종료·OS/장치 실패 자체를 막지는 못한다. 로컬 계산은 네트워크 연결 없이 가능하지만 PC 전원이 켜져 있어야 한다.

평가는 검증된 complete marker와 고정 suite를 받는다. 이 명령은 인터페이스/fixture만 검증했고 아직 capability 평가를 실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local evaluate --checkpoint runs/baseline_authorized_001/complete_000001.json --suite configs/development_eval.yaml
```

매 optimizer 시도 뒤 전체 model/Adam/scaler/RNG/sampler/phase/LR 진행/노출/history를 저장한다. SHA/크기/내부계약을 확인한 뒤 complete marker를 게시한다. 다음 시도는 이 검증 이후다. 저장 실패는 다음 시도를 막고 이전 정상본을 보존한다. 최근3개 untagged와 명시적 initial/parent/final tag만 유지하며 다른 run이나 사용자 자료는 회전 대상이 아니다. C: 여유20GiB와 새 임시 payload 예상 크기를 매 시도/저장 전에 검사한다. 다른 앱의 디스크 소비까지 통제한다는 보장은 아니다.

`local_checkpoint_verified=true`, `backup_to_drive_verified=false`, `evaluation_complete=false`는 서로 독립이다. Windows directory fsync 보장은 하지 않는다. 상세는 `docs/FDT_HANDOVER.md`, `docs/FDT_CODE_AUDIT.md`, `docs/FDT_NEXT_EXPERIMENTS.md`.
