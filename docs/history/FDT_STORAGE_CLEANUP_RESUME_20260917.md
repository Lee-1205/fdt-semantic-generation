# FDT 저장공간 정리 재개 기록

상태: **BLOCKED_BEFORE_DELETE**

사용자가 명시한 삭제 범위를 승인된 범위로 기록했다. 그러나 2026-09-17 현재 command runner의 Codex 사용량 제한(`ordinaryUsageAllowed=false`, primary window 100%)으로 PowerShell 실행이 자동 검토에서 거절되어, 삭제 전 preflight나 삭제 명령을 실행할 수 없었다. 우회 실행은 하지 않았다.

## 마지막으로 확인된 읽기 전용 상태

- C: 여유 공간: 36.83GB
- C: 사용 공간: 893.87GB / 전체 930.70GB
- 원본 FDT: `<USER_HOME>/Documents/Codex/2026-06-26/dk/outputs/FDT_RLM`, 136.72GB
- `runs`: 130.11GB
- `prepared_data`: 6.32GB
- Phase-2 workspace: 약20MB
- 1GB 이상 `.pt` 55개 SHA-256 확인: 완전 동일 hash 0개
- 삭제 실행: 0개

## 다음 실행 시 순서

1. 삭제 전 free/used/total을 새로 읽고 이 파일에 기록한다.
2. Phase-2 report, patch manifest, `AUDIT_RESUME.md`, `astra_phase2_evidence`, Git bundle/patch 존재를 검사하고 가능하면 `FDT_ASTRA_PHASE2_EVIDENCE.zip`을 만든다. evidence 디렉터리는 절대 삭제하지 않는다.
3. 승인 목록에 있는 대상만 path/size/SHA-256/reparse/link 상태로 manifest에 기록한다.
4. 승인된 `.pt`만 삭제한다. manifest/config/log/json/csv/md/txt, prepared_data, source code, V20, α=.20 `latest.pt`, 520M promoted base, bridge r10 100M/200M milestone은 건드리지 않는다.
5. 삭제 후 free space를 읽고, 100GB 이상 증가했으면 즉시 중단한다.
6. `FDT_STORAGE_CLEANUP_20260917.md`와 `FDT_STORAGE_DELETED_FILES_20260917.json`에 실제 결과만 기록한다.

## 승인된 범위 요약

- A: `runs/pytest*`, `runs/_pytest*`, `runs/test_tmp` 전체.
- B: α=.20 run의 `latest_recovery.pt`만.
- C: α=.024/α=.025 두 run의 `*.pt`만.
- D: 지정된 과거 v4 main/transition run의 checkpoint/recovery. bridge r10 100M/200M milestone 보존. Stage A 10M milestone만 조건부 삭제.
- E: 지정된 과거 대형 실험 run의 `*.pt`만. 520M promoted base 보존.
- F: 지정된 과거 소형 pilot run의 `*.pt`만. V20 관련 run 보존.
- G: prepared_data 전체 보존.

사용자 승인 범위를 벗어난 추가 삭제는 하지 않는다.
