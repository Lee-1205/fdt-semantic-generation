# 다음 FDT 학습 저장공간 후보 — 승인 대기

아직 삭제한 파일은 없다. 기존 사용자 삭제 승인 범위 밖이다.

| 삭제 후보 | 크기 | 보존 가중치 |
|---|---:|---|
| `outputs/astra_base_response_pilot_20260927/execution/mixed/recovery.pt` |4.47GiB|같은 폴더 `final.pt`|
| `outputs/astra_retention_repair_pilot_20260927/execution/mixed/recovery.pt` |4.47GiB|같은 폴더 `final.pt`|

합계9,598,064,702bytes =8.94GiB. 관측 C: 여유19.07GiB에서 약28.01GiB로 늘어날 수 있다. 새 실행 전 실제 여유를 다시 확인한다.

두 파일의 실제 SHA-256은 이전 저장 기록과 일치한다. 각각 단일 hardlink이며 reparse/symlink 속성이 없다. 상세 절대 경로·크기·SHA는 `FDT_STORAGE_CANDIDATES_CONTENT_20260928.json`에 있다.

삭제한다면 두 종료된 실행의 AdamW/RNG/optimizer 진행 상태로 돌아갈 수 없다. final.pt의 학습 가중치, 기존 모델·Exact bundle, 원문·평가·로그·SHA·보고서는 보존한다. 새로운 실행은 계획에 명시한 fresh AdamW로 시작하며, 실패한 terminal을 재시작하지 않는다.

승인 전 삭제하지 않는다. 승인된다면 두 정확한 경로와 실제 SHA/보존 final SHA/현재 프로세스 종료를 다시 확인하고, 한 개씩 삭제 manifest에 기록한다. 삭제 오류 시 범위를 확대하지 않는다. 원본 FDT source·prepared_data·V20·alpha.20·520M·bridge milestone·Phase-2 evidence 및 다른 프로젝트는 대상이 아니다.
