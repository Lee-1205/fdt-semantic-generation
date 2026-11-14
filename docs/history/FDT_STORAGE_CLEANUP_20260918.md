# FDT 저장공간 정리 결과 — 2026-09-18

사용자가 추가 승인한 speed r3/r4/r5/r6_fixed_validation/r7_syncfix/r8_fast_backend의 `latest.pt` 6개만 삭제했다. 삭제 전 각 파일 SHA-256과 단일 hardlink/reparse 아님을 기록했다. 디렉터리, 코드, 설정, 로그, 평가 결과, prepared_data는 삭제하지 않았다.

- 삭제 payload:10,188,018,050 bytes =9.49GiB =10.19GB.
- 실제 삭제 환경 C: 이전139,008,000,000 bytes free, 이후149,195,890,688 bytes free.
- 같은 환경의 관측 free 증가10,187,890,688 bytes. payload 총량과 작은 차이는 동시 파일시스템 활동 때문일 수 있다.
- 앞선 제한된 환경 조회는31,969,804,288 bytes였다. 두 환경 사이 차이의 원인은 UNKNOWN이며 이번 삭제 성과에 더하지 않는다.
- V20, α.20,520M, bridge100M/200M, Stage A final의 존재와 SHA를 기록했다. Phase-2 ZIP은 원래 SHA `df1906f2b7dd5e49fd86148fddefbb8dcd24e124c00abc7da6af9b22345d191c` 유지.
- 기존 승인 일반 checkpoint/recovery는 조사 시 이미 없었다. 누가 언제 제거했는지는 UNKNOWN이다. pytest 전체7.19MiB는 핵심 용량에 기여하지 않으며 metadata 보존을 위해 이 작업에서 삭제하지 않았다. Stage A10M은 다른 SHA로 중복 판정이 안 돼 보존했다.
- 삭제 manifest: `FDT_STORAGE_DELETED_FILES_20260918.json`. 과거20260917 보고서를 소급해서 삭제 성공 기록으로 바꾸지 않았다.

이번 확인된 삭제만으로100GB 추가 확보 목표를 달성하지는 않았다. 학습 실행 환경에서의 현재 여유는 시작 기준35GiB를 넘는다.

```text
CLEANUP COMPLETE
Freed: 10.19 GB (9.49 GiB)
C: free space: 149.20 GB (138.95 GiB)
Critical checkpoints preserved: YES
Phase-2 evidence preserved: YES
Prepared data modified: NO
```
