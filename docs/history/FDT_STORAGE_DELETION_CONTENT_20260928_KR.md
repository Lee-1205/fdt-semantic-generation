# 승인된 종료 실행 복구 파일 정리

사용자가 정확한 두 recovery.pt 삭제 질문에 **전부 승인해**라고 답했다. 이전 자동 검토 거부 후 이 명시적 답변을 근거로 다시 검증하고 실행했다. 범위를 추가하지 않았다.

- 삭제2개: `astra_base_response_pilot_20260927/execution/mixed/recovery.pt`, `astra_retention_repair_pilot_20260927/execution/mixed/recovery.pt`.
- 각 파일 SHA·크기·정확한 경로·상위 경로 reparse 부재·single hardlink·종료 상태·Python 프로세스 부재를 확인했다. 삭제 전 manifest를 기록하고 각 삭제 뒤 갱신했다.
- 두 실행의 final.pt SHA 일치 확인: BaseResponse `788519c13ec33d09c72764e35e749316b1b5ba8342d401b33a07590ac65afd31`, retention repair `710541bfd84c3f49eec34e2b098719de226f9e147252c14d0f5105a009d6190a`.
- 삭제한 두 과거 실행의 AdamW/RNG 상태 재개는 이제 불가능하다. 최종 가중치·teacher cache·원문·평가·로그·SHA·Phase-2 evidence는 보존한다.
- 삭제 전 C:24,083,619,840bytes, 삭제 후33,681,616,896bytes. 실측 추가 여유9,597,997,056bytes = **8.94GiB**, 삭제 후 C: **31.37GiB**. 다른 앱의 쓰기로 파일 크기 합계와 여유 변화가 소폭 다를 수 있다.
- 실패0, 범위 확대0. prepared_data 및 원본 source 수정0.

정확한 각 경로·크기·SHA·결과는 `FDT_STORAGE_DELETION_CONTENT_20260928.json`에 보존했다. 23:28:51 KST 후속 실제 프로세스0, GPU15,189MiB, C:33,681,604,608bytes로 새 학습의25GiB/12GiB 시작 기준을 통과했다.
