# FDT DIANA 지식 지속 사전학습 결과 — 2026-09-30

**새 optimizer 학습 미실행.** 6개 실제 첫 행 입력·7개 CPU 검사·짧은 읽기 전용 GPU 비교와 고정 리비전 LibreTexts 1문서 검토용 수집만 수행했다. 수집 결과는 `training_eligible=false`다. 4K 이상 경로는 legacy 위치 임베딩 2048 경계에서 차단되고, C: 여유는 약 15.8 GiB다. 또한 문서별 데이터 사용 조건, 대규모 중복·오염 검사, 새 학습 안전 기준이 아직 확정되지 않았다. 따라서 학습 결과 토큰 수, 지식 향상, 최종 checkpoint, 재개용 새 optimizer 상태는 없다.

보존된 기존 후보와 원래 실패는 [체크포인트 비교](FDT_CHECKPOINT_COMPARISON_KR.md)에, 실제 데이터 smoke와 차단 이유는 [데이터 감사](FDT_KNOWLEDGE_DATASET_AUDIT_KR.md)에 기록했다. 다음 실행은 [사전 계획](FDT_CONTINUED_PRETRAINING_PLAN_KR.md)의 준비가 끝난 별도 고정 예산 run이어야 한다.
