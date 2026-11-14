# FDT DIANA 지속 사전학습 실행 계획 — 2026-09-30

## 목적과 현 상태

주 목표는 다양한 고유 영어 문서의 causal next-token CE이며 소규모 QA 문구 암기가 아니다. 현재 [데이터 감사](FDT_KNOWLEDGE_DATASET_AUDIT_KR.md)와 [장문 준비 상태](FDT_LONG_CONTEXT_READINESS_KR.md) 때문에 optimizer 실행은 비활성이다. 실행 파일·설정에서 `pilot_training_enabled=false`다. 100M은 자료 시스템 검증의 조건부 목표이지 승인된 즉시 학습 예산이 아니다.

## 다음 개발 단계

1. 소량 스트리밍 자료의 문서별 라이선스·attribution을 검토하고 권리 불명 문서를 제외한다. FineWeb 개별 웹 문서, Common Pile 메타데이터의 실제 오류를 검사한다. 6개 전체 스트리밍 경로를 각각 제한적으로 실증한다.
2. 문서 내 장·절/문단/문장 추출을 강화한다. 표·수식·질문/답·책 머리말을 별도 타입으로 처리한다. 영속적 normalized hash+near-duplicate 인덱스와 기존 학습/평가 오염 검사를 만든 뒤 **문서 분리→청킹** 순서를 지킨다.
3. 실제 FDT tokenizer로 source별 collected/unique/train-exposed/context-only/loss-applied 토큰을 분리 집계한다. 초기 30/25/20/15/10 비율을 **고유 supervised 토큰** 목표로 적용한다. 부족분은 명시하고 대체하지 않는다. StackExchange는 별도 검토 후 ≤5%.
4. 2K 이하 안전 경로에서 서로 다른 길이의 문서/모델 비용을 실측한다. α 전환은 별도 복제 실험과 새 평가 계약 아래 수행하며 4K 이상 장문은 해당 forward/backward 검증 후만 연다.
5. checkpoint 출발점은 기존 α=.20, step128, step256의 보존 평가와 새 독립 자료를 함께 판단한다. step256의 기존 fixed52 실패는 그대로 둔다. 후보 optimizer 단계는 새 출력 디렉터리·SHA·고정 시간/고유 토큰 한도·중단 기준·독립 감독을 가진 별도 run으로 한다.

## 새 학습 계약 초안

- 기본 loss는 실제 supervised token 전체 분모의 causal CE. overlap, prompt, pad는 loss 제외. 실제 EOS·중복 노출·source별 내용 CE를 별도 기록. 짧은 문서의 row-mean 과대가중을 피한다.
- 작은 보존 샘플은 제한하고 신뢰할 수 없는 교사 출력을 정답으로 간주하지 않는다. Exact adapter/anchor는 초기 동결하고 Base 변경 뒤 ON/OFF 복사·거절을 재확인한다.
- 시작 전 새 holdout 문서, 독립 QA/근거 답변, 256/512 자유생성, 원래 fixed52 및 Exact 안전 검사를 동결한다. source별 의미 답변, 반복·공백·모순, held 문서 NLL, catastrophic retention 중단 기준을 **학습 전에 수치로 확정**한다. 과거 +1% 기준 위반을 소급 지우지 않는다.
- model+optimizer+RNG+scheduler+데이터 cursor·manifest를 첫 update/50분 이내/단계 끝에 durable 저장하며 실제 복구 다음 update 동일성을 확인한다. 기존 terminal run은 재개하지 않는다.
- C: 저장공간이 부족하다. 최소 두 개의 전체 복구본과 원문 근거를 수용할 충분한 여유를 확보하거나 새 출력 장치를 명시적으로 검증하기 전 장기 학습을 시작하지 않는다. 현재 D: 용량은 관측했으나 FDT 출력 대상으로 검증하지 않았다.

**현재 단계: 소량 데이터·평가 구현 진행, 새 optimizer run 없음.**
