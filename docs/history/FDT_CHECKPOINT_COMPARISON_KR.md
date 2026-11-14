# FDT DIANA 체크포인트 비교 — 2026-09-30

## 무결성과 구조

동일 FDT tokenizer SHA `4fade0c72e8673ba1e4319c8821593bb5e62a8c27792e2412de3bdc52d232862`, 동일 모델 구조 456,848,856개 저장 파라미터, `dim=1216`, 20층, 19 heads, fuzzy anchor 256/top8/router256, local window64, `rope_transition_alpha=.20`, `legacy_position_scale=1`, `legacy_position_transition_max_len=2048`로 확인했다. 3개 핵심 SHA를 다시 계산했다.

| 후보 | 실제 경로 | SHA-256 | 복구/과거 안전 |
|---|---|---|---|
| α=.20 parent | `FDT_RLM/runs/fdt_v4_1_424m_alpha020_retention_smoke_500k_20260907_r1/latest.pt` | `50e084717a0cce99ea29b9e695e402c15e0a0807f3c64db795eab6d26093436d` | 기존 기준 모델 |
| answer-signal step128 | `outputs/astra_answer_signal_pilot_20260930/execution/mixed/milestone_27279.pt` | `6f26c8fde606cc3849e525089a77b89de9c4977624070454e03c9da3d27eb9bb` | model-only. 기존 fixed52 row +0.82283%, 새 optimizer 필요 |
| answer-signal step256 | `outputs/astra_answer_signal_pilot_20260930/execution/mixed/recovery.pt` | `031145f797e6e201558cc9f5f15393e92d88062223d2718788d03f263dfe7a88` | model+Adam/RNG/cursor, 그러나 기존 fixed52 row +1.00354%로 **실패** |

step256 자유생성은 같은 가중치의 `final.pt` SHA `b2b3a1224fa4ff21db3dac1dd6784777ab1be4b5c143fc1d86e82427d13327ff`로 측정했다. 이전 실패 기록과 안전 중단은 그대로 유지한다.

현재 작업 디렉터리와 원본 FDT 루트에는 활성 `.git` checkout이 없다. 보존된 Git bundle 중 Phase-2 `FDT_ASTRA_PHASE2_SOURCE_20260917.bundle` SHA `b3699a86...7f2f68`/HEAD `ad8bdf8...`, data-repair `FDT_ASTRA_DATA_REPAIR_SOURCE_20260918.bundle` SHA `8d050f18...6aacaf`/HEAD `e79ff9c...`를 read-only로 확인했다. 원본/작업복사본의 `fdt_v4.py` SHA `0e881f62...8e4270`와 `config.py` SHA `f7607f3a...0cc8e`는 각각 동일했다. 이번 비교에는 모든 후보에서 같은 작업복사본 생성기(`generate_fdt_v4.py` SHA `d57d347c...42bf2c4`)를 썼다. 원본 생성기 SHA `199e59dc...5c68e0a`와는 다르므로 이 비교를 원본 CLI 그대로의 재현이라고 부르지 않는다.

## 이번 새 독립 개발 입력

세 모델에 같은 새 영문 이야기 시작 문장과 새로운 가상 기록 질문을 넣었다. Exact OFF, greedy 생성 설정, tokenizer와 코드 동일. 원문은 [생성 결과](../LOCAL_ASSETS.md)에 있다.

| 지표 | α=.20 | step128 | step256 |
|---|---:|---:|---:|
| 새 42-token 문단 NLL | 5.4086 | 5.3994 | 5.4100 |
| 새 52-token 근거 질문 전체 NLL | 3.5934 | 3.6069 | 3.6191 |
| 정답 `Nira Vale` 4-token 내용 NLL | **1.0379** | 1.1158 | 1.1265 |
| 실제 근거 답변 | `The gold telescope` | `The gold telescope` | `The Great Barrier` |
| 256-token 상한 이야기 실제 종료 | 74 tokens | 113 tokens | 73 tokens |

이야기 세 개 모두 원래 인물 Mara/봉인된 편지/망원경을 유지하지 못하고 카메라·엄마·보물 등으로 표류했다. 모두 EOS로 일찍 끝났으므로 **256개를 이어 쓸 수 있음이 입증되지 않았다**. 512-token 생성 및 장문 보존 평가도 아직 없다. 작은 새 문단의 NLL 차이는 우열의 통계적 증거가 아니다. 정답 내용 NLL은 부모가 가장 낮았고 세 모델 모두 자유생성 정답에 실패했다. [NLL 원시 결과](../LOCAL_ASSETS.md).

기존 answer-signal 패널에서는 step256이 공백 33/63→0, EOS 26/63→58, 부재 응답 0/8→8/8, Exact 복사 7/8 유지였으나 중복·설명 0/8, held QA 0/8, seen QA 1/8이었다. 이는 [기존 결과](../LOCAL_ASSETS.md)이며 이번 새 입력의 독립 결과와 섞어 성공률로 계산하지 않는다.

## 출발점 결정

**현재 지속 사전학습 출발점을 확정하지 않는다.** step128은 과거 fixed52 한도 안이지만 optimizer 복구와 자유생성 개선 근거가 없고, step256은 보존 가치가 있으나 기존 회귀 기준을 초과했다. α=.20을 안전한 비교 기준으로 유지한다. 먼저 데이터 사용 조건·장문 실행 경로·새 평가 계약을 확정하고 짧은 별도 단계에서 후보를 선택한다.
