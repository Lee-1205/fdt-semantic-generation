# FDT DIANA 장문 준비 상태 — 2026-09-30

## 확인된 코드 경계

모델 체크포인트 `max_seq_len=16384`는 **실제 사용 가능한 문맥 16K의 증거가 아니다**. 같은 설정에 `legacy_position_transition_max_len=2048`, `legacy_position_scale=1.0`, `rope_transition_alpha=.20`, `rope_transition_mode=output_blend`가 있다. [FDT 모델 코드](legacy/fdt_prompt_objective/src/fdt_rlm/models/fdt_v4.py)는 embedding 적용 시 `position_offset+length > legacy_position_embedding.num_embeddings`이면 오류를 낸다. 따라서 현 가중치·설정 그대로 4K/8K/16K를 forward하거나 생성할 수 없다. `max_seq_len`만 키우는 변경은 금지한다.

α=.20의 `output_blend`는 legacy/rotary 경로를 함께 계산하며, 기존 평가기는 중간 blend에 단일 KV cache가 없어서 자유생성에 full recompute를 사용한다. 이는 4K 이상 훈련·생성의 속도/VRAM 위험이다. Fuzzy anchor recency recurrence 및 Exact 경로도 길이별 실측이 필요하다.

## 실측과 미실측

- RTX 5060 Ti: 총 16,311 MiB, 현재 관측 free 14,894 MiB. 32-token 단일 GPU 생성과 세 모델의 짧은 새 평가가 실행됐다.
- 512-token 이하에서 실제 데이터 파이프라인의 길이·overlap·EOS 계약을 CPU로 확인했다.
- 1K/2K/4K/8K/16K의 실제 full-model forward/backward, activation VRAM, decode latency, checkpointing 및 state carry는 **미실측**. 4K 이상은 현재 위치 임베딩 경계가 선행 차단한다.
- 마지막 C: free 16,996,237,312 bytes = 약 15.8 GiB, D: 약 34.8 GiB를 관측했다. 같은 작업 중 C: 여유가 약 1 GiB 감소했지만 원인을 특정하지 못했다. Hugging Face 캐시 전체는 약 4 MB여서 그 변화의 근거가 아니다. 새 장기 run의 model+optimizer 복구본 여러 개를 C:에 안전하게 보관하기에 부족하다. 자료 이동·삭제는 하지 않았다.

## 안전한 다음 순서

1. alpha=.20 설정의 1K/2K forward/backward·생성을 먼저 실측하고 legacy 2048 경계를 재현한다.
2. 기존 가중치 복제본의 별도 실험에서 RoPE α→1, legacy scale→0 경로를 CPU 및 짧은 GPU 테스트로 검증한다. 출력 변화와 fixed52/새 QA·자연어 회귀를 확인하지 않고 주 학습으로 승격하지 않는다.
3. 4K→8K→16K 각각 실제 FDT forward/backward, generation, cache/recompute 비용과 문서 순서·경계·mask를 확인한다. 실패 길이는 `UNSUPPORTED`와 OOM/수치 오류를 구분한다.
4. overlap은 기본 연속성 방법이다. Fuzzy state carry는 문서 간 reset·오염 방지 검사 뒤 별도 단계에서만 도입한다.

**판정: 현 checkpoint의 검증된 장문 학습 길이는 2K 이하로 제한해야 하며 4K 이상은 미지원.**
