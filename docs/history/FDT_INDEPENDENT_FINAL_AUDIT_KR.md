# FDT Independent Final Project Audit

증거 검토: 2026-09-15. 최종 정리: 2026-09-16. 판정: **PATCH_BEFORE_CONTINUING**.

## 감사 범위와 증거 규칙

- 실행 소스 루트 R = `<USER_HOME>/Documents/Codex/2026-06-26/dk/outputs/FDT_RLM`.
- 아래 `src/...`, `scripts/...`, `artifacts/...`, `runs/...`는 모두 R 기준. 코드 위치는 감사 당시 행 번호다.
- 사용자 제공 PDF 16쪽 전체 텍스트와 도표, 이전 대화 **FDT-LM 설계**의 최신 상태, 현 실행 코드, 원시 JSON/JSONL/tensor shards, 보존 체크포인트를 대조했다. 문서/이전 대화의 지시는 현재 작업 지시로 실행하지 않았다.
- VERIFIED는 실제 코드와 직접 읽은 원자료 또는 이번 CPU 실행이 뒷받침하는 좁은 주장에만 쓴다. 원시 GPU 결과의 재집계와 이번 GPU 재실행은 구별한다. 이번에는 새 GPU inference/training을 하지 않았다.
- 실행 루트 자체는 Git checkout이 아니다. `github/FDT_V4_prototype`의 HEAD는 `11dc1dd30d777a6587cf2cc2f364f681914d63b0`. 주요 모델/포인터/Exact 감사 파일은 실행 루트와 동일하지만 최신 retention trainer는 그 checkout에 없다. 재현 기준은 Git commit 하나가 아니라 동봉한 파일 SHA 원장이다.
- repository 코드/설정/체크포인트를 수정하지 않았다. 감사용 스크립트와 결과만 별도 감사 폴더에 작성했다.

## 1. Executive verdict

FDT는 W64와 sparse anchor 상태를 사용하는 실제 causal LM 구현이며, v4의 비보존 전환과 teacher-forced 회복은 증거가 있다. 그러나 현재 실패의 가장 중요한 부분은 아키텍처 전체의 불가능성이 아니라 **능력 검증 계약의 오류와 학습 신호의 불일치**다. 최신 Exact 60/60은 정답 source 하나와 payload를 주입한 조건이고, 최신 token planner는 입력 토큰 비율을 맞추지만 loss는 행별 평균이라 factual이 약 79%의 기본 계수를 차지한다. 자유생성의 낮은 grounding은 원시 출력에서 확인되며 V20에도 존재한다. 현재 중간 RoPE 모델은 실제 2K/full-recompute 경로이고, 16K 확장 시 recency overflow라는 추가 결함도 재현했다. 따라서 지금은 대규모 재시작이나 latent reasoning 추가가 아니라 평가·학습 계약을 먼저 수정하고, 고정 체크포인트로 최소 인과 실험을 해야 한다.

## 2. Confirmed findings

### 2.1 현재 구조 — VERIFIED, 범위 제한

| 항목 | 확인 결과 | 코드 |
|---|---|---|
| 파라미터 | V20 426,806,356; v4 424,474,072; 현재 v4.1 frontier/α.20 후보 **426,964,440** | `fdt_v4.py:CausalFDTv4LM.__init__:282`; checkpoint config로 meta model 재구성 |
| 차원 | vocab24576, hidden1216, 20층, local heads19, tied embeddings | 최신 α.20 config:20–60 |
| 로컬 attention | 모든 층 W64 causal, 자기 위치 포함 최근64 | `fdt_v3.py:CausalWindowAttention`; `fdt_v4.py:112–114` |
| fuzzy 경로 | 짝수10층, anchors256/top8/router256/cosine/T=.25 | `fdt_v3.py:V3FuzzyAnchorLayer.route:870–990` |
| membership | 선택 top8 내부 softmax, 합1; 독립적인 고전 fuzzy membership과 동일하지 않음 | 같은 함수:921–987 |
| 상태 | long/recent 두 weighted summaries와 mass; 캐시의 numerator 필드가 normalized=True일 때 실제로 weighted mean | `fdt_v3.py:1177–1181,1225–1373` |
| persistence | 토큰 간/cache 내 상태 유지. 새 prefill은 새 상태. 세션 간 영속적 지식 DB나 latent reasoning loop가 아님 | `fdt_v4.py:565–651` |
| Exact | commit/proposal/pointer/cursor 및 explicit payload 구현 존재 | `lexical_pointer.py:94,227,344,1059,1560` |
| diffusion/reasoning | 해당 Base forward는 autoregressive logits; 별도 reasoning-depth recurrence는 없음 | `fdt_v4.py:408–478,603–651` |

현재 총 파라미터와 v4의 차이 **2,490,368 = 2048×1216**은 보존 위치 테이블이다. 동결 파라미터와 optimizer가 실제 업데이트하는 파라미터 수는 구별해야 한다.

### 2.2 v4 200M 실패 이력 재판정

원시 official JSON 5개를 직접 재집계했고 대응 checkpoint SHA를 모두 다시 계산하여 일치시켰다. 입력52행 SHA도 동일하다.

| 추가 active tokens | token-weighted NLL | paired row-mean ΔNLL vs V20 | loop-free /100 | whole exact /30 |
|---:|---:|---:|---:|---:|
| 0 | 5.406830 | +2.206970 | 8 | 0 |
| 9,831,728 | 5.320523 | +2.120919 | 7 | 0 |
| 36,370,475 | 5.122101 | +1.919687 | 6 | 0 |
| 100,005,238 | 4.772419 | +1.572600 | 4 | 0 |
| 200,102,827 | 4.583535 | +1.426579 | 1 | 0 |

1. **비보존 전환: YES.** 0-token checkpoint부터 큰 paired regression. 새 geometry뿐 아니라 learned position 제거 등 전환 묶음의 영향이다. RoPE 하나의 순수 효과로 단정하지 않는다.
2. **적응 및 둔화: YES.** NLL이 실제 회복한다. 마지막 약100M의 token NLL 회복은 약.189로, 직전 약64M의 .350보다 작다. 단 LR·curriculum 변경이 있으므로 architecture의 학습 한계나 asymptote를 증명하지 않는다.
3. **teacher-forced/free-run 분리: YES.** loop-free 8→1과 NLL 회복은 공존한다. 단 평균 trigram 반복률은 .6563→.5782로 감소했다. 따라서 ‘모든 생성 지표가 단조 악화’는 틀리다. endpoint 비교의 통계 유의성도 별도 paired 분석 없이 주장하지 않는다.
4. **Exact train metrics≠end-to-end: YES.** 모든 official cell에서 첫 토큰부터 실패. independent60행도 whole exact0. 학습의 cursor metric은 실제 cursor 실행조차 아니다.
5. **long row/token 왜곡: YES.** 원시 shards와 cursor에서 128행 long source를 총5,233회 사용, 64,241,664 active tokens. 총 base200,102,827의 **32.10%**, long row 약40.9회분이다. 설정 row share는2%다. natural 약7.38회, factual 약5.23회 노출도 확인된다.
6. **512 cache defect: YES, 역사적 독립 결함.** 512→513 decode logit error .001687437, local-KV .026515007, tolerance .0003. 단 짧은 공식8-step 체크는 PASS였다. 이후 group64 repair는 같은 v4 checkpoint에서 경계·장문 state/logits PASS와 route identity 일치 자료가 있다. ‘현재도 모든 경로가 똑같이 실패’라고 반복하면 안 된다. 이 수리의 최신 중간α 후보 적용 가능성은 별개다.

자료: `artifacts/fdt_v4_capability_trajectory_20260829/*official_fp32_eval.json`, `artifacts/fdt_v4_200m_audit_20260829/`, `artifacts/fdt_v4_1_cache_repair_20260830/`.

### 2.3 v4.1 Stage A 및 최신 상태

- **Stage A 완료:** 19,720,000 active tokens, step5000. 논문의 진행 중/중간 수치가 현재 상태를 대표하지 않는다.
- **α=0 보존:** 변환 직후 group0/quantum0에서 보존32행 logits bit-exact, 생성4행 동일이라는 raw 증거가 있고 전환 코드가 뒷받침한다. 전체 입력·dtype·하드웨어·미래 fine-tuned checkpoint에 대한 보편적 동일성은 아니다. group64 configured 경로는 max logit delta .000333726으로 ‘허용오차 통과’이지 bit-exact가 아니다. 감사 코드는 exact control에서 group size를0으로 강제한다(`audit_fdt_v4_1_behavior_transition.py:426`).
- **Stage A isolation:** Exact loss/recovery/UL/long-source를 끈 것은 전환을 분리하는 합리적 선택. 다만 LM optimizer update와 α ramp는 동시에 일어나므로 static α/no-update 및 α0/update controls가 있어야 효과를 분해할 수 있다.
- **Stage A final fixed52:** token NLL3.575405, paired row Δ+.034492. 재검사 row NLL3.356223 vs V20 3.321731 = **+1.0384%**, token mean 기준은 약+.6975%. 과거 ‘+.974% PASS’는 row delta와 token 평균을 섞은 해석이다. 뒤의 dual-aggregate 검사는 이를 구별하고 있다. artifact `status:PASS`가 capability gate PASS와 동의어가 아니다.
- **현재 frontier:** `runs/fdt_v4_1_exact_pointer_pilot_20260831_r3/latest.pt`, SHA `96439F59…BFE`.
- **최신 후보:** α.20/500,516 tokens/step602 COMPLETE, SHA `50E08471…36D`. raw fixed52 row3.347885(+.7873%), token3.567541(+.4759%)는 기존+1% 기준 안. 그러나 **동일 α.20의 무업데이트 frontier보다** row+.002254/token+.001318 나빠졌다. retention 통과는 capability 향상이 아니다. 최신 후보 generation/Exact/cache 종합 승격 근거는 UNKNOWN.
- **일정:** 사용자 문자열의 ‘…0.75→10→…’는 중복 붙여넣기로 보인다. 코드가 허용하는 α는[0,1]; literal10은 거부된다. 0→.25→.5→.75→1은 실험 구간일 뿐 최적 간격 증거는 없다. α1이라도 legacy_scale1이면 완전한 RoPE 전환이 아니다.

### 2.4 최신 학습 신호 재집계

최신100k source rows의 실제 tensor를 읽고, 저장 seed/cursor로 소비행을 복원했다.

| source | active tokens | supervised tokens | 소비행 | 기본 row-loss 계수 비중 |
|---|---:|---:|---:|---:|
| natural |225,301|88,201|457|9.49%|
| factual |174,976|17,750|3,786|78.61%|
| retrieval |50,157|3,617|176|3.65%|
| Python |25,036|11,553|120|2.49%|
| JSON |25,046|11,467|277|5.75%|
| 합계 |500,516|132,588|4,816|100%|

**계수 비중은 실제 loss 값의 비중이나 gradient norm 비중이 아니다.** 현재 loss는 각 행의 EOS-weighted CE를 평균한 뒤 행 수로 나눈다. 실제 loss-weighted contribution은 source별 numerator/denominator 로그가 없어 UNKNOWN. active-token planner만으로 이 값을 복원하면 안 된다. CPU 반례에서 같은 두 행의 현재 loss .567294와 global token mean .301635가 달랐다.

각 source20k행은 이번 full-token-row 해시 기준 unique이고 fixed52 완전행 중복0. 소비행은 epoch0. 그러나 source/document/substring overlap, 삭제된 과거 payload와의 중복, globally novel 여부는 UNKNOWN. manifest 자체도 과거 payload 일부가 삭제돼 재스캔할 수 없다고 적는다. 반복 사용한 fixed52는 개발용 validation이다. 최종 sealed holdout으로 볼 수 없다.

## 3. Likely but unconfirmed diagnoses

- **base capability·conditioning 부족:** V20도 동일 fixed100 greedy raw protocol에서 loop-free6/100. transition만 제거하면 해결된다는 가설에 반례다.
- **loss allocation/objective mismatch:** natural45% active지만9.49% row-loss 계수; factual 짧은 답변/EOS가 상대적으로 강하다. 이 불균형이 현재 quality의 어느 비율을 설명하는지는 A/B 필요.
- **decoder 영향:** 동일 Stage A에서 ngram penalty13이면 loop-free100/100, 그런데 원시 문장은 여전히 주제 이탈·무의미 변형. 반복 억제 성공과 의미 품질은 분리된다.
- **self-prefix mismatch와 fuzzy feedback:** 가능하지만 아직 교차 continuation/상태 개입으로 분해하지 않았다.
- **데이터 다양성:** v4 replay는 확인. 현재 local novelty만으로 주제·문서·추론 형태 다양성이 충분한지 알 수 없다.
- **EOS 문제:** gold-length Exact 평가가 종료를 가리고, factual의 짧은 supervised span 및 EOS2배 가중이 있다. 실제 조기/미종료 원인 비중은 UNKNOWN.

## 4. Claims that are currently unsupported

보류·축소할 주장:

1. ‘현재 Diana는424.47M/실사용16K/incremental KV 지원’ — 현재 전환 configuration과 다르다.
2. ‘Exact60/60으로 distractor retrieval와 whole-chain 해결’ — 정답만 등록된 조건이다.
3. ‘cursor_continuation_rate가 실추론 성공률’ — ground-truth mapping 통계다.
4. ‘token-normalized curriculum으로 학습 신호도 해당 비율’ — row-normalized loss와 다르다.
5. ‘200M globally novel tokens’, ‘source epoch0이면 전체 계보 novelty’, ‘fixed52 무누수 최종시험’.
6. ‘penalty_enabled=false면 penalty-off’ — raw artifact 일부는 동시에 ngram13 적용이다.
7. ‘Stage A 보존 통과가 generation, Exact, full RoPE,16K를 검증’.
8. ‘기억이 의미 관계를 인과적으로 표현하며 현재 그 공간에서 사고한다’ — learned routing/state는 구현돼도 semantic relational reasoning 증거는 없다.
9. ‘Transformer 대비 보편적 성능/속도 우위·독창성 확정’. 이 감사는 포괄적 novelty 검색을 수행하지 않았다.

## 5. Code audit

| file / function / location | problem | severity |
|---|---|---|
| `scripts/audit_fdt_v4_exact_copy.py/exact_span_metadata:289–313;free_generate:333–348` | 정답 key 하나+gold payload+gold generation length. retrieval/termination을 우회 | P0 평가 |
| `src/fdt_rlm/lexical_pointer.py/_explicit_copy_training_loss:1285–1288,1375–1378` | pointer accuracy는 token equality; cursor rate는 정답 연속성 | P0 측정 |
| 같은 함수:1255–1261 vs `mix_next_logits:1636–1667` | train에는 anchor-match score bias, inference에는 없음; global denominator vs 선택집합 softmax | P0 계약 |
| `src/fdt_rlm/models/fdt_v4.py/build_exact_memory:545–547` | FP32 model 평가에서도 memory keys FP16 저장. confidence 보정/경계 비교 필요 | P1 수치 계약 |
| `scripts/train_fdt_v4_1_retention.py/chunked_weighted_lm_loss:724;train:2131` | active-token schedule와 row-weighted objective가 다른 배분 | P0 회계/P1 목표 |
| `src/fdt_rlm/models/fdt_v4.py/forward:155–164;step:195–201;_transition_embeddings:396–402` | 중간α cache 불가, legacy table 유효시2048 초과 ValueError; 출력 예산 검사가 max_seq_len만 사용 | P0 runtime contract |
| `src/fdt_rlm/models/fdt_v3.py/_sparse_chunked_prefix_forward:195–204;__init__:825–833` | 현재 recency reference512/bias4에서 position11335부터 FP32 inf;16K summary NaN 재현 | P0 장문 확장 |
| `scripts/evaluate_fdt_v4.py/per_row_teacher_forced:614–630;evaluate_teacher_forced:727–765` | all-token NLL과 answer-only loss 구별 필요; 각 모델 별 truncation 후 row-index만 비교해 token identity 미보장 | P0 일반 평가 계약 |
| `scripts/audit_fdt_v4_cache_integrity.py/main:409–411` | CLI 기본0이 checkpoint의 prefix-stable group64를 덮어씀; 실제 배포 configuration 감사가 아닐 수 있음 | P0 감사 configuration |
| `scripts/generate_fdt_v4.py/main:124–128,140,169–176,235` | CUDA에서 bf16·greedy, 공식 FP32/Exact payload audit와 경로 다름; CLI는 span map만 전달 | P1 product parity |

`exact_memory_copy_cursor` config는 선언되지만 조사한 generation 경로의 cursor 활성 여부 제어에 사용되지 않는다. ablation 이름만 바꾸고 cursor가 계속 켜지는 실험을 방지해야 한다.

## 6. Free-generation root-cause analysis

### 가능성 순서 — 확률 수치로 가장하지 않음

| 순위 | 가설 | 현 증거 / 가장 싼 구분 |
|---:|---|---|
|1|insufficient base capability|V20도 약함. 동일 prompt의 V20/α0/Diana continuation|
|2|weak conditioning/grounding|정답이 prompt에 있어도 숫자·인물 불일치. 사실 한 개만 바꾼 counterfactual prompts|
|3|objective mismatch|실제 supervised/row 계수 불균형. correct accounting 고정α tiny A/B|
|4|decoder calibration|penalty13으로 loop metric만 개선. greedy vs T.8/top-p.9 2seed|
|5|self-generated-prefix distribution mismatch|4-way crossed continuation, gold-prefix control|
|6|insufficient data diversity|v4 반복 실재. 새 문서/새template sealed set에서 범주별 차이|
|7|generation protocol defect|dtype/cache/Exact/penalty flags 상이. 동일 token IDs/logits부터 parity|
|8|positional transition regression|v4 역사적 증거 강함; v4.1 전체 원인으로는 부족. same weights α0/.20/.25|
|9|EOS/termination issue|종료 계측 부족. EOS p/rank, censoring, premature EOS 분리|
|10|fuzzy-memory read/write feedback|생성 중 write-freeze/read-off 인과 개입. 아직 추측|
|11|evaluator defect만으로 낮은 내용 품질 설명|Exact 평가 결함은 확정이지만 raw 문장 자체도 나쁨. 의미 품질 실패 전체를 설명하지 못함|

우선순위는 현재 semantic-quality 문제 기준이며 원인들은 동시에 작동할 수 있다.

### 최소 실험 세트

**E0 — 실행 계약 통일, optimizer0.** checkpoint/tokenizer/prompt token IDs/precision/α/legacy scale/Exact mode/backend/penalties/stop 조건을 모두 기록. V20과 Diana를 모두 FP32 full-recompute로 비교하여 cache backend confound를 제거한다. product bf16은 별도 track. gold-prefix next-token parity, prompt boundary/BOS/EOS, effective context 검사부터 한다.

**E1 — 4-way prefix 교차.** 개발32 prompts(자연문 continuation, 근거제공 QA, counterfactual identity, 간단 구조변환 각8). prefix32 tokens와 continuation64 tokens; 처음에는 greedy·penalty0·Exact off. V20/선택 frontier가 만든 prefix를 각각 생성해 토큰 그대로 두 모델에 재-prefill한다.

- V20→V20; Diana→Diana; V20 prefix→Diana; Diana prefix→V20.
- prefix 생산2개와 suffix4개만 필요:32×(2×32+4×64)=**10,240 sampled tokens 상한**. full-recompute FLOPs/시간은 별도 기록한다.
- state/cache를 모델 간 옮기지 않는다. 동일 token string을 각 모델이 다시 읽는다. EOS로 짧아진 prefix는 별도 strata로 보고 gold length까지 억지로 채우지 않는다.
- gold prefix→두 모델을 추가한8개 진단행으로 둘 다 나쁜 prefix를 ‘좋은 control’로 오해하지 않는다.
- V20 prefix에서 Diana만 회복하면 prefix trajectory 영향; 양쪽 소비자가 Diana prefix에서 함께 나빠지면 prefix 손상이 강한 설명; prefix 생산자 무관 Diana만 나쁘면 소비 모델/위치/decoder 영향. V20 prefix가 더 좋다는 가정은 blind rating으로 먼저 확인한다.

**E2 — E1의 문제8행에만 decode/α controls.** T.8/top-p.9,2 seeds를 작은 pilot 값으로 사용. same checkpoint α0/.20/.25, 전부 legacy_scale1. greedy에서 temperature만 바꾸면 argmax가 같으므로 calibration 실험이 되지 않는다. Exact off/on과 BF16/FP32는 별도 paired track.

**E3 — grounding/memory 개입,8행.** 핵심 인물/값만 교체·삭제하고 답이 따라 변하는지 검사. 이어 각 소비 모델에서 prompt-prefill state 유지 후 generated-token fuzzy writes만 차단, fuzzy read 차단, local-only ablation을 별도 비교. 중간α는 현 cache API가 없으므로 먼저 동등한 replay/reference instrument가 필요하다. 최근/장기 상태를 섞어 바꾸지 않는다. write-freeze는 인위적 분포 변화이므로 악화만으로 feedback 가설을 반증하지 않는다.

**E4 — 독립 Exact multi-record chain.** 아래7절 계약. 이 실험은 semantic generation 점수와 합산하지 않는다.

### raw output 직접 평가

이번 감사에서 Stage A official 각범주 첫2행과 product penalty13의 고정 index0/1/10/30/50/75/99를 직접 읽었다. 확률 표본·blind human benchmark는 아니다.

- Eliot 질문은 두 인물을 명시하지만 답은 `the right to be a human being`: loop-free여도 질문에 답하지 않음.
- JSON은 age74를72로, record_id를 다른 값으로 바꾸며 유효 syntax를 출력.
- access code `LV320072`가 `LV320067`로 변형.
- product penalty13에서는 축구 문맥이 ‘bad mangoes’로 이탈하고, 기술 문맥이 일반적인 ‘following code’ 문장으로 바뀜. loop-free100/100이 의미 품질을 보장하지 않는다.

다음 평가는 모델·α·decoder를 숨긴 무작위 A/B, 가능하면 독립2인으로 relevance/coherence/specificity/grounding/termination 각각0–2점. grounding은 prompt의 근거 span과 대조하며 창작 continuation에 없는 단일 정답을 강제하지 않는다. termination은 정상 EOS/조기 EOS/예산 소진으로 분리한다. 평가자 불일치·paired win/tie/loss·신뢰구간을 남기고 자동 loop score와 합쳐 하나의 PASS로 만들지 않는다.

decoder와 생성 퇴화의 관계는 [Holtzman et al.](https://arxiv.org/abs/1904.09751)이 비교 실험의 근거를 제공하지만 FDT의 원인을 증명하지 않는다. generated prefix 뒤에 원래 gold suffix를 무조건 붙이는 처방은 의미 충돌과 학습 목표 편향을 만들 수 있다. [Huszár의 scheduled sampling 비판](https://arxiv.org/abs/1511.05101)을 고려해 유효한 recovery target만 구성한다.

## 7. Exact Memory verdict

**일부 복사 장치는 개선됐지만 end-to-end 해결은 검증되지 않았다.**

필수 성공 사슬:

**source top1 → first token → cursor commit → cursor continuation → whole string exact**

- v4: whole0/30, independent0/60, 실패.
- v4.1 구60행: whole20/60; explicit-span57/60; explicit registered standalone payload60/60. 이는 동일 계약의 단조 개선 실험이 아니다. seed/registration/payload/종료조건이 바뀐다.
- 최신60/60은 CPU 호출로도 `eligible_keys=1`과 gold payload 전달을 확인했다. 잘못된 source를 탈락시키는 retrieval 자체가 시험되지 않는다. gold target 길이로 생성이 끝나므로 over-copy/정상 종료도 미검증.
- train query는 teacher-forced hidden; 모든 source position에 anchor bias를 포함한 loss. inference는 proposal/global fallback·FP16 keys·다른 score·confidence threshold·실제 selected-token commit. mismatch가 남아 있다.

**권고: 고신뢰 deterministic hard-copy.** exact-identity 목적에는 검증된 source/span 선택 후 LM 재경쟁을 없애는 방식이 더 적합하다. 단 .9/.9/margin1을 충분한 calibration 증거로 간주하지 않는다. source absence·ambiguous duplicate·non-copy query에서 거짓 commit을 함께 측정하고 불확실하면 abstain/base 경로로 돌아간다. soft mixture는 탐색/학습 보조로 남길 수 있으나 lossless identity 보장은 아니다.

정당한 external registration은 모든 records에 동일 규칙으로 적용하고 정답 label을 보지 않아야 한다. target뿐 아니라 distractor payload도 저장하며 질문이 source를 선택해야 한다. parser를 제공한다면 ‘parser 지원 memory’라는 범위를 명시한다. 토큰4/8/16/32/64와 문자4/8/16/32/64는 별도 grid다. 원자료의60행은 문자 길이 grid이다.

## 8. Required patches

이번 감사에서는 아래 수정 명세만 제시하며 적용하지 않았다.

### P0 correctness

**A. Exact 평가 계약 수정**
- why: oracle retrieval/종료 누설 제거.
- files: `audit_fdt_v4_exact_copy.py`, `lexical_pointer.py`의 계측, 관련 tests.
- expected: 정답과 무관한 모든 record 등록, source-ID top1/first-token/실제commit/continuation/whole/termination 별도 기록.
- risk: 기존60/60 수치와 직접 비교 불가. 종전 결과는 삭제하지 않고 oracle-assisted로 명시.
- test: distractor shuffle, query swap, absent/duplicate source, random pointer, gold-free max budget.
- pass: source가 N개일 때 실제 N개가 경쟁; query swap에 정답 source가 바뀌고 random selector는 chance 수준; oracle fallback0. deterministic copy transport는 모든 등록 payload byte/token 일치 및 span 밖 출력0.

**B. 학습 회계·objective 명세**
- why: active budget와 row-loss allocation 불일치.
- files: `train_fdt_v4_1_retention.py:661–724,763,2102–2152`, data manifests.
- expected: source별 active/supervised/EOS-weighted denominator/CE numerator/coefficient 로그. global supervised-token mean 또는 명시한 source-normalized objective 중 하나를 config로 고정.
- risk: objective를 바꾸면 effective LR와 기존 loss 수치가 바뀜.
- test: 같은 batch의 분할·padding·short-group 재배치와1-token/긴행 반례.
- pass: 명시 수식과 loss/gradient 일치; 계수 합1; token accounting 정확 일치. 기존 checkpoint를 덮어쓰지 않고 별도 optimizer pilot.

**C. effective context·recency 안전성**
- why: 선언16K/실제2K 및11335 overflow.
- files: `fdt_v4.py:_transition_embeddings`, `fdt_v3.py` recency forward/backward/state, generation/evaluator preflight.
- expected: 현재 제한을 명시적으로 노출; recency는 상대좌표/재스케일링으로 동등한 weighted mean을 유지해 overflow 방지. α와 legacy scale 전환은 분리.
- risk: recurrent normalization·gradient·cache rounding 변화.
- test:2K/4K/8K/11334/11335/16K의 forward/backward, 원 FP64 수학 reference, alpha endpoints.
- pass: 전 범위 finite; 수학 reference와 사전고정 tolerance; 기존2K capability gate 유지. 숫자 상수 변경으로 overflow를 숨기지 않는다.

**D. evaluator/protocol 통일**
- why: mask/분모/backend/default override 및 penalty 표기 혼선.
- files: `evaluate_fdt_v4.py`, `audit_fdt_v4_cache_integrity.py`, `generate_fdt_v4.py`, `audit_fdt_v4_repetition_development.py`.
- expected: all-token/answer-only·row/token NLL 병기; 같은 실제 token tensor/hash로 paired; inherited group64를 기본 보존; 구체 decoder/stop/정밀도 설정 기록; execution PASS와 capability PASS 분리.
- risk: 과거 aggregate와 수치 차이.
- test: 길이 다른 checkpoint, empty/invalid row, ngram>0 with repetition1, preserved group64.
- pass: 조용한 truncation/누락0; full token identity 동일; penalty가 하나라도 켜지면 raw-off가 아님; 동일 setup 재현.

**E. cache/routing 통합 계약**
- why: 최종logits만으로 prefix route 변화를 통과시킬 수 있음.
- files: `fdt_v3.py`, `fdt_v4.py`, cache/route auditors.
- expected:63/64/65,511/512/513,2K/4K/8K/16K에서 prefix-complete route IDs/order/membership/input/KV/summary/mass/첫분기/최종logits 기록.
- risk: deterministic grouping 비용. 중간α에는 단일-KV를 억지로 허용하면 안 됨.
- test: teacher-forced replay, generated replay, 긴 prefills+65decode, tail chunks, varied masks, near-top8 ties, repeat full-recompute control.
- pass: selected IDs/order mismatch0; scalar tensor errors는 사전 tolerance 이내(기존 FP32 logit/state3e-4 기준을 사후 완화하지 않음); generated IDs 동일; 최신 후보의 unsupported는 PASS가 아님.16K max에서는16383→16384와 정확한 overflow reject를 구별.

### P1 capability

**F. Exact score/commit train–inference 정렬**
- why: anchor score bias와 candidate normalization·commit 목적 불일치.
- files: `lexical_pointer.py`, exact pilot trainer, `fdt_v4.py:build_exact_memory`.
- expected: 공통 score function·source position objective·first-step competition·실제cursor rollouts; 모든 후보 기준 confidence와 non-copy negatives.
- risk: threshold/LR 재조정 필요, pointer가 base LM을 오염할 수 있음.
- test: fixed teacher prefix에서 동일 candidate scores/order, pointer-only freeze-base pilot; 위치/형식/distractor OOD.
- pass: 먼저 oracle transport100%; multi-record pilot의 chain success와 false-commit CI를 함께 통과. 개발 pilot 목표≥95% whole, absent-query0 false commits/최소100행을 임시 기준으로 사전등록하되 production 보증으로 해석하지 않음.

**G. 원인에 맞춘 generation 학습**
- why: loop score 억제만으로 grounding은 회복되지 않음.
- files: retention trainer, generated-prefix builder, generation evaluator.
- expected: E1–E3 결과에 따라 broad continuation/grounded counterfactual/valid recovery 중 한 축만 변경.
- risk: 잘못된 prefix에 gold suffix를 붙이면 모순 학습; EOS 남용/망각.
- test: 동일 parent/data order/α의 작은 A/B, blind raw rating, category NLL/termination.
- pass: pilot은 개선 신호와 안전성; 확대는 독립 holdout에서 paired preference 개선 신뢰구간이0을 넘고 기존 retention gates 충족. loop 단독은 승격 조건 아님.

### P2 performance

**H. cache·prefill·Exact scan 최적화**
- why: 중간α full-prefix 재계산 및 Python online recurrence/full scan 비용.
- files: local output-blend attention/cache, sparse scan kernels, `lexical_pointer.py`.
- expected: 필요성이 입증되면 dual-branch attention cache; vectorized prefix/state, 동일 semantics의 bounded retrieval.
- risk: routing 경계·수치 동등성 훼손.
- test: 같은 checkpoint/길이/정밀도/실제 생성 토큰에서 latency·peak memory·cache/routing parity.
- pass: 모든correctness/quality gate 유지 후 end-to-end latency 유의미 개선. kernel tok/s만으로 채택하지 않음.

## 9. Next training plan

1. **즉시 optimizer0.** frontier와α.20 후보 및 V20를 고정하고 P0 A/B/D를 먼저 수정·검사한다. long-context 사용 전C, cached 배포 전E가 필수다.
2. **E0–E4 no-update 실험.**32행 screen에서 지배적 원인을 찾고 고정 개발100행으로 확장. 이미 사용한 fixed52/fixed100을 최종 sealed 평가로 재명명하지 않는다.
3. **한 가지 변수만0.25–1M supervised-token pilot.** actual input tokens·시간도 별도 상한. α는 고정; Exact pointer pilot은 base를 동결. large-model dynamics가 중요한 generation은 작은 unrelated proxy 성공만으로 전이하지 않는다.
4. **curriculum pilot 범위:** short base의 supervised-token budget natural50–70%, factual15–30%, grounded retrieval5–15%, Python+JSON5–10%에서 합1이 되는 몇 조합만 사전 선택. 이는 권장 탐색 범위이며 입증된 최적 비율이 아니다. long0에서 시작하고 exact는 독립 pilot. long benefit과correction 통과 후2K→4K→8K→16K, 총 active tokens의1–5%부터 단계적으로 조사하되 실제 supervised 및 loss contribution을 함께 제한한다.
5. **확대 조건:** 원래+1% dual retention gate·blind generation·Exact chain·cache contract를 각각 충족하고 sealed holdout에서 개선 확인. α전환은 고정간격 강행 대신 drift headroom로 결정. 현α.20의 tiny LR 학습은 동일α no-update보다 NLL이 나빠졌으므로 그것만으로 추가100M을 지출하지 않는다.

## 10. Top 5 experiments

1. **gold-free multi-record Exact + random/absent controls:**60/60 주장의 정보 손실을 가장 싸게 복구.
2. **same-runtime four-way continuation + gold-prefix control:**base/positional/self-prefix를 분해, 최대10,240 sampled tokens screen.
3. **matched decoder 및 counterfactual grounding:**greedy/top-p·FP32/BF16·prompt value 교체를 소수행에 교차.
4. **cache/recency 경계 regression:**현재11335 반례 및 full-prefix routing identity, 구group64 수리와 최신후보 구별.
5. **고정α token-objective A/B:**동일 novel shard/order, corrected loss budget vs 기존row budget. 결과가 없으면 더 큰 학습을 시작하지 않음.

fuzzy read/write ablation은2–3에서 residual ambiguity가 남을 때만 추가해 compute를 절약한다.

## 11. Stop conditions

- gold label/source/length에 접근하는 평가가 계속 필요하면 end-to-end capability 승격을 중단한다.
- NaN/Inf, token/cursor provenance 불일치, route IDs/order mismatch가 있으면 해당 학습·배포 경로 중단.
- decoder를 통일하고 exposure를 바로잡은 두 사전등록 bounded pilots에도 blind relevance/grounding이 개선되지 않으면 동일 objective 확장 중단.
- source top1이 chance 수준에 머물면 cursor tuning 중단, retrieval representation/negative distribution 재설계.
- Exact hard-copy false commit이 개선되지 않으면 자동hard-copy 채택 중단; abstention 유지.
- fuzzy read/write ablation이 반복 OOD 평가에서 무영향이거나 제거가 지속적으로 낫다면 relational semantic-memory 중심 주장을 낮추고 local-only/동일compute baseline과 architecture 재검토.
- α를 올리거나 legacy table을 제거할 때 headroom를 반복 소진하고 독립 품질이 회복되지 않으면 quarter-alpha 전환 경로 중단. base 전체 재시작은 그 뒤 비교 실험으로 판단.
- 무실패60행, syntax PASS, loop-free100% 중 하나만으로 전체 연구목표 성공을 선언하지 않는다.

## 12. Final decision

# PATCH_BEFORE_CONTINUING

이 결정은 현재학습 자동재개·424M scratch restart·전체architecture 폐기를 뜻하지 않는다. **새 capability 판정을 신뢰할 수 있도록 평가/회계/실행계약부터 고치라는 결정**이다. 핵심상수 `W64 / anchors256 / top8 / router256 / T=.25`는 유지한다. latent recurrent reasoning은 Base 안정 이전에 추가하지 않는다.

### 내가 compute budget을 직접 지불한다면

1. 모든 checkpoint를 보존하고 optimizer 지출을 동결한다.
2. oracle Exact evaluator와 loss accounting을 먼저 수정한다.
3.32행 crossed continuation·decoder/grounding controls를 돌려 지배적 원인을 고른다.
4. 현재11335 overflow 및 cache/runtime 계약을 해결하고 원인별 작은 단일변수pilot만 수행한다.
5. 독립 의미품질·Exact chain·retention·correctness가 함께 통과할 때만 추가학습 예산을 승인한다.

## 검증 및 한계

- 기존 관련 CPU tests: **36 passed**. 테스트 통과는 위oracle/row-weight/16K 반례를 부정하지 않는다.
- 이번 신규 CPU 검증: oracle key count1/gold payload; row/token loss 반례; 현재configuration의 position11335 FP32 overflow 및 실제16K sparse summary NaN; checkpoint config 기반 parameter counts.
- 보존trajectory0/9.83/36.37/100/200M 및 StageA 파일SHA 모두 원시평가와 일치. V20/frontier/latest candidate SHA도 확인.
-100k source rows의 full-token-row uniqueness, 소비4,816행 exposure 및 fixed52 exact-row overlap0을 확인. 문서/subspan/삭제된historical payload leakage는 UNKNOWN.
- 최신α.20의 새 free-generation/cross-prefix/GPU cache/Exact run은 실행하지 않았다. 따라서 그 결과나 fresh capability PASS를 만들어내지 않는다.
- 기존recovery가 존재한다는 것과 실제 crash-resume replay가 검증됐다는 것은 다르다. 이번에는 recovery restore를 실행하지 않았다.
- 자동 추가 실행은 예약하지 않았다. 중단 후 `AUDIT_RESUME.md`로 이어갈 수 있으며, 원본학습을 임의로 재개해서는 안 된다.

## 감사 증거 묶음

`FDT_AUDIT_EVIDENCE.zip`과 `evidence/`에 코드 SHA 원장, 변경 여부 재검사, 원시 trajectory 재집계, checkpoint SHA 대조, 데이터 회계, CPU 반례와 36개 테스트 결과를 보존했다. 재현 스크립트는 원본 경로를 읽으며 결과를 실행 위치의 `work/`에 쓴다. 새 감사 디렉터리에서 `work/`를 만들고 Python의 `-B` 옵션으로 실행한다. PyTorch가 필요한 스크립트에는 CPU PyTorch 환경이 필요하다. 모델이나 대용량 데이터는 묶음에 복제하지 않았다.
