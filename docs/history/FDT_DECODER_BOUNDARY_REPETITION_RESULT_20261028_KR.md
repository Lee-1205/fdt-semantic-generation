# 디노이징·디코더 경계와 강한 반복 벌점 재계산

최신 사람 지시 “반복에 패널티를 강하게…디퓨전 모델 특성…의미와 언어 중 어디”, “디코더가 이전 diana의 구조를 일부 따라가므로…디코더 문제일 가능성”에 따라 이력·코드·1차 연구를 먼저 확인하고 한 번의 작은 읽기 전용 진단을 수행했다. **현재 우선 수정할 근거는 latent→언어 디코더 연결부에 있다.** 자연문서 의미 자체가 이미 완성됐다고 단정하지 않는다. 기존 관계 예측의 높은 성과와 초기 학습의 NLL 개선은 보존한다.

## 실제 구조와 DIANA 이력

로컬 DIANA 명칭의 `work/run_diana_checkpoint_compare.py`와 `run_diana_nll_compare.py`는 기존424M/FDT-v4계열과24k BPE 모델을 비교했다. 당시 causal 생성의 반복·조기/늦은 EOS와 작은 보존 gate 문제는 이미 기록돼 있었다. 그 과거 소스의 window/RoPE/anchor 구조와 현재 구조를 동일하다고 간주하지 않는다.

현재 실제 호출 계보는 `research/text_context_decoder_v2.py::ContextDecoder` → `src/fdt_local/model_masked.py::Core/Attn/FF`다. 새 causal self-attention2층·RMS·SwiGLU·cross-attention과 prefix의 learned position을 사용한다. DIANA와 AR 계열의 성격을 공유하지만, 현재34개 decoder tensor를 옛 DIANA decoder에서 직접 복사했다는 근거는 확인되지 않았다. V2에서 새로 학습한 decoder를 V3/V4/V5/V6/V7이 이어받은 계보는 확인됐다.

현재 경로는 원문 byte context → VP16단계 → detached latent + 원문 context → AR byte decoder다. 따라서 AR 반복 손실을 추가해도 **latent 경로를 통해 denoiser로 직접 역전파되지는 않는다.** Decoder와 raw-context 경로는 학습되고 VP는 별도 v/clean/byte 손실로 학습된다. 반복 벌점을 확산의 모든 noisy state에 똑같이 주는 변경은 선택하지 않았다. 큰 t의 정상 잡음을 반복 오류로 취급할 근거가 없기 때문이다.

## 작은 실제 분리 측정

V7 최종 저장 모델을 그대로 사용했다. 기존 관측 dev16개에서 동일한 free proposal을 재사용하고 원문 memory/VP memory를 각각 제거·교환했다. 같은 prefix에 대한 logits 비교이며 새 자유생성 benchmark가 아니다. 별도4개 **훈련 행**의 clean-codec 복원은 oracle 용량·인터페이스 진단이며 추론 성과로 세지 않는다.

| dev16,990개 실제 target symbol | Byte NLL |
|---|---:|
| 정상 context + VP | 2.267184 |
| VP0, 원문 유지 | 2.279083 (+0.011900) |
| 다른 행의 VP로 교환 | 2.267334 (+0.000150) |
| 원문 memory0, VP 유지 | 5.218659 (+2.951475) |
| 다른 행의 원문 memory로 교환 | 2.268329 (+0.001145) |
| 둘 다0 | 6.196318 |

현재 반복 prefix의2064위치에서 VP를 다른 행과 바꿔도 argmax가100% 같았고, 원문을 교환해도100% 같았다. 원문을0으로 만들면 달라지므로 context 경로가 수치적으로 필요하긴 하지만, 이 작은 패널에서는 **내용 구분보다 공통 분포·언어 prior로 작동하는 현상**을 지원한다. 독립 의미 이해가 좋다는 증거가 아니다. 네 행 안에서의 교환, 좁고 이미 관측된 문맥이라는 한계를 둔다.

훈련4행에서는 free VP 조건의 decoder 복원 정확도가34.84%, **정답 byte를 완전히 담은 clean codec 조건도33.08%**였다. Decoder가 완전한 코드 정보를 활용해 복원하는 계약을 아직 배우지 못했다는 근거다. Clean codec은 현재 free-proposal 훈련과 다른 입력 분포이기도 하므로 decoder 전체의 표현 불가능성을 증명하지는 않는다. Free VP의 단순 pointwise codec readout 정확도8.48%는 특정 suffix byte와의 비교일 뿐 의미 정확도로 부르지 않는다.

## 위치정보 연결부의 구체적 구조 문제

현재 `ContextDecoder.context()`는 latent에 타입 벡터만 더하고 memory 위치 태그를 더하지 않는다. Prefix query 쪽에는 위치 embedding이 있지만, cross-attention memory의 byte codec에는 index를 명시하지 않는다. 기본 clean target은 byte마다 동일한 고정 직교 codec이어서, 순서가 다른 같은 byte 집합을 별개로 식별할 정보가 없다. Cross-attention은 위치를 별도로 표시하지 않은 key/value 집합의 순열에 불변이다.

실제 checkpoint decoder를 CPU에 strict load해 두 행/32query 위치를 계산했다. Latent129위치를 역순으로 바꾸자 최대 logit 차이는 **2.38e-6**, argmax 일치는100%였다. 가상의 고정 memory 위치 태그를 같은 위치에 추가한 대조에서는 역순 변경의 최대 logit 차이가0.3722로 커졌다. 이는 수정 후 언어 성능이 좋아졌다는 결과가 아니라, **latent 배열의 위치를 실제로 읽을 수 있게 만드는 조건**의 확인이다. Denoiser가 벡터 자체에 부수적인 위치 정보를 담을 수는 있으나, clean codec 목표는 이를 명시적으로 보존하지 않는다.

따라서 단어 반복을 억제하는 것과 함께, 위치·내용이 보존된 latent를 decoder가 문장으로 복원하는 훈련을 우선한다. 과거 V4에서도3개 의미 head가 모두 맞은23개 질문쌍 중3개는 생성 의미가 틀렸으므로 의미→언어의 독립 문제 이력이 있었다. 반대로 새 wording에서는 정확한3head가2/32뿐이어서 원문 의미 읽기도 별도 과제였다. 현재 V7은 V4 semantic reader를 쓰지 않는 다른 자연문서 분기이므로 옛 관계100%를 현재 문서 latent의 완성 점수로 옮기지 않는다.

## 반복 벌점 강도

16개 같은 loop에서 반복4단어를 닫는432개의 byte span을 계산했다. 이는16개 상관 경로의27개 반복 위치씩이며432독립 사례가 아니다. 반복된 ` the` span의 결합 확률 q 평균은**0.021106**이었다. Byte별 argmax가 계속 같은 단어를 선택해도 전체 단어 결합 확률은 작아 span unlikelihood가 약해질 수 있다.

손실을 `β × mean_spans[-log(1-q)]`로 정의하면 동일한 선택 byte logit에서의 gradient와 정상 active-symbol CE의 gradient 비율은 `β × N/K × q/(1-q)`다. 현재 저장 loop에서 계산한 중앙값은 다음과 같다.

| β | 같은 위치 AR CE 대비 gradient 비율 중앙값 |
|---:|---:|
| 0.1 | 0.0102 |
| 1 | 0.1021 |
| 4 | 0.4084 |
| 10 | 1.0209 |

따라서 **β≈10을 강한 훈련 후보**로 잡았다. 실제 전체 parameter gradient 비율이나 새 모델에서의 최적 계수라는 뜻은 아니다. 반복을 닫는 마지막 byte의 logit만2 낮추면 이 저장 경로의 해당 argmax는 모두 바뀌지만, `the`를 틀린 철자로 바꾸는 편법이 될 수 있다. Runtime의 모든 반복 byte에 큰 벌점을 주지 않고, 훈련용 word/span 부정 표적과 정상 내용 CE를 함께 쓴다. 실제 새로운 rollout 효과는 아직 미측정이다.

## 확인한 1차 연구와 구현

[CoDAR 원문](https://arxiv.org/abs/2603.02547)은 연속 latent와 이산 언어 사이의 contextual decoder를 별도로 학습하며, decoder 복원과 확산 생성의 영향을 나눠 본다. 해당 논문은 contextual embedding을 사용하는 반면 현재 후보는 고정 byte codec이므로 순서·복원 계약을 그대로 가정할 수 없다. [DITTO 원문](https://arxiv.org/abs/2206.02369)은 반복한 훈련 문장으로 자기 강화를 줄이는 방법이고, [Unlikelihood 원문](https://arxiv.org/abs/1908.04319)은 부적절한 반복 확률을 줄이는 목적을 제안한다. 논문 효과를 우리 모델의 측정 결과로 간주하지 않는다.

`research/text_decoder_alignment_v8.py`에 별도 memory position table, clean+제한 잡음 codec의 **훈련 전용 복원**, 강한 joint-span unlikelihood 후보를 구현했다. 기존34개 decoder tensor를 그대로 보존하고 memory position66048parameters만 추가한다. 실제 부모 strict import 후 모든 기존 tensor 동일을 확인했다. 관련 CPU 계약3개 통과: 위치 교환의 구별·부모 보존, span loss 값/gradient/범위, 훈련 전용 clean 복원과 eval 거부다. 이는 CPU 후보이며 V8 optimizer·runner·실행 plan·GPU 학습 완료가 아니다.

다음 한정 학습은 이 연결부 복원과 반복 부정 표적을 정상 내용 학습에 섞는다. 기존 모델·Adam/scaler/모든 RNG를 보존하고 새 memory parameter의 optimizer 추가를 명시한다. 좁은 품질 gate로 다시 멈추거나 광범위 baseline을 반복하지 않는다. 실제 driver의 필요한 phase·복구 차이와 비용만 확인하고 고정 예산에서 학습을 이어간다. 추론에는 예측 latent만 쓰며 clean-codec/oracle 점수를 자유 생성 성공으로 세지 않는다.

## 실행 근거

`runs/decoder_boundary_readonly_20261028_v1`: ONE COMPLETE/optimizer0/modelcall140=encode5+denoise80+decoder50+readout5, worker3.554초, 새120초/reserve30/cap160 CLOSED다. Model before/after SHA `e34ff721a0eb440191f0928bd4a8d76bd5cfb1c9a46e59f37dacf0a04e8f7f6d` 동일, 원본 checkpoint도 불변이다. Trace SHA `fe6517f9756b7bdadcaf9085542b606a8ffd3d6beb315c82a0c20b83e1b27e30`, plan SHA `7d44a5520126fb13db62382a48015c48f3c91300d043132b5e02fb706a436966`다. 추가 CPU decoder forward4회는 위 순열 검사이며 GPU 호출과 구분한다.

Owner15676→37716 생성13:46:03.771/.797KST를 Root가 직접 관측했다. Worker40092→31288은 소유 감독기의 CIM identity/parent/command·생존 검증 기록이 있고, Root가 다음 조회한13:46:21에는 이미 종료해 worker 생존을 직접 포착하지 못했다. 이 한계를 숨기거나 같은 GPU 진단을 재실행하지 않았다. Supervisor exit0/COMPLETE와 모든 owned/proxy 부재·freshPython0를 확인한 뒤 CPU 분석했다. 실제4CPU 경계·schema cleanup 검사/worker preflight/독립OS validate를 통과했으며 원본·모든 학습 자산·E2final 접근 금지·root-only 원칙을 유지했다.
