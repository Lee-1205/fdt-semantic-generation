# V10 다음 한정 학습 계약 — CPU component 준비

**최신 사람 지시로 우선순위 변경:** 핵심은 언어→입력 의미 추출→출력할 특정 의미의 조건부 디노이징→언어화다. 아래 byte-state alignment 후보는 언어 연결부의 보조 연구로 보존한다. 이것을 본래 의미 개선으로 바꾸어 부르거나 자동으로 다음 GPU 학습 대상으로 삼지 않는다. 다음 주 계약은 `FDT_SEMANTIC_OUTPUT_PIPELINE_NEXT_CONTRACT_20261028_KR.md`다. 후보를 삭제하거나 이력을 소급 실패로 바꾸지 않는다.

V9연결부는 정답 codec990/990, 반대 순서8/8을 복원했다. 실제 예측 VP의 출력은 여전히 비문이다. 다음 목적은 이 디코더를 고정한 상태에서 VP와 language tail을 복원 가능한 상태 분포에 맞추는 학습이다. 범용 의미·의식 상태의 완성을 주장하지 않는다.

현재 `research/text_state_alignment_v10.py`와 `tests/test_state_alignment_v10.py`의4CPU검사가 통과했다. 실제 V9최종 모델을 CPU strict copy했고 state digest는 `de63e1f723b2fde510c2ce2b1a37f81598ffe8d5677438c49242e9019dbf6d57`, 부모 모델 forward0/CUDA초기화false다. 전체100,994,050parameter 중 VP/language-tail41,924,608개를 학습 대상으로 하며 그중 language tail14,158,848개다. 디코더9,314,562개는 동결한다. 새 parameter0이다.

4검사는 frozen decoder를 통과하는 CE gradient가 VP velocity/projection/language tail에 연결됨, 직접 prompt memory는 내용0임, t=1에서 denoiser 입력이 target에 의존하지 않음, decoder 가중치 불변/모든 trainable gradient 존재, clean-vMSE 수학적 관계, eval에서 학습 target API 거부를 확인한다. 이는 tiny CPU component 검사이며 fullworker/AMP/scaler/모든 RNG의 실제 복구나 GPU 능력 검사가 아니다.

다음 ROOT는 후보를 다시 작성하거나 V9평가를 재실행하지 말고 실제 driver와 owner를 완성한다.

- V9source SHA `daaaaf0b780e93844efd045321a27a0a96769c21014d426858640c32810181b2`의 모든145Adam/scaler4096/tracker498/RNG를 보존한다. Optimizer group을 새 requires_grad 목록으로 걸러 decoder의 동결 moments35개를 유실하지 않도록 한다. 언어40개·VP70개가 다시 갱신되고 decoder35개 moments는 동결 상태로 보존돼야 한다.
- 현재 component objective는 `vMSE+.25cleanMSE+.5VPbyteCE+.25frozenDecoderCE`, β0, EOS CE4다. Noisy gold는 정상 diffusion 학습 target에만 쓰며, decoder 조건은 **같은 샘플의 예측 clean state**다. 예측 clean→decoder를 detach하지 않는다. 직접 raw prompt→decoder는 이 훈련 손실에서0, raw prompt→VP condition 경로는 살아 있다. 실제 추론에는 gold/길이/역할이 없다. 독립 자유 표본+임의 gold suffix 학습은 하지 않는다.
- 기존3271훈련행/512문서/held444행 분할을 유지한다. Prospective4유한 epoch seed91–94/최대1636attempt/13,084행/1,538,700targetUTF8bytes, batch8/micro4를 검토한다. Skip도 attempt/cursor/노출을 소비하고 되감지 않는다. 실제 driver 확인 뒤 고정하며 새로운 문서가 늘었다고 하지 않는다.
- 각 micro encode1+denoise1+decoder1=3calls,4epoch훈련 예상9816이다. 최종 기존 dev NLL1998+자유생성 최대1100에 필요한 소수의 initial/final latent logging을 더한 비용을 **실제 구현에서 계산한 뒤** 새 cap 약14000/900초/reserve90/output6GiB/start40GiB/GPU12GiB 내로 고정한다. 아직 실행 plan이나 열린 예산은 아니다. V9의 unused1459call/시간은 재사용하지 않는다.
- 초기·최종의 소수 동일 원문에서 실제 예정된 forward의 latent norm/codec projection/고잡음 t-bin loss를 저장한다. 큰 broad baseline/여러 solver trace/gradient probe를 반복 선행 조건으로 만들지 않는다. 최종 freeVP와 관련성 있는 문장·EOS·반복을 함께 보고, teacher-noisy reconstruction을 실제 자유 생성 성공으로 세지 않는다. 한 자유 표본 조건부 NLL은 marginal likelihood가 아니다.
- Fullstate에서 실제 frozen ownership 변경을 명시하고 decoder/model/moments/RNG를 보존한다. Import/마지막7행/유한순서/AMP skip/정확 next update 복구, 새로운 plan schema의 worker preflight·독립 OS validate·postPopen cleanup에 필요한 변경만 검사한다. 이전9개 CPU검사를 다시 무한 반복하지 않는다.
- 실제 worker/fullstate/owner가 완성되고 ROOT가 코드·비용·SHA·고정 clock/call/노출/resource를 검토한 뒤 ONE 정상 실행한다. 하드웨어 재승인 질문은 불필요하다. 실행 중 sealed code 변경·삭제·rotation·자동 재시작·상한확대는 하지 않는다. Terminal 뒤 allowned/proxy 종료부터 확인한다.

후보의 actual worker/fullrestore/plan/seal/GPU/실제 optimizer trial은 현재 **없다**. 증거는 `research/artifacts/text_state_alignment_20261028_v10_cpu/root_candidate_review.json`이다. 기존 모델·전체 Adam/scaler/RNG·원문·실패를 보존하고 E2final/registry/에이전트는 사용하지 않는다.
