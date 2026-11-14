# 언어·메모리·사고·출력 경계 상세 점검

2026-11-13 최신 지시는 **언어 입력·출력 가중치를 고정한 상태에서, 추후 의미 디퓨전의 사고 가중치만 학습**하는 것이다. 이 지시를 별도 계약으로 구현·검사했다. 지금은 디퓨전 학습을 시작하지 않았다. 최근 V77의 추가 퇴행은 고정된 상류에 대한 출력기 학습에서 발생했다는 근거가 강하다. 동시에 과거 교차 실험에는 추출·결합 쪽 오류와 두 경로의 상호작용도 있었다. 확인 범위를 나눠 다음 설계를 정한다.

## 1. 현재 실제로 실행되는 경로

선택한 연구 기준은 V75이고 최신 학습 후보는 V77이다. 다음 설명은 V75의 실제 모델을 CPU에서 복원해 확인했다.

```text
영어 원문
  → 9층 언어 encoder
  → μ/ρ/confidence와 token value
  → K_P prior + 입력별 K_C long/recent N/D
    + K_E invalid NULL, 내용·복사0
  → global 2개 + 16바이트별 local fuzzy state
  → 순서별 fuzzy-role 결합(role_projection도 이쪽)
  → bound Commit 확정
  → LayerNorm + MLP 위치별 신경 출력기
  → Byte/EOS logits → UTF-8 제약 출력
```

현재 `FuzzyRoundTripV64.verbalize()`는 `readout.from_commit()`을 부른다. 예전 `encoder.decoder`의 AR 모델이나 그 answer-prefix 경로는 부르지 않는다. **따라서 예전 DIANA 계열 AR 디코더의 반복 문제를 현재 출력기의 원인으로 그대로 가져오지 않는다.** 현재 출력기는 각 local state와 global2개를 받아16개 위치의 Byte/EOS를 동시에 예측하는 별도 MLP다. 자기 출력이 다음 token 입력이 되어 증폭되는 AR 반복 경로도 없다. 문장 복원 오류가 없어졌다는 뜻은 아니다.

`role_projection`은 이름상 readout 안에 있지만 실제로는 Commit 생성 시 적용된다. 지금 출력기만 학습한다는7개 tensor에는 이8,192파라미터 결합층이 포함되지 않는다. 추후 입력·출력 동결에는 이 결합층도 반드시 포함한다.

## 2. 전체 파라미터를 실제로 분류했다

실제 전체 모델123,497,080파라미터/285개 parameter tensor를 원래 V75+필수 standalone V74 상태에서 정확히 복원했다. 각 이름·shape·개수를 독립 대조하고 전체 model/buffer digest를 비교했다.

| 부분 | 파라미터 수 | 현재 경로/추후 사고 학습 때 정책 |
|---|---:|---|
| 입력 언어9층 |63,913,728|현재 입력 추출에 사용, 추후 전부 고정|
| 현재 마지막 언어 출력기 |2,908,192|현재 사용, 추후 전부 고정|
| fuzzy 순서 결합층 |8,192|Commit 생성 시 사용, 추후 고정|
| 이전 AR decoder |18,095,876|보존돼 있으나 현재 언어화에서 미호출, 고정|
| 이전 typed 사고 모듈 그룹 |28,262,025|현재 codec에서 디노이징 미호출, 그 상태로 사고가 수행된다고 세지 않음|
| 퍼지 추출·메모리 및 기타 상속 부분 |10,309,067|실제 경로와 미호출 부분을 구분, 추후 전체 codec 동결에 포함|

이 분류의 마지막 두 그룹을 전부 현재 활성 추론 파라미터라고 말하지 않는다. 현재 전체 NN 수와 실제 학습 대상2,908,192개도 다르다. V77 plan에 상속된17,677,856이라는 설명 필드가 남았던 문제는 이전 결과 보고서에 공개했고 sealed plan은 보존했다.

현재 encoder의11단계 상속 MRO와 실제 custom module 정의의 합집합14개 파일을 SHA/AST/class 목록으로 정리했다. 모든285개 parameter를 검사했지만 모든 역사적 연구 분기를 처음부터 다시 실행하거나 모든 자연언어 의미를 검사했다는 뜻은 아니다. 원래 core 파일 묶음 SHA `eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352`도 불변이다.

## 3. 세 메모리의 실제 상태

**K_P 기본 퍼지:** 학습된 canonical anchors와 prior를 μ/정규화ρ/confidence로 읽는다. 인증된 장기 N/D writer·사실 통합은 아직 없다. 기본 prior가 곧 참인 지식 저장소라는 해석은 하지 않는다.

**K_C 입력별 퍼지:** 문맥 token별 value를 앵커에 다르게 누적하는 long/recent N/D와 감쇠·gate를 사용한다. 입력마다 snapshot을 새로 만들고 다른 문서를 batch에서 섞지 않는다. 전체 입력의 global summary와16바이트별 ordered local 기여가 함께 bound Commit에 들어간다. 현재 일회성 입력 snapshot은 추후의 장기 기억 통합과 다르다.

**K_E Exact:** V64 override가 source id/version/SHA만 남기고 raw 문자열·span·입력 의존 key/value는 만들지 않는다. 유효 항목은0인 NULL이다. 상속된 옛 ExactSource 생성 코드는 현재 override를 통과할 때 실행되지 않는다. 미래에 모델 내부에서 명시적인 정확 복사 의도를 처리할 때의 선택·permit 정책은 별도 과제다.

세부 코드에서 확인한 점은 NULL value0도 bias가 있는 `condition` 투영을 거치면 상수 벡터가 될 수 있다는 것이다. 실제 V75에서 투영된 NULL norm1.40913,local branch의 최종 상수 offset norm0.38316이었다. 이것은 입력 내용을 운송하는 Exact 검색이 아니다. 파라미터와 local 구조가 같으면 입력 byte 정체성과 무관한 offset이다. 기존 가중치나 추론을 이 점검을 이유로 수정하지 않았다. **Exact 기여 벡터가 무조건 수치0이라는 말과 Exact에 내용·복사가0이라는 말을 구분한다.**

## 4. 의미 저장과 출력 오류에 관한 근거를 구분했다

| 확인한 이력 | 지지하는 결론 | 범위 |
|---|---|---|
| V71 Commit×출력기 교차8사례 |시각 `:`와 Cyril 오류는 출력기 변경만으로 재현; `27→2&`는 상태·결합 변경만으로 재현; Inez·17은 두 변경의 상호작용|오류를 본 뒤 고른 사례, 전체 오류 원인 비율이 아님|
| V74 상류 고정·출력기만 학습 |같은 상태에서 주 수량42/64→63/64,기존 양수kg21/72→64/72|해당 복원에 필요한 정보가 상태에서 읽힐 수 있었음|
| V75~V77 상류 tensor 불변,수정64개 Commit bitwise 동일 |이 시기의 새 출력 변화는 출력기 학습에서 발생|기존 상태의 모든 의미가 완벽하다는 증명은 아님|
| V77 아주 긴 문맥 |262KiB2개 전체 길이·UTF-8/EOS·대부분 관계 보존,수량28/5436필드 손상|작성 반복 문법·이미 관측된 자료,자연 장문 전체 이해가 아님|

V74의 성공은 단순히 정보가 사라졌다는 설명보다 **읽어내는 출력기의 정렬 문제**를 우선 고칠 좋은 근거다. V77에서는 최근 새 오류의 발생 위치를 출력기 쪽으로 좁힐 수 있다. 다만 V71의 상태·결합 오류 이력까지 출력기만의 문제로 지우지 않는다.

과거 V62의 역할/관계 의미 정렬 실험은 별도 binder 분기다. 관계210/256,전체 사실42/256처럼 관계와 개체 결합 성능이 달랐다. 이 점수를 현재 OWT codec의 의미 추출 정확도로 옮기지 않는다. 좋았던 관계 결과는 보존하면서 현재 경로의 의미성은 따로 평가한다.

## 5. 복원 가능성과 추상 의미 공간은 같은 주장인가

현재 Commit은 global2개뿐인 압축 벡터가 아니다. 입력16바이트마다512차원의 local state가 늘어나고16개 순서 role은 각각32차원의 결합 기여를 가진다. 원문 bytes를 Exact로 복사하지는 않지만, 문자·순서 정보를 상당히 보존하는 신경 codec이다. **높은 복원은 정보 가용성의 근거이며,곧바로 표현이 바뀌어도 동일한 추상 의미를 갖는 공간의 인증은 아니다.**

실제로 임의의 가역 부호화도 완전 복원은 가능하다. 의미를 위해서는 같은 뜻의 표현 변화와 의미를 바꾸는 최소 변화가 구별돼야 한다. 추상 표현 식별에 필요한 모델·자료의 가정에 관한 [Locatello 등의 원 논문](https://proceedings.mlr.press/v97/locatello19a.html)도 참고했다. 그 연구의 비지도 disentanglement 정리가 현재 FDT의 불가능성이나 실패를 증명한다는 식으로 적용하지 않는다.

언어9층은 층당48바이트 causal window라 직접 local receptive field의 상한은 이전423바이트다. 아주 긴 입력의 native/stream 동등성은 이 같은 계산을 chunk별 K/V로 실행한 것이다. 전체 입력의 fuzzy N/D/global pooling이 추가로 읽히지만256KiB 전체에 dense Transformer attention을 한 것은 아니다. local Commit도 길이에 따라 증가한다. 장문 복원과 임의 장거리 사고를 구분한다.

현재 강화할 검사는 새 무작위 대형 baseline보다 **누가/무엇을/누구에게,관계,부정,수량·단위,시간·순서**의 미학습 조합과 동치 표현이다. 원문과 완전히 같은 문장만 정답으로 요구하지 않고,이전처럼 실제 source/output를 읽어 다른 값·식별자 손상·표면 차이를 나눈다. 기존 관측 dev를 새 블라인드 표본으로 세거나 평가 정답을 TRAIN으로 넣지 않는다.

## 6. 사고 가중치만 학습하게 하는 코드 계약

새 `research/frozen_language_boundary_contract_v1.py`는 codec 전체를 `requires_grad=False`+eval로 고정하고 stale gradient를 지운다. Encoder·K_P/prior·K_C 관련 가중치·위치 결합·terminal·언어 출력기·상속된 미사용 module·모든 buffer까지 고정 상태 digest에 포함한다. 별도 reasoner의 이름 allowlist와 optimizer 소유 파라미터가 정확히 같아야 하며 공유 parameter·같은 storage view·중복 소유·나중의 객체 교체를 거부한다. buffer 변화나 잘못된 `.train()` 전환도 잡는다.

중요한 구현 차이는 **출력기 가중치 동결과 출력기 연산의 gradient 차단은 다르다**는 점이다. 나중에 언어 손실을 사고 모델에 보낼 경우 frozen decoder 연산은 autograd 안에 두어 `loss→decoder 입력 의미→reasoner`의 기울기를 유지해야 한다. Decoder 자체의 gradient와 optimizer 소유는0이다. 전체 decoder를 `no_grad`로 싸면 이 경로까지 끊어진다. 이는 [PyTorch autograd 공식 설명](https://docs.pytorch.org/docs/main/notes/autograd.html)의 동결 방식과 일치한다.

Tiny CPU11검사에서 이유 없는 입출력 update,stale gradient,공유 storage,외부 frozen parameter가 optimizer에 들어가는 경우,buffer 쓰기,training mode 변경,학습 이름 누락·중복,parameter 객체 교체를 검사했다. 작은 fixture의 시험 update는 디퓨전 학습 결과가 아니다.

이어 실제 V75 전체 크기 모델을 정확 복원한 뒤 random latent를 별도512×512 선형 시험 모듈에 통과시키고 현재 실제 출력기로 읽어 gradient 경계를 검사했다. 실제 출력기 forward1/입력 encoder0/원문 평가0/디퓨전0이다. 선형 시험 모듈에서 AdamW1회만 수행해 **기존123,497,080파라미터 및 buffer는 전후 digest가 같고 시험 모듈만 달라짐**을 확인했다. 새로운 사고 모델 능력·학습 trial·출력 개선으로 세지 않으며 시험 가중치 checkpoint도 만들지 않았다.

CPUworker3.875초/OSowner6.875초,새90초CPU+30초종료reserve/32MiB 한도 안에서 완료했다. Owner가 worker native26084,parent21468의 생성/명령/실행 파일/생존을 검증하고 exit0/allownedgone을 확인했다. ROOT는08:50:47.883에 freshPython0와 모든 owned/proxy 부재를 먼저 저장한 다음 결과를 읽었다. ROOT가 네 PID의 동시 생존을 봤다고 주장하지 않는다. CPU owner에는 Torch가 없었고 CUDA 초기화false다. 실제 full worker/scaler/RNG 재시작이나 실제 의미 디퓨전 학습은 이번 검사가 아니다.

## 7. 디퓨전은 이 경계에서 천천히 설계한다

```text
X ── 고정 E·K_P/K_C/K_E ──► 입력 의미 Zx,읽기 전용 메모리
                                       │
작업/질문 의미 ────────────────────────┤
                                       ▼
                   별도 사고 디퓨전 Rθ만 학습
                  noise → 목표 답변 의미 Zy*
                                       │
                             답변 의미 Commit 확정
                                       ▼
                                고정 언어 출력기 D
                                       ▼
                                      답변 Y
```

학습 목표의 의미는 입력 의미 `Zx`를 그대로 정리하는 데 한정되지 않는다. **입력 의미와 작업 조건으로부터,답변이 될 특정 의미를 만들도록** 학습한다. 지도 데이터가 있을 때 고정 encoder로 정답 답변의 `Zy=E(Y)`를 만들고 이것을 학습 표적으로만 사용한다. Denoiser의 조건에는 입력 X의 의미·질문·허용 메모리만 들어가고 정답 Y의 bank/역할/clean latent를 조건으로 몰래 주지 않는다. Noisy target은 학습 corruption 경로에만 있다. 추론은 정답 없이 noise에서 답변 의미를 생성한다.

이전 typed core의4관계+16역할+bias의21field basis와 현재512차원 codec Commit은 schema가 다르다. 옛 `denoise()`는 `query_roles/facts/query_mask` 등의 typed 입력을 요구한다. **현재 Commit을 그 함수에 그대로 넣는 연결은 올바른 새 사고 경로가 아니다.** 이전 관계 전이·time/refiner 가중치는 보존하되 재사용하려면 별도 사고 모듈로 복사하고 codec과 storage/parameter를 공유하지 않으며,질문·memory 읽기·출력 Commit geometry를 새로 명시해야 한다. 새 조건 projection·read router·길이 예측 등이 필요하면 모두 사고 모듈 소유로 한정한다.

메모리 snapshot의 값·canonical anchor·prior는 고정하고 입력별 내용도 사고 도중 덮어쓰지 않는다. 다만 사고 모듈이 만든 query로 메모리를 읽는 연산은 query 쪽으로 미분될 수 있어야 한다. snapshot을 detach하는 것과 read 연산 전체를 no_grad로 감싸는 것을 혼동하지 않는다. 새로운 read/query router를 학습한다면 별도 사고 모듈 소유이며 원래 추출 router를 공동 사용해 업데이트하지 않는다.

답변 길이가 입력 길이와 같지 않을 수 있다. 답변 의미의 slot 수/중단을 사고 모듈에서 결정해야 하며 target byte 길이를 추론 입력으로 주거나 입력 local slot 개수를 무조건 정답 길이로 쓰면 안 된다. Decoder는 승인된 schema의 확정된 답변 Commit만 읽는다. 현재 codec-only phase와 미래 reasoned Commit의 타입/epoch/메모리 출처 연결도 명시적으로 검증해야 한다. 단순히 phase guard를 제거하지 않는다.

연속 latent를 denoise해 출력과 연결하는 [Diffusion-LM 원 논문](https://arxiv.org/abs/2205.14217)은 참고했다. token embedding diffusion이 현재 퍼지 의미 Commit과 같은 공간이라거나 독립 사고를 보장한다고 보지 않는다. 처음에는 작은 관계·반사실·수량 과제를 이용해 고정 `D(E(Y))`의 언어화 한계,사고 출력 latent의 사실,최종 자연어 사실을 각각 측정한다. 잠재 MSE만 낮아지거나 유창한 문장이 나온 것으로 사고 성공을 세지 않는다.

## 8. 다음 실행의 순서

우선 V75 기준의 언어 출력에서 숫자/부호 margin 보존을 고치는 기존 다음 계획을 유지한다. 이 언어 안정화 단계에서는 선택한 출력기만 학습할 수 있으며 **아직 사고 학습 단계가 아니다**. 사고 학습으로 단계가 바뀌면 그 시점에 확정한 입력·메모리·결합·출력 전체 checkpoint를 새로 봉인하고 모두 동결한다. 그 뒤 같은 trial에서 codec fine-tuning을 섞지 않는다.

새 사고 trial의 실제 runner/목적/지도 쌍/schema/clock·call·노출·출력cap/전체 optimizer/scaler/RNG 복구는 아직 없다. 이번에 만든 것은 동결 소유권 계약과 점검 보고서·설계 정책이다. 새 GPU 실행 없이 기존 모든 원본·가중치·Adam/RNG·실패·게임을 보존했다. C여유 약111GiB를 유지했다. 향후100GiB+출력/temp/STOP+8GiB headroom과 독립OS 감독을 충족한 별도 한정 실행만 한다.

근거: `research/artifacts/language_reasoning_boundary_20261113_v1_cpu/{plan.json,actual_model_fixture/actual_boundary_audit.json,root_independent_review.json,tiny_contract_review.json,cpu_preflight_owned/final_status.json,ROOT_native_terminal.json}`. 고정 endpoint digest는 `bdcc74d6d4c50097559ae62957b21c84cfa9e6ed07c47a1f194a75b32e9fa9b2`다. 연구 기준V75 SHA `f01e851284df0bd971498ee709b81158abb4ce157bf7e19578c146614411c124`,필수 standalone V74 SHA `734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7`를 보존했다.
