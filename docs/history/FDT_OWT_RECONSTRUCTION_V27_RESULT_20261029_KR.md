# V27 — 과거 byte 손상15→50% 훈련 완료, 전체 복원 개선은 확인되지 않음

과제는 X→원래 퍼지 앵커의 입력 의미 Z(X)→같은 입력 X̂다. 출력 목적 의미 디노이징은 별도 동결했다. 모델 구조·원래 퍼지 μ/ρ·anchors/value-prior-condition/2views-onlywriter·원문 context 우회 없음·BOS자체prefix 추론을 유지했다. 자유창작 평가나 의미 정답 벡터를 추가하지 않았다.

## 실제 변경과 학습

이번 단일 목적 변경은 **훈련용 past-byte masking**이다. 새attempt1=.15,4096=.50까지 선형 증가,4097–8192는.50을 유지했다. CleanCE.5+corruptCE.5, fullB×129uniform draw수·순서, BOS/EOS/PAD 보호, 과거 byte만 손상, strict deterministic elementwiseCE를 유지했다. 같은 가중치의 teacher 추론과 parameter keys 불변을 CPU에서 확인했다. 손상 비율.50은 한 번의 후보이며 최적값으로 선언하지 않았다.

`runs/owt_fuzzy_reconstruction_20261029_v27`에서 **8,192attempt / 8,190update / AMPskip2(at4202,5350)**를 완료했다. 누적79,844attempt / **79,817update**다. V26의 model·전체63Adam/scaler·모든 RNG·history·전체 ancestry를 정확히 불러와 보존했다. 원래 fixed15% source objective도 ancestry에 명시했다. 새 유한 order351–352/LR구간/clock/call allocation은 과거 예산과 구분했다.

같은8,192train문서/32,768구간, 기존4,096문서 prefix와64dev문서/256입력·정답·순서를 유지했다. 두 finite epoch65,536행/targetUTF8 **5,852,956bytes**, decoder byte+EOS감독11,836,984symbols를 소비했다. 새 문서/평가 panel/다운로드/E2final·registry접근0이다.

실제545.154623초, encode8,232+decoder18,267+denoise0=**26,499/max28,000calls**. 새900초/reserve90초/output6GiB allocation은 CLOSED이며 미사용1,501calls·시간을 재사용하지 않는다. Source누적240,202calls는 별도 읽기 전용1,936calls를 제외한다(합계242,138). 별도 평가 metadata와 모든 과거 실패/원본/fullAdamRNG를 보존했다.

## 실제 schedule와 실행 계약

CPU7검사35.63초/실제 workerCPUpreflight를 호출한 독립OSvalidate 뒤 ONE 실행했다. 새 schedule 경계·p1 old15% 동일성·fixed RNG draw·강한 mask의 포함 관계·미래 byte/길이에 대한 과거 mask/logit 인과성·동일 추론/훈련 전용 guard를 확인했다. 전체 source component 불변 import/enabledscaler AMPskip/fullstate exact-next/마지막7행/복구 시 probability와 actualschema post-Popen 자연종료·예외 cleanup·UNCONFIRMED도 확인했다. TinyCPU는 실제 CUDA 복구나 모델 능력 증거가 아니다.

저장 history의 실제p는1=.15/4096=.50/4097=.50/8192=.50으로 모두 사전 계약과 일치했다. 첫 epoch평균p.325,mask951,090/eligible2,926,478; 둘째p.50,mask1,461,893/eligible2,926,478. 총mask **2,412,983/5,852,956=41.2267%**다. Bernoulli 가정의 p-weighted 기대 mask수2,414,382.63과 구분한다. 전체 CUDA random draw를 CPU에서 재생한 주장은 하지 않는다.

Owner43072→23040(08:49:41.677/.703KST), worker12448→8088(08:49:59.028/.044KST)의 생성·parent·command·생존을 ROOT가08:50:09에 확인했다. Worker-ownerexit0/COMPLETE/allownedgone,08:59:27KST Python0/C48,495,431,680B 확인 후 분석했다. 독립OS1초 감독/즉시 STOP-clock-callcap/FS최대1초+큰 쓰기 전 강제 검사/자원 reserve와 임시 sleepguard finally 해제를 유지했다. ROOT직접/에이전트 없음이다.

## 보조 예측과 상태만의 재복원

같은 관측 dev23,207byte+EOS symbols:

| 상태 | V26 NLL | V27 NLL | V26 보조 정확도 | V27 보조 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력 | .992906 | **.987024** | 70.1125% | **70.6339%** |
| 다른 입력 | 5.512020 | 5.977553 | 25.2295% | 23.3766% |
| 상태 제거 | 4.464589 | 4.880755 | 33.4899% | 31.2837% |

정상 correct16,271→16,392, **+.52139pp/NLL-.005882**다. EOS253/256은 유지했다. 정상-교환 NLLgap4.990528/정상-제거3.893730은 현재 입력별 상태 정보 사용을 뒷받침하지만 완성된 의미 공간·독립 추론을 입증하지 않는다. 보조 정확도70.63%는 정답 과거가 주어진 byte예측이며 의미 보존율이 아니다.

ROOT가 dev16·고정 train16을 모두 읽었다. 두 panel 모두 UTF8/EOS16/16/exact0/충실한 전체 의미 복원0/확인된 부분명제0이다. Train의 NATO 이름과 `Once the first season is` 조건 도입부가 남았지만 drive구입·사용/네트워크 침입이나 시즌 종료·더 이상 qualifying tournaments가 없다는 내용은 빠졌다. 이전 V26의 정확한 `She then told the 25-year-old`는 이번에 나이가20으로 바뀌어 **그 최소 부분도 유지되지 않았다**. 인명·나이·행위·부정·관계의 손실을 따로 읽었다. 불완전한 원문 구간에 완전한 정답 명제를 만들어 붙이지 않았다.

| 상태만의 같은 입력 복원 | V26 dev16 | V27 dev16 | V26 train16 | V27 train16 |
|---|---:|---:|---:|---:|
| chrF | 22.9883 | 23.1231 | 31.8275 | 29.2930 |
| 단어 F1 | 12.2822% | 12.0294% | 29.5042% | 26.3069% |
| 내용어 F1 | 3.2971% | 2.8950% | 17.9170% | 14.6310% |
| 정답 시작 UTF8byte | 3.3125 | 3.0000 | 11.5000 | 9.1875 |
| byte edit 유사도 | 25.5036% | 24.9362% | 38.8246% | 36.4996% |

Dev 자기 원문 unique top1은11/16으로 같다. **이번 결과로 stronger-mask가 전체 복원을 개선했다고 말할 수 없다.** 보조 지표만 보고 성공으로 승격하지 않았다. 같은 출발점의15% control을 병렬로 학습한 실험이 아니므로 차이를 mask만의 고립된 인과 효과로 단정하지 않는다. 추가 학습·order/LR구간·반복 사용 dev/작은 표본/비맹검 판독 한계가 있다. 그렇다고 초기 데이터의 이 결과를 전체 FDT·퍼지 방식의 실패나 한 모듈의 단독 원인으로 일반화하지 않는다.

## 고정50% 구간의 학습과 저장 검산

첫 epoch cleanCE.679193/corruptCE1.671059,둘째cleanCE.536820/corruptCE2.008332다. 첫 epoch에는 난도가 올라가므로 두 corruptloss를 같은 조건의 퇴행으로 읽지 않는다. 추가 forward 없이 저장 history만 나눠 본 **고정p.50** 구간에서4097–6144→6145–8192 onlinecleanCE.549744→.523896/corruptCE2.018285→1.998380/mixed1.284014→1.261138이었다. 가중치·order가 변하는 online평균이며 고정 final trainNLL이나 인과 실험은 아니다. 근거 `root_hold_phase_analysis.json`.

최종 fullstate656,063,217B SHA **be85ab3b6bdd21b70ce9942d40a704a0a04a157245e3336446bde401f5da4648**, algorithm **626462998312ab4634db9ecaff5eb5ec676cbb94ec17cc5cfe87eba283b52fb1**. PlanSHAd427da55242ad8e8fa6e388cfc12778d0429b34f437a80ca20f3c0113ff0991c.

ROOT CPU는 import0/mid4096/final8192의 SHA/safeload/재귀digest, source전체component동일성424b083bc6240fcec286b2d0513a0ac86aff7ddb2472ec036e286d8515449470, 63Adamstep79,817/scaler65,536tracker842/모든 RNG/history/cursor/order/전체 ancestry·외부ledger·sourceoldobjective·동결 reasoning-prefix-basis/finite model·Adam을 확인했다. 모든 실제historyp/노출과 저장23,207logits×3의 NLL/argmax·call수도 재구성했다. 추가 model forward0/새 CUDA 복구 재실행0이다.

근거는 run의 `root_result.json`, `root_state_reconstruction_similarity.json`, `root_all32_state_reconstruction_reading.md`, `root_hold_phase_analysis.json`, native JSON, `scripts/review_owt_state_reconstruction_v27_cpu.py`다. V26·V25·이전 원본/core/typedpartial9/a877/Phase2/archive/fullAdamRNG/raw/실패를 모두 보존했다.

다음 한정 목적은 **ramp를 다시 시작하지 않고 p.50을 고정한 추가 finite학습 한 번**이다. 현재50% 난도에서 학습한 것은 한 epoch4,096attempt이며 onlineloss는 아직 낮아지고 있어 적응 구간을 따로 관측한다. 비율을 더 높이거나 여러 hyperparameter run을 자동 반복하지 않는다. 현재 whole복원 미개선/퇴행을 기준점으로 유지하고, 고정50% 후속에도 개선이 없으면 같은 목적을 무한 반복하지 않는다. 새V28모듈/worker/owner/plan/seal/GPU는 아직 없다. 다음 실행도 source전체AdamRNG를 정확히 보존하고 필요한 CPU delta·actualpreflight·독립OSvalidate·새고정cap·자원/native0 확인 후 ONE 실행한다.
