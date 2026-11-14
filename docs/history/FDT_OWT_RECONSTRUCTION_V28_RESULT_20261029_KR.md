# V28 — 고정50% 훈련 완료, 표면 복원 일부 개선·전체 의미 복원 미완료

언어→원래 퍼지 입력 의미→같은 입력 재복원만 학습했다. 출력 목적 의미 디노이징은 별도 동결했다. 원래 μ/ρ·anchors/value-prior-condition/2views-onlywriter·원문 context 우회 없음·BOS자체prefix 추론은 유지했다. 무조건 창작/일반 LM 완성 평가로 해석하지 않는다.

## 실제 실행과 관측 범위

`runs/owt_fuzzy_reconstruction_20261029_v28` ONE 실행에서 **8,192attempt / 8,188update / AMPskip4(at190,2908,3107,7217)**를 완료했다. 누적88,036attempt / **88,005update**다. V27의 끝난 ramp를15%로 다시 시작하지 않고 처음부터 끝까지 train past-byte p=.50을 유지했다. 모델 구조·cleanCE.5+corruptCE.5·엄격 결정성·AMP·추론·corpus는 같았다.

실제538.894128초, encode8,232+decoder18,292+denoise0=**26,524/max28,000calls**였다. 새900초/reserve90초/output6GiB 예산은 CLOSED이며 미사용1,476calls·시간을 재사용하지 않는다. Source누적266,726calls와 별도 읽기 전용1,936calls는 구분한다(합계268,662). Same8,192train문서/32,768구간의 두 epoch361–362로65,536행/targetUTF8 **5,852,956bytes**를 소비했다. 기존4,096문서 prefix/64dev문서256행·정답·순서는 유지했다. 새 다운로드/평가 panel/E2final·registry접근0이다.

CPU7검사39.97초/실제 workerpreflight를 호출한 독립OSvalidate 뒤 ONE 실행했다. SourceV27schema/완료p50·전체component/이전ramp계약·외부ledger, enabledscaler AMPskip/fullstate exact-next/마지막7행, 처음부터p50·V27endpoint와 같은 RNG/보호된 과거 mask·미래 byte/길이 인과성·같은 추론/actualschemaOS post-Popen cleanup을 확인했다. 준비를 GPU성능으로 세지 않는다.

Ownerproxy34364는09:18:56.140KST에 시작했다. 실제 owner13784/workerproxy28588/native27616의 parent·command·identity와 native 생성09:19:15.033KST는 독립OSowner 기록으로 검증됐다. **ROOT가 다음 직접 조회한09:45:59에는 이미 종료됐으므로 V28 live 생존을 직접 포착하지 못했다.** 이 한계를 숨기거나 관측을 위해 같은 학습을 재실행하지 않았다. Worker-ownerexit0/COMPLETE/allownedgone와 freshPython0 확인 후 분석했다. Supervisor의 마지막 cachedprogress가 final reconstruction인 것은 오래된 값이며 freshworkerprogress는complete다. `root_native_launch_observation.json`도 legacy이름과 달리 이번에는 postterminal 관측이다. 독립OS감독·즉시 STOP/clock/callcap/자원보호·임시 sleepguard finally 해제를 유지했다.

## 학습 조건과 보조 예측

History p는 모든8,192회에서.50이다. 첫 epochmask1,464,146/eligible2,926,478,둘째1,462,984/2,926,478,총**2,927,130/5,852,956=50.0111%**다. Decoder byte+EOS감독11,836,984symbols, Bernoulli 가정 기대mask2,926,478과 실제 draw합계는 구분한다. Online epochcleanCE.628226→.483687/corruptCE2.032113→1.972563/mixed1.330170→1.228125는 학습 중 평균이며 고정 final trainNLL은 아니다.

같은 관측 dev23,207byte+EOS symbols:

| 상태 | V27 NLL | V28 NLL | V27 보조 정확도 | V28 보조 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력 | .987024 | **.996612** | 70.6339% | **71.1768%** |
| 다른 입력 | 5.977553 | 6.371324 | 23.3766% | 22.0752% |
| 상태 제거 | 4.880755 | 5.229656 | 31.2837% | 30.1762% |

정상 correct16,392→16,518, **+.54294pp**, 그러나 **NLL+.009587 악화**다. EOS253/256 유지. 정확도 상승만으로 전체 확률 예측이 개선됐다고 보고하지 않는다. 정상-교환 NLLgap5.374712/정상-제거4.233044는 상태 정보 사용을 뒷받침하지만 완성된 의미 공간·독립 추론·특정 모듈 원인 증거는 아니다. 정답 과거를 주는 보조 정확도71.18%는 의미 보존율이 아니다.

## ROOT의32개 직접 판독

Dev16과 고정 train16 모두 UTF8/EOS16/16/exact0/충실한 전체 의미 복원0/확인된 부분명제0이다. Train에서 `Outside the court`, Eduardo, NATO, Craig Charles 같은 부분은 더 보존됐다. 그러나 Brazil은 Brazau가 아니고, 현금 요청·카드 수수료 회피·drive 사용/침입·네 사람의 joining 관계는 빠졌다. 나이25가20으로 바뀐 이전 telling-recipient 부분도 돌아오지 않았다. 원문 자체가 불완전한 구간에 완전한 정답 명제를 만들어 붙이지 않았다.

| 상태만의 같은 입력 복원 | V27 dev16 | V28 dev16 | V27 train16 | V28 train16 |
|---|---:|---:|---:|---:|
| chrF | 23.1231 | 22.6713 | 29.2930 | 34.9578 |
| 단어 F1 | 12.0294% | 12.2564% | 26.3069% | 30.1430% |
| 내용어 F1 | 2.8950% | 2.1946% | 14.6310% | 21.7660% |
| 정답 시작 UTF8byte | 3.0000 | 3.1250 | 9.1875 | 13.8125 |
| byte edit 유사도 | 24.9362% | 25.9390% | 36.4996% | 42.2928% |

Dev 자기 원문 unique top1은11→12/16이다. Train 표면 복원은 개선됐지만 dev는 지표가 엇갈렸고 **전체 의미 재복원 개선은 확인되지 않았다**. 약속한 한 번의 p50hold 후속이 끝났으므로 같은 목적을 세 번째로 자동 반복하거나 mask비율을 더 높이지 않는다. 동시에 작은 데이터/훈련과 반복 관측 dev로 전체 FDT·퍼지 실패나 extractor/decoder 단독 원인을 단언하지 않는다. 비교는 같은 출발점의 독립 control이 아니라 추가 학습·order/LR구간이 섞인 순차 관측이다.

## 저장 검산과 다음 구체적 변경

최종 fullstate658,087,089B SHA **bc0fe9d59cf39d59290b64377d40e53e6f214c083eaaa3206b2bcf84961c4246**, algorithm **a61b9ddc88efab59947f0458a78d9423b27c04d0fa60f79e39fd080065f7575c**. PlanSHAb9bb06b30fc791508ade5c8df42612d61778a79ce078c64e955618ddd13b8eb4.

ROOT CPU는 import0/mid4096/final8192 SHA/safeload/재귀digest, source전체component동일성abbd6d93a0d07ed0e1e6fdebc8f335aff2dd607b176e753d491762b1ad42fb2e,63Adamstep88,005/scaler32,768tracker975/모든 RNG/history/order/cursor·전체 ancestry·외부ledger·source끝난ramp·constantp50/동결 reasoning-prefix-basis/finite model·Adam을 확인했다. 모든 실제historyp·노출/저장23,207logits×3 NLL·argmax·실제 call수도 독립 검산했다. 추가 model forward0/새 CUDA복구재실행0이다. 근거 run의 `root_result.json`, `root_state_reconstruction_similarity.json`, `root_all32_state_reconstruction_reading.md`,nativeJSON와 `scripts/review_owt_state_reconstruction_v28_cpu.py`다.

다음 판단을 위해 실제 code와 이전 V7/V9 결과를 좁게 다시 읽었다. 현재 decoder는2개의causal Core와 semantic-memory cross attention이며, V9의 정답 byte-index codec 주소를 unordered fuzzy의미2views에 사용하지 않는다. V9oracle100%는 이번 의미 복원 성과가 아니며, V7의 rawcontext 우회도 현재 경로에는 없다. 이를 근거 없이 현재 decoder의 단독 고장이나 정상으로 단정하지 않는다.

현재 입력 언어 encoder9개block 중 **마지막7·8 두 개와pool만** OWT학습에 연결돼 있고 앞7개/embedding은 동결이다. 다음 후보는 구조와 inference·원래 fuzzy·p50loss를 유지하면서 **기존 block5·6 두 개도 학습 가능하게 하는 것**이다. 원인 확정이 아니라 별도 학습 범위 가설이다. [ULMFiT의 단계적 unfreezing 연구](https://aclanthology.org/P18-1031.pdf)는 이 방향의 선행사례지만 과제·구조가 다르며 이 byte-fuzzyAE의 성공이나 최적 층 수를 보장하지 않는다.

`research/owt_extraction_tail_v29.py` 후보와 `scripts/review_owt_extraction_tail_v29_cpu.py`를 ROOT가 작성했다. 실제 V28source CPU safeload/strictcopy 후 **모든 model tensor hash96facf5ae63fd526fe6e648a11dbdfaac3ea209ebadd01c88bec05bfe60ee57d가 동일**, 총111,485,711params 유지/newparams0,기존63trainable tensor24,150,914params에 block5·6의18tensor14,158,848params만 thaw,후보trainable38,309,762다. Embedding/앞5blocks·reasoning은 계속 동결한다. CPU생성 RNGscope복구/CUDAinitializedfalse/실제모델forward0/optimizer생성·update0이다.

이 후보는 **CPU trainability inventory만 완료**했다. 기존63Adam 전체정확보존+새18tensor빈Adam그룹/새frozen-scope digest/전체 driver/scaler/RNG exactrestore/실제 plan/preflight/owner/seal/GPU는 아직 없다. 예상추가FP32Adam moments113,270,784B·기존payload기준771,357,873B는 실측 새 checkpoint가 아니다. 구체적 후속은 `outputs/FDT_OWT_EXTRACTION_TAIL_V29_NEXT_20261029_KR.md`다. 모든 원본/core/typedpartial9/a877/Phase2/archive/fullAdamRNG/raw/실패를 보존했다.
