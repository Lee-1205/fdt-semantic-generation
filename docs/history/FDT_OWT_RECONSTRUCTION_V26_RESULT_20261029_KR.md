# V26 — 8,189회 추가 갱신, 보조 예측 70.11%, 완전복원 격차의 저장 증거 계산

과제는 X→원래 퍼지 앵커의 입력 의미 Z(X)→같은 입력 X̂다. 출력 목적 의미 디노이징은 별도 동결했다. 원문 context 우회·정답 prefix/길이/역할을 BOS부터의 복원에 주지 않는다. 기존 파일명 `generation/free`는 무조건 창작이 아니라 이 상태 조건 재복원이다.

## 실제 학습·종료

`fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261029_v26` ONE 실행에서 **8,192attempt / 8,189update / AMPskip3(at3896,5531,7586)**를 완료했다. 누적71,652attempt / **71,627update**다. 같은8,192train문서/32,768구간의 두 finite epoch341–342,65,536행/targetUTF8 **5,852,956bytes**를 소비했다. 기존4,096문서 prefix와64dev문서/256행·정답·순서는 유지했다. 다운로드0/새 panel0/E2final·registry접근0이다.

모델·원래 퍼지 구조·loss(clean.5+15%손상prefix.5)·AMP·엄격 결정성·추론은 V25와 같았다. Source 전체63Adam/scaler32,768tracker324/모든 RNG·history·ancestry를 보존하고 별도 읽기 전용1,936calls 참조를 fullstate와 source ancestry에 명시했다. 새 local order/cosine 구간·clock/call allocation은 과거 예산과 구분했다.

CPUdelta5PASS29.09초 뒤 독립OSvalidate가 실제 workerCPUpreflight를 호출해 PASS했다. 중복 standalone preflight를 추가하지 않았다. Source importer·전체component/외부ledger·enabledscaler AMPskip/maskRNG exact-next/마지막7행·finiteorder와 실제 schemaOS post-Popen 자연종료·예외 cleanup·UNCONFIRMED 경로를 확인했다. 변경하지 않은 자원 guard 계약은 재사용했다. TinyCPU는 실제 CUDA 복구나 모델 능력의 증거가 아니다.

실제542.774689초, encode8,232+decoder18,278+denoise0=**26,510/max28,000calls**였다. 새900초/reserve90초/output6GiB 예산은 CLOSED이며 미사용1,490calls·시간을 재사용하지 않는다. Source누적213,703calls와 별도 읽기 전용1,936calls를 구분한다(합계215,639). Mask877,944/eligible5,852,956=.1500001, 두 decoder loss byte+EOS감독11,836,984symbols다.

Owner40596→8712(08:17:29.411/.437KST), worker21860→13064(08:17:45.314/.330KST)의 생성·parent·command·생존을 ROOT가08:17:54에 관측했다. Worker-ownerexit0/COMPLETE/allownedgone/08:26:56KST Python0/C50,243,002,368B 확인 후 분석했다. 자원 전체 검사515회/독립OS1초감시/즉시 STOP·clock·callcap/큰 쓰기 전 강제 검사/임시 sleepguard finally 해제를 유지했다. ROOT직접/에이전트 없음이다.

## 보조 예측과 재복원

같은 관측 dev23,207byte+EOS symbols:

| 상태 | V25 NLL | V26 NLL | V25 보조 정확도 | V26 보조 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력 | .995085 | **.992906** | 69.7333% | **70.1125%** |
| 다른 입력 | 5.267533 | 5.512020 | 25.4277% | 25.2295% |
| 상태 제거 | 4.259455 | 4.464589 | 33.8346% | 33.4899% |

정상 correct16,183→16,271, **+.37920pp/NLL-.002179**다. EOS보조예측251→253/256. 정상-교환 NLL차이4.519113/정상-제거3.471682는 현재 입력별 상태 정보 사용을 뒷받침한다. 정답 앞부분을 주는 보조 정확도를 의미 보존율로 부르지 않는다. 보조 개선 폭은 작아졌지만 이것으로 전체 FDT·퍼지 방식의 실패나 extractor/writer 단독 원인을 단정하지 않는다.

학습 중 epoch평균 cleanCE.739771→.592252/corruptCE1.261526→1.164223/mixed1.000649→.878237은 계속 낮아졌다. 고정 final trainNLL은 아니다.

ROOT가 dev16·고정 train16을 모두 읽었다. 두 panel 모두 UTF8/EOS16/16/exact0/충실한 전체 의미 복원0이다. Train15의 **“She then told the 25-year-old”**는 같은 주어·말한 행위·수신자/나이를 보존한 최소 부분만 별도로 인정했다. 경찰 신분과 움직이지 말라는 지시는 여전히 없다. Train14의 V25 정확한 두 인명은 이번에는 사라졌다. 이름·나이·도입부 일부의 개선과 퇴행을 구분한다. 불완전한 원문 구간에 완전한 정답 명제를 만들어 붙이지 않았다.

| 상태만의 같은 입력 복원 | V25 dev16 | V26 dev16 | V25 train16 | V26 train16 |
|---|---:|---:|---:|---:|
| chrF | 22.1957 | 22.9883 | 31.3716 | 31.8275 |
| 단어 F1 | 11.0381% | 12.2822% | 25.9326% | 29.5042% |
| 내용어 F1 | 1.6161% | 3.2971% | 14.4357% | 17.9170% |
| 정답 시작 UTF8byte | 3.0625 | 3.3125 | 11.5000 | 11.5000 |
| byte edit 유사도 | 25.6593% | 25.5036% | 38.4491% | 38.8246% |

Dev 자기 원문 unique top1은11/16으로 같다. 표면지표의 일부 개선을 전체 의미 복원 성공으로 부르지 않는다. 반복 관측 dev/작은 표본/비맹검 ROOT 판독/다수 설계 선택의 한계를 유지한다.

## 기존 자료로 계산한 첫 오류 위치

추가 모델 호출 없이 같은 저장logits와 raw 복원문만 사용했다. 전체256teacher행의 argmax가 전부 맞는 행은0, 첫 오류 전 길이는 평균6.1328/중앙값4tokens였다. 고정16개에서는 teacher의 첫 오류 위치와 복원 출력의 첫 오류 위치·그때 선택한 symbol이 **16/16 모두 일치**했다. 두 경로의 처음부터 맞는 길이는 모두 평균3.3125bytes다. 첫 오류 뒤 위치에서 teacher는 매번 원래 정답 과거를 다시 제공받아68.7573%(14,701/21,381)를 맞췄다.

이는 보조 정확도와 BOS 복원 사이의 격차를 보여 주는 기존 증거다. 첫 오류가 나기 전에는 두 경로의 과거가 같으므로 **첫 오류 자체까지 모두 잘못 생성한 과거의 누적으로 설명할 수 없다**. 입력 상태→해당 byte를 읽는 매핑도 아직 부족하다. 첫 오류 뒤 teacher와 자체 prefix의 차이는 추가 훈련 목적을 검토할 이유지만 단일 원인 증명은 아니다. 근거 `research/artifacts/owt_v26_prefix_diagnostic_20261029_cpu/result.json`, `scripts/review_owt_prefix_gap_v26_cpu.py`; 추가 forward0/optimizer0/새 평가행0이다.

이전 V18–V19 이력과 실제15% past-prefix 손상 코드를 먼저 다시 읽었다. [Scheduled Sampling](https://proceedings.neurips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)은 정답 과거로 학습하고 자체 출력으로 실행하는 차이를 다룬다. [Bowman 등(2016), §3·§6.1](https://aclanthology.org/K16-1002.pdf)은 decoder 입력 단어를 줄여 잠재 정보 사용을 학습하는 방법과 그 tradeoff를 연구했다. 현재 모델은 해당 VAE·KL목적이 아니며 word와 byte도 다르다. 따라서 그 dropout 비율이나 성공을 그대로 옮기거나 현재를 posterior collapse로 확정하지 않는다. 현재의 강한 상태교환 손실 차이도 함께 고려한다.

## 저장과 다음 한정 목적

최종 fullstate654,041,585B SHA **0bfdf57f12351f1fc38d144149ff2f42f9badc8dcc45dc769df72b8cd5096423**, algorithm **ac6b291422687514189e1a700ade6aab3677396ff780800445c98a459337e7bb**. PlanSHAda039bc87deedaf8ecb6b0ce7a195fbe6406700debaed264c282dd6ca1710e80.

ROOT CPU는 import0/mid4096/final8192의 SHA/safeload/재귀digest, source전체component동일성bfd98693bda6af69a02b73c5999b115aa1354ea381a92313a9c74b44989b04e3, 63Adamstep71,627/scaler32,768tracker606/모든 RNG·history·cursor·order·전체 ancestry·외부ledger/동결 reasoning-prefix-basis/finite model·Adam을 검산했다. 저장23,207logits×3의 NLL/argmax와 call수도 재구성했다. 새 CUDA 복구 재실행0이다.

다음은 현재 architecture·원래 퍼지 앵커·inference·corpus·clean손실을 유지하면서 **훈련용 과거 byte 손상 비율을 더 높이는 유한 curriculum**의 CPU 계약을 준비한다. 단순히 같은8,192회를 다시 도는 대신 상태에서 언어를 읽는 훈련 압력을 조정하는 가설이다. 정답 의미 벡터나 추론 시 골드 정보는 추가하지 않는다. 세부 계약은 `outputs/FDT_OWT_RECONSTRUCTION_V27_NEXT_20261029_KR.md`다. 현재 V27모듈/worker/plan/seal/GPU는 없다. 모든 원본·core·typedpartial9/a877·과거 fullAdamRNG·raw·실패는 보존했다.
