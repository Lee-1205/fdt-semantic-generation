# V25 — 같은 입력의 퍼지 추출·재복원 학습 8,188회 추가 갱신

언어→원래 퍼지 앵커의 입력 의미→같은 입력 복원만 학습했다. 출력 목적 의미 디노이징은 별도 동결했다. 원문 context 우회·정답 prefix/길이/역할을 BOS부터의 상태 조건 복원에 주지 않았다. 자유창작·일반 LM 평가로 해석하지 않는다.

## 실제 추가 학습과 보존

`fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261029_v25`에서 **8,192attempt / 8,188update / AMPskip4(at1531,3547,4573,7868)**를 완료했다. 누적63,460attempt / **63,438update**다. V24에서 이어지는 fullAdamW/scaler/모든 RNG·history·전체 ancestry를 불러왔고 새 유한 order331–332/cosine 구간과 clock·call allocation을 구분했다. 과거 미사용 예산을 되살리지 않았다.

같은 immutable8,192train문서/32,768구간의 두 epoch로65,536행/targetUTF8 **5,852,956bytes**를 소비했다. 두 decoder loss의 byte+EOS감독11,836,984symbols, prefix mask879,285/5,852,956=.1502292다. 기존4,096문서 prefix와64dev문서/256행·정답·순서를 유지했다. 다운로드0/새 panel0/E2final·registry접근0이다. 모델·퍼지 구조·입력 경로·목적함수·AMP·엄격 결정성·추론은 V24와 같다.

실제538.354545초, encode8,232+decoder18,281+denoise0=**26,513/max28,000calls**였다. 새900초/reserve90초/output6GiB allocation은 CLOSED이며 미사용1,487calls·시간을 재사용하지 않는다. 저장 source누적187,193calls와 별도 과거 읽기 전용1,936calls를 구분했다(합계189,129). 별도 평가 참조는 이번 fullstate의 metadata에도 보존했다.

CPU delta5검사24.87초: actualV24schema 전체 component·source 불변성, enabledscaler AMPskip와 mask RNG의 fullstate exact-next/마지막7행, 유한 order/caps, actualschemaOSpost-Popen 자연종료·예외 cleanup·UNCONFIRMED를 확인했다. 변경하지 않은 V24 자원 guard 검사를 다시 반복하지 않았다. 실제 workerCPUpreflight/독립OSvalidate 뒤 ONE 실행했다. TinyCPU 계약은 실제 CUDA 복구 재실행이나 모델 능력 증거가 아니다.

Owner25664→25184(07:48:50.812/.837KST), worker2780→39304(07:49:05.279/.294KST)의 생성·parent·command·생존을 ROOT가07:49:15에 확인했다. Worker-ownerexit0/COMPLETE/allownedgone,07:58:26KST Python0/C52,308,021,248B를 확인한 뒤 분석했다. 자원 전체 검사513회, 독립OS1초 감시·즉시 STOP/clock/callcap·강제 큰 저장 전 검사·임시 sleepguard finally 해제를 유지했다. ROOT직접/에이전트 없음이다.

## 보조 예측과 실제 복원은 다르게 움직인다

같은 관측 dev23,207byte+EOS symbols:

| 상태 | V24 NLL | V25 NLL | V24 보조 byte 정확도 | V25 보조 byte 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력 | 1.015962 | **.995085** | 68.9447% | **69.7333%** |
| 다른 입력 | 5.019234 | 5.267533 | 26.1516% | 25.4277% |
| 상태 제거 | 4.024280 | 4.259455 | 34.5973% | 33.8346% |

정상 correct16,000→16,183, **+.78856pp/NLL-.020877**다. EOS보조예측252→251/256도 기록한다. 정상-교환 NLL차이4.272448/정상-제거3.264370은 현재 입력별 상태 정보 사용을 뒷받침한다. 정답 앞부분이 주어진 보조 예측이며 의미 보존율69.73%가 아니다. 이것만으로 extractor·writer 단독 원인, 완성된 의미 공간이나 독립적인 추론 능력을 입증하지 않는다.

Online epoch cleanCE .804494→.655888/corruptCE1.312207→1.214852/mixedloss1.058350→.935370은 학습 중 평균이며 고정 final trainNLL이 아니다. 보조 개선 폭은 V24보다 작다. 절대 노출량·학습 추세를 고려하되 전체 FDT·퍼지 방식의 실패로 단정하지 않는다.

## ROOT가 직접 읽은32개

Dev16과 고정 train16 모두 UTF8/EOS16/16/exact0/충실한 전체 의미 복원0/확인된 부분명제0이다. Train의 **Craig Charles와 David Emanuel 두 이름은 정확히 보존**됐다. 그러나 join이 and로 바뀌고 Richard Coles/Aasmah Mir가 빠져 관계 명제가 복원된 것은 아니다. `forces and`와 `Iraq`도 남았지만 diplomats/reaffirming/U.S.는 없어졌다. 원문 자체가 불완전한 구간에 완전한 정답 명제를 만들어 붙이지 않았다.

| 상태만의 같은 입력 복원 | V24 dev16 | V25 dev16 | V24 train16 | V25 train16 |
|---|---:|---:|---:|---:|
| chrF | 22.7484 | 22.1957 | 29.2000 | 31.3716 |
| 단어 F1 | 11.5348% | 11.0381% | 21.8352% | 25.9326% |
| 내용어 F1 | 1.0870% | 1.6161% | 13.2230% | 14.4357% |
| 정답 시작 UTF8byte | 3.0625 | 3.0625 | 9.1875 | 11.5000 |
| byte edit 유사도 | 26.1214% | 25.6593% | 37.0443% | 38.4491% |

Dev 자기 원문 unique top1은12→11/16이다. 학습 표본의 이름·도입부/표면 유사도는 개선됐지만 dev 전체 복원은 함께 개선되지 않았다. 모든 조건이 좋아졌다고 보고하지 않는다. 반복 관측 dev·작은 표본·비맹검 ROOT 판독·표면유사도의 한계를 유지한다.

## 실제 저장 상태 검산과 다음 단계

최종 fullstate652,206,321B SHA **1240d032c0f6b75f946e898568cad0dfd491517c30b9c7ddd63c84c93a751b5d**, algorithm **43641872b1ca61a682948ab65e4b89a6da18b8053ad594a533bbf0ec93ce92e5**. PlanSHA39e07bc7f6bcc9267ffdf7bbba591ce3ddd66f7d96940136e5dfd12aa86e616f.

ROOT CPU는 import0/mid4096/final8192의 SHA/safeload/재귀digest, source전체component동일성be69a99b6e80f2e177cc612ed111257640bcf661c91668a95120fadc9fb24737, 63Adamstep63,438/scaler32,768tracker324, 모든 RNG/history/order/cursor·전체 ancestry, 동결 reasoning-prefix-basis, finite model·Adam을 확인했다. 저장23,207logits×3의 NLL/argmax와 실제 call수도 독립 재구성했다. 추가 model forward0/새 CUDA 복구 재실행0이다.

근거는 run의 `root_result.json`, `root_state_reconstruction_similarity.json`, `root_all32_state_reconstruction_reading.md`, native JSON과 `scripts/review_owt_state_reconstruction_v25_cpu.py`다. 모든 원본·core·typedpartial9/a877·이전 fullAdamRNG·raw·실패를 보존했다.

다음은 최신 V25 fullAdamRNG를 기준으로 현재 의미 추출·언어화 학습을 계속하되, 낮아지는 clean/corrupt loss와 학습 표본 복원 개선이 held 복원으로 이어지는지를 새 유한 노출에서 관측한다. 검증만 반복하거나 teacher 정확도를 완료 게이트로 삼지 않는다. 보조 개선 폭 감소와 train/dev 복원 차이는 보존하며 단일 모듈 원인으로 단정하지 않는다. 현재 새 V26worker/plan/seal/GPU는 없고 모든 모델 프로세스가 종료됐다. 다음 실행은 새로운 finite order/clock/callcap·실제 source-schema import/필요한 CPU delta·actualpreflight·독립OSvalidate/자원 확인 후 ONE 실행하며 이번 예산을 재개하지 않는다.
