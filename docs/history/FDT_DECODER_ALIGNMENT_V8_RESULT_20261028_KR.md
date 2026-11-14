# V8 첫 연결부 학습 — 위치·복원을 먼저, 반복 벌점은 β=0으로 분리

사람이 제공한 V7 해석과 연구 순서를 참고하되 그대로 한 번에 결합하지 않았다. 첫 V8 실행은 새 memory position을 **0으로 초기화**해 부모의 초기 출력 함수를 보존하고, **VP·언어 encoder 전체를 동결**한 채 decoder의 clean/noisy codec 복원만 학습했다. 반복 손실 구현과 훈련 전용 부정 자료는 보존했으나 β=0이라 이번 실행에는 반복 부정 forward·gradient가 없다. β=10의 최적성이나 효과를 선행해 주장하지 않는다.

실제 **1636attempt/1636update/AMPskip0**, 기존 실패 하나를 포함한 누적2863attempt/2862update다. 같은512훈련 문서의3271행을 순열71–74로 네 epoch 읽어13084행/1538700positive target UTF8 bytes/2048EOF를 추가 노출했다. 새 문서나 BPE token 수가 아니다. 현재100994050parameters 중 이번 학습 대상은 decoder9314562parameters이며, 동결 VP·언어층에 새 gradient를 학습했다고 부르지 않는다.

## 연결부 진단과 실제 생성 결과를 구분

| 항목 | 학습 전 → V8 후 |
|---|---:|
| 관측 held16, clean **정답 codec** teacher 복원 정확도 | 33.94 → 40.10% |
| 같은 noisy codec σ=.15 teacher 복원 정확도 | 33.84 → 40.40% |
| 같은 clean codec teacher NLL | 2.26573 → 1.97140 |
| 저자가 만든 반대 순서8문장, **oracle codec/원문0** 자유 복원 exact | 0/8 → 0/8 |
| 실제 예측 VP 조건 dev444행 NLL | 2.32475 → 2.10436 |
| 실제 예측 VP 조건 byte accuracy | 32.4102 → 36.4045% |
| 같은 실제16prompt 자유생성 반복4단어 평균 | 93.10 → 49.21% |
| 실제16prompt 고유 출력 | 1 → 14 |
| 실제16prompt UTF8 / EOS | 16/16 유지 / 0/16 유지 |

위의 clean/noisy 복원과8문장 통제에는 정답 정보가 있는 codec을 제공했다. **Decoder 인터페이스 진단이며 VP 의미 이해·실제 자유 생성 점수가 아니다.** 마지막 실제 dev NLL·자유생성은 모두 예측 free VP만 사용했다. 새 memory 위치는 고정 capacity129의 일반 index이며 실제 추론에 정답 byte 위치·정답 길이·gold 의미를 주지 않는다.

분리한 첫 학습만으로도 byte 예측·복원·반복 지표가 개선됐다는 초기 신호가 있다. 그러나 출력은 여전히 `and/to/the`가 이어지는 수준이며, 자연스러운 문서 이어쓰기와 정확한 내부 사실의 표현은 입증되지 않았다. 실제 EOF argmax는 부모1/64→0/64였다. Zero-VP 대조는 EOS16/16이지만 `ther/therer` 등의 깨진 표현으로, 이를 정상 문장 생성이나 의미 충실도 성과로 세지 않는다.

특히 oracle의 A/B, C/D 등 순서를 바꾼8문장이 모두 정답과 달랐다. 출력이 달라졌다는 사실만으로 의미의 인과적 전달이 성공했다고 하지 않는다. 위치·복원 수정 전체가 한 학습 묶음이므로 그 안의 위치 태그/paired conditioning/dropout 각각의 단독 효과도 분리해서 주장하지 않는다. 기존 typed 관계 예측100%/99.9452% 성과는 유효한 별도 관측으로 유지하며, 현재 문서 latent의 의미 완성을 뜻하지 않는다.

## 실제 학습 방식과 보존

기존34개 decoder tensor·VP·144개 populated Adam state·enabled scaler·모든 저장 RNG를 복원하고,66048개 새 memory-position parameter만0에서 추가했다. 복원 훈련의 latent는 **같은 훈련 target**의 codec에 σ≤.15 잡음을 더했으며 비활성 위치는 PAD codec 대신 별도 잡음으로 채웠다. Decoder의 원문 context는 훈련 중 행별50% dropout을 사용했다. EOS CE 가중치4, LR base3e-5/cosine1636, clip1을 사용했고 중간 품질 gate로 멈추지 않았다.

임의의 free VP 표본과 다른 정답 문장을 무조건 CE로 붙이던 불일치를 이 실행의 decoder 훈련에서는 사용하지 않았다. VP denoiser·언어층은 그대로 보존해 첫 연결부 효과와 denoiser 변경을 섞지 않았다. β=0인 첫 대조군이므로 반복용 top8 훈련 단어 부정 자료3271행/102094span은 준비됐지만 실제 negative 훈련 노출로 계산하지 않는다. 추론에 반복 규칙·강제 EOS·정답 template를 넣지 않았다.

Root의 저장 상태 계산에서 VP 전체 tensor 및 VP의 기존 Adam moments가 import0과 최종1636 사이에 정확히 같았다. Decoder의 기존34개 Adam tensor는step2862, 새 memory tensor는step1636, 동결된 기존 VP40개는1226/70개는1078을 유지했다. Scaler는1024에서 정상 growth interval2000을 지나 **2048/tracker862**가 됐다. 초기 Root 분석은 tracker가 선형2862일 것으로 잘못 가정해 assertion을 냈고, growth 규약을 반영해 고쳤다. 학습 오류나 GPU 재실행은 없었다.

새 위치 table의 최종 RMS는0.001532, 행별 norm 평균0.03372였다. 이는 학습된 위치 구분 신호의 수치이며 성능 판정은 아니다. Codec vector의 전체 norm은sqrt512인 구조이므로 위치 신호가 내용 신호에 비해 작은 점은 다음 **더 명시적인 위치 주소 지정/정렬** 후보를 검토할 근거다. 강한 반복 벌점부터 무작정 더하는 순서로 돌아가지 않는다.

## 실행·계산 근거

새900초/reserve90/cap11000calls/output6GiB를 봉인하고 한 번 실행했다. **187.844초, encode3395+denoise1840+decoder4486=9721calls**로 COMPLETE다. 학습 중 VP denoise는0이며1840회는 실제 예측 VP 개발평가/생성에만 사용됐다. EOF에 따른 각 generation batch의 실제 decoder 호출 수를 저장 byte/EOS에서 재구성해 전체9721과 일치시켰다. 이 예산은 CLOSED이며 남은1279calls·시간을 재사용하지 않는다.

Native owner43160→18348는14:14:10.450/.476KST, worker41224→29028은14:14:19.393/.410KST에 생성됐고 Root가 생성·parent·command·생존을 직접 관측했다. Worker/owner exit0/COMPLETE,14:17:33KST freshPython0/모든 owned·proxy 종료를 확인한 뒤 상세 결과를 읽었다. Worker 임시 sleepguard는 finally에서 해제됐다.

새 driver에 필요한7CPU 계약은 부모 component 보존·zero-init 출력 동일성·enabled scaler skip·전체 RNG와 epoch 마지막7행 exact restore·VP 동결·강한 span 수식·실제 plan schema/post-Popen cleanup이다. 실제 worker CPU preflight와 독립 OS validate도 통과했다. Worker는 import0/409/818/1227/1636의5fullstate를 저장하며 SHA/safeload/재귀 digest로 검증했다. Root는 마지막 상태와 import0의 VP·moments 보존, 전체 history/노출·저장 logits/raw만 계산했고 추가 model forward/GPU 호출은0이었다.

Plan SHA `f52f2c3bcc2f1f0f39d2afc37351f7e77739e14e563bc941abd14892800cd350`, 최종815352301B SHA `9b7f5bf310922c6d7261ef75b30091b1252dc51c66bf61d92ed440534dd61258`, algorithm `8195500995628858c1da6026febdd8c87828f7406ad6098e6836ad4190236515`다. 이전 model/Adam/scaler/RNG component import digest는 `5723ae6b5875733bad42ee3fc5258aa6b1a89a0eae95067e2346e8a863b19295`다. 실제 근거는 `fdt_local_102m_20261026/runs/text_decoder_alignment_20261028_v8/root_result.json`, code_snapshot native 관측, OS final이다.

## 다음 연구 판단

현재는 원문 경로를 갑자기 제거하거나 VP 의미 상태가 완성됐다고 가정할 단계가 아니다. 작은 위치 태그를 더 학습하는 후보와 query–latent의 상대 위치를 명시적으로 연결하는 후보를 비교해, clean/noisy codec 및 반대 내용의 정확한 복원이 가능하도록 학습한다. 충분한 복원이 생긴 뒤 같은 출발점에서 β=0/1/4/10 등 제한된 반복 벌점 비교를 진행한다. 새 objective와 비용이 구체화되면 필요한 복구·safety 검사만 하고 새 한정 학습으로 이어간다.

장기 목표는 `Z*=SemanticProcess(X)`를 언어 출력 전에 정하고, `P(Y|Z*,E)`가 그 상태와 허용된 정확 기억을 충실히 표현하는 것이다. 현재 byte-codec 복원은 그 연결부의 한 실험 단계이며 독립 의미 연산의 완성을 뜻하지 않는다. 실제 내용·관계 변경이 올바른 문장 변경으로 이어지는 평가를 유지하고, 원문 우회 의존성은 그 능력이 생긴 뒤 통제해 줄인다. 모든 과거 모델·Adam/RNG·원문·실패는 보존하고 E2final/registry에는 접근하지 않는다.
