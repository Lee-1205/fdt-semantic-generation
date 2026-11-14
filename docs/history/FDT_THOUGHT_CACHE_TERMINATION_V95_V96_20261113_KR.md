# V95 고정 의미·메모리 자료와 V96 종료 학습 목적

2026-11-13, ROOT 직접 수행. 에이전트 사용 없음.

현재 의미→언어의 구체적인 병목은 짧은 답의 **문장 내부 종료 위치**다. 이번 31개 답에서는 정답 문장의 모든 내용 바이트가 그대로 나타났지만, 신경망 EOS가 늦게 선택되어 뒤에 불필요한 글자가 붙었다. 이것을 디퓨전 실패나 전반적인 의미 손실로 판정하지 않았다. 실제 디퓨전 학습은 아직 시작하지 않았다.

## 실제로 수행한 범위

- V88 전체 코덱 123,497,080개 파라미터와 모든 버퍼를 동결했다. 입력 인코더·K_P·K_C·K_E·의미 결합·출력기·기존 미사용 분기도 변경하지 않았다. V91의 새 오타를 피하기 위한 기존 V88 선택을 유지했다.
- 기존 V5 자료의 영어 TRAIN에서 384개 상관 표현, 192개 역전 쌍, 32개 사실, 19개 이름쌍 그룹을 선택했다. greater/less/unknown 각 128행이다. 각 개발 분할은 32행이고 unknown 상수 기준선은 50%다. 새로운 사실이나 독립된 384개 그래프를 만든 것이 아니다.
- 입력 특징은 id/prompt뿐이다. 정답·관계·원래 사실·역전 여부는 loss/score 쪽에만 있다. 네 유한 순서는 각각 TRAIN384행을 한 번씩 포함한다. 미래 네 epoch는 1,536회 노출이며 새 문서 1,536개가 아니다.
- 기존 canonical 이름쌍 분할을 보존했다. dev_pair 이름쌍 그룹은 TRAIN과 다르고, dev_surface/dev_position은 의도적으로 훈련 사실을 공유하는 표현·위치 통제다. 이미 관측한 합성 개발 자료이며 blind benchmark가 아니다.

입력 480개는 `E(X)`와 해당 입력에서 형성된 K_C long/recent를 저장했다. K_P anchors/prior는 고정 공통 파일이다. K_E는 invalid NULL, 내용 0, 복사 0으로 확인했다. 문장 전체 Exact 저장, raw H, 정답 span, 정답 역할, 고정 답변 조립은 추론 특징에 넣지 않았다. 해시는 출처 확인용이며 문장 회수 기능이 아니다.

31개 고유 정답의 `E(Y)`는 **별도 loss 전용** 파일이다. 정답을 인코딩하면서 만들어진 K_C는 버렸고 입력 조건에 합치지 않았다. 개발 정답도 평가 표적이므로 보존했지만 향후 훈련 표적은 TRAIN 행에 연결된 것만 사용해야 한다. 개발 정답 파일 전체를 별도 fitting 대상으로 쓰면 안 된다.

## 실행과 독립 확인

새 실행 `runs/thought_cache_20261113_v95`는 읽기 전용으로 한 번 완료했다.

| 항목 | 실제 값 |
|---|---:|
| 입력 인코딩 | 480 |
| 정답 loss용 인코딩 | 31 |
| 고정 출력기 호출 | 31 |
| 총 모델 호출 | 542 / 상한 542 |
| optimizer / 디노이징 | 0 / 0 |
| worker 시간 | 18.669초 |
| tensor 저장량 | 75,529,151바이트 |
| 새 전체 출력 상한 | 128MiB |

새 owner 300초, 계산 cutoff 240초, 종료 여유 60초는 CLOSED다. 남은 시간과 호출을 재사용하지 않는다. Plan SHA는 `be9f031242c687d0650b327535fb5b386b7e0f051b748989eb1f6897880708ef`다.

ROOT가 native/proxy 24896→22944→4672→23380의 생성 시각·부모·명령·실행 파일·동시 생존을 관측했다. worker와 owner exit0/COMPLETE, 모든 owned/proxy 부재와 fresh Python0를 **21:57:22 KST, 분석 전에** 확인했다. 첫 launch 관측 도구의 exit1은 당시 아직 없던 progress.json 조회 때문이며, 네 실제 프로세스 관측 파일과 후속 정상 실행은 보존했다.

실제 전체 코덱 CPU preflight는 E2회+D1회, optimizer0, 모든 이전 텐서와 저장 Adam/RNG 원본 확인, frozen SHA 불변, CUDA false로 통과했다. 이는 CUDA 학습 복구 시험이 아니다. ROOT는 별도 CPU 검토에서 480개 source cache와 모든 target/trace의 SHA·출처·형태·NULL Exact·정답 분리를 확인했다. 저장된 표본의 258클래스 logit으로 argmax·모든 UTF8 상태 선택·gold log probability도 재계산했다. 모든 위치의 258클래스 전체 logit을 저장했다는 주장은 하지 않는다.

## 출력 내용을 직접 읽고 판정한 결과

31개 예상 문장과 실제 출력을 모두 읽었다.

| 측정 | 결과 |
|---|---:|
| 올바른 관계/unknown 문장 전체가 출력의 prefix로 존재 | 31 / 31 |
| 그 문장 내용 바이트 | 599 / 599 |
| 출력 전체 일치 | 0 / 31 |
| UTF8 및 나중의 EOS | 31 / 31 |
| 정답 종료 위치에서 EOS를 선택하지 못함 | 31 / 31 |
| 추가 부분이 공백뿐 | 1 / 31 |

예: `C is larger than D.` → `C is larger than D.Cmyra  r`, `It is unknown.` → `It is unknown.  `.

출력기의 16바이트 local patch 중간, 즉 14/19/20번째 바이트에 필요한 EOS가 약했다. 해당 EOS와 가장 높은 byte logit 차이는 평균 -5.0120, 범위 -11.7959~-0.4462였다. 다음 structural terminal patch는 EOS를 내지만 그 전의 불필요한 글자를 제거하지 못한다. **완성된 내용을 prefix에서 보았다는 것과 전체 출력의 의미 충실도는 구분한다.** 내용 뒤의 잡음까지 정상 문장이라고 세지 않았다. 이 결과는 gold `E(Y)`를 직접 준 출력기 진단이며 입력으로부터 올바른 답 의미를 추론했다는 증거가 아니다. 긴 문장·임의 문장의 완전한 의미 보존 증명도 아니다.

따라서 V94의 latent MSE 목표에만 맞추면, 이 표적을 정확히 학습해도 알려진 종료 오류를 물려받을 수 있다. 메모리나 디퓨전 의미 형성을 원인으로 먼저 단정할 필요가 없다.

## 출력 가중치를 바꾸지 않는 후속 수정

고정 실제 출력기의 복사본으로 마지막 부분 patch 3개만 CPU 계산했다. full codec 인스턴스·새 E·GPU·SGD는 없었다. 출력기 파라미터는 전부 requires_grad false, gradient None, 전후 SHA 동일이다. 의미 잠재값에 대한 EOS 손실 기울기는 세 경우 모두 존재했다. 이 방향으로 0.001만 움직이는 수치 시험에서 EOS CE 5.102966→5.099261, byte CE 0.013636→0.013633이었다. CPU와 저장 GPU logit 최대차는 0.0000992로, 관측 EOS margin보다 훨씬 작았다.

이는 **고정 D를 통과해 새 디퓨전의 의미값을 학습할 경로가 있다**는 확인이다. 실제 종료 수정이나 일반 생성 개선이 검증됐다는 뜻은 아니다. 선택한 세 patch를 별도 학습 데이터로 fitting하지 않았다.

새 `research/thought_language_objective_v96.py`는 다음 TRAIN 전용 목적을 추가한다.

`L = L_VP-v + .25 L_clean + .5 L_byte + .5 L_EOS`

- byte CE와 EOS CE를 따로 평균내어, 많은 내용 바이트에 단 하나의 종료 신호가 묻히지 않게 한다. 계수는 첫 후보이며 최적값이라는 근거는 없다. TRAIN의 6,784 byte/384 EOS에서 문장별 별도 평균은 EOS 한 개당 byte 한 개의 14/19/20배 비중이고, TRAIN 문장 평균은 약 17.67배다. byte logit이나 파라미터의 gradient가 정확히 17.67배라는 뜻은 아니다.
- 고정 D는 미분 가능하게 유지하되, gradient는 예측 clean 의미를 거쳐 **새 thought 파라미터에만** 전달한다. 출력기 가중치 변경, latent detach, 언어 경로 no_grad는 사용하지 않는다.
- 정답 의미·EOS·PAD는 loss 계산에만 있다. 평가의 sample()은 목표·마스크·정답 길이를 받을 수 없다. 정답 길이로 출력 자르기, EOS 강제 삽입, 문장 치환, 고정 템플릿 조립은 추가하지 않았다.
- inference는 기존대로 입력 의미→조건화 잡음→각 step의 K_P/K_C/NULL K_E 읽기→최종 의미 확정→고정 언어화다. 학습 중의 보조 언어 손실을 추론 중 매 step 언어 출력으로 바꾸지 않는다.

집중 CPU 계약 4개가 2.70초에 통과했다. 실제 tiny thought 업데이트는 새 thought만 변경했고 D는 불변이었다. eval에서 정답 loss API 거부, 외부/오염 표적과 trainable codec 거부, EOS/PAD 분리, target-free sample 뒤에만 inference 언어화하는 기존 경계도 확인했다. 이 tiny 시험은 실제 3.2M thought 학습 성능이나 전체 worker/AMP/scaler/RNG 복구 완료 증거가 아니다.

## 다음 단계와 한계

자료와 종료 원인 확인은 끝났고 같은 평가를 반복하지 않는다. 다음은 V96 목적을 실제 thought-only driver에 연결하여 finite cursor, fresh AdamW, enabled scaler, 모든 RNG와 loss/call/exposure budget, STOP 시 정확한 fullstate, 불변 전체 codec digest를 보존하는 한정 실행이다. 입력·메모리·출력 가중치는 계속 전부 동결한다. 다른 의미/메모리를 주었을 때 **관계가 올바르게 바뀌는지**와 최종 문장 충실도를 별도로 기록해야 한다.

현재 코드 구성요소와 CPU 시험을 GPU 학습 완료로 보고하지 않는다. 실제 driver/복구/owner/새 plan·상한은 아직 없다. 384 TRAIN의 네 finite epoch가 첫 후보지만 실제 비용을 계산하기 전 cap을 봉인하지 않는다. C:는 100GiB 이상을 유지했고 후속 확인 2026-11-13T22:10:39.3745746+09:00 기준 105.82GiB였다. 전역 여유 공간은 이후에도 변동하므로 새 실행 직전에 확인한다. 다음 시작은 새 전체 출력량+임시 checkpoint+8GiB 여유까지 다시 계산한다. 기존 weights/AdamRNG/raw/실패/게임 삭제 없음.

참고한 원 논문: [Diffusion-LM](https://arxiv.org/abs/2205.14217)은 연속 잠재 벡터의 단계적 언어 생성을 다룬다. [Progressive Distillation](https://arxiv.org/abs/2202.00512)은 확산의 파라미터화와 샘플링을 다룬다. 이 논문들이 FDT의 메모리 설계, 이번 EOS 계수 또는 성능을 보증하지는 않는다. 이번 원인 판정의 직접 근거는 로컬 코드와 저장 출력, 실제 수치 계산이다.

근거: `runs/thought_cache_20261113_v95/{root_result.json,root_cache_and_termination_review.json,ROOT_native_launch_observation.json,ROOT_terminal_native.json,final_status.json}` 및 `research/artifacts/thought_cache_20261113_v95_cpu_v1/`.

## 저장 공간의 일시적 변동 — 후속 관측

연구 프로세스가 모두 종료된 뒤 C: 전역 여유 공간이 22:10:39의 105.82GiB에서 22:11:16의 97.16GiB로 급감했다. 이 시점에 100GiB 조건이 충족되지 않았으므로 새 계산을 보류했다. 연구 파일의 약 75.53MB 캐시 저장량과 전역 변동량을 같다고 가정하지 않았고, 전역 변동의 원인은 확인되지 않았다. 다른 앱이나 게임을 종료·삭제하지 않았다.

종료 기록과 owned 종료 증거가 있는 raw 로그에만 적용하는 한정된 NTFS 무손실 압축 범위를 준비했다. 그러나 실제 정리 시작 시 여유 공간이 이미 104.83GiB로 회복되어, 101GiB 이상이면 즉시 끝내는 조건에 따라 압축 대상 0개·실제 회수 0바이트로 정상 종료했다. 원래 파일 내용, 가중치, 데이터, 게임은 바꾸거나 삭제하지 않았다. 재확인 2026-11-13T22:15:46.5924450+09:00의 C: 여유는 104.83GiB, Python 프로세스 0개다.

이는 정리로 7GiB를 회수했다는 결과가 아니다. 전역 공간이 자체적으로 변동한 관측이며, 다음 학습 시작은 보호 100GiB+8GiB 시작 여유+새 출력/임시 저장량을 충족해야 한다. 현재 이를 충족한다고 가정하지 않는다. 근거: research/artifacts/closed_raw_compression_20261113_v3/{scope.json,root_result.json,ROOT_terminal_native.json}.