# 첫 사고 디퓨전 학습과 잡음 구간 진단

ROOT 직접 진행, 에이전트 없음. 새 사고 모듈만 **실제로 192회 학습**했다. 기존 언어 입력·퍼지 메모리·의미 결합·언어 출력의 모든 가중치와 버퍼는 변하지 않았다. 정확한 답 의미가 출력으로 전달됐다고 확인할 수는 없지만, 원인을 전체 FDT의 실패로 확대하지 않고 실제 학습 구간과 추론 경로의 차이로 좁혔다.

## 실제 실행과 고정 경계

`runs/memory_thought_20261114_v102`는 CPU 4 threads로 한 번 완료했다. GPU 여유가 당시 11,319MiB로 GPU 시작 기준 12GiB보다 적어, **CUDA를 초기화하지 않는 별도 CPU 프로필**로 실행했다. GPU 기준을 낮춰 실행하거나 다른 앱을 종료하지 않았다.

| 항목 | 실제 값 |
|---|---:|
| 새 사고 모듈 | 3,224,322 parameters |
| 고정된 기존 전체 코덱 | 123,497,080 parameters + 모든 buffers |
| attempt / update / AMP skip | 192 / 192 / 0 |
| TRAIN 노출 | 384개 상관 표현 × 4 finite epochs = 1,536행 |
| target UTF8 노출 | 27,136 bytes |
| 사고 모듈 / 고정 출력기 / 새 입력 인코더 호출 | 4,864 / 1,744 / 0 |
| 총 모델 호출 | 6,608 / 상한 6,800 |
| worker 시간 | 68.605초 |
| 실제 GPU 사용 | 0, CUDA initialized false |

기존 V95의 입력 의미와 입력별 K_C long/recent, 고정 K_P anchors/prior를 재사용했다. K_E는 invalid NULL, 내용 0, 복사 0이다. 원문 전체를 Exact에 저장하거나 정답·역할·정답 길이·고정 답변 조립을 inference에 주지 않았다. 학습 표적은 별도 frozen `E(Y)`이며, 실제 평가는 원문에서 얻은 입력 의미와 메모리만으로 무작위 공간을 조건화한 뒤 16단계 디노이징하고, 완성된 의미를 고정 D로 언어화했다.

선택한 전체 코덱은 V88이다. model/buffer SHA `42a2032db3137ffb9f876585cf26979854d2555e19639fa5c2f1e8d61201b96a`가 전후 같았다. 원래 typed 관계 예측, 이전 언어 복원 모델의 가중치·실패 기록·Adam/RNG도 보존했다. 이번 새 사고 모듈의 0점과 과거의 좋은 structured 관계 성능을 같은 모델의 퇴보로 비교하면 안 된다. 그 과거 시험을 다시 실행했다는 주장도 하지 않는다.

새 고정 한도 600초, model cutoff 480초, 종료 여유 120초, 전체 출력 256MiB는 CLOSED다. 남은 192 calls나 시간을 재사용하지 않는다. Plan SHA `b255671af492846d2d46c098864bcfd80ad0b2d90a06a1226609e3d0db9aba1a`.

ROOT는 owner 30872→2816, worker 28364→36440의 생성 시각·부모·명령·실행 파일·동시 생존을 관측했다. worker/owner exit0·COMPLETE, 모든 owned/proxy 종료 및 fresh Python0를 **00:35:11.905 KST, 분석 전에** 확인했다. 실제 학습 중에는 상세 분석을 하지 않고 OS 감독만 유지했다.

## 결과 문장을 읽어 본 판단

정상 final 96개 출력 모두를 직접 읽었다. 단순한 전체 일치뿐 아니라 요구한 관계나 unknown을 명확히 전달하는지도 확인했으나, 확실히 올바른 답으로 인증할 수 있는 것은 **0/96**이었다. 대체로 `Cod`, `2`, `I E` 같은 짧은 문자열이나 단어 비슷한 잡음이다. 단문은 원래 질문과도 대조했다. 이 질문들은 이름 하나를 고르라는 문제가 아니라 첫 대상의 상대 크기를 묻는다. `F`와 `H`를 비교하는 질문의 `s` 한 글자는 smaller의 약어일 가능성은 있지만, 명확한 의미 정답으로 인증하지 않고 불확실 사례로 남겼다. 반대로 `B`가 `A`보다 작은 경우의 `>`는 기호를 상대관계로 해석하면 반대 답이다. 자동 문장 추출기가 이 기호를 놓치므로 자동 추출 점수와 수동 판단을 구분한다. 이 측정을 내부 잠재 상태에 아무 정보도 없다는 증거로 확대하지 않는다.

| 같은 고정 noise 평가 | 초기 byte NLL | 학습 뒤 byte NLL | 요구 관계 문장 | EOS / 조기 EOS |
|---|---:|---:|---:|---:|
| dev_pair 32 | 15.4054 | 5.8358 | 0/32 | 32/32 · 23개 조기 |
| dev_surface 32 | 14.3513 | 6.7169 | 0/32 | 31/32 · 22개 조기 |
| dev_position 32 | 14.7665 | 5.8829 | 0/32 | 32/32 · 27개 조기 |

전체 평균 NLL은 14.8411→6.1452다. EOS는 10/96→95/96으로 늘었지만, 그중 72개가 목표 문장 길이보다 먼저 종료했다. 정답과 같은 길이인 6개도 내용이 틀렸고, 17개는 늦게 종료했으며 1개는 EOS가 없었다. 따라서 **EOS 수 증가를 올바른 종료 개선으로 세지 않았다.** 요구한 사실이 담긴 답변을 만드는 목표는 아직 미달이다.

TRAIN의 처음 8개 group→마지막 8개 group 평균은 latent loss 1.6170→1.3725, byte CE 3.3034→0.5242, EOS CE 3.1674→0.3648이다. 반면 실제 무작위 시작의 16-step 출력 NLL은 약 6.15다. 훈련 손실이 낮아진 것과 실제 답 생성 사이에 큰 차이가 있다.

고정 8행의 메모리 통제에서는 K_C를 0으로 만들면 NLL +0.06839, 출력 3개 변경; K_P를 0으로 만들면 +0.01000, 출력 변경 0이었다. source Z에는 정상 메모리의 영향이 이미 남아 있고 답변 자체도 아직 틀리므로, 이 작은 통제를 유용한 의미 추론이나 독립적인 메모리 인과 효과의 증명으로 삼지 않았다.

자료는 기존의 관측된 A–H 영어 비교·부정 표현이다. canonical pair 분할은 유지했지만 surface/position 통제는 TRAIN 사실을 공유한다. 1,536개의 새 그래프, 일반 언어모델 학습, 새 blind benchmark나 긴 문장 사고 성능은 아니다.

## 독립 재구성

`scripts/review_memory_thought_v102_cpu.py`는 새 모델 forward 없이 다음을 검증했다.

- 초기/96/192의 3개 fullstate SHA, safeload와 재귀 digest. 전체 기존 코덱의 동결 digest도 원래 source로 독립 비교했다.
- Adam step 192, enabled scaler 1024와 tracker 192, gradient 잔여 없음, 모든 실제 row order·노출·history·cursor.
- 초기 RNG로 1,536회의 실제 time/noise 추출을 다시 계산하여 최종 Torch RNG와 일치. Python/NumPy RNG 불변, CUDA RNG 없음.
- 저장된 208개의 FINAL meaning/score trace SHA와 UTF8 decode, argmax, gold log probability, 모든 UTF8 상태 선택 및 표본 258-class logit. 모든 위치의 전체 258개 logit을 저장했다는 뜻은 아니다.

유용한 새 사고 가중치와 populated Adam/scaler/모든 RNG/history/cursor는 `checkpoints/state_0192.pt` 39,042,045B, SHA `24170dc8ad81b52a9b9d0abd6421f9203b702512c1c034533a05e5cde286d79c`로 보존했다. 단순한 실패 폐기나 자동 승격은 하지 않았다.

## 추가 계산으로 좁힌 원인

먼저 기존 저장 자료를 계산해 잡음 노출을 복원했다.

| 훈련 time 범위 | 실제 노출 | 해당 구간을 한 번이라도 본 TRAIN 행 |
|---|---:|---:|
| t ≥ .90 | 139 | 122 / 384 |
| t ≥ .95 | 79 | 76 / 384 |
| t ≥ .99 | 20 | 20 / 384 |
| t = 1 | 0 | 0 / 384 |

실제 sampler는 t=1의 입력 의미 조건화된 순수 잡음에서 시작한다. 기존 훈련은 t∈(.001,.999)를 무작위로 뽑았고, 4 epochs에서는 대부분의 TRAIN 행이 높은 잡음 구간을 경험하지 못했다. 학습량 전체 1,536행과, 실제 시작 구간에서 조건을 읽는 유효 노출은 달랐다. 연속 t 훈련에서 정확히 t=1을 보지 않는 것만으로 구현 오류가 되는 것은 아니다. 여기서는 근처 t≥.99도 20행뿐이고 실제 경로의 성능 차이가 커서, 잡음 구간별 노출을 보장하는 수정 후보를 검토한 것이다. 이것은 학습 초기에 대한 구체적인 부족 근거이며, 충분히 학습한 디퓨전이 원천적으로 안 된다는 근거가 아니다.

별도 NEW 읽기 전용 CPU 진단 V103에서는 **TRAIN 4행만**, t=.05/.5/.95에서 정답 의미에 잡음을 준 경로와 실제 target-free 16-step 경로를 비교했다. 76 thought+16 D=92 calls, optimizer0/E0/GPU0, 약 7.281 worker초로 끝났고 모든 source/새 thought 가중치가 불변이었다. owner 종료 및 Python0를 확인한 뒤 분석했다.

- 낮은 잡음 t=.05: 4개 모두 정답 문장 본문이 prefix로 보였다. 일부 뒤붙임은 기존 D의 부분 patch EOS 한계와 연결된다.
- t=.5: `C is larger than F.`가 고정 D와 새 thought만으로 정확히 나오는 사례가 있었다. 다른 사례는 `lmaller` 또는 마침표/공백 오류가 남았다. **이 경로에는 noisy gold가 있으므로 실제 추론 성공으로 세지 않았다.**
- t=.95: 네 경우 모두 정답 문장 본문을 만들지 못했다.
- 실제 무작위 시작: 이 TRAIN 4행도 `2`, `iza`, `I E`, `I`로 나왔다. unseen 개발 분할 탓만으로 설명되지 않는다.

또한 출력 의미의 capacity 35 slots 중 v-MSE를 직접 적용한 것은 4~5 slots뿐이고, 나머지 30~31개 tail slots는 직접 loss에서 빠졌지만 self-attention에는 참여했다. 동일한 가중치를 공유하므로 간접 영향까지 없었다는 뜻은 아니다. 이 영역의 RMSE도 약 1.17로 남아 있었다. **고잡음/시작점의 부족과 tail의 학습·추론 불일치**가 다음에 직접 다룰 구체적인 문제다. 둘 중 하나가 모든 오류의 유일한 원인이라고 분리 증명한 것은 아니다. 단순히 출력기 가중치만 잘못됐다고 결론 내리지도 않았다.

V103은 실제 학습 감독과 달리 준비용 launcher가 hash 준비 과정에서 Torch를 import했다는 flag가 남았다. 독립 Popen watchdog과 종료 검증은 수행했으나 Torch-free owner 증거로 포장하지 않는다. 진단을 로그 수정 목적으로 재실행하지 않았고, 다음 진단 launcher는 표준 hashlib만 사용한다. V102 실제 학습 owner는 Torch-free였다.

## 준비한 수정 V104 — 아직 추가 학습 아님

새 `research/thought_grid_tail_v104.py`와 별도 curriculum을 만들었다. 기존 384 TRAIN 행과 192개의 역전 쌍은 그대로이며 자료를 다시 수집하거나 개발 행을 편입하지 않았다.

1. 실제 inference의 16개 time grid `1,15/16,…,1/16`를 훈련에 사용한다. 16 finite epochs에서 **각 TRAIN 행이 모든 time을 정확히 한 번씩** 경험하며, 역전 쌍은 같은 time과 batch에 둔다. t=1에서도 입력은 입력 의미+독립 잡음뿐이라는 CPU 검사를 통과했다. 학습 표적이 model input에 남는 편법은 없다. 16점 목적은 원래 연속 t 평균과 동일하지 않으며 실제 sampler 시간점의 학습을 우선하는 후보다.
2. 기존 valid-slot v/clean loss와 별도로, 채워져 있는 tail의 frozen EOS 의미에 대한 v/clean loss를 0.05배 추가한다. body와 tail을 각각 평균내어 tail 30개가 내용 4~5개를 압도하지 않게 한다. 0.05는 첫 후보이고 최적값이나 개선의 증거는 아니다.
3. 원래 forward/sample 연산과 파라미터 이름·수가 그대로다. V102 가중치가 전부 strict copy됨을 확인했다. 입력·메모리·출력은 계속 전부 고정하고 사고 가중치만 학습한다. 정답 길이로 inference mask를 만들거나 출력 자르기·문장 치환을 추가하지 않았다.

새 CPU 계약 5개가 2.40초에 통과했다: 모든 row/time 노출, t=1에서 target-independent 입력, tail의 실제 gradient, 기존과 동일한 inference 연산, checkpoint/scaler/Adam의 정확한 READY 복구다. 실제 크기 V104 forward/학습은 0이며, tiny CPU 검사를 성능 향상으로 세지 않는다.

후속 후보는 384×16=6,144행 / 768 attempts / target 108,544 UTF8 bytes다. 새 자료 6,144개가 아니다. V102의 마지막 가중치를 warm start하되 목적이 달라 fresh Adam/scaler를 쓰고, 기존 populated Adam/RNG는 원본대로 보존한다. 기존 V102의 같은 가중치·연산·noise 평가를 baseline으로 재사용하면, 새 훈련 12,288 calls + final/controls 1,904 = 14,192 calls로 계산된다. 이는 **다음 한정 실행의 비용 후보**이며 아직 새 worker/plan/seal/실행 증거가 아니다. 전체 codec baseline이나 31개 target ceiling을 다시 선행 조건으로 돌리지 않는다.

## 복구·저장 공간

실제 학습 전에 READY에서 GradScaler가 지연 초기화되지 않은 문제와, optimizer restore가 같은 CPU Adam tensor를 공유해 checkpoint dictionary를 바꾸는 문제를 수정했다. CPU 7개 계약, 실제 크기 source/Adam/scaler/모든 RNG 복구, 실제 shared 평가 API를 확인했다. 최초 오류와 독립 ROOT 검토에서 거절된 V100 CPU profile도 보존했다. 일반적인 저장/복구 원칙은 [PyTorch AMP 공식 안내](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#saving-resuming)를 참고했고, 이번 두 경계의 실제 문제와 수정 근거는 로컬 재현 시험이다.

새 저장 공간 작업은 종료 증거가 있는 **raw 로그 167개만** NTFS LZX로 투명 압축했다. 논리 SHA와 CPU safe-load는 전후 같았고 물리 3,191,859,322B, 약 2.97GiB를 확보했다. 삭제 0, 가중치 변경 0, 게임 변경 0이다. 실제 학습 시작 전 C: 100GiB+8GiB 여유+전체 새 출력 한도를 충족했다. 마지막 모델/진단 프로세스는 모두 종료됐고, 새 실행 전에는 전역 여유 공간을 다시 확인한다.

주요 근거:

- `runs/memory_thought_20261114_v102/{result.json,root_saved_review.json,root_result.json,ROOT_native_launch_observation.json,ROOT_terminal_native.json}`
- `research/artifacts/thought_noise_gap_20261114_v103_cpu_v1/`
- `research/artifacts/thought_grid_tail_20261114_v104_cpu_v1/`
- `research/artifacts/closed_raw_compression_20261114_v4/`
