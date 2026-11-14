# V21 — 문장→퍼지 입력 의미→동일 문장 재복원 추가 학습

원래 퍼지 앵커의 μ/union/ρ/value/prior/condition과 추출된 두 view만 받는 복원기를 유지했다. 원문 context 우회 전달은 없다. 목표는 동일 입력 X→Z(X)→X̂이며 자유창작·추론·출력 목적 의미 디노이징을 함께 학습하지 않았다. Reasoning/diffusion 호출 0, 관련 가중치는 동결됐다. 기존 `free/generation` 파일명은 정답 앞부분 없이 BOS와 자체 복원 prefix로 같은 문장을 재복원한 기록이다.

## 추가 학습과 관측 개선

실제 **4,096 attempt / 4,095 update / AMP skip 1회(attempt 2120)** 완료. AE 누적 31,444 attempt / **31,436 update**다. V20 전체 가중치·63 Adam 상태·scaler·모든 RNG·중첩 이력을 정확히 이어받았다. 새 allocation의 finite order와 cosine LR 구간은 별도로 시작했으며 과거 미사용 예산을 이어 쓴 것이 아니다.

같은 4,096 train 문서/16,384 선택 문장·구간을 두 finite epoch(291–292)에 노출했다. 32,768행, target UTF8 2,918,756 bytes, 두 decoder loss의 UTF8 감독 5,837,512회/byte+EOS 5,903,048 symbols다. 손상 prefix 437,173/eligible 2,918,756 = 14.9781%. Skip도 순서·노출을 소비했다. 새 다운로드·held 편입·E2 final/registry 접근 0이다. Online clean CE 평균은 첫 epoch .973869→두 번째 .808089, 손상 prefix CE 1.499371→1.383544, 혼합 loss 1.236620→1.095816이었다. 고정 final train NLL과는 다른 온라인 값이다.

기존 64 dev 문서/256문장, active byte+EOS 23,207개를 그대로 사용한 보조 측정:

| 추출 상태 | V20 NLL | V21 NLL | V20 byte 정확도 | V21 byte 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력의 정상 상태 | 1.286866 | 1.269213 | 62.2183% | 63.0801% |
| 다른 입력의 상태로 교환 | 4.111317 | 4.300625 | 30.0254% | 29.3963% |
| 상태 제거 | 3.205514 | 3.377847 | 38.1092% | 37.7343% |

정상 correct 14,439→14,639, EOS 251/256 유지다. 정답 앞부분을 받는 조건에서 +.86181pp 개선이며 의미 보존율 63.08%를 뜻하지 않는다. 정상-교환 NLL 차이 3.031412, 정상-제거 차이 2.108634는 입력별 추출 상태의 정보가 예측에 쓰인다는 근거다. 현재 AE에 원문 context 우회는 없으므로 과거 V7의 직접 원문 경로 진단을 그대로 현재 원인으로 적용하지 않는다. 완전한 의미 추출이나 decoder 단독 원인까지 입증한 것은 아니다.

## 동일 입력의 상태만으로 복원한 결과 직접 판독

개발 16개와 사전에 고정한 **이미 학습한 16개 문장·구간**을 구분해 모두 읽었다. 추론에 정답 byte·정답 길이·외부 역할을 주지 않았다. 두 표본 모두 UTF8/EOS 16/16, exact 0/16, ROOT의 충실한 의미 재현 판독 0/16이다. ROOT 판단은 비맹검·정성 판단이며 독립적인 자동 의미 정확도가 아니다. 이름/도입부/행위 조각은 따로 기록했고, 올바른 전체 관계가 없는 경우 부분 명제로 높여 세지 않았다.

| 상태 조건 복원 지표 | V20 개발16 | V21 개발16 | V21 훈련16 |
|---|---:|---:|---:|
| chrF | 21.0753 | 21.3720 | 27.7533 |
| 단어 F1 | 10.9277% | 13.9512% | 23.6227% |
| 내용어 F1 | .6250% | .6579% | 12.0755% |
| 정답인 시작 UTF8 byte 평균 | 2.3125 | 2.9375 | 8.6875 |
| byte edit 유사도 | 25.6070% | 24.7838% | 35.9692% |

V21 개발 복원은 자기 원문 unique top1이 11/16(V20 9/16), 상태 교환 chrF 15.1818/제거 16.5786보다 정상 chrF 21.3720이 높았다. 그러나 표면 점수와 주어·행위·대상·부정·관계 보존은 구분한다. 작은 lexical 개선을 완전 복원 성공이라고 보고하지 않는다. Train/dev 입력이 서로 다른 16개이므로 훈련 점수가 더 높은 것을 학습 노출의 단독 인과 효과로 확정하지 않는다.

직접 읽은 예:

- 훈련 원문 `Eduardo is asking customers to pay in cash to avoid bank fees from credit card machines.`에서 복원은 `Eduardo said is a construction ...`이었다. 이름은 정확히 남았지만 현금 요청·수수료 회피 관계는 빠졌다.
- `The tweets were flagged by a source and sent my way.`는 `The tweets were always gened ...`로 복원됐다. 주어와 도입부는 남았지만 제보자가 표시해 전달했다는 행위는 복원되지 않았다.
- `“It just goes to show you their exact moral compass.`는 첫 `“It just goes to show you`를 정확히 복원했지만 뒤의 도덕적 판단은 다른 내용으로 바뀌었다.
- `She then told ... she is a cop and instructed him not to move.`에서도 `She then told`는 남았으나 경찰 신분·움직이지 말라는 금지가 빠졌다.
- 개발 문장은 이름·행위·대상 관계가 대부분 누락됐다. 첫 문장의 `didn't`가 남더라도 `didn't acknowledge`와 `didn't played`는 같은 명제가 아니다.

관측한 훈련 구간 하나는 `forces and diplomats in Iraq, reaffirming that U.S.`로 원문 선택 자체가 불완전했다. 자료 builder는 `(?<=[.!?])\s+`로 경계를 나누므로 마침표·약어와 원문 단편에 대한 한계가 있다. 이 구간에 존재하지 않는 완전한 정답 명제를 만들어 평가하지 않았다. 전체 corpus의 불완전 문장 비율을 측정한 것은 아니며, 이것만으로 현재 복원 문제의 원인을 단정하지 않는다. 기존 immutable 자료·개발 분할·과거 결과는 보존한다.

**이미 본 훈련 구간에서도 전체 복원이 어렵다**는 관측이 추가됐다. 따라서 개발 표현의 일반화만으로 문제를 설명하기는 어렵다. 추출-압축·언어화·자체 prefix에 따른 오차 누적·학습량이 함께 후보이며, 이 실험만으로 어느 하나를 단독 원인으로 정하지 않는다. 초기·작은 고유 corpus이고 정상 상태의 정보 이용과 teacher 예측은 개선됐다. FDT·퍼지 방식 전체의 실패 판정은 하지 않는다.

## 실행과 보존

7 focused CPU 검사 12.71초 PASS, 실제 worker CPU preflight/독립 OS validate PASS 후 ONE 실행했다. Owner19092→40068(21:24:39.565/.594KST), worker35660→1964(21:24:49.584/.601)를 ROOT가 생성·parent·command·생존 관측했다. 실제 worker 321.0989초, encode4,136+decoder10,069+denoise0=**14,205 calls**. 새 max15,300/900초/reserve90/output6GiB 예산은 CLOSED이며 미사용 1,095 calls·시간은 재사용하지 않는다. Worker-owner exit0/COMPLETE/allowned gone, 21:30:24KST Python0을 확인한 뒤 분석했다. 분석 후 21:35:31에도 Python0/Cfree43,730,939,904 bytes였다.

Plan SHA `01e73a60bb4f01865e07138e57f2bdec07e735b686d0adc65feb3a3ee880b910`. Final `state_4096.pt` 645,122,225 bytes/SHA `32c8ba9af2381a58147eb6db9c9551211c7f218440ce81d7a175634b4aad82d2`, algorithm `9fdd540d19026309b4542a28ff71f64e67ab949812cb920d31cc2a282e86112e`다. Import0/mid2048/final4096 및 source·모든 이전 Adam/scaler/RNG/history/raw/실패를 보존했다.

ROOT가 세 fullstate의 SHA/safeload/재귀 digest/source 전체 component 정확성/동결 reasoning-prefix-basis/finite model-Adam/63Adamstep31436/scaler131072tracker1976을 확인했다. Source tracker70부터 모든 성공·skip을 재생했고 두 유한 epoch 순서·노출·mask·손실 합성·저장 23,207 logits×3조건 argmax/NLL·raw byte/EOS·전체 call을 CPU로 독립 재구성했다. 추가 model forward 0이다. 근거는 run의 `root_result.json`, `root_state_reconstruction_similarity.json`, `root_all32_state_reconstruction_reading.md`, native JSON과 `scripts/review_owt_state_reconstruction_v21_cpu.py`다.

다음은 같은 언어→퍼지 입력 의미→언어 복원 목표의 더 긴 유한 학습량을 검토한다. 현재 추출기와 복원기의 공동 학습을 유지하고 reasoning/출력 의미 디노이징은 분리한다. 알려진 원문 구간의 불완전성은 한계로 남기며 새로운 자료 처리 개선은 원본·split·과거 결과를 바꾸지 않는 별도 artifact로 다룬다. 무관한 자유창작 gate·같은 GPU 평가 반복·자동 승격은 추가하지 않는다.
