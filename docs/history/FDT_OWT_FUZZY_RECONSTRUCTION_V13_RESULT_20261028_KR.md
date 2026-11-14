# OWT 퍼지 추출·복원 V13 — 문서 확장 후 추가 학습

원래 퍼지 앵커/비배타적 소속도/두 의미 뷰/원문 우회 없는 복원 구조를 그대로 유지했다. 훈련 문서를512→1,024개, 문장을2,048→4,096개로 늘리고 **2,048회 추가 학습**했다. AE 경로 누적은4,096업데이트다. 추론/출력 의미 디노이징은 이번에도 동결됐다.

## 같은 검증 문서에서의 결과

관측 개발64문서/256문장/23,207기호를 그대로 유지했다. V12최종 모델과 동일한 함수·batch·상태 교환 조건이므로 저장된 V12결과를 초기 참조로 사용했고 추가 초기 GPU평가는 하지 않았다.

| 조건 | V12 NLL | V13 NLL | V12 byte 정확도 | V13 byte 정확도 |
|---|---:|---:|---:|---:|
| 정상 추출 상태 |1.976292|**1.744130**|43.1465%|**49.2567%**|
| 다른 문장 상태로 교환 |2.366906|2.525478|34.4465%|35.0024%|
| 상태 제거 |2.231898|2.179626|36.6269%|39.2080%|

정상 teacher-forced byte 정확도는**6.1102pp** 올랐다. 정상 상태를 교환했을 때의 NLL 비용은0.390614→**0.781348nats**, 제거했을 때는0.255606→**0.435496**으로 늘었다. 문장별 추출 정보를 더 이용하는 관측이다. 정답 prefix를 준 바이트 점수이며 자유 생성이나 보편적 의미 정답 점수와는 구분한다.

같은16검증 문서의 자유 복원은 완전 일치**0/16**을 유지했다. 정상·교환 UTF8는16/16이고 EOS는16/16→**15/16**으로 한 문장 하락했다. 제거 상태는 UTF8/EOS16/16이며 동일한 출력1개를 반복했다. 정상·교환은 각각16개 출력이지만 비문과 잘못된 내용이 남는다.

저장된 문자열로 사후 계산한 정규화 byte 편집 오류 평균은 정상0.8153→0.7889다. 이는 미리 정한 성공 기준이 아닌 추가 관측이며, 의미적 충실도 점수로 부르지 않는다. 원문 완전 복원은 여전히 학습 중이다. 전체 방식의 실패로 단언하거나 추론 학습으로 조기 이동하지 않는다.

## 데이터와 상태 보존

기존512훈련 문서·2,048문장을 그대로 앞부분에 보존하고 새512훈련 문서·2,048문장을 추가했다. 새 문서 중89개는 이미 저장된 source pages에서 회수했고423개는3,011,392bytes의 제한된 추가 다운로드로 확보했다. 기존 OpenWebText revision `79d93d786212f7344586290adb811d4ae6a1762c`와 각 page의 `x-revision`을 확인했다.

기존64dev문서/256문장/정답·순서는 정확히 동일하다. 문서 SHA split과 문서·문장 exact 중복0을 유지했다. 추가 near-duplicate 검사나 과거 pretraining 중복 여부는 여전히 UNKNOWN이다. 물리 훈련 sentence UTF8는364,178bytes=기존181,203+새182,975다. 이번4유한 epoch은16,384행/1,456,712targetUTF8bytes를 노출했으며 기존·새 문서가 각각8,192행/724,812·731,900bytes다. 문장 경계 EOS를 문서 EOF로 부르지 않는다.

V12의 모든 가중치/63populatedAdam/scaler2048 tracker48/CPU·CUDA·Python·NumPy RNG를 정확히 가져왔다. Import 전후 algorithm component digest `b598b3f8bcf3a583de4ee3745b6cc42f4a9d2b007fdf222232da8d9fe3e66632`가 동일하다. Source history/order/exposure/calls를 ancestry에 보존하고 새 cursor/attempt/call allocation은0에서 시작했다. 새4epoch LR schedule을 명시했으며, 닫힌 V12시간·잔여 호출을 재사용하지 않았다.

최종63Adam tensor state의 step은 모두4,096, scaler는4096/tracker96이다. 원래 동결된 reasoning/diffusion/prefix와 basis는 import 상태 대비 정확히 동일하다. 새 parameter·anchor reset·강한 균등 사용 벌점·추론 학습은 없다. 검증에서 활성화된 앵커 범위는15→11/64, 뷰당 top-k8, 소속도 합 평균2.1715→2.2677이다. 집중 경향은 계속 기록하되 이것만으로 의미 붕괴나 좋은 의미 분화를 단정하지 않는다. 상태 전체의 내용 사용 증가는 앵커 각각의 해석 가능성을 증명하지 않는다.

## 실제 실행과 검증

- 6focusedCPU delta 검사: 전체 모델/Adam/scaler/RNG import와 source object 보존, 새 fullstate exact next update·마지막7행·누적/새 call 분리, 기존 held·train prefix 보존·4유한 epoch, 실제 새 schema의 postPopen 자연종료·예외·UNCONFIRMED 처리. 실제 worker CPU preflight와 독립 OS validate도 통과했다.
- ONE 실행: owner20124→15072 생성15:53:40.065/.071KST, worker31764→17340 생성15:53:47.901/.916KST. ROOT가 생성·parent·command·생존을 직접 관측했다. Worker/owner exit0/COMPLETE/allowned/proxygone,15:56:28KST Python0 확인 뒤 분석했다. 임시 sleepguard는 finally에서 해제된다.
- 실제144.989초, 새 encode2084+decoder3376+denoise0=**5460calls/max6500**. 새900초/reserve90/output6GiB 예산은CLOSED, 남은1,040calls/시간은 재사용하지 않는다. 종료 후 Cfree약59.6GB다.
- Plan SHA `826e04b4ebd0b44d17bfad69aeee32fc9968ae4a94517aa34575beca3d368b1a`.
- 최종639,961,777B SHA `56b769013a38d9672dcbc684640228594da8d87375a8c6067f450c26dc1e6ad3`, algorithm `4b2ac7da1d07befad4867f81b3fc9b0767f29df5438076b649deca17e686c9cc`.
- Root가 source/import의 전체 algorithm components 동일, 최종 SHA/safeload/digest·동결 tensor·모든 optimizer/scaler·4epoch order/cursor/노출을 독립 확인했다. 전체 saved logits23,207기호×3조건과 원래 target bytes/EOS/PAD·fuzzy degree·raw 생성·실제 호출 수를 CPU 재구성했다. FP16저장 logits의 argmax 집계도 worker와 일치했다. 추가 모델 forward0이다.

근거는 `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261028_v13/root_result.json`, import/native 관측 및 저장 raw/fullstates다. CPU review의 첫 import 경로 오류는 분석 스크립트에서 수정했으며 학습·원본 파일·저장 결과를 바꾸거나 GPU 재실행하지 않았다. 모든 원본·과거 결과·전체 Adam/RNG·실패를 보존했고 E2final/registry에는 접근하지 않았다.

다음은 같은 분리 구조로 추출·복원 학습량을 더 늘리는 한정 실행이다. 의미 기반 출력 디노이징은 별도 후속 단계로 남긴다.
