# V7 초기 학습 경과 — 추가818회 업데이트 완료

사용자의 “학습량이 아직 모자라니 검증에 너무 목매달지 말고 문제 해결되면 계속 학습해”에 따라 실제 학습을 이어갔다. 중간 품질 gate나 추가 baseline 반복으로 멈추지 않고, 기존 V6 가중치·populated Adam·scaler·모든 RNG를 복원해 두 유한 epoch를 마쳤다. 수치·gradient 소유권·디스크·실행 종료 보호는 유지했다. 에이전트는 사용하지 않았다.

**추가818attempt/818update/AMPskip0, 누적1227attempt/1226update**다. 이전 실패149 하나는 그대로 보존했다. 같은512문서의3271행을 새 순열62/63으로 두 번 노출해6542행/769350target UTF8 bytes/1024EOF를 추가 학습했다. 각 epoch 마지막7행도 포함했다. 기존 SGD 목표383651bytes를 합친 누적 노출은1153001bytes이며, 새 문서나 BPE token 수가 아니다.

| 같은 관측 dev의 경과 | V6 → V7 |
|---|---:|
| Byte NLL | 2.43479557 → 2.32475041 |
| Byte accuracy | 31.37912 → 32.41016% |
| 문서별 NLL 개선 | 64/64문서 |
| 실제 EOF에서 EOS 평균 확률 | 6.5182 → 21.4904% |
| 실제 EOF에서 EOS argmax | 0/64 → 1/64 |
| 비EOF 오종료 argmax | 0/53086 → 18/53086 |
| 자유 greedy UTF8 | 16/16 → 16/16 |
| 자유 greedy EOS | 0/16 → 0/16 |
| 자유 greedy 반복4단어 비율 평균 | 60.97 → 93.10% |
| 자유 greedy 고유 출력 | 7/16 → 1/16 |

내용 예측은 더 학습됐고 종료 확률도 상승했다. 한편 자유 greedy는 ` and the the ...`에 더 집중됐다. 전체 구조나 모델의 실패라고 단정하지 않으며, 이른 단계의 likelihood 학습과 자유 rollout의 차이로 기록한다. 반복 개선이나 문장 생성 성공을 선행해 주장하지 않는다. 기존 typed 관계 예측100%/99.9452% 성과와 원래 가중치는 별도로 보존한다.

훈련 목적 변경은 실제 EOF의 CE 가중치4다. 정상 symbol 수로 정규화해 microbatch의 기존 active-symbol 가중치와 일치시켰다. VMSE/clean 손실·Gold 없는 free VP proposal·추론 경로는 유지했다. 언어 LR base5e-6/기타1.5e-5의 새818attempt cosine을 명시했으며 기존 Adam moments를 초기화하지 않았다. 따라서 추가 학습량·새 LR·CE 가중치의 단독 인과 효과를 각각 주장하지 않는다. 실제 argmax 평가와 자유 생성에는 EOS를 강제로 넣지 않았다.

새900초/reserve90/max35000calls/output8GiB 실행에서 **519.542초, encode1751+denoise29652+decoder2779=34182calls**로 마쳤다. 이 실행은 CLOSED이며 남은 호출·시간을 재사용하지 않는다. 기존 V6 final 자료를 SHA로 묶어 baseline으로 재사용해 추가 baseline GPU 호출은0이었다.

ONE native owner17424→7412(13:26:39.467/.494KST), worker37000→29332(13:26:47.421/.437)의 생성·parent·command·생존을 Root가 관측했다. Worker/owner exit0/COMPLETE,13:35:41KST 모든 owned/proxy 종료 및 fresh nativePython0 확인 후 결과를 읽었다. 마지막 C 여유는74772762624B였다. Worker 임시 sleepguard는 finally에서 해제했다.

새 경계·수정에 필요한7개 CPU 계약과 실제 worker preflight/독립 OS validate만 실행했다. 기존 감독 로직은 schema/cap/status 외에는 V6와 같았으며, 새 실제 schema의 post-Popen 자연종료·예외 cleanup·UNCONFIRMED 경로를 포함했다. 초기 pytest 기본 Temp 권한 때문에4개 fixture setup error가 있었고 새 workspace 임시 경로에서7개 모두 통과했다. 학습 실행 오류는 없었다.

Worker는 import0/256/512/768/818의5개 fullstate를 SHA·safeload·재귀 digest로 저장 검증했다. Root는 최종 fullstate 하나의 전체 SHA/safeload/재귀 digest와818행 history·두 유한 순열·노출을 확인하고 저장 logits/raw로 위 지표를 계산했다. 같은 넓은 검사를 계속 반복하지 않았다. 최종 Adam144개 state 중 이전74개 step1226/나중 연결70개 step1078, scaler1024/tracker1226이다. 추가 모델 forward/GPU 호출은0이다.

Plan SHA `9ab49dbbe6386851f7c1703b47282ec2d6a44ee2dc5584fd97740dbd21a7ca73`. 최종814127285B SHA `edcfb82588fb5c7fb2281fd2b2702b340bd91d4073e28e52e93f94ac0eb063dc`, algorithm `1018d434e424ca31629cc0fb56d82972a341c7d73f3c1cd6380a537505844a46`, worker result SHA `d08fabe54625478106cb9e9f3bca4d3bc14f036433f6f217af13882da1a1c87a`. 실제 근거는 `fdt_local_102m_20261026/runs/text_language_continuation_20261028_v7/root_result.json`과 native 관측·OS final이다.

다음 학습 방향은 기존 원문 train 범위에서 언어 노출을 늘리며, 이전에 확인한 반복 훈련 해법을 작은 보조 목적에 반영하는 것이다. 동일512문서의 likelihood epoch만 무작정 반복하지 않고, train 원문에서 만든 반복 부정 표적과 정상 내용 CE를 함께 다루는 계약을 우선한다. 실제 dev 문장·모델 평가 보고서·E2final은 훈련에 넣지 않는다. Top-p 비교는 작은 별도 생성 조건으로 다루고 학습 성과와 구분한다. 새 objective의 단위·복구 검사와 고정 비용을 정한 뒤 새 한정 학습으로 이어간다. 하드웨어 재승인이나 광범위 검증 반복을 선행조건으로 삼지 않는다.
