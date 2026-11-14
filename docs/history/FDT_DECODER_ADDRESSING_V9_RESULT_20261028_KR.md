# V9 실제 추가 학습 결과 — 2026-10-28

V9는 **1,636시도/1,636업데이트/AMP skip0**을 완료했다. 현재 Adam 경로의 누적은4,499시도/4,498업데이트이며 이전 실패1시도를 지우지 않았다. 별도4유한 epoch, 같은512훈련 문서의13,084행 노출/정답 UTF8 1,538,700bytes다. 새 문서13,084개나 LM token 수가 아니다.

| 측정 | 초기 V9 | 최종 V9 |
|---|---:|---:|
| 관측 held16, **정답 clean codec** 바이트 복원 |571/990(57.68%)|990/990(100%)|
| 같은 정답 codec+σ=.15 잡음 |572/990(57.78%)|990/990(100%)|
| 원문 없이 정답 codec만 준 반대 순서8문장 exact |0/8|8/8|

최종 출력은 `A is taller than B.`와 `B is taller than A.` 등 변경한 내용을 각각 정확히 복원했다. **이것은 정답 정보를 넣은 연결부 진단이다. VP가 관계를 추론했거나 일반 언어 생성에 성공했다는 점수로 사용하지 않는다.** 같은 panel의 반복 관측, 짧은 ASCII 문장, 기존 관측 개발 자료의 한계도 유지한다.

실제 원문→예측 VP→디코더 경로는 아직 개선되지 않았다. 개발444행/53,150기호에서 조건부 teacher-forced byte NLL은 V8참조2.104359에서8.319530으로 높아졌고 byte accuracy는36.4045%에서10.6679%로 낮아졌다. 실제 생성16개는 모두 정상 UTF8이지만 EOS1/16, 문장이 아닌 글자 혼합이 관측됐다. 반복4-word 비율0만으로 성공으로 세지 않는다. Zero-VP EOS16/16도 `Jjuckn's!` 같은 무의미 출력이므로 언어 성공이 아니다.

V8참조는 **동일한 V9 초기 함수의 baseline이 아니다.** Address prior가 처음부터 함수를 바꿨다. 이 NLL은 원문과 한 번 추출한 잠재 표본을 조건으로 준 값이며, 잠재를 적분한 주변 언어 likelihood도 아니다. 정답과 다른 자연스러운 이어쓰기가 가능하다는 점과 별개로, 실제 저장 출력의 비문자열·비문 문제는 확인된다. 원인을 VP 하나의 실패나 FDT 전체 실패로 확대하지 않는다.

이번 학습에서 VP와 언어 인코더는 계속 동결됐고 디코더9,314,562parameter만 학습됐다. 전체100,994,050parameter, 새 parameter0, β0다. V8의 모든 가중치/145 populated Adam state/scaler/모든 RNG를 가져왔다. Root는 최종 SHA/safeload/재귀 digest, 동결 VP와 Adam 불변,4epoch의 각3271행 순서, scaler4096/tracker498, 저장 logits의 복원 점수와 전체53,150기호 NLL·원본 target bytes·EOF·PAD, raw 생성과 실제 호출 수를 CPU에서 독립 재구성했다. 추가 모델 forward0이다.

- 실제191.858초, encode3395+denoise1840+decoder4306=**9541calls/max11000**. 새900초/reserve90/output6GiB 예산은CLOSED, 남은1459call/시간은 재사용하지 않는다.
- owner14172→42940 생성14:44:32.888/.895KST, worker43960→14996 생성14:44:41.778/.793KST. Root가 생성·parent·command·생존을 직접 확인했다. Worker/owner exit0/COMPLETE,14:47:59KST 전체 owned/proxy 종료와 Python0 확인 뒤 분석했다. 종료 시 Cfree66,025,865,216B, 임시 sleepguard는 worker finally에서 해제된다.
- 새 plan SHA `05d389a57cea35d5dd083ede4289bff1182aa3f470d52bd28fec96bfda27ae44`.
- 최종815,776,813B SHA `daaaaf0b780e93844efd045321a27a0a96769c21014d426858640c32810181b2`, algorithm `f3afd7c1b20b2a74361e419f7a7748f476023d764e7b1a10cbb132a2691d1ef6`.
- 근거: `fdt_local_102m_20261026/runs/text_decoder_addressing_20261028_v9/root_result.json`, `root_target_integrity.json`, `code_snapshot/root_native_launch_observation.json`, `root_terminal_native_observation.json`.

이전 두 실행 전 거부는 원문 그대로 보존했다. 사람이 다시 학습 시작을 명시한 뒤 같은 정상 exec 경로의 봉인과 실행이 받아들여졌다. 보안/승인 설정 변경이나 다른 실행 경로 우회는 없었다. 원래 거부의 정확한 구성요소와 사유는 여전히 미확인이다.

다음은 이 연결부를 계속 정답 codec으로만 학습시키는 반복 대신, 고정된 V9디코더를 통해 VP가 생성할 상태의 분포와 언어 인코더 조건을 학습하는 작업이다. 세 경로 검토와 V10후보는 별도 문서에 남겼다. 과거 typed 관계100%/99.9452% 성과, 모든 기존 모델/AdamRNG/실패/원문을 보존하며 E2final/registry에는 접근하지 않았다.
