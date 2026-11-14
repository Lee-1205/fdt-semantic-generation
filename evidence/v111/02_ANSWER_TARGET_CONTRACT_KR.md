# 02. 답변 의미 표적과 누수 경계

2026-11-14. ROOT 직접, 에이전트0. VERIFIED=이번 실제 코드/실행·독립 재계산, OBSERVED=기존 저장 관측, INFERRED=해석, PROPOSED=미실행 제안.

## VERIFIED

`X→고정 E→Zx`, `Zx+K_P+K_C+NULL K_E+독립 ε→Rθ×16→Z_hat→고정 D→Y_hat`을 실행했다. **표적은 E(Y), E(X)가 아니다.** source packet과 loss-only answer packet은 manifest/파일 경계가 다르다. 선택134개는 원래 TRAIN384에서만 왔고 DEV 편입0, fact key31, 답변 의미27종, greater38/less42/unknown54다. 원래 전체480행의 정답은 사전 저장 분석에서 유한 수치 세계로 독립 재구성됐다.

[조건·표적 계약](../../research/artifacts/answer_meaning_objective_20261114_v1/contract.json)의 row ID→원래 정답 답변 hash→실제 answer semantic packet을 연결했다. source와 target 파일의 SHA, codec version, epoch, dtype, mask, 원문 답변 label, EOS/PAD를 확인했다. `Data.target()`은 원래 답변 문자열의 SHA로 **훈련 전용 packet**을 읽으며 질문 원문으로 치환하지 않는다. 이 lookup은 loss/사후 score 경로에만 있고 생성 결과를 직접 반환하지 않는다.

| 항목 | 실제 계약 |
|---|---|
| 최종 의미/표적 | FP32 `(1,35,512)` |
| global/local |2 /33, 합35 |
| mask | bool `(1,35)`, loss-side only |
| labels | int64 `(34,16)`, bytes→EOS256→PAD257 |
| UNKNOWN14byte | body index0–3, tail4–34 |
| positive19/20byte | body0–4, tail5–34 |

body에는 global2, 실제 local1또는2, frozen terminal1이 들어간다. index는0부터다. terminal 이후 tail은 같은 frozen terminal 벡터로 채워져 있다. **local33과 전체35를 용량 불일치로 비교하지 않는다.**

주 경로는 정답 packet을 읽지 않고 target-free16단계를 먼저 실행한다. 이후 E(Y)와 끝점 loss를 계산한다. 보조 VP 경로만 noisy E(Y)를 사용한다. 실제 추론은 target/정답 mask/답변 길이/역할/template 인수를 받지 않는다. 평가의 target packet도 최종 의미와 D 출력이 끝난 다음 score에 사용한다. source ID/hash를 NN embedding으로 넣지 않았다.

정확한 loss는 행마다 body의 slot/512좌표 평균 MSE, tail의 별도 평균 MSE를 내어 `body +.05 tail +.1 VPaux`다. VPaux는 body의 velocity MSE+.25 clean MSE, 그리고.05배 별도 tail 평균을 합한 기존 목적이다. batch는8행 동일 비중 gradient accumulation이다. 언어 CE/EOS CE 로그0은 미사용 자리표시자이며 실제 언어 loss0이라는 뜻이 아니다.

K_P는 고정 anchor/prior, K_C는 입력별 long/recent snapshot이다. θ가 매 단계 읽는 query/projection만 학습한다. K_E는 invalid NULL/key-value0/copy0이며 전문 복사 기능을 만들지 않았다. source의 전체 원문·token H·답변 retrieval·gold 역할·정답 길이는 D나 R에 새로 공급하지 않았다.

## OBSERVED

입력134개는 앞선 source 왕복 검사에서 전체 byte+UTF8+신경 EOS가 통과했다. 이 사실은 새 답변 생성 성과가 아니다. 사전검사에서27개 E(Y)를 실제 D에 직접 넣어 모두 정답 prefix를 확인했지만 종료 잡음은 남았다.

## INFERRED / PROPOSED

이 경계에서는 올바른 정보를 가진 입력이 있어도 R이 답변 의미를 형성하지 못할 수 있다. 원문의 복원 가능성이 관계 변환의 쉬운 학습을 보장하지 않는다. 미래의 별도 codec 개선 가능성은 열어두되 이번 실험에서 E/D를 공동 학습하지 않았다.
