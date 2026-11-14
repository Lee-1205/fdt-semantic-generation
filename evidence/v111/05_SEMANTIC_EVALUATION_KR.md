# 05. 의미 정확도와 질문 역전 평가

2026-11-14. ROOT 직접, 에이전트0. VERIFIED=이번 실제 코드/실행·독립 재계산, OBSERVED=기존 저장 관측, INFERRED=해석, PROPOSED=미실행 제안.

## VERIFIED

학습 전후 각112개는 같은 row별 noise seed·고정 D·고정 decoding으로 비교했다. ROOT는 TRAIN 전체80개와 DEV48개 긍정 출력의 대상·관계 내용을 직접 검토했다. 명확히 올바른 긍정 관계를 확인하지 못했다. 이는 단지 전체 문자열 exact가0이기 때문이 아니다. 주어·목적어가 빠지거나 다른 문자이며 관계어도 깨졌다.

| 패널 | NLL 전→후 | body MSE 전→후 | 후 byte argmax | 후 긍정 정상 내용 |
|---|---:|---:|---:|---:|
|TRAIN probe16|12.2188→2.1505|1.8645→1.0061|63.69%|0/8|
|dev_pair32|13.7161→2.9855|1.8353→1.0750|59.25%|0/16|
|dev_surface32|13.7368→4.4394|1.8874→1.2635|39.00%|0/16|
|dev_position32|12.1226→6.1199|1.8989→1.4821|33.44%|0/16|


전체 TRAIN134를 별도 공통-pair-noise로 평가한 결과도 **긍정0/80, 실제32긍정쌍 모두 정답0/32**였다. 전체 TRAIN 적합을16probe로 추정하지 않았다. 이134개는 학습 전 전수검사와 같은 noise 비교가 아니라, 학습 후의 별도 paired 검사다.

UNKNOWN의 자동 keyword 점수는 확정 의미 정확도가 아니다. 주 평가의 TRAINprobe5/16, dev_pair8/32, surface3/32, position2/32가 자동 partial grammar에 잡혔지만 모두 UNKNOWN 쪽이며 `unknowng`, `unknownene` 같은 깨진 문자열도 포함한다. 전체 TRAIN은31/54 UNKNOWN이 substring에 잡혔다. 사람이 의도를 읽을 수 있는 일부 오타는 있지만 미인증 범주로 분리했다. 전체 byte 일치만 강요하지 않되 깨진 문자열의 의도를 target에서 채워 넣지 않았다.

확실한 정답 prefix는 주 평가 DEV에서3/96(모두 dev_pair UNKNOWN), 전체 문자열은2/96였다. 전체 TRAIN 별도평가에서는prefix4/134, fullstring1/134였고 모두UNKNOWN이다. prefix가 맞아도 뒤의 언어 안정성은 별개다. **UNKNOWN54개를 전부 옳다고 가장 관대하게 보아도, 긍정0/80인 현재 전체 TRAIN은 상수 UNKNOWN54/134=40.30%를 넘었다고 할 수 없다.** DEV도 같은 이유로 상수50%를 넘는 관계 능력이 입증되지 않았다.

## 실제 예시

| scope | 기대 답 | 실제 출력 |
|---|---|---|
| TRAIN 긍정 |A is larger than B.|`A  s laa`|
| TRAIN 긍정 |H is larger than B.|`H is largeer hhe\n W.Cgeaa r aa`|
| DEV pair 긍정 |C is larger than D.|`D ism iake`|
| DEV surface 긍정 |B is smaller than A.|`It is unknown.unn'  e  s  awinee Etio.mo `|
| DEV pair UNKNOWN |It is unknown.|`It is unknown.`|
| TRAIN UNKNOWN |It is unknown.|`It is unknown.  A C`|

## VERIFIED: 동일 잡음 역전쌍의 반응

전체32긍정쌍에 같은 ε를 주었다. source 표현 차이 RMS는 최소.35168/평균.44151로 실제 입력 차이가 있었다. 생성 의미 first4slot 차이 중앙값은.05063, 해당 정답 answer-code 차이 중앙값은1.77113, 개별 비율의 중앙값은**.02816**이었다.32쌍 중9쌍은 실제 출력 문자열도 같았다.

이 비율은 **목표 query-role 코드에 대한 반응 크기 진단**이지 의미 정확도2.8%가 아니다. `A>B`와 `B<A`는 같은 canonical 사실을 표현하므로 역전짝의 정답이 서로 반대 사실이라는 뜻도 아니다. 다른 표현으로 같은 사실을 정확히 답했다면 의미적으로 인정할 수 있지만, 그 사실만으로 질문 역할을 변환했다고 증명되지는 않는다. 현재는 양쪽 모두 제대로 된 긍정 내용을 생성하지 못했다.

## INFERRED

공통 답문 형태는 배웠지만 대상과 관계의 결합은 아직 부족하다는 해석이 지지된다. TRAINprobe 긍정8개의 ` is `4byte는29/32가 맞아졌지만 주어0/8, 목적어2/8, 관계 단어 전체0/8이었다. 전체 TRAIN 긍정80개에서도 주어 argmax5/80, 목적어6/80, 관계 단어0/80이다. 이는 바이트 정확도 상승을 의미 이해로 대체할 수 없는 구체적 이유다.

표현이 모든 정보를 잃었다거나 Diffusion 원리가 불가능하다는 결론은 아니다. 현재 업데이트 수·표현·loss·조건 분리 학습에 관한 결과다. NN 재실행 없이 비교한 TRAIN 평균 표적은 같은16probe body MSE.69385였는데 모델은1.00609였다. 기본 fit도 미수렴이다.

## PROPOSED

다음은 조건 반응을 분리해 강화하는 좁은 통제 실험이다. 데이터·epoch를 무조건 크게 늘리거나 새 AR decoder를 붙이지 않는다. 구체적인 후속은07에 기록했다.

[전체224 저장 점수·출력](../../docs/LOCAL_ASSETS.md) · [TRAIN134 원문·출력](../../docs/LOCAL_ASSETS.md) · [실제58쌍 반응](../../docs/LOCAL_ASSETS.md)
