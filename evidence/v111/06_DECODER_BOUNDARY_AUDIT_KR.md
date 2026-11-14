# 06. 고정 언어화 경계와 생성 의미의 차이

2026-11-14. ROOT 직접, 에이전트0. VERIFIED=이번 실제 코드/실행·독립 재계산, OBSERVED=기존 저장 관측, INFERRED=해석, PROPOSED=미실행 제안.

## VERIFIED

이번 C 사전검사에서 실제 사용27종 E(Y)를 고정 D에 직접 넣었다. **정답 prefix27/27, 전체 문자열0/27**, 신경 EOS/UTF8는27/27이며 추가 문자는1–12byte였다. 과거31종 결과를 새 측정27종의 성적으로 자동 승계하지 않았다. 정답 길이 crop이나 template 조립을 사용하지 않았다.

| 경로 | 행 | whole exact | correct prefix | no EOS |
|---|---:|---:|---:|---:|
| 직접 E(Y)→D |27|0|27|0|
| 학습 전 R→D, 전체 main probe/DEV |112|0|0|0|
| 학습 후 R→D, 같은 main 조건 |112|2|3|19|
| 학습 후 R→D, TRAIN전체 paired |134|1|4|30|

생성134개는 noEOS30/early16/late76, 나머지12개는 길이가 같지만 내용 정답은 대부분 아니다. 본 평가112개는 noEOS19/early20/late70이며 길이 일치도 내용 정답과 구별했다. UTF8는112/112와134/134다. UTF8 규칙이 정상 문자열을 만드는 것과 의미가 맞는 것은 다르다.

byte4gram 반복 비율 평균은 main최종0.0128, 전체TRAIN0.0180였다. 이는 저장된 출력 바이트의 반복성 지표이며 AR feedback loop를 측정한 수치가 아니다. 모든 원시 출력은 길이를 자르지 않고 보존했다. 길이 초과 평균은 별도 JSON에 있지만, 내용이 틀린 문장을 ‘정답 뒤 추가byte’라고 표현하지 않는다.

샘플로 저장한 처음2patch의 full258 logits는32byte를 덮는다. 이번 정답은14/19/20byte라 **모든 정답 byte와 EOS 점수**를 독립 재계산할 수 있었다. 길게 발생한 뒤쪽 tail의 모든258logit을 저장한 것은 아니며, 그쪽은 모든 legal UTF8 decision trace로 전체 decode/EOS를 독립 검증했다.

## INFERRED

직접 정답 의미는 관계 문장을 읽는데 생성 의미는 대상·관계부터 깨지므로 **현재 실패를 D의 기존 종료 오류만으로 설명할 수 없다.** 반대로 D의 종료 한계가 해결됐다는 뜻도 아니다. 순수 MSE로 정답 의미를 정확히 만들더라도 그 표적의 late EOS를 물려받을 수 있다.

현재 생성 tail MSE는 main probe약.524, 전체TRAIN약.536이다. 같은 terminal을 두면 되는 tail도 남아 있으므로, 아직 decoder가 읽기 좋은 잠재 상태에 충분히 도달하지 못했다고 보는 것이 타당하다. 이것만으로 내부에 어떠한 관계 정보도 전혀 없다고 증명되지는 않는다.

## PROPOSED

먼저 θ의 조건별 분리와 기본 fit을 조사한다. 향후 의미 내용은 맞는데 종료만 나쁜 단계가 실제 확인되면, 별도 θ-only frozen-D 충실도/EOS 목적을 검토한다. 이번에는 D를 열거나 새 decoder, 높은 반복 penalty, 정답 길이 crop을 추가하지 않았다.

[실제27 oracle readout](../../docs/LOCAL_ASSETS.md) · [경계 통계](root_decoder_boundary_statistics.json)
