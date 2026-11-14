# 2026-10-28 V4 追加 결과

새reader512+realizer512를root직접완료·원시검증했다. 관측pair패널exact37.5→68.75%/질문대상읽기100%로개선됐지만,새wording은40.625→34.375%로하락했다. UTF8/EOS80/80유지,긍정exact12/22·1/22로일반화목표미달이다. 현재학습프로세스는없고모든owned종료/weights·AdamRNG보존. 원래typed102M은불변이며연구hybrid110,376,455parameters다. 상세FDT_TEXT_SPAN_ALIGNMENT_RESULT_20261028_KR.md. 다음은FDT_GENERATION_CORPUS_NEXT_20261028_KR.md의표현·위치변화와출처확인된자연언어corpus CPU준비다. 아래V1–V3는역사적결과로보존한다.

# 생성 언어모델 연구 재개 결과 — 2026-10-28

사용자지시에따라에이전트없이root가직접연구·구현·검사·학습·원시검증을재개했다. 세별도실험이정상종료됐고현재학습/추론프로세스는없다. 지속연구자동화는ACTIVE다. 이전closedQueryCount9/128실험을재시작하거나남은예산을쓰지않았다.

| 별도 연구 분기 | 실제 학습 | 핵심 결과 |
|---|---:|---|
| VP byte 문장 생성 | 256/256, skip0 | EOS/일부문자패턴학습, 문장exact0/80. 실패보존. |
| VP+문맥byte decoder | 512/512, skip0 | UTF8/EOS80/80정상. dev exact10/32·13/32. 개체/방향오류남음. |
| 원문질문 semantic reader+decoder | 512/512, skip0 | UTF8/EOS80/80유지. dev exact12/32·13/32, 의미동치15/32·16/32. 긍정문장exact2/22·3/22로목표미달. |

고정템플릿이나외부정답역할을추론에넣지않고원문과잡음에서신경망byte/EOS를생성하는경로가구현됐다. 현연구모델은별도109,778,455parameterhybrid로,원래typed102M을보존하고동결VP와learnedreader/causaldecoder를결합한다. **생성형식의개선과정확한의미생성은구분한다.** 현재일상대화·요약·범용언어모델완성수준이아니다. 좁은A–H한·영합성비교/부정및이미관측한개발패널이다.

세run모두root가nativePID/생성/parent/command·OSowner·실제progress·terminal/allowned종료를확인했다. 각240raw생성과5fullcheckpoint를CPU에서독립재구성/전체SHA·safeload·RNG/cursor/동결source검증했고추가GPU평가를하지않았다. CPU검사는V1 10개/V2 7개/V3 7개와원문runtime3개가통과했다. 모든가중치/AdamW/scaler/사용RNG/history/실패/raw를보존했다. E2final접근0이다.

현재코드경로는research/text_generation_runtime_v3.py의load_models/generate_text다. Runtime은정답/regex역할해석/고정답변생성/검색을쓰지않고모델출력을보정하지않는다. 이API의CPU검증3개와기존패널GPU검증을구분하며새raw입력CUDA사용은새한정readonly계약으로한다.

다음은질문대상읽기와semanticfactor→문장충실도를분리해정렬하는구체적후속연구다. Reader가맞아도생성이틀린사례3개가있으므로실현계약도검사한다. CPU에서query span/역전짝/teacher-to-predicted조건일관성을준비한뒤root검토를거친새한정실행으로진행한다. 이후어휘·자연문장·긴응답corpus와독립평가로확대한다. 예전한도연장·동일실험자동재시작·목표미달결과폐기는하지않는다.

상세근거: FDT_TEXT_GENERATION_PILOT_RESULT_20261028_KR.md, FDT_TEXT_CONTEXT_DECODER_RESULT_20261028_KR.md, FDT_TEXT_SEMANTIC_ALIGNMENT_RESULT_20261028_KR.md. 다음계약: FDT_TEXT_GENERATION_NEXT_CONTRACT_20261028_KR.md. 최신실제상태: PROJECT_STATE 첫CURRENT.
