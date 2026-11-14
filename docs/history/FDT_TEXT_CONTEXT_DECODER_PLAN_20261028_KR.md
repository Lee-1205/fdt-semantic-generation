# 별도 V2: 확산 latent의 공동 byte 문맥 decoding

V1의 실제0/80 exact·한글UTF8오류 이후 root가 직접 선택한 한정 연구다. V1학습/가중치/실패/900초terminal을 보존하고 새V2출력·SHA·예산을 쓴다. 확산을 버리지 않는다. V1 VP모델 전체를 동결하고 그 실제16step Gaussian→latent 제안을 새 causal byte decoder의 문맥으로 전달한다. 원문 byte context도 함께 제공한다. 따라서 생성이 개선돼도 diffusion 단독 이득/순수확산 생성/일반LM 완성으로 부르지 않는다.

새2layer causal decoder는 이전byte+BOS만 보며 미래정답을 보지 않는 CPU causal 검사, VP passive cache와 원래 sampler의 값·17calls 완전 일치 검사, gradient/zero-latent 통제 검사를 한다. 데이터모델score를 보기 전에 정한 기존split/panel을 유지한다. Teacher forcing은 shifted 정답byte의 학습이며 실제생성은BOS에서greedy로EOS/40byte까지 수행하고 정답/템플릿/검색을 쓰지 않는다.

고정 train pool은 기존3008행에서seed29로선정한1024행이며4개의고정순열/최대4096row exposures,512attempt/최대512success다. 새로운 driver에서decoder만freshAdamW/LR3e-4/clip1/AMP1024로학습한다. V1optimizer를재개하지않는다. 부모계보는strict V1final589023727B/SHA0e3947eea37d5141ef299044c7c24369aeb79a18048f283b526cb29c82ce2163. 원래typedpartial9는변경하지않는다.

Train1024와고정평가80원문의VP16step문맥을정답없이계산·저장한다. 예상VP2346calls(138batch×17),decoder학습512calls,평가baseline/final/zero-latent각최대400calls로합계최대4058calls,새cap5000이다. 새owner900초/reserve90초/output8GiB/C36GiBstart·항상C20GiB+2GiBtemp/GPU12GiB다. 기동·CPUpreflight·cache·모든평가·저장을포함한다. 모델메서드직전 actualcounter를소비한다. 0/128/256/384/512fullstate검증저장·중단시fullstate,실행중회전삭제없음. 자동재시작·초과epoch·옛미사용예산재사용금지.

미리정한개발목표는두dev각wellformed80%이상·문장exact25%이상이다. 학습/검증미달도모두보존한다. 최종zero-VPlatent통제는원문context는그대로두고latent만0으로만들므로문맥디코더가실제latent에얼마나의존하는지제한적으로본다. 이통제도단일학습인과증명은아니다. 같은sourcegraph·답변문장중복과관측dev선택을명시한다. Root가새code/실제planschemaCPU/workerpreflight/독립ownervalidate/native/resources를검토후한번만봉인·실행한다. 에이전트/E2final접근0.
