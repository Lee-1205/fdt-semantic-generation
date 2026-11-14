# 최신: V4 두phase 실행·검증 완료, 이후 CPU 범위

V4는global1024/1024/skip0/2020calls/52.192s COMPLETE/allownedgone다. 관측pair22/32개선,새wording11/32퇴행을보존한다. 더이상V4driver를준비하거나옛예산을재개하지않는다. 다음은FDT_GENERATION_CORPUS_NEXT_20261028_KR.md와PROJECT_STATE첫CURRENT를우선한다. 아래CPU/계약설명은이전준비역사다.

# 2026-10-28 CPU 준비 진전

V4 span/pair자료와neuralcandidate/phase분리CPU10검사·실제V3CPUimport·1104행독립검증을마쳤다. 최신artifact는research/artifacts/text_span_alignment_20261028_v4_cpu_v2,root_cpu_review.json이다. 아직worker/OSowner/preflight/봉인/GPU없음. 다음은configs/text_span_alignment_20261028_v4_cpu_policy.json의두phase256/512경계와복구를실제driver로구현하는것이다. 상세FDT_TEXT_SPAN_CPU_READY_20261028_KR.md/PROJECT_STATE첫CURRENT를우선한다. 아래문서는선행설계근거다.

# 다음 구체적 연구 — 읽기와 실현 계약 정렬, CPU 준비부터

최신사용자“재개 연구 심화 언어모델 생성까지 알아서 진행할것”에따라root가직접계속한다. 다음작업은현재V3의실제오류에근거하며,새실행이아직준비·봉인·시작됐다고하지않는다. 기존V1/V2/V3의900초/상한은모두닫혔다.

관측사실은(1)질문주어/목적어head가56–59%수준,(2)세semantichead가모두맞은devpair10개중생성의미오류3개,(3)한·영UTF8/EOS는80/80정상이다. Reader를계속학습하는것만으로해결된다고단정하지않는다. V3는초기오답softfactor와gold답변을같이학습했으므로,realizer가factor의의미를무시하거나다르게해석하도록압력을받았을가능성이있다. 이는기록된입출력에근거한가설이며확정원인이아니다.

다음CPU준비는두계약을분리한다.

1. 원문질문의실제두대상byte span과질문역전짝을offline명시한다. Span/target/label은훈련손실·사후score용이며추론feature에정답역할·regex결과를넣지않는다. Learnedpointer/reader는원문byte표현과위치에서질문대상을예측한다. 같은fact/query역전짝의방향일치와부정scope도체크한다. 기존train/dev경계를보존하고관측transfer/E2final을편입하지않는다.
2. 의미factor와문장사이의실현계약을독립확인한다. Reader가맞은조건과틀린조건을분리기록하고오답조건+정답문장쌍을무조건같은실현학습으로섞지않는후보를작성한다. 필요시별도teacher-semantic조건훈련후predicted-semantic조건전이를명시한다. **Gold조건은훈련용으로만허용하는별도설계선택**이며추론/평가에정답factor를주면생성성능으로세지않는다. 신경망byte생성을고정문장template로대체하지않는다.

후속코드/자료/CPU/실제preflight/독립OSvalidate/정확한planSHA/새output/시간·call·노출·phase상한/복구/RNG/stops/native/resources를root가검토후에만별도한정학습을봉인한다. 이번문서는승인대기질문이아니라구체적실험코드검토이전상태다. 기존복구가능checkpoint와모든실패/원문은보존한다. 자동초과epoch·재시작/기존남은예산재사용금지.

최종평가는unknown상수기준선,긍정관계exact·의미동치,질문대상span/ID정확도,맞는factor일때생성충실도,UTF8/EOS를분리한다. 같은observeddev/상관자료임을밝힌다. 의미결합이안정되면미관측어휘/이름·다양한자연문장과긴응답으로범위를넓히고출처·권리·정확한분할/실제byte·token노출을검증한별도언어corpus를준비한다. 현재109.78M연구분기를범용LM완성으로부르지않는다.
