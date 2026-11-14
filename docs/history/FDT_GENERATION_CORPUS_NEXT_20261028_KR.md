> 2026-10-28 업데이트: 아래 CPU 자료 계약은 완료됐다. 새 자료·후보·15CPU검사·실제 부모 복사 검토는 FDT_LANGUAGE_CORPUS_CPU_READY_20261028_KR.md, 다음 실제 worker/OS owner 범위는 FDT_LANGUAGE_CONTINUATION_NEXT_CONTRACT_20261028_KR.md를 따른다. 아직 새 GPU/optimizer trial/plan/seal/launch는 없다.

# 다음 CPU 범위 — 표현·위치 일반화와 실제 언어 corpus

V4는기존질문형식에서대상추출100%·문장22/32로개선됐지만미학습wording은11/32로퇴행했다. 모델능력은여전히A–H짧은비교·부정에한정된다. 언어encoder63.9M가계속동결됐다는공백도그대로다. 같은형식의synthetic optimizer만더쓰며범용생성이완성됐다고하지않는다.

다음root직접CPU작업은두자료계약을준비하는것이다. 에이전트/새GPU/optimizer/봉인은아직없다.

1. 질문표현·어순·문맥앞뒤길이를변화시키는prospective훈련자료를별도작성하고,byte위치암기와언어표현coverage를구분할통제를둔다. 기존관측devwording의실제prompt/정답행을train에복사하지않는다. 표현가족·literal·이름쌍의겹침을명시하며이미관측한dev가새blindbenchmark인것처럼말하지않는다. 비교의방향·부정scope·질문대상span을offline독립검사하고NN입력에는GT/regex역할을넣지않는다.
2. 기존로컬자연언어corpus의출처·권리·원문경로·기존train/dev분할을먼저작게확인한다. 관련역사문서는outputs/ASTRA_DOCUMENT_DATA_AUDIT_20260926_KR.md, outputs/FDT_KNOWLEDGE_DATASET_AUDIT_KR.md, outputs/ASTRA_CONTENT_FOLLOWUP_RESULT_20260930_KR.md다. 코드/사용자원본/oldraw를보존한다. 이전QAannotation을전부사실정답으로승격하지않고,원문continuation학습과사실QA평가를분리한다. 과거held문서/기사단위train-dev경계가있으면재섞지않는다. E2final과프로젝트모델평가보고서를traincorpus에넣지않는다.

자료는출처별권리와language·문서수·물리적UTF8bytes·중복정규화hash·문서단위split·문맥/target길이·학습노출예상을기록한다. 대규모다운로드/유료cloud/외부게시를하지않는다. 필요한작은추가자료만별도근거와계약으로준비한다. Source권리가불명확하면그자료를성급히학습에넣지않고,근거가확실한자료나명시적authoring대안을준비한다. 원문요약·작문능력은새corpus와prospective평가없이주장하지않는다.

후속architecture/목적은자료를보고root가결정한다. 원래typed102M·VP·V1–V4가중치/AdamRNG/raw/실패는그대로둔다. 언어encoder를계속동결한신규span읽기만일반언어학습이라고부르지않으며,실제로언어부분을학습할경우정확한trainable계층과목적·손실·token/byte·RNG를별도고정한다. Diffusion을성급히버리지않고query/latent/time-noise/decoder정렬을계속구분한다. Teacherfactor를평가에주는것이나고정문장template로답을만드는것을언어생성성공으로세지않는다.

새worker/ownedOSowner/CPU·실제preflight·독립validate/newSHA/output/고정time-call-노출상한/복구/stops/native/resources를root가검토하기전GPU실행하지않는다. 완료된V4의미사용시간·780call을재사용하지않는다. Rawpointerlogits·후보mask도다음실제worker에서보존하며이로그추가때문에동일V4평가를재시도하지않는다.
