# V4 실제 실행 계약 — reader와 realizer 분리

원래V1/V2/V3과QueryCount9/128은닫힌채보존한다. V3의실제span/개체방향오류와맞는factor의잘못된실현을분리하기위한새한정학습이며,기존실험시간을늘리는것이아니다. Root직접/에이전트없음.

새worker는scripts/run_text_span_alignment_20261028.py,CPU검증가능한상태엔진은scripts/text_span_engine_v4.py,독립OSowner는scripts/supervise_text_span_alignment_20261028.py다. 기존V3파라미터를명시적으로warmstart하고새pointer3개key만초기화한다. VP/source와factorvalueembeddings/types는동결한다. 전체연구모델110,376,455=VP91,633,920+reader/decoder18,742,535parameters다. 범용NER/LM완성이아니다.

고정reader512attempt(최대512update)에서query/pointer/byteposition/관계·언어head만학습한다. Loss는spanCE+relationCE+.25languageCE+.1질문역전확률일관성이다.177역전짝을batch안에유지한고정4epoch/4096cachedrow순서를쓴다. Decoder/factorvalues/VP는동결한다.

그뒤reader전체를동결하고freshdecoderAdamW로realizer512attempt/고정4epoch를학습한다. 처음256attempt는훈련전용teacherfactor,뒤256은4예측이모두맞는행만predictedsoftfactor를쓰고나머지훈련행은teacherfallback한다. 평가/추론은항상predictedfactor만쓰며teacher/정답span/regex역할/정답행선별을금지한다. `.eval()`상태의teacher/transitionmethod는예외를낸다. Teacher/predicted/fallback실제행수를기록한다.

Reader와decoder는phase별freshAdamW/LR3e-4/cosine512/clip1이다. **ONE enabledGradScaler(scale1024/growth2/backoff.5/interval2000)는phase경계에서재설정하지않는다.** Torch CPU/CUDA·Python·NumPy도seed31초기화후경계에서재설정하지않으며order만reader31/realizer32고정이다. 두optimizerstate를함께보존한다. AMPskip은attempt/order/cursor를소비하고3회면중단한다. 실패attempt도실제호출·cursor·오류history로남긴다.

새owner900초(기동·실제CPUpreflight·자원검사·저장·모든검증포함)/reserve90초,global1024attemptmax,actualreader+decoder2800callcap이다. 최대2520예상=reader학습512+transitionreader256+decoder학습512+baseline410+readerphaseheadcheck10+final410+zeroVP410. 각실제모델메서드직전차감한다. VP/languageencode·denoise는0이며검증된기존cache를자료로읽는다. 옛미사용call을가져온것이아니다.

Fullstate는reader0/256/512,realizerglobal512/768/1024에별도directory로검증저장한다. Phase512중복filename을회피한다. Model/두optimizer/scaler/모든사용RNG/phase별attempt·update/order·cursor/history/exposure/callbudget/baseline·headcheckSHA를보존한다. 중단시90초reserve안에별도emergencyfullstate를보존하며기존같은step을덮어쓰지않는다. 실행중삭제/회전/자동재시작/초과epoch없음. C36GiBstart/GPU12GiB/상시C20GiB+2GiBtemp/output8GiB를지킨다.

Root검토의CPU증거는새11focusedruntime검사와선행10candidate검사다. 실제단계엔진의CPUAdam/scaler로정확한fullstate·nextstep복구,reader512→realizer0(동일global512)경계에서populatedreaderAdam/빈freshdecoderAdam/scale·RNG유지·grad정리,teacher256→matched257,실패attempt기록,Infgradient로실제AMP3skip/cursor24소비와scale1024→128,잘못된order/빈scaler거부를확인했다. CPU평가에서gold를바꿔도동일생성/teacher호출금지·80행전체채점·headcheckdecoder0도검사했다. 실제plan-schema fullowner postPopen자연종료/예외cleanup/expired-noPopen과REALWindowsproxy-child/CIMexit CPU경로도통과했다. 이들은CUDA복구나GPU성능증거가아니다.

실제workerCPUpreflight와독립ownervalidate를draft에서통과했다. 새절대clock/SHA를한번봉인한뒤같은sealedplanvalidate와freshnative/resources를확인해ONEhiddenownedlaunch한다. 준비·봉인은실행증거가아니다. 실제nativePID생성·parent·command/생존과progress/latest/final을관측하고종료후모든owned/proxy부재확인뒤판단한다. Actualtraining동안root상세모델분석은중단한다.

고정관측80행에서exact/긍정exact/UTF8/EOS/byte·질문span·ID/관계head와zero-VP를모두보존한다. 목표는각dev75%exact/60%긍정exact/90%wellformed이며미달도결과를버리지않는다. 같은A–H한영합성·관측dev/이름·답변중복·선택편향이있고새blindbenchmark/일반LM/독립확산추론으로포장하지않는다. E2final접근0/이전raw·AdamRNG·실패/source보존.
