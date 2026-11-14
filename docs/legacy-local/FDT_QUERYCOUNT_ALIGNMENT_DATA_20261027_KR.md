# 질문 수 정렬 자료와 다음 한정 학습 준비 — 2026-10-27

원래 QueryMix의 train1024/dev64 graph, facts, 분할,16query 순서를 유지하고 prefixQ1/4/8/16을 만들었다. Train4096view/dev256view지만 실제graph는1024/64개 그대로이며 새 독립 graph나 blind평가가 아니다. 기존transfer16개graph는train에서제외했다. 이 자료는 질문 수1·4가 빠져 있던 정제 학습 문맥을 보완하기 위한 후보이며 원인이나 개선 효과는 아직 검증하지 않았다.

Root가 builder/sampler 전체 코드와4352저장행의source동일·prefix·정답/hops·hash·split·transfer제외를 독립 재구성했다. Agent7/root7 CPU검사PASS로 원래oracle/독립BFS,5feature/활성mask/용량/all8IDs와Sampler복구를 확인했다. CPU자료 준비이며 model forward0/optimizer0/새GPU0이다.

| 질문 수 | train view행 | train query support [unknown,greater,less,conflict] | dev query support |
|---|---:|---|---|
|1|1024|[305, 228, 235, 256]|[20, 12, 16, 16]|
|4|1024|[1182, 966, 924, 1024]|[83, 54, 55, 64]|
|8|1024|[2381, 1916, 1847, 2048]|[162, 113, 109, 128]|
|16|1024|[4667, 3820, 3801, 4096]|[322, 207, 239, 256]|

모든후보view를합치면train29696/dev1856활성위치이며중복prefix를포함한다. 원래서로다른basequery는16384/1024개다. Premise물리적phrase UTF8bytes105337/6601, view반복421348/26404이며LMtoken이아니다. Class완전균형이나독립성을주장하지않는다.

Seed25의한정sampler는epoch당128batch,각8개서로다른sourcegraph에Q1/4/8/16을각2행씩배정한다. 한epoch에서1024개sourcegraph를한번씩방문하므로실제행1024/query7424이며모든4096후보view를소비한것이아니다. Querycontext행균형이지query위치·gradient기여동일이아니다. RNG/cursor JSON복원은정확한nextbatch를재현한다.

다음은저비용감독agent의CPU ONLY 새학습worker/독립OSsupervisor/config/필수계약/authorized:false계획준비다. 아직GPU/optimizer/seal/launch는없다. 제안output runs/querycount_alignment_20261027_v1, 모델은기존query-bound helper와actual102237197개수를유지하고a877모델가중치만warmstart/freshAdam·scaler·모든RNGseed25/localcounter0다. Originalphase2whitelist/VPobjective/LR3e-5/clip1/동결backbone은유지하며cosinehorizon256을보존한다. Parent256/active128/Role8_128/E2_128/E1_1440 ancestry와새counter를구분한다.

새prospective한도는128success/128attempt/한sampler epoch/900초reserve120/8192실제top-levelbatchedencode+denoise calls(학습·baseline·진단·최종포함)이다. Row1024/query7424/fact12288상한이며AMPskip도시도·노출을소비하고cursor를되돌리지않는다. Epoch끝에목표미달이면INCOMPLETE/STOPPED로닫고추가epoch를자동실행하지않는다. 매시도현재model+Adam/scaler/모든RNG/sampler/cursor/history를검증저장하고32/64/96/128 milestone을보존한다.

같은seed17의Q별dev64·기존6개보존panel·관측transfer16각Qview를baseline과32/64/96/128에서비교한다. 최종17/29/43은seed17비교와분리한다. 이미높은기존control은max(.94,자기boundbaseline-.02) accuracy/macrobaseline-.05,새qdev/transfer는상대baseline-.02/macro-.05로2연속실제회귀시중단한다. Baseline부터낮은새panel에절대94하한을보존기준으로강요하지않는다. Ground/init.95·동결·수치·소유·반복AMPskip·폭증·진행/저장정체·디스크/시간중단은유지한다. 새qdev와관측transfer모두오류0/기존보존통과면목적이미충족으로optimizer0종료한다. 아니면최대128회이고이전실험의94%zero-SGD규칙을그대로가져오지않는다.

개발목표는기준오류가있으면Q1/Q4 dev의count-balanced오류율절반이하,각newqdev99%이상과Q8/Q16·기존보존통과다. 관측Q4누락교정은별도보고하며그한사례만으로성공이라하지않는다. Baselineceiling·새context/sample/classweight변화를밝히고하나의변경만의인과이득·범용LM/NER/vNext완성을주장하지않는다. 동일replay무한연장/광범위baseline·gradientprobe를다시선행조건으로두지않는다.

Root가실제code/CPU검사/workerpreflight/독립validate/새planSHA/고정stops/owned감독/freshnative/C20GiB+임시checkpoint·8GiBheadroom/GPU12GiB를확인한뒤만ONEhiddenlaunch한다. CPU초안authorized:false는사람승인대기가아니다. 계획·위임은실행증거가아니며실제training중root분석은멈추고agent+OS가감독한다. 원래모델/모든source/raw/weights/AdamRNG/oldterminal을보존하고E2final은접근하지않는다.

BuilderSHA37330356aadb225bc15f12213986448d961cd4fcddd65bfb26480163c0bf2c22
Manifest 실제파일SHA8ec8617f8a3f1a9d381d7d24fd48abdafa6a34eb9ad7a891140ad6215685742b
Manifest self-excluding payloadSHA2280742593a0d882bf3768d72a968e7edf9709419631459e0a2bca5475f89bdf
근거: research/artifacts/querycount_alignment_20261027_v1/manifest.json, root_cpu_review.json 및 tests/test_querycount_alignment_v1.py.
