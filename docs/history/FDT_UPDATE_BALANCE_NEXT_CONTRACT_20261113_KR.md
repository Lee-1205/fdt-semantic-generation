# V76 이후: V75 기준에서 실제 문맥 update 비중 보정

V76 ONE1024/1024/skip0/4,960calls/275.169초는 완료·CLOSED/all4ownedgone-beforeanalysis다. 새 단위/EOS와 보관량은 일부 회복됐지만 최대 장문 수량16→36/OWT decoded108→122,새64개 A/B동시 보존58/64 유지였다. 따라서 다음 초기 NN은 V75로 유지한다. V76 모든 상태와 이득도 보존한다. 미사용240calls/시간으로 추가 학습하지 않는다.

먼저 같은1,024행을 그대로 또 학습하지 않는다. V76의 부호행75%/unsigned12.5%/OWT12.5%가 V75의40/40/20%와 크게 달랐다. 각 window 평균 손실의 한 행당1update라 OWT바이트77.9%가 있어도 update보존 비중은12.5%였다. 자료에서 계산한 숫자 class weight도3.65→2.85였다. 다음은 이 두 요인의 실제 효과를 분리할 계약을 만든다.

- 실제 finite row/order 또는 같은 optimizer update 안의 그룹별 gradient 누적에서 unsigned/signed/OWT 기여를 명시한다. scalar 가중치만 바꾸고 clip/Adam 뒤에도 그 비중이 유지된다고 가정하지 않는다.
- Base symbol weight를 현재 자료에서 다시 계산하는 정책과 기존 V75 weight 고정 후보를 구별한다. 이름·숫자·부호·단위 보존과 일반 영어 손상을 따로 기록한다.
- 새로운 unsigned 여러 문맥 자료를 보강하더라도 실제 held prompt/이름/gold를 TRAIN에 넣지 않는다. OWT 재노출을 쓰면 같은128원문의 반복횟수·실제 UTF-8노출을 명시하고 새 문서 수로 부르지 않는다.
- END 평가32행의 소유격+관사 중복을 다음 별도 revision에서 고친다. 이미 실행한 V76 평가64행/정답/점수는 바꾸거나 제외하지 않는다. 수정된 새 표면이 과거 의미·이름을 공유한다는 관측 이력도 밝힌다.
- 중요한 목적은 일반 수량이 `22→+2`처럼 바뀌는 간섭을 억제하면서 V76의 단위 누락·잘못된 부호 교정 이득을 다시 얻는 것이다. 디노이징/Exact 복사로 우회하지 않는다.

현재 연구 선택은 `configs/research_base_selection_v76_review_20261113.json`의 V75 parent-required runtime이며 저장 base는 여전히 standalone V74다. 실제 다음 worker를 만들 때 선택 파일을 확인하고 최신 완료 모델이라는 이유만으로 V76에서 자동 초기화하지 않는다. 모든 부모 Adam/scaler/RNG를 그대로 보존하고 새 optimizer·목적을 명시한다.

새 자료/코드/SHA/실제 schema와 bounded복구/CPUpreflight/독립ROOTreview/새 actualcost-clock-call-row-epoch-outputtempSTOP/resource 검토 후 ONE bounded trial만 한다. 기존 V76 engine/helper를 재작성하거나 광범위 baseline/gradient audit를 반복 선행조건으로 만들지 않는다. C100GiB보호+8GiBheadroom+새 output예산/GPU12GiB를 만족해야 한다. 현재 약112GiB의 여유이며 별도 재승인 질문은 필요 없다. 이 문서는 새 trialplan이 아니고 V77자료/worker/봉인/GPU는 아직 없다.

ROOTONLY/noagents/K_Pprior+K_Cdynamicfuzzy/K_EExactNULL/복사0/Commit-onlydecoder/sourceHraw-GTprefix-GTlength 금지/원본·게임·유용weightsAdamRNGraw실패 보존을 유지한다. 언어→의미→언어의 의미 보존과 긴 입력 안정화 이전에 디노이징으로 넘어가지 않는다.
