# V78 결과 — 일부 수량 개선, 장문 사건번호 보존 미회복

ROOT가 V78의 실제 한정 학습과 저장 상태·출력 검토를 완료했다. **다음 연구 초기 기준은 V75로 유지하고 V78을 기본 후보로 승격하지 않는다.** V78은 같은 관측 미학습64문장에서 두 수량 보존54→56/64로 개선됐지만,기존 장문의 사건번호·부호 간섭이 커졌다. 전체 FDT나 의미 형성의 실패를 선언하는 결과가 아니라 이번 출력기 변경의 이득·손실을 구분한 결정이다.

## 실제 변경과 학습

언어→의미→언어의 마지막7개 readout tensor/2,908,192파라미터만 V75에서 fresh AdamW1e-5로 학습했다. 입력 언어·K_P/K_C·순서 결합·나머지 전체 tensor는 고정했다. K_E는 invalid NULL/내용0/복사0,decoder는 bound Commit만 읽는다. 의미 디퓨전0이며,추후 사고 학습에서는 입력·메모리·출력 전체를 동결하고 별도 사고부만 학습한다는 최신 사람 지시는 그대로다.

새 TRAIN 문장을 추가하지 않았다. 기존1,024개 원문을 teacher에서1회 읽고 V77과 정확히 같은1,920head 노출을5개씩 묶었다. **384attempt/384update/AMPskip0**,worker312.383624초다. 각 group은 unsigned/signed/OWT/unsigned/signed이고,같은 가중치 상태에서 각 row loss .2를 backward한 뒤 unscale/clip1/Adam을 한 번만 한다. 행 비중40/40/20이며 corpus byte 비중이나 파라미터 gradient mass를 보장하는 말은 아니다. V77의 개별1,920Adam update와 구분한다.

누적이 끝난 뒤 unscale/clip/step하는 순서는 [PyTorch AMP 공식 문서](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)의 effective-batch 처리와 대조했고,이번 코드 자체도 CPU gradient 동등성·enabled scaler 복구로 검사했다. 이 일반 원칙이 V78의 성능 개선을 보장하는 것은 아니다.

Teacher는1,010,943 UTF-8바이트,head는 반복2,720,189바이트/중요86,108바이트다. V75의 base symbol weight를 그대로 고정했다. Teacher가 맞힌 TRAIN body 위치 중 초기 gold 대 strongest non-gold margin≤3인 위치에 min(margin,1)을 유지하는 ReLU hinge를 추가했다. 중요/비중요 계수는.25/.10이며 기존 cap8 오답 CE를 유지했다. Teacher 정답·margin·경쟁 class는 loss에만 쓰고 추론 입력으로 주지 않는다. 현재 strongest competitor를 매번 다시 계산해 경쟁 class가 바뀌어도 제약한다. 계수의 최적성을 확인한 실험은 아니다.

## 같은 관측 문장 결과

| 지표 | V75 기준 | V77 | V78 |
|---|---:|---:|---:|
| 같은 수정64개 A/B 모두 분명한 값·단위 보존 |54/64|54/64|56/64|
| 같은64개 주 수량 A |64/64|64/64|64/64|
| 위 kg 보관량 B |10/16|12/16|12/16|
| 위 온도 B |14/16|16/16|16/16|
| 위 거리 B |14/16|12/16|14/16|
| 위 부호 없는 개수 B |16/16|14/16|14/16|
| 관측 V75 DEV 보관량 B |14/32|22/32|20/32|
| 기존 양수kg72구간,분명한 값·단위 |59|58|60|
| OWT16개 decoded 바이트 오류 |108|110|114|
| 최대 장문2개 수량·보관량 변화 |16|28|28|
| 위 사건번호 고정 구간 변화 |10|17|25|
| 위 이름 변화 |2|2|2|
| 위 물품명 변화 |6|7|6|
| 위 decoded 바이트 오류 |58|63|76|

수정64개는 V77과 같은 원문·같은16개 상관 quartet·같은 관측 이름이다. 새 블라인드 자료가 아니다. V76의 관사 중복 수정 전64개와 직접 점수를 비교하지 않는다. V78에서 `+71.25 km` 두 행이 정상으로 돌아왔지만 `75→7;` 두 행은 남았다. `+50.25 kg→450.25 kg`의 실제 크기 변경도 남았다. 별도 수량이 유지돼도 `Review→Reviet`,`7109;→71090` 같은 다른 표현·식별자 손상이 있으므로56/64를 문장 전체 의미 정확도라고 말하지 않는다.

같은64개의 NLL은0.00668718→0.00608507,전체 문자열 일치는44→42/64였다. 바이트/NLL과 특정 사실 보존을 각각 남겼고 전체 일치를 주목표로 삼지 않았다. 숫자·단위가 그대로 읽히는 whitespace/양수 표기 차이는 같은 값으로 처리한다. 추가 기호로 형태가 깨진 경우와 실제 다른 값은 구분하며 사후 문자열 수리를 하지 않는다.

관측 V75 DEV 보관량32개는20개가 분명한 값·kg 유지,6개가 다른 값,2개는 값·단위 동시 손상,4개는 숫자 중심부·kg는 읽히지만 기호 손상이다. 주 수량64개는 전부 유지됐다. 관측 V74 DEV 주 수량61→62/64,별도 보관량34→36/64였다. 기존128문장은 decoded54→55/일치85→84,표현32개는8→10/25→22,retained English36개는3→3/33→33이었다.

## 아주 긴 문장·문서와 실제 읽기

262,042바이트 문서와262,029바이트 한 긴 문장,TRAIN 최대16,384바이트의 약16배 입력을 실제 source에서 다시 추출했다. Native/stream4,096바이트 및 기존 동일 upstream Commit/current head와 fresh source의 parity도 통과했다. 전체 길이·UTF-8·EOS는 두 입력 모두 유지됐다.

V75→V78 수량·보관량 변화16→28은1교정/13신규/15잔여다. V77→V78은3교정/3신규로 총28이 같지만 같은 오류 집합은 아니다. 새로 `27→+7`,`5→+` 두 구간이 생겼다. V76→V78은8교정/신규0/28잔여다. 이름은 V75 대비1교정/1신규이며 `Aveline→Eveline`가 남았다. 사건번호는 V75 대비1교정/16신규, V77 대비2교정/10신규로25곳이 됐다.

실제로 읽은 `193→1+3`,`421→4+1`,`1292→1+92`는 사건 식별자를 보존했다고 할 수 없다. 수량28/전체5,436필드,사건번호25/2,718개 상관 사건이라는 규모를 함께 기록한다. 장문 전체가 붕괴했다고 과장하지 않지만 중요한 국소 사실의 퇴행은 존재한다. 제한 문법의 before/after·출발/도착 색·부정 유무는 변화0이었다. 다만 `drawer→draber`처럼 장소 명사 자체의 오타도 있으므로 제한된 위치 필드 변화0을 자연언어 장소 표현 전체가 완전하다는 뜻으로 확대하지 않는다.

ROOT는 수정64개 비일치22문장,관측 V75 DEV 비일치14문장,장문 변경73개 사건의 전체 국소 source/output와46종 필드 변화,앞·중간·끝6구간을 읽었다. 약52만 바이트 전체를 단어별로 수동 의미 채점했다고 하지 않는다. 이미 관측된 반복 작성 문법 장문이며 자연 장문·임의 장거리 사고·고정 메모리 무손실성의 인증이 아니다.

추가 점검에서 **고정된 gold 숫자 구간만 비교하면 인접 문장부호가 숫자로 바뀌어 합쳐지는 경우를 놓친다**는 점도 확인했다. `episode1089,→episode10893`, `episode53,→episode533`는 고정 구간1089/53 자체가 같아도 실제 식별자가 바뀐다. 두 사례는 V77/V78에 있고 V75에는 없었다. V78에는 `145+`,`169+` 두 malformed token도 추가됐다. 기존 사건번호25/17 점수를 소급 수정하지 않고 별도 adjacent-token 위험으로 보존했다. 이 score-only 검사나 regex를 추론 경로에 넣지 않았다.

## margin 보존이 왜 충분하지 않았나

실제 teacher의 맞힌 중요45,063위치 중 weak margin 대상은1,943개(약4.31%)였다. 맞힌 비중요964,665위치 중 대상은2,995개다. 반복 head 노출에서 중요 margin 위반101회/비중요478회가 실제 기록됐다. 즉 손실이 실행되지 않았다는 설명은 맞지 않는다. Group 평균 중요 margin 항은0.00402044,비중요0.00298059였다. 실제384group 중249개가 clip1을 거쳤다. 이 수치들은 byte 반복 횟수·scalar loss이며 파라미터 gradient 기여 비율은 아니다.

그러나 TRAIN 일부 위치에 대한 제약이 미학습 Commit의 결정 경계를 보장하지는 않는다. 장문 잘못된 `+` 출력은 V75 8개/V77 21개/V78 36개,숫자→`+`는8/21/34개였다. 같은 상류·동일64개 Commit과 모든 upstream tensor는 불변이므로 이번 새 출력 변화는 readout에서 생겼다. Weak-margin 보존과 같은-update 묶기만으로 이 간섭을 해결하지 못했다. 이것은 상류 의미가 모두 완전하다는 증명도,의미 디퓨전 실패도 아니다. 실제 디퓨전은0이다.

## 코드·복구·실행 검토

준비 중 clamp_min의0 경계 subgradient가 teacher와 같을 때도 logits를 밀던 문제를 CPU에서 발견해 ReLU로 고쳤다. 실패 테스트와 코드 원문을 남겼다. 실제 CPU 첫 preflight는 현재 pool weight라는 설명용 진단 필드를 report에 두 번 감싸면서 값이 달라지는 문제로 encode/readout/optimizer0에서 중단됐다. 원본 plan·코드·실패를 남기고 idempotence를 고친 별도execution_review_v2에서 통과했다. GPU를 실패 후 반복 실행한 것이 아니다.

경계 gradient는 [PyTorch autograd 정의 원문](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)도 확인했지만,핵심 근거는 실제 로컬 버전에서 실패를 재현하고 equality의 loss·gradient가 모두0인 것을 통과시킨 CPU 검사다.

9개 focused loss/gradient CPU 검사,작은 실제 grouped engine의 populated7Adam/enabledCPUscaler/모든RNG/정확 다음 group·overflow5행 소비·teacher/한 micro 뒤 STOP 상태 보존과 자동 replay 금지 검사를 통과했다. 전체 모델의 실제 CPU preflight는 encoder2/readout2/optimizer0/CUDAfalse였고,ROOT가1,017pins와initial/partial teacher 상태를 별도로 재구성했다. Canonical model_scope7/2,908,192·단일 출력 추정으로 이전 상속 설명 필드 문제도 차단했다.

실제 owner22684→34400 생성09:26:30.685/.710,worker6164→31288 생성09:26:38.821/.835의 parent/command/exe/생존을 ROOT가09:26:56.861에 확인했다. 학습 중 ROOT 상세 분석을 중단하고 ownedOS 감독만 했다. Worker-owner exit0 뒤09:32:16.183에 모든4owned/proxy와 freshPython0을 먼저 확인한 다음 분석했다. 에이전트는 쓰지 않았다.

새384attempt/6,752calls(encode3,076+readout3,418+chunks258)/wall900·modelcut780·종료reserve120/1.25GiB 예산은 CLOSED다. 미사용248calls·시간을 재사용하거나384회를 자동 연장하지 않는다. 실제 longblock2,322/denoise0,학습 GPU peak3,334,297,088바이트,teacher3,264,918,528바이트였다. 검토 당시 owned 출력1,007,390,554바이트는 cap 이내,종료 C118,165,704,704바이트/약110.05GiB로100GiB를 지켰다.

ROOT는1,019sealed pins,initial/teachercomplete/final3개 linked fullstate를 독립 재구성했다. V75 warm exact/상류 전부 불변/active7 변경/7Adam step384/enabledscaler tracker384/모든 저장RNG/order·cursor·노출/384group·1,920micro loss와cap·margin 통계/teacher1,024개 packet을 검증했다. Raw470평가+parity1의 SHA/targets/argmax/NLL/독립 UTF-8 FSM/출력 문자열을 재구성했다. Teacher 전체 logits는 저장하지 않았으므로 teacher 전체 argmax나 strongest competitor margin을 raw에서 독립 재계산했다는 주장은 하지 않는다. 사후 모델 forward0이다.

## 선택과 다음 범위

다음 초기 기준은 V75다. V78의 두 수량56/64·이름/물품 개선과 모든 상태·Adam/scaler/RNG·raw를 보존한다. V78을 다음 기본으로 자동 승격하거나 실패 자료를 삭제하지 않는다. 다음에는 같은7개 tensor를 같은 방식으로 반복 학습하는 대신,기존 출력기 앞쪽 feature/norm까지 함께 움직이는 범위를 줄여 **frozen feature 위 마지막 출력 선형층2개 tensor만 학습하는 후보**를 CPU 계약으로 검토한다. Weak teacher만 고르는 보존 범위도 TRAIN 기준으로 재검토하되 dev/장문 정답을 TRAIN에 넣지 않는다. 아직 이 다음 후보의 실제 모델/GPU 학습은 없다.

이것은 언어 출력기 수리이며 추후 사고 디퓨전에서는 안정화된 encoder·메모리·결합·decoder 전체를 고정한다는 정책을 변경하지 않는다. 언어→의미→언어 전체 완성·범용 의미 공간 무손실성·자율 사고 완료를 선언하지 않는다.

Plan SHA `3bac6f51460e4a518d26b1f83de71158a021bd7f4d487117aca0c6b79bf1425a`.
최종36,007,558바이트 linked runtime SHA `e2e2cd004b6b96f6347d71e89f06dbc7bdb4b7b0ef3fe1483ee3cfff90cffca5`.
전체 재구성 algorithm SHA `a0d934f83c085e72eed4ee5f0b97b0e24292a0cd943b7327f59374e22224d313`.
필수 standalone V74 base SHA `734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7`.

근거: `runs/competing_margin_20261113_v78/{root_result.json,root_saved_review.json,root_meaning_review.json,root_manual_reading_cards.json,root_training_and_boundary_review.json,ROOT_terminal_native.json}`.
