# V75 이후: 두 수량의 독립 보존과 단위·문장 끝 결합

V75는 actual640/640/skip0,3,360calls/211.319초를 완료했고 모든 owned/proxy 종료 뒤 세 parent-linked 전체 복구 상태·407raw와 실제 의미 읽기를 검토했다. 새900초/3,500calls/1GiB 예산은 닫혔으며 미사용140calls·시간은 재사용하지 않는다. 새 V76 자료·worker·plan·GPU는 아직 없다. 기존 CPU 데이터/engine/linked utility를 처음부터 다시 만들지 않는다.

양호한 결과는 보존한다. 관측 최대 장문2개의 숫자 오류29→16/이름5→2/decoded110→58,새 미학습64개의 주 수량63→64/64 및 이름144구간 유지가 확인됐다. 전체 unsigned32문장은 학습 전후 모두 완전 동일했다. 그러나 새 미학습의 별도 보관량은20→14/32로 퇴행했고,기존 양수kg 보존도64→59/72였다. V74와V75를 모두 남기고 한 패널의 개선으로 자동 승격하지 않는다.

가장 구체적인 자료 빈틈은 TRAIN640개에서 `kg.`/`degrees C.`로 끝나는 입력이0개라는 점이다. EOS 위치16개를 덮었지만 단위/EOS 결합은 덮지 않았다. 두 미학습 출력은 `+2.25 kg.`의 끝4바이트 ` kg.`를 잃었고 저장 raw에서 실제 조기 EOS를 확인했다. 이는 이전에도 있던 오류로 이번 학습이 새로 만든 현상이라고 보고하지 않는다. 새64개 Commit은 전후 전부 비트 수준으로 같았으므로 이번 출력 변화는 readout 학습에서 생겼다. 다만 모든 필요한 정보가 latent에 완전하게 있었다거나 상류가 무조건 옳다고 가정하지 않는다.

다음 CPU 자료는 독립 수량 A/B의2×2 조합,부호가 의미 있는 증감/양수 물리 보관량,행위자·서술 순서,중간/끝 위치,여러 단위와 EOS의16개 위치를 함께 교차한다. A만 바뀌면 B가 보존되고 B만 바뀌면 A가 보존되는 짝을 명시한다. 한 값/부호만 다룬 짝을 독립적인 두 사실 보존의 증거로 확대하지 않는다. 실제 관측 held prompt·gold·이름을 TRAIN에 편입하지 않으며 새 평가 명세와 노출을 학습 전에 고정한다. 같은 OWT TRAIN 분할·원본도 보존한다. source label/span/pair는 loss/score 전용이며 모델 feature에는 원문 id/prompt만 둔다. 일반 이름·숫자를 Exact 표나 문자열 검색으로 복구하지 않는다.

기존 고정 상류/실제 원문 재인코딩/분리 교정·보존 손실을 기본 후보로 두되,또 같은640회를 돌리는 것만을 해결책으로 삼지 않는다. 실제 TRAIN의 문장 끝·보관량 오답과 readout 기울기 방향을 새 좁은 CPU 계약에서 확인하고 필요한 변화만 반영한다. 상류를 바꿔야 하는 구체적 근거가 생기면 그때 별도 단계와 원문/Commit epoch를 정의한다. 광범위 baseline·gradient probe·반복 trace를 무한 선행조건으로 만들지 않는다. 실제 출력·관계·단위·수량을 읽고 의미 보존과 표면 차이를 구분한다.

후속의 초기 NN은 V75 parent-linked 최종에서 정확 복원할 수 있다. 기존 standalone V74를 NN 저장의 필수 base로 유지하고 V75 active7개를 합친 전체 모델 digest를 확인한다. helper는 linked 부모를 재귀적으로 읽지 않으므로 V75 참조 파일을 standalone 부모처럼 지정하지 않는다. 새 runtime 역시 standalone V74 base+active7+모든 새 optimizer/scaler/RNG/phase/cursor/history/calls를 보존할 수 있다. 부모 V74/V75의 모든 populatedAdam/scaler/RNG는 그대로 남기며 새 목적·예산·optimizer 정책을 명시한다.

자원은 별도로 다시 검토한다. V75 종료 시 C약107.9GiB였고23시 재확인은114,092,699,648바이트/약106.25GiB다. 현재100GiB 보호선은 유지하지만100GiB+8GiB 시작 headroom+새 output cap 조건은 부족하다. deadline이나 cap을 늘리거나 유용한 모델·raw를 자동 삭제해 맞추지 않는다. 이전에 정리가 승인된 재생성 가능한 캐시만 범위를 확인해 정리할 수 있고 게임·사용자 원본은 건드리지 않는다. 현재 확인한 NVIDIA DX/GL 캐시 중24시간 초과 후보는 약1.27GiB다. 새 실행의 byte/temp/STOP 예산과 실제 시작 free를 충족하기 전에는 GPU launch하지 않는다. 이것은 하드웨어 재승인 대기가 아니다.

새 자료SHA/actual schema/전체 phase·teacher/head in-flight/AMPskip/parent-linked full recovery/실제 owned CPU preflight/ROOT 독립 validate/새 fixed time-call-row-output cap/native 중복0/OSowner 감독 후 ONE 숨김 실행만 한다. 실행 중 sealed code를 바꾸지 않고 terminal 뒤 모든 owned/proxy 종료부터 확인한다. ROOTONLY/에이전트 없음,원문→K_P prior/K_C dynamic fuzzy→Commit→언어,K_E ExactNULL/복사0,원문 H/바이트·GTprefix·GT길이 우회 금지를 유지한다. K_P 인증 장기 writer가 없다는 한계를 숨기지 않는다. 언어→의미→언어의 내용 보존을 안정화한 뒤 디노이징으로 넘어간다.
