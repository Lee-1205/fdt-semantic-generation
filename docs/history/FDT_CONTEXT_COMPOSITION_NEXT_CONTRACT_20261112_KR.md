# V73 이후: 문맥 조합·이름 정체성 보강 범위

V73은 실제960회 학습과 미학습 영어128/장문10의 검증·원시 재구성·의미 읽기를 완료했다. 새 예산900초/5,600calls는 닫혔고 미사용254calls·시간은 다시 쓰지 않는다. 새 단계의 자료·engine·plan·GPU는 아직 없다. 하드웨어 승인 대기가 아니라 ROOT가 다음 자료/목적을 구체화할 단계다.

최신 사람 지시대로 미학습 문장과 장문을 충분히 포함한다. 다만 한 번 평가한 패널은 이후에는 관측 보존 패널로 기록한다. 새 형식·새 이름·새 사실 조합과 기존 패널을 섞어 blind 성능처럼 보이지 않게 구분한다. actual source→K_P/K_C→Commit→언어 경로만 평가하고,이전 Commit 재사용을 바뀐 추출기의 성능으로 보고하지 않는다. K_EExact는 내용0/NULL/복사0,디노이징0을 유지한다.

현재 가장 구체적인 수정 대상은 다음 두 가지다. V73 TRAIN에는 signed-unit 표현이+degrees C와-kg에만 몰렸고,+kg의 일부는 실제 출력에서 다른 크기나 음수로 바뀌었다. 또한 중요 이름 구간에서 내부y/마지막z가 비었으며 미학습 Zephyr/Boaz의 이름 변화가 새 장문 이름 변화453개 중442개를 차지했다. 이는 전체 모델의 의미 형성 실패를 선언할 근거가 아니라 특정 문맥 조합과 정체성 보존을 보강할 근거다. 모든 문자와16개 위치의 coverage만으로 충분하다고 가정하지 않는다.

먼저 prospective 자료에서 부호±×단위kg/온도×측정값/증감/보유량 문맥을 교차한다. 실세계 물리량과 증감량을 구분한 자연스러운 문장도 넣고,학습 목적은 원문의 내용을 보존하는 것임을 유지한다. 원문이 이상한 값을 적었다고 inference에서 임의로 고치지 않는다. 이름은 새로운 이름을 사용해 y/z의 단어 처음·내부·끝과16개 위치를 교차하되,실제 held Zephyr/Boaz 또는 그 평가 문장을 TRAIN에 편입하지 않는다. 인명 전체를 Exact나 외부 문자열 표로 회수하는 기능을 추가하지 않는다.

부호만 다른 최소짝·대상만 다른 최소짝은 원문 조건이 바뀌면 의미와 출력도 달라져야 한다는 감독으로 사용한다. 각 짝은 상관 view이며 독립 문서 수로 세지 않는다. id/prompt feature와 gold span/fact/pair loss target을 분리하고,금표 정보를 Commit·decoder inference feature로 넘기지 않는다. source byte/H 우회와 pre-bank factor head의 높은 점수로 실제 언어 사이클을 대신하는 이전 문제를 되풀이하지 않는다.

같은 고정-head 학습960회를 그대로 더 돌리지 않는다. 새 자료와 실제 V71/V72/V73의 출력기·상류 분리 결과를 바탕으로 다음 학습 경계를 정한다. 고정 추출기 readout 단계와 고정 readout 추출 단계를 순서대로 검토할 수 있지만,단계별 optimizer/scaler/RNG/finitecursor·teacher 기준과 source embedding epoch를 분리해야 한다. 상류가 바뀐 뒤에는 그 이전 Commit cache를 학습 feature로 사용하지 않는다. 모든 단계에서 실제 정상 Commit→언어 gradient 경로와 frozen ownership을 확인한다.

다음 실행은 C여유약112.3GiB에서100GiB를 유지해야 한다. 이전 states/raw를 자동 삭제하거나 old cap을 늘리지 않는다. 새 상태 수·raw 보존 범위·STOP 복구본·임시 쓰기를 실제 byte 단위로 계산하고8GiB 시작 headroom을 확보한다. 필요 없는 동일 baseline·광범위 audit를 다시 반복하지 않고,이미 관측한 장문을 보존 지표로 쓰며 새 후보를 판별할 표현·문맥에 검증을 집중한다. 사람의 미학습/장문 지시는 유지하되 매 반복마다 같은 거대한 raw를 무제한 추가하지 않는다. 자료/모델/실행은 준비된 사실과 실제 실행을 구분해 보고한다.

코드·자료 SHA/실제 schema/전체 phase 및 populated optimizer/scaler/모든 RNG 복구CPU/actual worker owned preflight/ROOT독립 pin·native·자원 검토/새고정 time-call-row-epoch-output caps 후 한 번만 봉인한다. 현재의 유용한 V67/V70/V72/V73 전체 가중치·AdamRNG와 모든 실패를 보존한다. 목표는 여전히 영어 언어→의미→언어의 내용 보존을 높이는 것이며,언어 사이클 전체 완료·일반 무손실성·디노이징 준비를 앞서 선언하지 않는다.
