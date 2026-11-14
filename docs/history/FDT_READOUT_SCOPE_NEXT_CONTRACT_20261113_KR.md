# 다음 언어 수리 범위 — 앞쪽 readout feature 동결과 최종 선형층 후보

V78은384group/1,920실제원문micro/6,752calls/312.383624초로 완료·종료·독립 검토됐다. 같은64개 두 수량 보존54→56/64 이득을 보존하지만 장문 사건번호17→25·바이트63→76·잘못된plus21→36으로 전체 보존 목적을 충족하지 못했다. V75가 다음 초기 기준이며 V76~V78 모든 상태·Adam/RNG/raw/실패를 보존한다. 모든 옛예산·미사용248call은 CLOSED다.

최근 실험에서 상류는 모두 고정됐지만 마지막 readout7개는 LayerNorm·첫Linear/SiLU feature와 최종Linear를 함께 바꿨다. 다음 구체적 질문은 **앞쪽 feature까지 움직이는 것이 이미 분리된 문자/수량의 결정 경계를 흔드는가**다. 추가 broad baseline이나 같은384/1,920회의 자동 반복을 선행조건으로 삼지 않는다.

새 CPU 후보에서는 V75의 전체 추출·K_P/K_C·순서 결합과 readout terminal/LayerNorm/첫Linear를 고정한다. trainable은 `readout.readout.net.2.weight`와`.bias` 두 tensor로 명시한다. 이는 아직 확정된 해결책이나 실제 학습 성과가 아니다. 실제 parameter 수·storage alias·optimizer 소유·gradient가 두 tensor에만 있는지 계산하고 canonical scope에 넣는다. 기존7Adam trajectory를 잘라 이어 쓰지 않고 새 목적의 optimizer를 별도로 정의한다.

Margin 학습의 범위도 재검토한다. V78은 맞힌 중요45,063개 중1,943개만 weak teacher anchor 대상으로 삼았으며 TRAIN 위반101회에도 미학습 경계는 보호하지 못했다. 처음부터 teacher가 강하게 맞힌 중요 위치도 current strongest competitor가 gold를 위협하면 제약하도록 하는 후보를 검토한다. 초기 margin target은 과도하게 큰 값으로 복제하지 않고 명시한 cap 안에 두고,teacher 오답은 anchor하지 않는다. 실제숫자·부호·사건번호·이름·주변 구두점은 TRAIN loss-only 위치/표적이며 NN추론 feature로 주지 않는다. 기존 관측 dev/장문의 오답 source/gold를 학습에 편입하지 않는다.

같은 update 안 signed/unsigned/OWT 누적을 유지할지,최종 선형층의 볼록한 CE 부분을 이용할지는 실제 CPU gradient·step 계산 후 결정한다. 학습률·step 수·anchor계수·출력 범위를 한꺼번에 막연하게 확대하지 않는다. 앞쪽 feature가 고정되면 그 feature cache를 사용할 수는 있지만 source/embedding/head-prefix SHA와 입력 경계를 분리하고 활성 gradient가 끊기지 않는 경우에만 명시적으로 계약한다. 현재 V78에는 feature cache가 없었으며 이 후보가 이미 실행됐다는 뜻은 아니다.

평가는 의미 보존을 우선한다. 고정 gold 숫자 구간만 보아 `1089,→10893` 같은 인접 합침을 놓치지 않도록 token 경계 위험도 사후 점수에 분리 기록한다. 기존 field 점수는 소급 변경하지 않고,읽히는 숫자+기호 손상/다른 숫자/단위 누락/식별자 합침을 구별한다. 제한된 작성 장문의 색·순서·부정 필드0변화를 자연어 전체 정확도로 부르지 않는다. 실제 source/output 읽기를 계속한다.

현재 입력·출력 분리 상세 감사와 future thought-only 동결 계약은 완료됐으니 같은 검사를 다시 만들지 않는다. **디퓨전으로 단계가 바뀌면 encoder·메모리·결합·출력 전체를 봉인·동결하고 별도 사고 가중치만 학습한다.** 지금 제안은 그 전의 언어 출력기 수리다. 원래 typed21field basis와 현재512dim Commit 연결/답변 의미의 slot·stop 예측/입력 메모리만 조건으로 쓰는 계약은 설계 상태이며 디퓨전 trial0을 유지한다.

다음 실제 실행 전에는 새 코드/dataSHA·fullstate/두 tensor optimizer·scaler·모든RNG·finite order·중단복구·실제 CPU preflight·ROOT 독립 검토·새 clock/call/row/epoch/outputcap·native0·C100GiB+출력tempSTOP+8GiB/GPU12GiB·ownedOS 감독을 고정한다. 준비를 실학습이나 개선으로 세지 않고 재승인 질문 없이 승인된 연구 내에서 ROOT가 진행한다. 현재 다음 후보의 worker/plan/seal/GPU는 아직 없다.
