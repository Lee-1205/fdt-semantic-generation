# V73 다음 범위: 고정 언어화 기준에 맞춘 실제 추출·결합 개선

V72는 실제 완료됐고900초/2,200call 예산은 닫혔다. 남은90calls·시간을 쓰거나 같은640epoch를 자동 반복하지 않는다. 아직 V73 자료·후보·engine·plan·preflight·seal·GPU는 없다. 아래는 ROOT가 다음 구현에 적용할 구체적 목적이며 하드웨어 재승인을 기다리는 문서가 아니다.

현재 목표는 영어 언어→의미→언어의 원문 내용 보존이다. 단어 표면의 전체 일치만으로 다음 단계를 막지 않는다. 관계·방향·부정·수량·시간·참조 대상을 읽고 판단한다. 디노이징은 아직 별도이며 이 단계에서 시작하지 않는다. K_P prior+K_C 동적 퍼지+K_E 휴면 NULL을 유지하고 Exact에 일반 문장·이름·숫자를 넣지 않는다. decoder는 실제 Commit만 받아야 한다.

V71의 선택 사례에서는 마지막 readout만 교체해도 발생하는 변화와 상류 Commit+binding을 바꿔야 나타나는 변화를 모두 봤다. V72는 상류를 고정해 출력기 일부를 실제 개선했다. 그런데 `27→2&`, `Inez` 같은 오류와 다른 값으로 바뀌는 수량6곳이 남아 같은 head-only 반복으로 모든 문제를 해결할 근거는 없다. 반대로 고정 Commit에서 출력이 크게 좋아졌으므로 상류에 유용한 정보가 있다는 구체적 근거는 생겼다. 완성된 추상 의미나 정보 손실 부재를 증명한 것은 아니다.

다음의 우선 후보는 **V72 최종 readout7개 tensor를 고정하고 언어 추출의 마지막2block/value/condition/role_projection만 조정**하는 분리 단계다. 기존 joint 활성30tensor 중 마지막7개를 제외한23개를 예상한다. 실제 전체 모델에서 소유권·수·기울기 연결을 CPU로 산출해 확인하기 전 확정 숫자로 실행하지 않는다. 원문을 매 update 실제 인코딩하고 K_P/K_C→boundCommit→고정 readout의 loss가 상류에 전달돼야 한다. V72 Commit cache로 이를 대체하면 상류 학습이 끊기므로 금지한다. old Adam을 재개하지 않고 별도 fresh optimizer를 쓰며 이전 전체 states/RNG를 보존한다.

우선 CPU에서 다음 prospective 자료를 준비한다. 중요 구간에서 빠진 대소문자·이름 내부 문자와 plus/clock/currency 표현을16개 위치에 분산한다. 기존 미학습 평가의 실제 prompt/gold/이름 전체를 훈련에 옮기지 않고 새 이름·수량·관계의 독립 작성 조합을 만든다. 6/7/8/9를 보지 않았다는 잘못된 원인을 가정하지 않는다. 같은 문맥에서 수량·대상·방향만 달라지는 짝을 포함해 값이 달라지면 복원 내용도 달라져야 한다는 감독을 구성한다. 이는 상관 작성 view이며 독립 자연 문서 수로 부풀리지 않는다. 자연 OWT 보존 자료는 기존 TRAIN split 안에서만 사용하고 held문서는 편입하지 않는다.

자료는 features id/prompt와 loss/score targets를 분리하고, 임의 padding으로 gold 위치를 feature로 표시하지 않는다. 자연스러운 앞·뒤 문맥 변화와 길이/잔여 위치 조합을 사전 생성해 실제 위치 coverage를 계산한다. 새 중요 문자와 부호가 모든 위치를 덮는지, literal duplicate와 기존 observed dev overlap, 문장 의미·방향·부정·수량 labels가 실제 원문과 일치하는지 CPU로 좁게 검토한다. 특정 gold span은 loss·사후 score 전용이며 decoder 입력에 건너가지 않는다.

초기 후보 loss는 기존 실제 복원 CE+중요 표현 CE를 기본으로 한다. V72 초기 정답의 TRAIN noncritical probability를 유지하는 anchor를 쓸 경우, teacher probability/정답 여부는 초기 모델의 별도 loss packet에만 둔다. 상류가 바뀌므로 features Commit을 frozen cache로 고정해서 학습하지 않는다. 의미적으로 같은 여러 표현을 하나의 압축 vector로 무조건 당겨 local form 정보를 없애는 과거 V68 초안이나 pre-bank factor head 우회를 자동 도입하지 않는다. 별도 factor head 정확도로 실제 언어 복원 성공을 대신하지 않는다.

역사적 분리 fine-tuning·보존 기법은 방법 후보를 주지만 우리 구조의 최적성은 보장하지 않는다. [ULMFiT 원문](https://aclanthology.org/P18-1031.pdf)은 층별 학습률·점진적 해제를 다뤘다. [Two-tower의 부정 결과 원문](https://aclanthology.org/2026.insights-1.5/)은 다른 과제에서 점진 해제가 오히려 불리할 수 있음을 보여준다. 따라서 이번 분리는 선행연구가 보장한 만능 처방이 아니라 V71/V72 자체 관측에서 선택한 한정 가설이다. 조건을 한 번에 무제한 확장하지 않는다.

실제 code·data가 나오면 호출 수·행/epoch·시간·출력/STOPstate·복구 비용을 계산해 새 상한을 고정한다. C100GiB 유지와 예상 출력+temp+8GiB 여유/GPU12GiB를 지킨다. 실제 전체 phase/optimizer/scaler/모든 RNG/cursor/AMPskip 복구CPU계약, 실제 worker owned preflight, ROOT 독립pin/resource/native0 검토 후 새 output/clock/planSHA로 한 번 봉인해 실행한다. 기존 owner의 postPopen 예외 cleanup·CIM 부분행·naturalexit 규율은 검증된 helper를 유지한다. 준비를 실행이나 성능 개선으로 세지 않는다.

평가는 같은 출력 기준이 고정된 상태에서 내용 변경을 비교한다. 새 held 표현의 모든 행을 filter 없이 평가하고, 관측 OWT/기존 영어·최대길이의 수량·이름·시간 손상을 함께 읽는다. 상류가 바뀌므로 V72의 old long Commit을 그대로 재사용해 새 상류 능력이라고 보고하면 안 된다. 길이 검사는 변경된 추출 가중치로 실제 streaming 경로를 다시 계산해야 한다. 새 사실 위험·회복 가능한 표면 오타·수량 변경을 분리하고 모든 결과와 불확실성을 보존한다. 자동 승격/무제한 추가 epoch/디노이징 진입은 하지 않는다.
