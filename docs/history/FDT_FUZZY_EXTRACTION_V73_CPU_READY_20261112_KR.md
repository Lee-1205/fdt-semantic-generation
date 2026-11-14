# V73 추출·결합 보강 CPU 자료/후보 준비 결과

V72 실제640회 학습과 원시·의미 검토를 완료한 뒤,발견한 중요 문자·부호·위치 편중을 보강하는 **다음 CPU 자료와 학습 경계 후보**를 준비했다. 새 V73 전체 모델의 optimizer trial/GPU/engine/실행 plan/seal/launch는 아직 없다. V72의 새32행 내용 위험과 장문 수량·이름 손상이 줄어든 실제 결과와 이 준비를 구분한다.

새 불변 자료는 `research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2`다. TRAIN960행은 새 작성832행/416쌍과 기존 OWT TRAIN128행이다. DEV32행을 별도 작성했고 기존 평가 문장·정답을 TRAIN에 편입하지 않았다. 실제 원문1,392,317 UTF-8바이트/중요 구간39,730바이트/최대16,384바이트/한 유한 epoch 순서를 보존했다. 416짝은 같은 문맥에서 주요 수량만1 차이 나는 작성 view이고 독립832개 자연 문서가 아니다. 자연스러운 앞 문맥과 시간·온도·지급액·이동·부정 절의 위치를 바꿨다.

중요 구간의 대문자A–Z/소문자z/plus/clock colon이 모두16개 byte-position 기여 위치를 덮는다. 여기서 position은 행위자 같은 추상 의미 역할이 아니다. 새 이름32개가 아니라 TRAIN26개/새DEV8개이며 실제 문장 안에서의 이름 구간만 offline loss/score에 표시한다. 새DEV 이름 전체와 기존 held 이름 `Elric/Mavis/Torin/Yvette/Stellan/Brynn/Lucan/Odette/Inez`는 TRAIN에 없다. NN feature는 id/prompt뿐이고 pair/family/facts/span/위치 label은 별도 target이다.

초기 CPU revision1은 주요 수량을6/7에만 집중시켰다. ROOT 자료 검토에서 불필요한 값 편중 위험을 발견해 **실행 전에** 원본 자료와 정확한 builder를 보존하고 새 revision2를 만들었다. revision2의 주요 값은0..99가 각각8회 또는9회 나온다. 통화는 불필요한 leading-zero 그룹을 피하도록100+q 그룹을 사용한다. 이전 revision1을 덮어쓰거나 GPU 성능 실패로 부르지 않는다.

독립 CPU 검토는 TRAIN+DEV992행의 실제 문자열·UTF-8 span·이름·작성 절의 수량/대상/부정 표현을 파싱해 재구성하고,416짝의 metadata와 순서의 인접성,128개 OWT TRAIN의 정확한 원문 동일성,기존 observed dev literal/새DEV 이름과의 분리,전체 위치 coverage와 finite960 cursor 순서를 확인했다. 문장 작성 가족과 앞 문맥은 반복되며 관측된 문제를 보고 설계했으므로 blind benchmark가 아니다.

`research/fuzzy_extraction_v73.py`는 V72 언어 출력기 마지막7개 tensor를 고정하고,실제 추출기의 마지막2 language block/value/condition/role_projection만 활성화한다. 실제 V72 fullstate를 CPU safeload하고 전체 모델 tensor를 strict copy했다. 실제 활성 수는 **14,769,664파라미터/23개 tensor**다. 부모7Adam/scaler/모든 RNG를 포함한 전체 recursive digest가 전후 동일했다. 원래 core를 바꾸지 않았고 실제 부모 모델 forward0/CUDA 초기화false다.

작은 CPU fixture에서는 정상 원문→K_P/K_C→bound Commit→고정 출력기의 loss가 추출·결합23개 tensor까지 전달되고,고정 출력기7개와 다른 동결 tensor가 그대로인 것을 확인했다. enabled scaler/fresh23Adam의 작은 한 update를 수행했다. 이것은 실제 큰 모델의 학습이나 전체 driver/scaler/RNG/phase 복구 시험이 아니다. 첫 진단은 매 호출 clone을 반환하는 `reader_tensor()`의 사용되지 않은 clone에 gradient를 붙여 실패했다. 해당 진단 source/incident를 보존하고 CPU test만 stored Commit tensor의 gradient를 관측하도록 고쳤다. production Commit 경계·clone 동작·기존 worker는 바꾸지 않았다.

다음 실제 구현은 두 단계로 나눠야 한다. 먼저 V72 초기 고정 모델로 TRAIN의 teacher 확률·정답 여부를 loss 전용 packet에 저장하고,그 다음 실제 원문을 매번 새로 인코딩해 고정 언어 출력기의 gradient를 상류로 보낸다. 상류가 바뀌는 훈련을 V72 Commit cache로 대체하면 안 된다. teacher 생성 cursor/packet SHA/완료 경계와 live960 cursor/fresh23Adam/scaler/모든 RNG/AMPskip 행·노출 소비까지 실제 engine의 정확 복구를 구현한다. 이 전체 engine/owned preflight는 아직 없다.

prospective 비용은 teacher960×2+live960×2+새DEV32 baseline×2+최종118행×2+장문 stream256≈4,396calls이며 실제 code의 parity/진단/호출 위치를 검토한 뒤 새 고정 cap을 확정해야 한다. 기존 V72의2,200cap을 늘리거나 미사용90calls를 가져오는 뜻이 아니다. 상류가 바뀌므로 마지막 장문2개도 새 가중치로 원문 streaming을 실제 실행해야 하고,old Commit만 재사용해서 새 추출 성능이라고 보고하지 않는다.

예상 fullstate/원시 평가/추가STOPstate 용량은4GiB 경계에 가까워질 수 있다. 실제 저장 크기를 계산해 새 output cap과 C100GiB+예상 출력/temp+8GiB 시작 여유를 검토한다. 현재 C약117.8GiB이며 원본·이전 weights/AdamRNG/실패를 삭제하지 않았다. 새 plan-schema/전체 복구CPU/actual worker owned preflight/ROOT 독립 pin·native·resource 검토 뒤 기존 자율 하드웨어 승인 안에서 한 번만 실행한다. 지금 CPU 준비를 실제GPU 실행이나 개선 능력으로 세지 않는다. 에이전트0/디노이징0/Exact 내용·복사0이다.

증거는 새 artifact의 `root_data_review.json`, `root_candidate_review.json`, `coverage.json`, `orders.json` 및 `scripts/review_fuzzy_extraction_v73_cpu.py`다. 앞 계약의 구현 전 설명보다 이 준비 결과가 최신이다. 다음은 자료를 다시 만드는 일이 아니라 실제 teacher/live engine·복구·OSowner·실행 schema를 구현하는 일이다.
