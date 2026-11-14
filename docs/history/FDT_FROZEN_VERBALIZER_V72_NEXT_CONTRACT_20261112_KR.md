# V72 고정 추출·결합 → 언어 readout 분리 학습 계약

V71의8개 교차 사례에서 마지막 readout 변경만으로 재현되는 퇴행,Commit·결합 변경으로 재현되는 퇴행,두 변경의 조합에서만 나타나는 퇴행을 각각 관측했다. 다음 한정 학습에서는 **추출·결합의 가중치를 고정하고 마지막 readout7개 tensor만 조정**해 경계가 동시에 움직이는 문제를 줄인다. 이것은 선택한 단계의 가설이며 모든 문제를 출력기에만 돌리는 결론이 아니다.

현재는 CPU 자료·loss·작은 모듈 기울기 계약만 완료됐다. 실제 V72 engine/runner/OS owner/실행 plan/seal/GPU는 아직 없다. 사람의 하드웨어 재승인을 기다리는 단계가 아니라 ROOT가 실제 driver와 복구 계약을 구현할 단계다.

부모는 V70의 독립 검증된 fullstate `state_0512.pt`,SHA `809414362bc7ac93b85afbadbb9cba9af67730a74d0ebfe0b744c78ce872f3d4`다. V67과 V70의 모든 가중치·Adam/scaler/RNG·실패·원시 출력은 보존한다. 다른 체크포인트 조합을 이미 승격된 제품 모델로 바꾸지 않는다.

`freeze_extraction_and_binding`은 전체 모델을 동결하고 `readout.readout.*`7개 tensor만 활성화한다. `readout.role_projection`도 이미 Commit을 만드는 쪽이므로 동결한다. K_P/K_C/Exact 세 구조를 유지하고,각 입력마다 실제 다른 문맥 퍼지 메모리를 만드는 계산을 고정 가중치로 수행한다. 메모리 내용을 모든 입력에서 같은 상수로 고정하는 뜻이 아니다. Exact는 휴면 NULL/내용0/복사0,의미 디노이징0이다.

TRAIN은 `research/artifacts/frozen_verbalizer_20261112_v72_cpu`의640행이다. 기존 V70 TRAIN512행의 id/prompt를 그대로 유지하고 새128행을 추가했다. 한 유한 epoch의640개 순서,954,100 UTF-8바이트와43,992개 중요 바이트를 CPU로 독립 검토했다. 새 DEV32행의 이름8개는 전체 TRAIN에 없다. V66/V69 평가 문장·정답을 TRAIN에 넣지 않는다. 표현 가족과 관측 개발에 따른 설계 선택은 공개하며 blind 일반화로 부르지 않는다.

완전한 수치 표현 mask는 시간·날짜·부호·소수·통화·범위·단위를 포함하는 원문 구간을 훈련 손실에서만 선택한다. 기존 OWT에는 구문상 숫자 표현 heuristic을 사용하므로 semantic gold/범용 NER라고 하지 않는다. 이름/숫자/regex/span·teacher 정답 여부를 NN 추론 feature로 주지 않는다. 추론 API는 원문→고정된 실제 추출·결합→Commit→언어 readout을 유지한다.

제안 손실은 V70의 기본·중요 표현 CE와 `0.05×Bernoulli KL`이다. 초기 readout이 **TRAIN에서 실제 정답을 맞힌 비중요 body 위치만** 확률 anchor로 사용한다. 중요 구간·PAD·초기 오답 위치는 anchor에서 제외해 교정을 막지 않는다. 평가에는 matched/gold 필터를 전혀 쓰지 않는다. 계수와 readout LR3e-5는 새 한정 가설이며 최적값이나 선행연구로 보장된 설정이 아니다.

입력 인코더와 결합 가중치가 고정되므로 TRAIN에서 실제 생성한 bound Commit을 저장해 decoder 학습에 재사용할 수 있다. cache feature에는 Commit 메모리값과 provenance만,정답 문자열·byte H·GT 길이·역할·teacher 확률은 별도 TRAIN target에 둔다. cache는 `encoder+role_projection` 전체 digest와 source id/version/입력 SHA에 묶는다. 이 상류가 바뀌는 후속 단계에서는 cache를 무효화하고 새로 만든다. 마지막 readout만 바뀐 것을 상류 임베딩 변경으로 혼동하지 않는다.

실제 driver에는 추출 cursor/cache refs·SHA,teacher 기준·정답 mask,head cursor/order/history/exposure,모델 전체·fresh7Adam/scaler·모든 RNG·call ledger·phase와 objective signature를 함께 저장한다. cache 도중·cache 완료→학습 경계·populated Adam 이후의 정확한 다음 update와 AMPskip의 행/노출 소비를 CPU에서 검증한다. 같은0번 상태의 directory 충돌을 피하고 실행 중 삭제·rotation하지 않는다. 아직 이 전체 driver 복구 검사를 완료한 것은 아니다.

제안은 source640행을 한 번 인코딩하고 초기 readout의 TRAIN 확률을 얻은 뒤 head640attempt를 한 번만 수행하는 것이다. skip도 행/노출을 소비하며 되감기·extra epoch를 하지 않는다. source 인코딩 노출,초기 teacher readout 노출,학습 readout 노출을 따로 기록해 세 번 새로운 자료를 본 것처럼 세지 않는다. 새 loss용 fresh7Adam을 사용하며 V70의 populated30Adam trajectory를 재개하지 않는다.

기존 V70의 관측 평가 Commit은 상류 가중치/입력/결합 digest가 동일할 때 최종 readout 비교에 활용할 수 있다. 이를 새 raw prompt의 end-to-end 평가라고 포장하지 않는다. 새 복합 수치 표현 DEV의 원문 경로,작은 수의 실제 원문→메모리→언어 재실행으로 cache와 실제 경계의 일치를 확인하고,기존 영어/자연 문장/최대 장문의 수량·이름·시간·부정·단위 손실과 퇴행을 함께 읽는다. 동일한 baseline을 광범위하게 반복하지 않는다.

새 model-call/time/output 상한은 실제 code 비용을 계산한 뒤 사전 고정한다. 기존 V67/V69/V70/V71 예산과 남은 시간·호출을 재사용하지 않는다. 실제 plan schema/CPU 전체 phase 복구/actual worker preflight/독립 native·자원 검토/새 SHA/output를 ROOT가 확인한 뒤 ONE hidden launch한다. C100GiB 여유 보호,시작 시 예상 출력·체크포인트 임시 여유와8GiB headroom,GPU12GiB/독립 OS owner/모든 postPopen 예외의 verified-owned cleanup/terminal allowned·proxy 종료 확인을 유지한다. ROOT만 수행하며 에이전트는 쓰지 않는다.

V72 자료와 loss 후보를 성능 개선이나 GPU 실행으로 세지 않는다. 현재 언어→의미→언어 개선이 진행 중이고,정보의 추상 의미화 전체 또는 보편적 무손실성을 인증하지 않았다. 의미 디노이징은 이 경로의 안정성을 확보한 뒤 별도 목적·계약으로 학습한다.
