# V78 grouped-margin 언어 출력기 한정 학습 준비

V77 뒤의 구체적 수정이다. 입력 추출·K_P/K_C·순서 결합·Exact NULL 정책은 고정하고 마지막 출력기7개/2,908,192파라미터만 V75에서 학습한다. 사람의 '디퓨전은 입력·출력 고정 후 사고부만 학습' 정책과 구분되는 현재 언어 출력기 안정화 단계다. 의미 디노이징0이며 새 디퓨전 모델·학습은 없다.

현재 teacher가 맞힌 TRAIN 위치의 gold 대 strongest non-gold margin을 loss-only로 저장한다. Teacher margin≤3인 body 위치만 보존하고 보존 표적은 min(margin,1),critical/noncritical hinge 계수.25/.10이다. 틀린 teacher는 기존 cap8 CE로 교정하며 그 오답 class를 정답으로 주지 않는다. 현재 가장 강한 경쟁 class가 바뀌어도 새 경쟁 class를 제약한다. 최적 계수 증명이나 실제 성능 개선 주장은 아니다.

초기 CPU 검사에서 clamp_min의 경계0에서도 gradient가 남아 이미 같은 teacher 상태를 밀어내는 문제를 발견했다. 원본·실패를 보존하고 equality에서 gradient0인 ReLU로 고쳤다. 실제 worker CPU 첫 준비는 report를 두 번 감싼 '사용하지 않는 현재 pool weight' 설명 필드가 달라져 encode/readout/optimizer0에서 중단됐다. 원본 계획·코드를 남기고 idempotence 수정·9번째 CPU 검사를 추가했다. 이들은 GPU 모델 학습 실패가 아니다.

새 head order는 같은 기존 V77 1,920노출을5개씩 묶은384group이다. 각 group은 unsigned/signed/OWT/unsigned/signed,각 row loss .2를 같은 parameter 상태에서 누적하고 unscale/clip1/Adam을 한 번만 한다. 부호/무부호/OWT gradient row 비중.4/.4/.2다. 각 window mean이므로 전체 corpus byte 비중과는 다르다. V77의1,920회 개별 Adam과 달리384회 group Adam이며 margin도 같이 바꿨으므로 단일 변수 인과 실험이 아니다. Seed는 기존 초기화 코드의77113을 유지한다.

TRAIN은 기존1,024고유 원문/teacher한번/5micro×384=1,920head노출이다. 실제 원문1,010,943/반복head2,720,189 UTF-8바이트/중요86,108바이트다. 새로운 TRAIN 문장·dev 정답 편입0. V75 symbol weight를 유지한다. 관측340패널+동일 수정64개 전후+기존262KiB2개 실제 source/stream parity를 보존한다. 같은 관측 자료이며 새 blind 자연 benchmark가 아니다.

9개 loss/gradient CPU 검사와 작은 실제 grouped engine 계약이 통과했다. Populated7Adam/enabledCPUscaler/모든 RNG/parent-linked 정확 다음 group,teacher중간/경계,overflow시5행 소비,group1micro뒤STOP시requested5/actual1 상태 보존·자동 replay금지를 확인했다. 현재 tiny engine은 실제 전체 모델이나 CUDA 재시작 증거가 아니다.

실제 worker CPU preflight는19.203초,encoder2/readout2/optimizer0/CUDAfalse로 통과했다. ROOT는1017pins와 실제 initial/partial linked fullstate를 별도로 재구성해 V75 warm exact,상류 고정,teacher packet 분리,canonical model_scope7/2908192와 단일 출력 추정을 확인했다. OS owner의 postPopen cleanup 본문은 검증된 V77과 동일하다. 모든 CPUowned/proxy/Python0 뒤 독립 검토했다.

새 prospective 한도는384attempt/max7,000calls(expected6,752=encode3,076+readout3,418+stream258)/wall900/modelcut780/종료reserve120/owned1.25GiB다. 단일 출력 추정1,158,797,952바이트에는 initial/teacher와final/STOP/temp파일을 포함한다. 시작 C100GiB+1.25GiB+8GiB=109.25GiB/GPU12GiB,상시C100GiB 보호를 유지한다. 이전 V77미사용248call·시간은 재사용하지 않는다. CPU 준비·봉인은 실제 GPU 실행이나 보존 개선이 아니다.

ROOTONLY/noagents/사용자 원본·모든 weights/Adam/RNG/raw/실패 보존. 다음은 새 clock/seal 후 자원·native 검토를 통과하면 ONE hidden 언어 출력기 trial이며 자동 승격·자동 연장하지 않는다. 디퓨전 단계는 추후 확정한 codec 전체를 고정하고 별도 사고 가중치만 학습한다.

근거: research/artifacts/competing_margin_20261113_v78_cpu_v1/manifest.json,root_engine_CPU_contract_v2.json,execution_review_v2/root_independent_validate.json. 실행 후보 runs/competing_margin_20261113_v78.
