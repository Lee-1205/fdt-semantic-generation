# V75 중요 내용 교정·보존 학습 실행 준비

ROOT가 V74의 실제 결과·코드·자료 이력을 먼저 확인하고 새 후속을 구현했다. V74에서 양수 수량이 개선됐지만 장문 숫자·자연문 보존은 퇴행했다. 이를 전체 의미 형성 실패라고 선언하지 않고,고정된 추출·퍼지 결합 상태를 읽는 출력기의 교정/보존 간섭을 이번의 구체적인 대상으로 삼았다. 학습 전의 이 문서는 새 모델 성능 결과가 아니다.

새 TRAIN640행은 일반 수량·사건번호256행,부호·단위·보관량256행,기존 OWT TRAIN128행이다.256개의 상관 최소짝이며 실제 원문969,013 UTF-8바이트/중요 구간29,114바이트/한 유한 epoch다. 새 DEV64행의 `Alaric/Blythe/Cyrene/Denzel`과 문장 형식은 TRAIN과 분리했다. 기존 실제 held prompt/이름도 TRAIN에 편입하지 않았다. 원문 입력에는 id/prompt만 두고 수량·이름·짝·span은 loss/score 쪽에 분리했다. 별도 검사로704행의 원문·구간·수량·행위자·단위·사건번호와 각 짝의 유일한 변경을 확인했다. 일반 수량 및±kg/온도 각각16개 바이트 위치를 덮었다. 위치를 추상 의미 역할로 부르지는 않는다.

준비 중 단순 substring 탐색이 보관량152 안의2를 주 수량2로 잡는 오류를 발견했다. 원본 builder를 `FDT_V75_CPU_PREBUILD_REV1_20261112.py`에 보존하고 숫자 경계를 적용한 뒤 독립 전체 문법 파싱으로 확인했다. 첫 출력 예산 계산도1GiB를 약3.6MB 넘음을 사전에 잡았다. 원래 계산 코드를 보존하고,Commit을 중복 저장하지 않는 raw 참조 파일의 컨테이너 오버헤드를 실제2,209바이트와 비교해 보수적인16KiB로 산정했다. 실행 cap이나100GiB 보호선을 높이거나 낮추지 않았다.

새 loss는 초기 정답 중요 구간 CE.10/초기 오답 중요 구간 CE.25,초기 정답 비중요 Bernoulli KL.10/초기 정답 중요 KL.25를 각각 따로 정규화한다. 기본 바이트/EOS/어려운 위치/잘못된 끝 EOS 손실은 유지한다. 잘못된 teacher 출력을 정답으로 학습하지 않는다.6개 CPU 수치·gradient 계약에서 PAD 제외,teacher와 동일할 때 KL0,중요 정답 확률 하락의 복원 방향,오답 teacher 비보존,비중요 구간 증가에 따른 중요 anchor 희석 방지 등을 확인했다. 계수는 후속 가설이며 최적값이나 성능 보장을 뜻하지 않는다.

이전 출력을 유지하며 새 학습을 병행하는 근거로 [Learning without Forgetting](https://arxiv.org/abs/1606.09282)을 확인했다. 원 논문은 다른 과제/모델의 연구이므로 현재 FDT의 계수를 보장하지 않는다. [Unlikelihood Training](https://arxiv.org/abs/1908.04319)도 확인했으나 이번 문제는 반복 단어만이 아니라 문맥별 수량 보존이므로,이번에는 일반 토큰에 강한 추가 반복 벌점을 넣지 않고 교정·보존 분리와 자료 문맥을 먼저 바꾼다. 이는 현재 관측에 대한 ROOT의 설계 판단이다.

전체 V74 NN123,497,080개를 정확히 복원하고 상류를 고정한다. 마지막7개 readout tensor/2,908,192파라미터만 freshAdamW LR1e-5로 학습한다. 부모의 populated7Adam/scaler/모든RNG는 그대로 보존하며 원래 trajectory를 재개하지 않는다. Teacher phase에는 정답 확률·정답 여부·label/mask만 저장하고 **Commit feature를 파일로 쓰지 않는다**. Head phase에서 각 원문을 실제 다시 인코딩한다. K_P prior/K_C dynamic fuzzy→Commit→언어 경로이며 K_E ExactNULL/내용0/복사0,디노이징0,원문 H/바이트·정답 prefix·GT길이 우회 금지를 유지한다. K_P 인증 장기 writer는 아직 없다.

작은 engine에서 teacher 중간/경계,부모 참조형 populated7Adam/enabledscaler/모든RNG의 정확 next update,AMPskip의 행·노출 소비,teacher/head in-flight 중단 상태 보존과 자동 재실행 거부를 확인했다. 실제 크기 CPU owned preflight15.656초에서도 초기·teacher 일부 runtime의 정확 재구성,실제 원문 encoder2/readout2와7개만의 gradient,장문 wrapper tensor 일치를 확인했다. 실제 optimizer0/CUDAfalse이고 모든 CPU owned/proxy 종료를 확인했다. 실제 CUDA 복구 증거는 아니다.

ROOT는614개 input pin과 새 코드 문법,실제 V74 부모·fresh 초기 상태,복구 상태의 source/teacher 분리,기존에 검증한 V74 owner와 동일한 postPopen cleanup 본문을 독립 확인했다. 제안된 실제 비용은 encode1,412+readout1,690+stream chunks258=3,360calls/blockwork2,322이다. 새 고정 한도는3,500calls/900초/780cutoff/120reserve/출력1GiB이며,기존 모든 terminal 예산과 분리한다. 초기/teacher경계/최종/추가STOP은 각각 필수 V74 부모를 참조하는 runtime으로 보존하고 standalone이라고 부르지 않는다.

원시 평가·teacher·모든 linked 상태·STOP·metadata/여유32MiB를 합한 보수적 예상 출력은1,072,809,248바이트로1,073,741,824바이트 cap 안이다. 기존 Commit에 대한 평가에는 SHA가 고정된 원본 참조를 남겨 중복 저장을 줄인다. 새 문장64개는 학습 전후 실제 원문 경로로 평가하고 기존276개·262KiB 장문2개는 이미 관측한 보존 패널로 보고한다. 장문은 원문부터 다시 실행하며 native/stream4096바이트 및 같은 현재 출력기로 읽은 기존 Commit과 비교한다. 같은 장문 raw를 두 번 저장하지 않는다.

시작 조건은 C100GiB보호+출력1GiB(임시/STOP포함)+여유8GiB=109GiB,시작GPUfree12GiB이며 상시C100GiB/GPU사용12GiB를 보호한다. ROOT의19:40:21 확인은 C117,084,520,448바이트/GPUfree15,161MiB/nativePython0였다. 새 실제 plan 봉인과 단 한 번의 숨김 launch는 이 준비와 별도 증거로 기록한다. 준비·CPU 통과를 GPU 성공으로 부르지 않는다. 자료·code·actual preflight·ROOT 검토는 `research/artifacts/critical_retention_20261112_v75_cpu`에 있다.
