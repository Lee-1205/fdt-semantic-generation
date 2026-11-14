# V74 이후 중요 내용 보존과 교정의 간섭 완화

V74는 실제640/640/skip0,2,592calls/227.721초와345raw/초기 linked runtime/두 standalone fullstates 검토를 완료했다. 모든 owned/proxy는 종료됐다. 새900초/2,700call/3.5GiB 예산은 닫혔고 미사용108calls나 시간을 재사용하지 않는다. 다음 단계 worker/실행plan/seal/GPU는 아직 없다. 같은 V74 평가를 추가 로그 때문에 다시 실행하지 않는다.

이번 readout-only 수정으로 같은 추출·퍼지 결합 상태에서 기존 양수kg 보존이21/72→64/72,새DEV 주 수량이42/64→63/64로 개선됐다. 기존 정보가 새 출력기로 더 정확하게 표현될 수 있다는 구체적 증거다. 반면 새DEV 별도 보관량은26/64에 머물고 장문 수량11→29/decoded74→110,자연 OWT decoded99→120으로 퇴행했다. 초기 학습을 전체 실패라고 선언하지 않고 유용한 V72/V73/V74와 fullAdamRNG/raw를 모두 보존한다. 한 조건의 점수만 보고 자동 승격하지 않는다.

다음 목적은 학습된 signed-unit 교정이 일반 사건·인명·보관량·문서 기호를 훼손하는 간섭을 줄이는 것이다. 이미 관측한 장문·DEV의 실제 prompt/gold/name을 TRAIN에 넣지 않는다. 새 TRAIN은 평가와 다른 이름·값·형식을 사용해 일반 unsigned수량/사건번호,부호가 의미를 갖는 증감,양의 물리 보관량,다른 서술 위치/종속절을 함께 덮어야 한다. 새 held의 평가 명세는 TRAIN 전에 고정하고 관측된 기존192short와 장문을 보존 패널로 별도 보고한다. 상관 최소짝·합성 반복·이름 중복을 독립 자연 문서 수로 세지 않는다. OWT 기존 TRAIN의 보존은 가능하지만 held 문서를 섞지 않는다. 추가 corpus나 E2 final/registry는 이번 준비의 필요 조건이 아니다.

코드 확인상 현재 별도 KL은 초기 정답 비중요 바이트만 보호한다. 초기 정답 중요 바이트33,509개가 제외되지만 일반/중요 CE는 이미 해당 구간에 작동한다. 따라서 단순 누락 버그로 고치고 성능을 보장하지 않는다. prospective 후보는 초기 정답 중요 구간의 별도 유지 항과 초기 오답 중요 구간 교정 항을 분리하고 각 항을 독립 정규화하는 것이다. 틀린 teacher의 내용이나 실제 평가 정답을 학습 기준으로 보존하지 않는다. 추가 보존항의 역할·계수·gradient는 작은 CPU 계약에서 확인하고 전체 NN 개선이나 실제 GPU 성공으로 세지 않는다. 같은512/640회 추가만을 계획으로 삼지 않는다.

현재 C여유는 약109GiB로100GiB 보호선과8GiB 시작 headroom을 고려하면 새 V74식3.5GiB 출력 trial을 바로 시작할 수 없다. 원본·fullstate·raw를 지우거나 floor를 낮추지 않는다. ROOT는 별도 CPU 작업으로 **필수 부모를 명시한 active-readout delta + runtime** 저장·재구성 유틸리티를 구현했다. 실제 V74 최종518,012,241바이트 fullstate가 V73 필수 부모를 참조하는35,539,462바이트 파일에서 동일 전체 재귀 digest로 복원됐다. 두 원본은 그대로 남았으므로 기존 공간을 회수한 것은 아니다. 실제 모델 forward/optimizer0/CUDAfalse이며 새 worker 복구·CUDA 재시작·학습 성능 증거도 아니다.5개 focused CPU 계약과 정확한 부모·변경 tensor·Adam/scaler/RNG/runtime 보존은 `outputs/FDT_LINKED_RUNTIME_V75_CPU_READY_20261112_KR.md`에 기록했다.

후속 출력 byte예산은 새 loss-only teacher packet,새 raw,임시 쓰기와STOP 복구본까지 다시 계산한다. 참조형 복구본은 standalone이 아니며 부모 SHA/재귀 digest/고정 tensor 정확 동일성과 재구성 fullstate digest를 확인한다. 부모의 populatedAdam/scaler/RNG는 건드리지 않고 새 optimizer trajectory를 명시한다. 동일 상류 epoch의 동일 기존 행이면 Commit feature를 읽기 전용 참조할 수 있다. 새 데이터에는 해당 행의 실제 원문을 인코딩해야 한다. 저장량이 큰 새 Commit cache를 유지하는 대신,고정 상류에서 teacher phase에는 loss-only 확률만 저장하고 head phase에서 같은 원문을 재인코딩하는 유한 순회 후보를 검토한다. 이는 prospective 설계이며 실제 worker는 아직 없다. 추가 encoder 호출을 비용에 포함하고 바뀌는 추출기에 과거 cache를 쓰지 않는다. 새 output cap은1GiB 안팎이 가능할지 실제 raw dtype/장문 수/파일 임시 상태까지 계산한 뒤 고정하며 아직 실행 예산으로 선언하지 않는다.

새 runtime을 만들 경우 phase/cursor/order/노출/calls/history/inflight/두 단계 경계/Adam/scaler/모든RNG를 정확 재구성하고 오류·AMPskip도 행과 예산을 소비하게 한다. 완전한 새 bounded schema/CPU 복구/actual worker owned preflight/독립 ROOT pin·native·resource검토/새 SHA·출력/고정 time-call-row-epoch-output caps가 갖춰져야 한 번 숨김 실행한다. 출력 cap은 현재 여유에서100GiB+8GiBheadroom과 임시/STOP 공간까지 포함해 지켜야 한다. 하드웨어 재승인 대기나 자동 반복은 없다.

언어→의미→언어를 완료한 뒤 디노이징으로 간다는 사람 지시를 유지한다. K_P prior/K_C dynamic fuzzy/K_E ExactNULL내용0/복사0,Commit-only decoder와 원문 H·바이트·GTprefix·GT길이 우회 금지를 지킨다. K_P 인증 장기 writer가 없다는 한계를 숨기지 않는다. 현재 source 복원과 특정 필드 보존은 추상 의미 형성 전체의 증명이 아니므로,내용·관계·대상·수량이 다른 미학습 문장을 읽고 출력의 보존 여부를 계속 구분한다. 의미 평가를 문자 유사도나 전체 문자열 일치 하나로 대신하지 않는다.
