# V79 이후: 출력 갱신의 보존 조건과 TRAIN coverage

다음 초기화는 V75이며 V79 자동 연장/승격은 없다. V75~V79의 코드·자료·실제 결과를 먼저 참조한다. 현재 목표는 영어 language→meaning→language의 안정화이며, 디퓨전은 아직 학습하지 않는다. 미래 디퓨전은 전체 입력·메모리·바인딩·출력 가중치와 buffer를 고정하고 별도 사고 모듈만 학습한다.

V79는 V78과 동일한 loss/weak margin/계수/그룹/1,920행 노출/384회/LR/seed에서 출력 prefix까지 고정했지만, 긴 수량 오류29/episode22로 V75의16/10을 보존하지 못했다. 따라서 다음 실험을 같은 2개 텐서에 같은 384회 더 적용하거나 λ만 올리는 작업으로 정하지 않는다. 입력 의미가 모두 완벽하다는 뜻도, Diffusion이 실패했다는 뜻도 아니다.

구체적인 다음 CPU 작업은 TRAIN 내 보존 대상과 수리 대상의 **실제 갱신 방향** 계약이다. 현재 correct weak margin<=3는 critical 중1,943/45,063위치만 선택했다. 반복 노출과 그룹 loss 평균은 unseen margin 보존 보장이 아니다. 기존 unsigned128/OWT128/signed768의 표현·자리·숫자 경계 coverage를 먼저 좁게 점검하고, 평가의 실제 문장/정답/장문 event를 TRAIN에 가져오지 않는다. 새로운 prospective TRAIN을 만들 경우 이름·숫자·어순/문장길이·정규화/중복/상관 view와 노출을 따로 기록한다.

학습 수리 gradient와 올바른 TRAIN 사실의 retention gradient를 구분하는 후보는 검토할 가치가 있다. [GEM 원 논문](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)과 [PCGrad 원 논문](https://arxiv.org/abs/2001.06782)은 gradient 제약/충돌 완화의 근거다. 이 논문이 현재 모델의 충돌이나 개선을 입증하는 것은 아니다. 논문의 episodic replay는 FDT K_E Exact memory 활성화와 전혀 다른 학습 표본 재사용 기법이며, K_E는 계속 invalid NULL/content0/copy0다.

**그대로 PCGrad를 Adam 앞에 붙여 안전하다고 부르지 않는다.** ROOT의 CPU 수치 반례에서 g_ret=[1,1], g=[2,-1]이면 dot=1로 raw-gradient 검사는 통과한다. 그러나 양의 diagonal preconditioner [.1,10]에 따른 Δθ=[-.2,10]은 g_ret·Δθ=9.8로 retention 손실을 1차적으로 증가시킨다. 모멘텀/weight decay도 실제 Δθ와 함께 검토해야 한다. 이것은 analytic toy 계산이며 실제 FDT gradient 계측이나 optimizer 구현이 아니다.

후보가 채택되려면 최종2개 optimizer 소유권/AMP unscale/부분 그룹/skip/거절된 step의 Adam·scaler·모든 RNG·cursor·노출 보존을 실제 driver CPU 계약으로 정한다. raw gradient projection만으로 보호를 주장하지 않고, 실제 Adam 제안 update 및 TRAIN 참조 손실/경계의 수치 변화를 기록한다. 같은 이미 관측한 개발 패널로 세부 threshold를 역산하지 않는다. 한정 실험마다 사전 고정한 목적/caps와 독립 검토를 갖고 실행하며, broad audit나 반복 baseline을 무한 선행조건으로 만들지 않는다.

현재 C: 약109.08GiB는 다음 동일 owner 시작조건109.25GiB보다 작다(100GiB 보호는 유지). 다음 GPU 예산을 무작정 다시 사용하지 않는다. 새 raw 저장이 float32라면 lossless BF16 roundtrip 여부를 실제 저장값에서 CPU 확인하고, 정확 복원되는 경우에만 새 저장 형식/실제 estimate/hard cap을 낮추는 경로를 검토한다. 원시값이 달라지면 그 형식은 거절한다. 기존 raw/체크포인트/AdamRNG/source/게임은 지우지 않는다. 준비는 새 GPU 실행이나 능력 향상 증거가 아니다.

아직 V80 worker/plan/seal/GPU는 없다. V79의 미사용248 calls/시간은 닫혔다. K_P prior와 K_C 장단기 fuzzy를 유지하고 raw input/GT prefix/GT length/정답 역할/regex를 출력 feature로 우회하지 않는다. K_P certified writer는 미완료다.
