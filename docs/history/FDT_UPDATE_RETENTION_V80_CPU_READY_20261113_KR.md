# V80 CPU 준비 — 수치 학습자료 편향 수정 및 실제 Adam 갱신 보호 후보 (2026-11-13 10:40 KST)

이번에는 V75~V79 실제 기록과 코드·자료를 먼저 확인한 뒤 TRAIN1024행/teacher packet1024개와 저장 raw471개를 CPU로 조사했다. **새 FDT 학습/추론·GPU 실행은 없다.** 현재 선택 모델은 계속 V75이며 V79 결과와 미사용 budget은 닫혀 있다.

구체적인 자료 빈틈을 확인했다. 합성 수치 TRAIN의 B는 kg/온도/km 각256행에서 모두 `+`였다. A는 `+`/`-`가 각각128행이었다. 무부호 수치는 items의 steel rings만 있고, 합성 A/B에는 한 자리 수량이 없었다. 수치 단위에서 A 정수부분10~30, B50~70, items는 A30~50/B70~90이었다. EOS와 byte-role 균형이 내용 조건까지 균형 잡았다는 뜻은 아니었다. OWT128행에는 다른 숫자가 있을 수 있으므로 '전체 TRAIN에 한 자리 숫자가 전혀 없다'고 말하지 않는다. 이 분포가 모든 오류의 유일 원인이라는 증거도 아니다.

새 별도 CPU artifact는1024행=896개 상관 합성view(224quartets)+기존128OWT를 유지한다. 숫자 단위3개에서 A/B의 `+`, `-`, 암묵적 양수를 독립적으로 조합하고1/2/3자리 범위를 모두 포함했다. Items는 자연스러운 무부호 count를 사용한다. 모든4단위가0~9와16개 EOS role을 포함하며 총84개 unit/sign/width cell이 있다. Decimal 표기도 .25/.50/.875로 다양화했다. 완전한 sign×width×EOS×표현 전체 factorial이나 균등한84cell 빈도는 아니다.

ROOT가 전체1024행의 원문·UTF8span·Decimal 값/부호/자리수와224개 quartet을 독립 검증했다. 같은 quartet에서 A/B 수치 밖의 모든 byte가 동일하다. 기존 OWT128의 원문/targets는 정확히 유지했다. 이미 관측한 개발 문장은 exclusion-only 검사에만 쓰고 실제 문장/정답/event를 TRAIN으로 옮기지 않았다. E2 최종 holdout/registry는 접근하지 않았다. Feature는 id/prompt만이며 goldspan/role/labels는 loss·score용이다.

준비한 순서는384그룹×5행=1920노출이며 각 그룹은 authored quartet4+OWT1이다. 각224quartet을 최소 한 번, 그중160개를 한 번 더 노출하고 OWT128은3회씩 노출한다. Unique signed600/unsigned296/OWT128, head signed1028/unsigned508/OWT384다. 원래 U/S/O/U/S와 달라졌으며 자료와 grouping을 동시에 바꾼 후보이지 optimizer만의 인과 비교가 아니다. 385개 next-group cursor의 JSON복원도 확인했다. Teacher1002559UTF8B/head2731825B이며1920개 새 문서가 아니다.

실제 AdamW 제안을 직접 검토하는 transactional primitive를 구현했다. 원시 gradient가 아니라 momentum/weight decay가 포함된 실제 Δθ를1,.5,.25,.125의 고정 후보로 시험한다. 일부 damping이 채택되면 제안의 Adam moments를 유지한다는 알고리즘을 명시했다. 전부 거절하면 가중치+Adam을 되돌리고 finite-attempt scaler 증가는 소비한다. 예외/STOP은 가중치+Adam+scaler를 복구하고 gradient를 비운다. 입력/RNG/row/cursor/calls는 호출 driver가 소비 상태로 보존하며 재생하지 않아야 한다. AMPskip은 reference를 호출하지 않는다.

Tiny 실제 CPU AdamW·enabled GradScaler·다음 update 복원10개 검사와 raw 저장6개 검사가 통과했다. 이는 **전체 FDT worker/모든 phase·cursor·cache·CUDA 복구 검사 완료가 아니다.** 해당 driver·owner·실제 preflight는 아직 구현하지 않았다. 과거 raw-gradient/Adam 반례는 그대로 적용되며 이번 실제 FDT task-gradient 충돌을 측정했다는 주장도 하지 않는다.

다음 driver용 TRAIN-only reference100개도 사전에 선택했다:84개 unit/sign/width cell의 최저 원문hash 행+길이4분위별 OWT4행=16행이다. 예상7487개 hidden patch/512BF16값7666688B다. 향후 frozen prefix activation은 Commit에서만 만들어 TRAIN 전용 feature 파일에, label/correct mask는 분리 target 파일에 저장한다. K_E memory나 추론용 문장 검색이 아니다. 초기 batched reference prediction을 실제 NN으로 만들고 별도 검증해야 하며 old rowwise logits와 같다고 가정하지 않는다. 이100개에서 보존했다고 unseen 의미가 보장되는 것도 아니다.

공간 문제도 값 보존을 우선했다. Raw472개 logit tensor 중468개만 FP32→BF16→FP32 bit-exact였고, long/parity4개는 달라졌다. 따라서 일괄 BF16 반올림은 거절했다. 새 저장 후보는 완전히 같은 FP32값에만 BF16 carrier를 쓰고 loader가 원래 FP32 dtype/비트를 복원한다. Full FP32 장문과 parity는 그대로 둔다. NaN canonicalization을 CPU 시험에서 잡아 원본 FP32 fallback으로 고쳤고 초기 실패/코드도 보존했다. Scalar/negative-zero/noncontiguous/corrupt-carrier도 검사했다.

전체471raw/1074tensor를 메모리에서 pack/save/weights-only-load/unpack하여 dtype·shape·모든 비트·metadata를 독립 대조했다. 기존 raw913032323B→새 형식 예상771456643B, 절약 가능141575680B(~135MiB)다. 기존 파일을 바꾸거나 지운 것은 없고 실제 C 여유가135MiB 늘었다는 뜻도 아니다. 새 loader는 NLL/FSM 전에 반드시 unpack해야 하며 기존 CPU reader의 직접 torch.load에 BF16 carrier를 곧바로 넘겨선 안 된다.

Prospective 고정 범위는384attempt 이하/4개 damping 후보/초기reference1+최대reference1536=기존6752에 더해8289calls(후보8500cap), owner900/modelcut780/reserve120초다. 새 raw 저장 및 reference24MiB reserve를 반영한 estimate1023415424B는1GiB hard cap 안이다. 시작109GiB=보호100+8headroom+1owned 후보, 현재C117116940288B~109.07GiB다. **아직 실제 worker 비용·임시/STOP·actual plan-schema·owner를 검증한 실행 plan이 아니다.** 부족하면 code/resource 목적을 다시 계산하며 과거 cap이나 deadline을 늘리지 않는다. 캐시는 소량만 남아 있어 삭제하지 않았다. 가중치/raw/게임도 삭제하지 않았다.

다음은 이 후보의 실제 FDT driver 연결이다. (1) teacher 중 reference100개 frozen-prefix feature를 분리 저장, (2) 초기 batched-reference correct mask 생성, (3) 실제 Adam candidate의 TRAIN reference 평가와 accepted/rejected/AMPskip/STOP 원자 상태 보존, (4) new raw loader와 평가·독립 reader 연결, (5) 실제 schema/CPU full restore·preflight/ROOT 검토 후 새 SHA/clock 한 번 봉인한 한정 학습. 준비·테스트·봉인을 실제 실행으로 세지 않는다. V75가 초기화이며 이미 완료한 boundary audit·old broad baseline을 다시 선행조건으로 만들지 않는다.

영어 language→meaning→language 단계/K_P prior+K_C dynamic fuzzy/Commit-only decoder를 유지한다. K_E는 invalid NULL/content0/copy0이며 일반 텍스트 복사 경로를 활성화하지 않는다. K_P certified writer는 아직 없다. 디퓨전은 아직 호출0이고 미래 학습은 전체 입력·메모리·바인딩·출력을 고정한 별도 사고 가중치만 대상으로 한다. 에이전트는 사용하지 않았다.
