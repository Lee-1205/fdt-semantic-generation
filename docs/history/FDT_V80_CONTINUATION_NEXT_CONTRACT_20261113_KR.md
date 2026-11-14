# V80 이후 연구 방향 — 학습 진전 보존, 표현·행동·숫자 경계 보완

다음 한정 언어 수리의 후보 부모는 V80이며 V75는 균형 비교와 복구 참조로 그대로 둔다. 전역 승격이나 언어↔의미 완성 판정이 아니다. 기존384attempt/247update/137reject/7569call은 닫혔고 미사용931call·시간으로 더 학습하지 않는다. 새로운 목적/예산/입력SHA/실제driver/CPU/owner 검토가 필요하며 하드웨어 재승인을 묻지 않는다.

이번 진전은 데이터의 독립 A/B 부호·자리수 coverage, quartet grouping, 실제 Adam update TRAIN guard를 함께 바꾼 결과다. 장문 수량15/5436,episode8/2718,action moved→roved5개가 V75의16/10/8보다 낮고 V79의29/22보다 회복됐다. 그러나 짧은 두 수량 clear는54→51/64다. 이51은 전체 문장 의미정확도가 아니다. 여덟 경우는 숫자 core와 unit이 그대로인 비표준 앞기호이고 네 경우는 실제 값 변화, 한 경우는75→7:의 digit 훼손이다. 오류 심각성을 과장하거나 실제 값/부호 손상을 무시하지 않는다.

다음 CPU 자료·코드 보완은 **TRAIN 표현과 predicate/identifier boundary 보존**으로 좁힌다. 현재 critical mask는 이름·수식에 집중하며 authored grammar의 recorded/count ed 중심 표현과 long moved 동사는 다르다. 이미 관측한 긴 event/개발 문장/정답을 그대로 TRAIN으로 넣지 않는다. TRAIN-only 이름·새 수치·새 어순으로 prospective 행동/부정/대상보존 문맥을 만들고, verb/주체/대상과 identifier 경계도 offline loss·score로만 지정하는 후보를 검토한다. 데이터·중복·문서/상관 view·실제노출을 기록하며 GT역할/정답문장을 NN feature로 주지 않는다.

V80에서 학습된 유용한 가중치를 버리고 같은 V75 시작을 무조건 반복하지 않는다. 새 부모로 V80을 쓰면 fullNN/247Adam/scaler/RNG/source cursor를 읽어 보존하고, 새로운 trial의 fresh Adam 여부·유한 순서·소비/거절/복구 정책을 명시한다. 기존 V75 loss-profile을 유지할 경우 source SHA와 profile SHA가 다르다는 관계를 실제schema에 명시한다. V79/V80 worker의 고정 parent SHA를 바꿔 그대로 실행하지 않는다.

TRAIN guard의100개 source/117600초기정답 byte·EOS위치 보존은 확인됐고14개오답(critical8/noncriticalEOS6)을 고쳤다. 이것이117600개 독립 사실이나 미학습 의미 보장은 아니다. 새로운 후보는 필요하면 TRAIN-only 표현·predicate reference coverage를 늘리되 무한 gate로 만들지 않는다. Hardtop1만으로 의미 보존을 보장하지 않고 실제 원문/출력의 값·역할·동작을 읽는다. 새 partial reference와 accepted/rejected/skip/STOP의 fullstate 계약을 유지한다.

현재 C는108GiB대이며100GiB 보호를 유지한다. 다음 실행은 현 여유에서 실제 owned output+8GiB headroom이 맞는 별도 고정예산을 계산한다. 새 raw의 bit-exact carrier는 유지하고 fullFP32 long/parity를 무단 반올림하지 않는다. 필요 시 **종료된 run의 논리 byte/SHA를 보존하는 가역 파일시스템 압축**을 별도 CPU 목적·정확 경로·전후 SHA/공간 측정으로 검토할 수 있다. 삭제 명령을 다른 도구로 우회하지 않는다. 캐시 삭제는 자동 승인 검토에서 차단됐고 실제 삭제0이다. 원본·raw·가중치AdamRNG·게임은 보존한다.

아직 다음 stage worker/plan/seal/GPU는 없다. 이미 완료한 전체 boundary audit와 같은 broad baseline을 무한 선행조건으로 반복하지 않는다. V80의 progress 문자열 loss_only_teacher_no_feature_cache는 상속 명칭이며 실제로 별도 TRAIN-reference feature cache는 존재했다. 다음 새 코드에서 명칭을 바로잡고 봉인된 V80 코드는 수정하지 않는다.

현재는 language→meaning→language의 안정화 단계다. K_P prior/K_C dynamic fuzzy를 유지하고 K_E는 invalid NULL/content0/copy0다. 미래 thought diffusion은 입력·memory·binding·출력 전체 가중치/buffer를 고정하고 별도 사고 가중치만 학습한다. 실제 thought trial 전에 당시 확정된 codec SHA를 새 정책에 고정해야 하며 V75용 정책을 V80에 자동 적용 완료로 보지 않는다. 이번에 디퓨전은 실행하지 않았다.
