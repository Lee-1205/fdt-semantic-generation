# 정렬 전이·질문 수 검증 자료 — 2026-10-27

CPU 자료 준비와 root의 저장 자료 재구성을 통과했다. 새로운16개 abstract graph에서 질문 수1/4/8/16의64개 상관된 view를 만든 것으로,64개의 독립 graph가 아니다. 원래 QueryMix1088개와 V2의 명시적 관측 제외112개, 합계1200개의 canonical graph와 겹치지 않는다. 모든 과거 자료에 대한 신규성이나 통계적 독립성을 주장하지 않는다.

고정 graph 후보 seed5271000부터53개를 확인해 네 original sampling stratum에 각4개를 채웠다. 최대 후보50000 한도 내다. Oracle별 query support8개 이상이라는 graph 선택 조건은 남아 있다. Graph 선택 registry를 기록한 뒤 seed20261027로56개 nonself pair에서16개를 중복 없이 뽑고 앞의1/4/8/16개를 사용했다. Model 오류로 선택하지 않았다.

| 활성 query 수 | Graph/view 행 | 활성 query 위치 | UNRESOLVED/GREATER/LESS/CONFLICT | Premise UTF8bytes |
|---:|---:|---:|---|---:|
|1|16|16|4/3/5/4|1671|
|4|16|64|12/18/18/16|1671|
|8|16|128|26/36/34/32|1671|
|16|16|256|49/77/66/64|1671|

총464개 활성 query 위치에는 prefix중복이 포함되고, 고유 base query는256개다. 실제 physical facts 문자열1671bytes는 질문 수가 늘어도 같다. 숫자 query ID의 JSON 문자 수를 언어모델 token이나 premise 입력 byte로 더하지 않는다. 실제 실행에서 view와model마다 다시 encoder를 읽는 노출량은 별도로 계측해야 한다.

Agent5CPU검사와 root5CPU검사, root의 전체64행 재구성이 PASS다. Root는 별도 BFS로 원래oracle의 label·shortest hops와 일치함을 확인했고, all8IDs premise 참여/물리·canonical고유성/제외집합/같은facts·queryprefix·labels·hops/5개 neuralfeature/16slot paddingmask/byte 용량을 확인했다. Gold·hops·stratum metadata는 모델 입력 밖이다. E2 최종 holdout은 접근하지 않았다. 모델 forward0/optimizer0이며 GPU 검증 결과는 아직 없다.

- Builder SHA5364fcdf6ade2308a60e549ec2b4fcb8bb696126aa18b122f4683c18f7bd47fa.
- 실제 manifest 파일 SHA9f428e54ae184f5ab6f66021ba6dfa23e6fd6678ea84d8e20f46a9bf9932ae67.
- Self 제외 payload hash67d82b2f45e5eda6dc3ca10a9026e4a22567420ca8ca2d0a45e5f2ae6b6d8da3는 실제 파일 전체 hash와 다르다.
- Registry SHA9802a0b00b2e912b659513163311b82656fcd3da98094677ee56dee1350b49f4.
- q01/q04/q08/q16 파일 hash는 manifest와 root_cpu_review.json에 고정돼 있다.

다음은 하나의 제한된 readonly 전이 검증 driver/독립감독/CPU계약/초안 계획 준비다. 원래 FDT100과 QueryBoundRefiner가 같은a877부모 weights를 읽고, 같은 view의 feature/noise에서 비교한다. 네Q view/각4batch, encode2+denoise8로 실제160batched calls, 상한200/owner120초/종료reserve30초를 제안한다. Full16 base noise를 같은 graph의 모든prefix와 두 모델에서 유지하며 active mask만 점수 분모에 쓴다. 파라미터·optimizer·checkpoint를 변경하지 않는다. 아직새계획봉인·GPU실행은 없고 root의 실 코드/실preflight/계획SHA/새출력/자원/owned감독 검토 후 한 번만 실행한다. 질문 수가 바뀌면 full initializer의 query context도 바뀔 수 있으므로 query-count 변화 자체를 하나의 연산 원인으로 단정하지 않는다.
