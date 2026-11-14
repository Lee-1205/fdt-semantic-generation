# QueryCount 부분 9회 가중치 — 별도 읽기 전용 평가 초안

최신 상태: Root 검토와7개 CPU 검사·실제 preflight·독립 validate 후 별도 읽기 전용 owner를 한 번 실행했지만, plan의 checkpoint_payload_sha256와 owner의 parent_payload_sha256 참조가 달라 작업자 Popen 이전에 실패했다. 모델 호출0/optimizer0/checkpoint0이며9회 뒤 능력 평가는 아직 없다. 원래 QueryCount 실험도 STOPPED9/9/AMPskip0/미완료로 닫혔다. 아래는 보존하는 사전 계획이며 재실행·남은 예산 재사용을 허용하지 않는다.

목적은 유용한 부분 가중치를 판단하는 것이다. 원래 학습 전 저장한 자기 bound baseline과 같은 seed17·패널·입력·DDIM4를 사용해 9회 뒤 Q1/Q4 오류와 기존 성능 보존을 비교한다. 128회 실험 완료, 최종 3seed 평가, 새 독립 성능 검증 또는 자동 모델 승격을 뜻하지 않는다.

## 고정 입력과 비교

- 소스는 querycount_alignment_20261027_v1/model/complete_000010.json 및 payload_000010.pt, 전체 SHA5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675, 636586945B다. 모델 state SHAce6ba1b3adf1e0b7ba1d3b8d7b17915e6514429cd6a518057dae1211958614db를 확인한다. 추론에는 model weights만 strict-load하며 저장된 전체 AdamW/scaler/모든 RNG/이력/커서는 원본에 보존한다.
- 같은 QueryBound helper/core/config와 원래 mathSDPA·TF32off·CUDA fp16·batch4·seed17+원래 batch 시작위치·4step rollout을 유지한다. 구조 변경, 새 noise seed, neutral-init, 학습 및 optimizer 생성은 없다. Gold는 score에만 사용하며 모델에는 원래 다섯 feature만 전달한다.
- 원래 baseline/summary.json 및 각 패널 raw를 SHA로 고정한다. 기존 6개 패널(primary QueryMix64, uniform64, old role128, normal16, large16, paired16), querycount dev64의 Q1/Q4/Q8/Q16, 관측 transfer16의 Q1/Q4/Q8/Q16을 비교한다. Q16 dev는 primary와 facts/query/순서가 정확히 같으므로 그 한 번의 raw·metric을 명시적 alias로 재사용한다. 이 동일성은 CPU에서 검증한다. 총 14개 이름/13개 실제 평가 패널이다.
- 같은 관측 dev와 상관 prefix view이며 fresh blind 자료가 아니다. Transfer16은 학습에 없지만 이미 여러 번 관측했다. Q4 사례는 관측 뒤 선택된 진단이므로 별도로 보고한다. E2 최종 holdout에는 접근하지 않는다.

## 별도 사전 한도

- 새 output 제안: runs/querycount_partial9_readonly_20261027_v1. 이전 run의 terminal·STOP·source·plan·logs·시간을 바꾸지 않는다.
- ONE 읽기 전용 invocation, owner 전체120초(기동·CPU preflight·자료/가중치 검증·자원검사 포함), 종료 여유30초. 예상 top-level batched encode+denoise700회, 절대상한720회. 별도 readout 호출도 계측한다. 원래 학습의 미사용7376call을 전용한 것이 아니다.
- 13개 패널의 graph-view304+192+64=560행을 batch4로 평가한다. 배치140개 × (encode1+denoise4)=700회이며 initializer/final readout은 각각140회다. querycount dev Q16 alias는 새 forward0이다. 5392활성 score 위치는 같은 graph와 prefix가 겹친 수치이며 독립 query 수로 부르지 않는다.
- optimizer0/SGD0/학습 checkpoint0. 결과 raw/summary/검증 기록만 쓰고 출력 총64MiB 이내로 제한한다. 모델 parameters와 buffers 및 global Torch RNG를 전후 비교한다. 수치·입력/mask/hash·디스크·시간·프로세스 안전 오류만 중단 사유다. 성능 목표 미달로 결과를 버리거나 자동 반복하지 않는다.
- 독립 owned OS 감독과 fresh native 중복 검사, C20GiB·시작8GiB 추가 여유·GPU12GiB를 확인한다. Outer 로그는 supervisor stdout/stderr만 쓰고 worker 로그는 owner가 xb로 만든다. 모든 post-Popen 예외 정리와 proxy/actual child 종료 확인을 CPU에서 검증한다. 종료를 증명하지 못하면 UNCONFIRMED로 남긴다.

## 판정과 다음 단계

Q1/Q4 dev 오류율의 같은 비중 평균을 baseline과 비교한다. baseline은 Q1 0/64, Q4 2/256으로 이미 ceiling에 가깝다. 원래 개발 목표인 오류율 절반 이하와 각 Q dev99% 이상 여부, Q8/Q16 및 기존 패널의 accuracy/macro F1 보존을 구분해 보고한다. 관측 transfer Q4 1/64 누락의 교정 여부는 부수적인 결과이며 그 하나만으로 성공이라 하지 않는다.

기존 qualified control 4개(uniform/old role/normal/large)는 자기 bound baseline 대비 accuracy max(.94, baseline-.02), macro baseline-.05를 비교한다. Primary/paired 및 새 Q dev/관측 transfer는 상대 baseline-.02/-.05를 비교한다. Ground/initializer .95 기준도 확인한다. 한 번 평가이므로 원래 학습의 2연속 회귀 중단을 통과했다고 주장하지 않는다. 실제 개선·새 오류·회귀·불확실성을 모두 보존한 뒤 다음 구체적 정렬 개선을 판단하며 추가 optimizer를 자동 승인하지 않는다.

Root가 완성 코드·CPU 계약·실제 worker preflight·독립 validate·새 plan SHA·고정 한도·native/resource를 검토한 뒤에만 standing 사용자 승인 범위에서 한 번 실행할 수 있다. CPU 계획이나 위임 자체는 실행 증거가 아니다.

## 실제 prelaunch 실패와 후속 제한

Sealed plan301531387d75d50a4bd9b402e30148559e3e4c60fef8e56a32dd4b560177b9cd, worker9d8be54671badff4c7dc2b498a40016555b52bc8b6052be46415e8bcad1341e3, owner1073c61f283dd014fc1785c1396c4274c75484357a2a79cc05e5509e0d8085a0를 보존한다. Native24700→19056 생성19:23:05.988/19:23:06.013KST를 root가 관측했다. 감독 stderr에 KeyError가 기록됐으며 supervisor state·worker 로그·progress·result가 만들어지기 전에 종료했다. Root와 agent는 모든 관측 프로세스 종료와 native Python0을 확인했다.

CPU mock이 옛 parent_payload_sha256 필드를 넣어 실제 계획과의 불일치를 가렸고 root 검토도 이를 놓쳤다. 새 v2 owner는 실제 계획 schema를 복사해 전체 supervise 상태 생성 경로를 지나는 CPU 회귀로 이 문제를 고친다. 원래 봉인 코드는 수정하지 않는다. 아직 새 봉인·GPU 실행은 승인하지 않았다.

원래 owner clock이 저장되기 전에 실패했으므로 native proxy 생성시각을 보수적인 시작점으로 삼아 절대 끝19:25:05.987614/worker cutoff19:24:35.987614KST를 유지한다. 그 시간이 지났으므로 새120초나 남은720call을 부여하지 않고 CPU 수리만 한다. 이 실패는 모델 성능 실패가 아니며 부분9 가중치의 효과는 아직 판단할 수 없다. 근거는 outputs/QUERYCOUNT_PARTIAL9_READONLY_PRELAUNCH_INCIDENT_20261027.json이다.
