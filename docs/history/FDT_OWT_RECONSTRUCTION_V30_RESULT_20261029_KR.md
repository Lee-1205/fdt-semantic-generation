# V30 — 같은 퍼지 입력 의미 재복원, 4block 후속 학습 완료

V29의 첫 추가 층 적응 한 epoch 뒤 같은 구조·자료·목적을 유지해 두 유한 epoch를 더 학습했다. 실제8,192attempt/8,188update/AMPskip4/621.591초이며 전체 종료를 확인했다. V29와 합쳐 이번 두 실행의 추가 갱신은12,282회, AE누적100,324attempt/100,287update다. 과거 실패와 누적36AMPskip+V18실패attempt1을 소급 고치지 않았다.

언어→원래 퍼지 입력 의미→같은 입력 재복원만 학습했다. 출력 목적 의미로 개선하는 추론·디노이징은 동결이다. 111,485,711total/38,309,762trainable, 마지막4languageblock/퍼지 μ·ρ·anchors/value-prior-condition/두 의미 view만 받는 decoder/p50 목적/추론 경로를 유지했다. 새 parameter·새 optimizer state·추가 층 해제는 없다. 원문 우회, 정답 prefix·정답 길이·외부 역할·정답 template를 state-only 복원 추론에 제공하지 않았다.

## 실제 실행

- Run `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261029_v30`, worker/owner COMPLETE/exit0/allownedgone. Source V29 finalSHA `97165ab6e7fcf9e6b75174b5b8eff80801c7f95925e49c3c8923b70cbd0d22ae`의 model/81populatedAdam/세group/scaler32768tracker642/phasefrozen/모든 저장 RNG를 exact로 가져왔다. V29최초 층 해제 기록과 전체 중첩 ancestry·과거자료·실패도 유지했다.
- 기존 문서8,192개/span32,768개에 seed381·382의 두 유한 순서, 선택65,536행/targetUTF8 5,852,956bytes/decoder지도11,836,984byte+EOS/masked2,924,372bytes. 원래 dev64문서256행과 old4096문서 prefix는 그대로다. 새 corpus 다운로드·새 blind패널·E2final/registry 접근은 없다.
- 실제 encode8,232+decoder18,286+denoise0=26,518calls/max28,000. 새900초/reserve90/output6GiB는 CLOSED, 미사용1,482calls·시간 재사용 금지다. 전체 학습 장부307,455calls는 따로 종료된 V23readonly1,936calls를 제외하며 합계309,391이다.
- CPU필요차이5PASS46.96초: 모든81Adam/3group exact import, enabledscaler AMPskip/정확한 전체 state next-update/마지막7행/RNG/order, 실제 schema/유한자료와 OS postPopen 세 종료 경로. 실제 worker CPUpreflight를 독립OSvalidate로 통과했다. CPU계약을 성능이나 실제CUDA복구 재실행으로 부르지 않는다.
- 시작40GiB 여유는 이전 CLOSED V20–V26 생성checkpoint21개의 무손실 파일시스템 압축으로 회복했고 낮추지 않았다. 모든 논리SHA·길이 exact/삭제0/이동0/소스·사용자원본·typed·V27–V29 제외 증거는 `research/artifacts/owt_checkpoint_lossless_space_20261029_v3`에 있다. 이 저장 방법 자체는 학습 개선이 아니다.
- PlanSHA `fce72e7a7e91773ae3ce8bdfbfd45cbd4f5c194e11287eca32bcf93670a12fbe`, 새start1793246242.4890554/deadline1793247142.4890554/cutoff1793247052.4890554. Owner5264→4892 생성12:57:22.734/.763KST, worker43872→30164 생성12:57:44.988/45.004KST. ROOT가 생성정보/parent/command/생존과 독립owner identity_verified를 관측하고 실제 학습 동안 OS/native/progress/latest/resources만 감시했다. 13:08:33 terminal/allownedgone/Python0을 먼저 확인한 다음 분석했다. 임시sleepguard는 종료 finally에서 해제됐다.

## 원문에서 추출한 의미만으로 복원한 문장을 직접 읽은 결과

ROOT는 같은 dev16문장과 고정 train16문장을 전부 읽었다. 두 패널 모두 UTF8/EOS16/16, exact0/16, **전체 충실 복원0/16**이다. 이는 무조건 자유 창작이 아닌 X→Z(X)→Xhat 평가다.

이번에는 train0의 `More people, some in critical condition`이 정확히 돌아왔다. **일부 사람들이 위독한 상태라는 부분 서술**은 보존됐다고 판단해 train16 중 부분 의미1건을 인정했다. 도착하는 행위와 늦은 금요일이라는 시간은 사라졌으므로 전체 문장 성공으로 세지 않았다. Dev의 확실한 부분 명제는 여전히0건이다. 이 수작업 판독은 이미 관측한 작은 패널에 대한 비맹검 판단이지 일반 의미 정확도가 아니다.

`NATO`, `Bill Haslam`은 다시 정확해졌고 `Craig Charles`도 유지됐다. 하지만 각 사건·다른 인명·합류 관계는 복원되지 않았다. Eduardo는 여전히 흐트러지고,25세는20세가 됐으며 경찰 신분·움직이지 말라는 지시도 없었다. `Outside the court`와 `It just goes to show you the exact` 같은 긴 첫 구절은 회복됐지만 뒤의 원래 의미가 이어지지 않았다. 원문이 문장 조각인 경우 가상의 완전한 정답 명제를 만들지 않았다.

| 사후 어휘 지표 | dev V29→V30 | 고정 train V29→V30 |
|---|---:|---:|
| chrF |23.0972→23.3863|32.3149→35.6401|
| wordF1 |10.4395→12.8883%|27.9395→30.5486%|
| content-wordF1 |1.6062→1.8470%|16.6967→24.1086%|
| 처음부터 맞은 UTF8bytes 평균 |2.75→3.375|12.5625→15.1875|
| byte편집 유사도 |25.5162→26.1976%|40.9750→42.6444%|

지표와 몇몇 조각의 회복을 남기되 전체 의미가 복원됐다고 해석하지 않는다. 문장마다 원문·실제 출력·주체/행위/대상·ROOT 판독을 `run/root_all32_state_reconstruction_reading.md`에 보존했다.

## 보조 teacher 지표와 해석 한계

| 같은 dev256행/23,207symbols | V29 | V30 |
|---|---:|---:|
| 정상 NLL |0.991277|1.016412|
| 정상 정답-prefix byte정확도 |71.1682%|71.5689%|
| 정확 byte/EOS 수 |16,516|16,609|
| 정답EOS위치 |253/256|253/256|
| 다른 행 의미로 교환한 NLL |6.407646|6.820012|
| 의미0 NLL |5.284431|5.636028|

정확도는0.40074pp 올랐지만 NLL은0.025134 악화됐다. 두 지표가 모두 좋아진 것으로 보고하지 않는다. 다른 행 의미/의미0에서 크게 나빠지는 것은 입력별 상태 사용을 지지하지만 완전한 의미 추출·decoder 무결성·독립 추론 능력의 증명은 아니다. 정답 이전 byte를 준 정확도71.57%를 의미 보존률이라 하지 않는다.

온라인 첫/둘째 epoch cleanCE .562848→.413813, corruptCE1.988449→1.933291, 혼합loss1.275648→1.173552로 내려갔다. 진행 중 가중치와 다른 순서의 평균이므로 고정 최종 trainNLL 비교가 아니다. 학습 자료의 부분 복원은 발전했지만 devNLL 악화는 남겨둔다. 데이터량/optimization/표현 압축/읽기 과정이 결합돼 있어 현재 자료만으로 어느 한 모듈이나 FDT전체의 실패를 단언하지 않는다. 과거 typed관계 성능과 원래 core/가중치는 별도로 보존됐다.

## 독립 CPU 원시 재구성과 다음 목적

ROOT reviewer `scripts/review_owt_state_reconstruction_v30_cpu.py`가 추가 model construction/forward0으로3fullstates의 SHA/safeload/재귀digest, 이름별 세 Adam group과 RNG/history/order/cursor/전체 ancestry를 확인했다. 초기81Adam 전부 exact, 최종 기존63step100,287·뒤에 추가했던18step12,282다. 새 AMPskip3460/5476/5959/7971과 scaler32768tracker221을 history로 재구성했고 모든 model/Adam은 finite, frozen reasoning/basis는 불변이었다. 저장23,207logits×3의 argmax/NLL, 모든 rawbytes/EOS 및 실제 batch마지막EOS 기반 call수도 재구성했다. 실제CUDA복구 재실행은 하지 않았다.

- Import0 772,873,413B SHA `1a0b350019c7dd524f77b3a09e60e7c08091e9196e3f39b4c6706ad0bfd2995b`.
- Mid4096 773,643,077B SHA `75a1c96aa9e1becf1a5aef25b2dc41cf57093742a3dc827854b9ff27becbb298`.
- Final8192 774,413,445B SHA `96480189836964c58153d7d002a27b25b279057af1555cd38950c754d724e504`, algorithm `1ce01437db667d46adebca638fc295e07d793d9268d8158e76094f4c3112ec81`.

다음은 같은 광범위 학습을 자동 반복하는 대신 **기존 훈련 문서64개/각 짧은 기존span1개에 집중해 이 경로 자체가 문장을 학습 복원할 수 있는지** 확인하는 유한 curriculum후보다. 퍼지/추론/decoder/가중치 구조를 바꾸거나 원문 우회를 넣지 않는다. 이것은 memorization을 포함한 학습 가능성 검사이며 blind LM 성능 또는 유일한 의미 정답의 증명이 아니다. 원래 전체 corpus·dev·이번 결과·모든 fullAdamRNG는 유지한다. CPU 자료만 `research/artifacts/owt_focused_reconstruction_20261029_v31_cpu`에 준비했고 **V31 runner/실행plan/preflight/seal/GPU는 없다**. 다음 계약은 `outputs/FDT_OWT_FOCUSED_RECONSTRUCTION_V31_NEXT_20261029_KR.md`다. 현재 Cfree 약40.91GB로40GiB시작 기준 미달이므로 새 실행 전 별도 저장 여유를 해결해야 한다. 기존 예산 재사용·기록 삭제·보호선 인하는 하지 않는다.
