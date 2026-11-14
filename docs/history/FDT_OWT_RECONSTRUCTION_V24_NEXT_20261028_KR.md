# 다음 한정 목적 — 보호 검사의 중복 I/O 비용을 줄이고 같은 추출·복원 학습 계속

최신 실제 source는 V23 partial `runs/owt_fuzzy_reconstruction_20261028_v23_final/emergency/checkpoints/state_7440.pt`다. SHA `c45b5efb9b57d3cba1bda234bda025295b6e275fe99dab378181d971c9360a69`, algorithm `0635fd824fab9908114c169eef2ac4ff15c78fc41ba09e3ba626c2c7212670c9`, model digest `9f31aecd0d5bf663e4bd63b5d124af22029bb376a49b364de1510193b7a8bb89`. Source schemaV23/local7440attempt7437update/cursor59520/global47076attempt47061update/63Adamstep47061/scaler65536tracker817/모든RNG/history/orders/전체 ancestry를 정확히 보존한다. 과거14AMPskip+1V18failedattempt도 유지한다.

학습 supervisor FAILED/exit2/time-reserve guard와 누적7,437update는 그대로다. 원래752미시도·5,680미사용calls·시간은 CLOSED이며 재개하지 않는다. 별도 읽기 전용평가1936calls/23.38초/optimizer0도 CLOSED/미사용464calls·시간을 재사용하지 않는다. 현재 source는 7,440회 실제 partial 상태이며 존재하지 않는 V23final8192를 찾지 않는다.

현재 관측 teacherNLL1.061039/보조 정확도67.5141%는 source를 반영한 읽기 전용run `runs/owt_partial_assessment_20261028_v23_readonly`의 저장 결과다. 같은 dev/function baseline은 이 결과를 사용하며 V22를 최신 기준으로 오인하거나 초기 GPU평가를 반복하지 않는다. Whole state-conditioned 복원은 dev/train각0/16으로 아직 어렵고 V22의 한 train 부분 명제도 이번에는 사라졌다. 보조 성능 개선과 실제 복원 불안정을 모두 남긴다. 원래 의미 추출→언어화 사이클을 유지하고 추론/출력 목적 의미 디노이징은 계속 별도 동결한다.

먼저 ROOT 직접 CPU로 작은 runtime 변경만 구현한다. 현재 `RecoveryBudget.check()`는 step 시작과 세 NN charge마다 디스크 사용량·전체 owned파일 크기를 순회한다. 종료 후32회 측정한 V23 비용은 owned_bytes3.398ms/disk .0168ms, 네 검사 약13.661ms/attempt다. V22도 비슷하므로 학습 지연의 단일 원인이란 주장은 하지 않는다. 절대 보호 I/O 반복 비용을 줄일 후보로만 삼는다.

새 예산 객체는 **STOP 존재·고정 cutoff·callcap을 매번 즉시 검사**한다. 비싼 disk/owned-size 검사만 monotonic 기준 최대1초 간격으로 제한하고 시작·복구 직후·큰 checkpoint 저장 직전에 전체 검사를 강제한다. 독립 owned OS owner의 기존1초 자원 검사·C20GiB 보호/22GiB 작업 여유·output cap은 유지한다. 구간 안에 큰 쓰기가 있을 때는 cache를 믿지 않고 사전 강제한다. Emergency 저장은 cooperativeSTOP 때문에 저장 자체가 차단되지 않도록 자원 검사와 STOP/clock 검사를 분리하고 원래 owner deadline까지의 저장 규칙을 보존한다. 새 자원 검사 시점의 최대 지연과 강제 경로를 실제 plan-schema에 명시한다. 보호를 조용히 생략하거나 무제한 캐시하지 않는다.

CPU 계약은 fake clock을 사용해 STOP/clock/call 즉시 중단,1초 만료 시 disk/output 실패 검출, 강제 checkpoint 검사, 복구 후 첫 자원 검사, post-Popen 예외 cleanup을 확인한다. 실제 model·objective·AMP·정밀도·NN 입력·fuzzy 구조는 바꾸지 않으며 source 전체 Adam/scaler/RNG/ancestry import와 exact-next 복구를 확인한다. 이미 통과한 unrelated gradient/broad audit/GPUtrace를 반복하지 않는다. CPU timing은 실제 GPU gain이나 완료 보장이 아니다.

같은 최종 artifact `owt_fuzzy_reconstruction_20261028_v23_final`8192train-doc32768행/old4096prefix/64dev256정답·순서를 유지한다. 다운로드0/새 품질 규칙0/새 평가 panel0이다. 새 finite2epoch/8192attempt/batch8/65536rowselections/targetUTF8 **5,852,956 bytes**=old2,918,756+new2,934,200, 두 decoder 손실UTF8감독11,705,912/byteEOS11,836,984를 후보로 한다. 원래 cleanCE.5+corruptprefixCE.5/.15mask/strict deterministic elementwise reduction/현재 AMP guard를 유지한다. 새 local clock/order/call allocation/finite cosineLR 구간만 source와 구분한다. 과거 미소비752회 시도를 이어 쓰지 않는다.

비용 후보24576train+128dev+1552devstatecontrols+520fixedtrain=26776expectedcalls/newmax28000/owner900초/reserve90/output6GiB/start40GiB/GPU12GiB다. 원래 V23 한도를 늘리거나 reset하지 않는다. 새 I/O 변경이 실제 속도를 개선하는지 새 한정 학습에서 측정하며 reserve guard에 닿으면 현재 상태를 저장하고 닫는다. Fullstates import0/mid4096/final8192+STOP을 새 directory에 보존한다. 고정 train16과 dev16은 SAMEINPUT 복원을 따로 저장하고 ROOT가 의미를 읽는다. Gold prefix/길이/역할·원문 context 우회는 추론에 쓰지 않는다.

코드·source/dataSHA·CPU·실제 workerpreflight·독립 OS validate·자원·native 중복0·새 고정 상한을 ROOT가 검토한 뒤 ONE 새 학습을 한다. 하드웨어 재승인을 묻지 않는다. 현재 V24 code/worker/owner/plan/seal/GPU는 **아직 없다**. 현재 모든 모델 프로세스는 종료됐다. CPU 예측을 GPU 개선으로 보고하지 않는다.

ROOT ONLY/noagents, 모든 source/core/typedpartial9/a877/Phase2/archive/사용자원본/weightsAdamRNG/raw/실패/자료 준비 이력 보존, E2final/registry/browser/mouse/keepalive/유료cloud/외부게시/사용자원본삭제/Windowsos.kill(pid,0) 금지, 자동재시작·olddeadline연장·unusedbudget재사용·실행중sealedcode수정삭제 금지를 유지한다. Timer30분 prompt는 정확히 재개다.
