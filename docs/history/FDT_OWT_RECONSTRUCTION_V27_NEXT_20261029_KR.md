# V27 다음 목적 — 원래 퍼지 상태에서 읽는 훈련 압력 조정

먼저 PROJECT_STATE 첫 CURRENT와 V26 결과를 읽는다. 최신 source는 `runs/owt_fuzzy_reconstruction_20261029_v26/model/checkpoints/state_8192.pt`, SHA0bfdf57f12351f1fc38d144149ff2f42f9badc8dcc45dc769df72b8cd5096423 / algorithmac6b291422687514189e1a700ade6aab3677396ff780800445c98a459337e7bb다. V26 actual schema/local8192attempt8189updatecursor65536/global71652attempt71627update/63Adam/scaler32768tracker606/모든RNG/history/orders/전체ancestry/외부readonly1936calls metadata를 정확히 보존한다. V26미사용1,490calls·시간/모든 과거 예산은 CLOSED다.

V26는 teacher70.1125%/NLL.992906으로 개선 폭이 줄었지만 trainclean·corrupt loss와 일부 표면복원은 계속 좋아졌다. Dev/train전체 충실복원은0/16이다. 최소한의 telling-recipient 부분1개만 별도로 인정했고 경찰 신분/금지 내용은 미보존이다. 이전 두 정확한 인명의 퇴행도 남긴다. 이 정도 초기 데이터·노출로 전체 FDT/퍼지 실패나 특정 모듈 단독 원인을 단언하지 않는다.

CPU 기존logit 계산에서 고정16개 teacher/상태 복원 첫 오류 위치·symbol이 모두 같았다. 즉 초기 상태→byte 매핑의 오류가 있고, 첫 오류 이후에는 정답 과거를 돌려주는 teacher와 자체 prefix가 달라진다. 상태교환 NLLgap4.5191은 상태 정보를 전혀 안 쓰는 설명과 맞지 않는다. 원인 하나가 확정됐다는 주장 없이, 현재15% 과거 손상보다 강한 훈련을 한 번 시험할 근거로 삼는다. 기존 V18–V19 strict deterministic CE 수정·과거 mask 훈련 이력을 보존하고 같은 광범위 gradient/baseline/trace를 다시 선행조건으로 반복하지 않는다.

제안한 단일 변경은 **훈련용 past-byte masking probability의 유한 schedule**이다. 처음.15에서 첫4,096attempt 동안.50까지 선형 증가하고 다음4,096attempt는.50을 유지한다. CleanCE.5 + corrupted-prefixCE.5, 원래 architecture/퍼지 μ·ρ/anchors/value-prior-condition/2views-onlywriter/inference를 유지한다. 임의의.50은 최적값이 아니라 한 번의 사전 고정 후보이며 논문75% worddropout을 byte에 그대로 적용한 값이 아니다. 새로운 출력 의미 디노이징·규칙 답변·gold 역할·rawcontext 우회를 추가하지 않는다.

구현 전에 정확한 schedule 정의를 고정한다(예: attempt1=.15, attempt4096=.50, 이후.50). Probability는 새 allocation의 attempt로만 결정하고 현재 fullstate attempt로 복구한다. 기존 fullB×129 uniform draw수·순서를 유지하고 BOS/EOS/PAD 보호·eligible past-byte 제한·미래 label/길이가 과거 난수에 영향을 주지 않는 계약을 유지한다. Changed mask threshold 이후의 trajectory 변화는 의도된 학습 목적 변경이며 sourceAdam/RNG 초기화가 아니다. 각history에 실제 probability/eligible/masked 수를 기록하고 fullstate에 objective/schedule 계약을 저장한다. Source는 구15% objective로서 정확히 검증하고 새 objective로 import한 뒤 모델 가중치·전체Adam/scaler/RNG 불변을 확인한다.

필요한 CPU delta만 ROOT가 직접 작성한다: schedule 경계와 복구, 미래 byte/길이 변화에 대한 과거 mask/logit 인과성, 전체 source component 불변 import, enabledscaler·AMPskip·fullstate exact-next/마지막7행, 실제 plan schema와 post-Popen cleanup. 기존 strict deterministic elementwiseCE/동결 reasoning·prefix·basis/독립OSowner와 immediateSTOP-clock-callcap/FSmax1초+강제 큰 저장/C20GiB+22workerreserve를 유지한다. CUDA실행/성능 개선으로 CPU 준비를 보고하지 않는다.

후보 실행 규모는 같은 immutableV23final8192train문서/32768행, 두 finite epoch/8192attempt/batch8/65536행/5,852,956targetUTF8bytes다. Encode1+decoder2/attempt라 train24,576calls, 기존 dev128/statecontrols1552max/fixedtrain520max로26,776expected/max28,000/새900초reserve90/output6GiB/start40GiB/GPU12GiB를 후보로 한다. 실제 코드 비용·schema·CPU·workerpreflight·독립OSvalidate·새SHA/output/자원/native0 확인 뒤 ONE 새 한정 학습만 한다. 한도를 늘리거나 기존 clock·calls를 재개하지 않는다. 현재 V27 code/runner/owner/plan/seal/GPU **아직 없음**이다.

비교 기준은 저장된 V26 결과이며 추가 초기 GPU baseline은 하지 않는다. Same256devteacher controls와16dev+고정16train의 상태 조건 복원을 저장해 ROOT가 전부 직접 읽는다. Higher-mask 훈련 loss는 old15% corruptloss와 조건이 달라 직접 동등 비교하지 않는다. 청결 보조 NLL/byte/EOS, 전체/부분 명제·인명/나이·부정/관계 보존, graded lexical 지표와 과거 정확한 부분의 손실을 구분한다. Teacher 개선만 목표로 성공 판정하거나 미달이면 자동 추가SGD/승격하지 않는다. 다음 훈련의 구체적 결과로 판단한다.

관련 primary 연구는 [Scheduled Sampling(2015)](https://proceedings.neurips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html), [Generating Sentences from a Continuous Space(2016),§3·§6.1](https://aclanthology.org/K16-1002.pdf)다. 이는 방향을 차용할 근거일 뿐 현재byte-fuzzyAE의 최적 비율·성공·원인 증명이 아니다. 현재는 KL을 쓰는 VAE가 아니므로 posterior collapse라는 진단으로 바꾸지 않는다.

ROOT ONLY/noagents/로컬학습·검사·이상중단 승인 유지/하드웨어 재승인 질문 없음. 모든 원본/core/typedpartial9/a877/Phase2/archive/weightsAdamRNG/raw/실패 보존. E2final/registry/browser/mouse/keepalive/유료cloud/외부게시/사용자원본삭제/Windowsos.kill(pid,0) 금지. Terminal 뒤 allowned/proxy 종료부터 확인하고 학습 중 ROOT는 상세 분석을 멈추고 OS감독만 한다. Timer30분 prompt는 정확히 재개다.
