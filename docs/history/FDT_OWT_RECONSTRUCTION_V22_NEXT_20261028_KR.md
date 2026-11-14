# 다음 한정 목적 — 같은 퍼지 추출·재복원 사이클의 학습량 확대

최신 사람 지시대로 원문 X→원래 퍼지 앵커 Z(X)→같은 원문 X̂를 학습한다. 추론·출력 의미 디노이징은 계속 별도·동결한다. Decoder는 두 추출 view와 BOS/자체 prefix만 사용하며 원문 context·정답 prefix·정답 길이·외부 역할은 추론에 주지 않는다. Teacher-prefix 점수는 보조 지표, 상태 조건 원문 복원은 같은 사이클의 결과로 기록한다. 자유창작·일반 LM 검증을 새로운 gate로 두지 않는다.

최신 실제 source는 V21 `model/checkpoints/state_4096.pt`, SHA `32c8ba9af2381a58147eb6db9c9551211c7f218440ce81d7a175634b4aad82d2` / algorithm `9fdd540d19026309b4542a28ff71f64e67ab949812cb920d31cc2a282e86112e` / plan `01e73a60bb4f01865e07138e57f2bdec07e735b686d0adc65feb3a3ee880b910`다. 全model/63Adamstep31436/scaler131072tracker1976/모든RNG/finite cursor/history/orders/exposure/전체 중첩 ancestry를 보존한다. Source global31444attempt31436update이며 새 local allocation은0이다. 과거7AMPskip+1V18failedattempt와 모든 실패도 유지한다. V21 미사용1095 calls·시간은 CLOSED다.

V21은 teacher-prefix NLL1.28687→1.26921/정확도62.22→63.08%로 개선됐다. 상태만 받은 복원의 훈련16문장은 개발16문장보다 이름·도입부·내용어를 더 보존했지만 둘 다 충실한 전체 복원은 아직 없었다. 이것은 현재 작은 표본의 학습 상태이지 퍼지·FDT 전체의 실패 판정이 아니다. 더 많은 학습을 원하는 사람 지시와 online loss가 감소한 근거에 따라, 이번에는 같은 목표에서 새4finiteepoch/8192attempt 후보로 학습량을 늘린다. Architecture/anchor/penalty/손상확률/정밀도/추론 경로를 동시에 바꾸지 않는다.

자료는 동일 immutable V20 4096train문서/16384선택문장·구간, 기존64dev문서256행 그대로다. 새 다운로드·held-to-train 편입·평가 panel 확대는 없다. 4epoch/65536rowselections/5837512targetUTF8bytes, 두 손실의11675024UTF8감독/11806096byte+EOSsymbols다. 실제 입력 분할 중 불완전 구간이 관측됐으므로 “모든 행이 완전한 문장”이라고 부르지 않는다. 원본·기존 split을 소급 수정하지 않는다. Sentence segmentation 개선은 필요할 때 새 train-only artifact로 별도 수행하고 기존 held를 섞지 않는다.

최근 실제 시간321.10초/4096attempt를 바탕으로8192attempt는 약640초 규모의 후보이며 무제한 학습이 아니다. Cost는 actual3calls/attempt라 training24576+finaldev128+devstatecontrolsmax1552+fixedtrainnormalmax520=**26776expectedcalls**다. 실제 worker 비용 검토 후 새max28000/owner900초/reserve90/output6GiB/start40GiB/GPU12GiB를 사전고정한다. 학습 중 final reserve에 닿으면 현재 상태를 보존하고 닫으며 deadline을 늘리지 않는다. Source import는 정확히 하되 새 finite cosine LR 구간/order/clock/call allocation이 구분됨을 기록한다.

Fullstates import0/mid4096/final8192+STOP현재상태를 새 output에 저장하고 삭제·rotation하지 않는다. 같은 dev/function의 baseline은 V21 저장 결과를 사용해 추가 초기 GPU 평가를 하지 않는다. 최종 dev16 및 이미 고정한 train16의 SAMEINPUT 복원을 분리해 저장하고 ROOT가 주어·행위·대상·부정·관계 보존을 직접 읽는다. 단어 조각·lexical 점수를 의미 확률로 부르지 않는다. Train16은 이미 본 자료이며 새 blind benchmark가 아니다.

필요한 변경 검사는 실제V21schema 전체 source/Adam/scaler/RNG/ancestry import, 4유한epoch/8192attempt/cursor·AMP 노출·정확 next 복구, actual cost·시간·자원·독립 owned OS post-Popen cleanup·worker preflight·독립validate다. 동일한 broad audit/gradient probe/반복 trace를 무한 선행 조건으로 만들지 않는다. ROOT가 새 코드·SHA·고정 상한·native중복0·C20GiB보호+checkpoint임시여유+8GiBheadroom을 확인한 뒤 ONE 새 한정 실행한다. 하드웨어 재승인을 묻지 않는다.

현재 V22 code/runner/owner/plan/seal/GPU는 **아직 없다**. 현재 학습/추론 프로세스는 없다. 준비를 성능 개선으로 보고하지 않는다. ROOT ONLY/에이전트 금지, 모든 source/core/typedpartial9/a877/Phase2/archive/사용자원본/weightsAdamRNG/raw/실패 보존, E2final/registry 접근 금지, browser/mouse/keepalive/유료cloud/외부게시/사용자원본삭제/Windowsos.kill(pid,0) 금지, 자동재시작·olddeadline연장·unusedbudget재사용·실행중sealedcode수정삭제 금지를 유지한다. Timer30분 prompt는 정확히 재개다.
