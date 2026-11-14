# 다음 한정 작업 — 확장된 OWT에서 같은 추출·복원 학습 계속

최신 실제 source는 V13최종 `runs/owt_fuzzy_reconstruction_20261028_v13/model/checkpoints/state_2048.pt`, SHA `56b769013a38d9672dcbc684640228594da8d87375a8c6067f450c26dc1e6ad3`, algorithm `4b2ac7da1d07befad4867f81b3fc9b0767f29df5438076b649deca17e686c9cc`다. V13은2,048추가/AE누적4,096업데이트 완료, 모든 owned/proxy 종료, 예산CLOSED다. 미사용1,040call/시간은 재사용하지 않는다.

정상 teacher byte 정확도43.15→49.26%, NLL1.97629→1.74413으로 개선됐다. Swap NLL비용.3906→.7813, zero비용.2556→.4355로 추출 상태 사용도 늘었다. 자유 exact0/16, EOS15/16으로 충실한 전체 문장 복원은 아직 학습 중이다. 활성 앵커15→11/64 집중은 보존하되 전체 의미 붕괴나 방식 실패로 결론 내리지 않는다. 데이터는 아직1,024문서/4,096짧은 문장/364,178고유 UTF8bytes다.

다음 ROOT는 같은 모델/원래 fuzzy routing/추론동결/원문우회금지를 유지하며 **학습량을 늘리는 목적**을 구체화한다. 새로운 진단·baseline·동일 CPU test 반복을 무한 선행 조건으로 만들지 않는다.

1. Source의 모든63Adam step4096/scaler4096 tracker96/모든 RNG를 새 importer로 그대로 가져온다. Source model·optimizer·group·RNG component digest를 비교한다. Source V13의 local counters2048와 ancestry V12업데이트2048를 모두 보존해 **AE global4096**을 유지한다. 다음 global을 source local2048로 잘못 계산하지 않는다. Source의 이미 닫힌 order/deadline/calls를 다시 열지 않는다.
2. 당장은 이미 확장한1,024train문서/4,096문장과 같은64dev문서/256문장을 사용해 학습량 효과를 본다. 새 문서 수를 늘렸다고 하지 않는다. Prospective8유한 epoch/최대4096새attempt/batch8/32,768행/2,913,424targetUTF8bytes를 실제 코드 비용과 함께 검토한다. 이전과 같은 base LR language1e-5/other1e-4의 새 유한 cosine schedule이며 moments는 유지한다. 임의의 큰 반복·anchor 균등화 벌점·anchor 재초기화를 추가하지 않는다.
3. 실제 예상 call은 train8192+final held3조건128+final16자유복원최대1552=9872다. 실제 driver를 확인한 뒤 cap약10500/900초/reserve90/output6GiB/start40GiB/GPU12GiB 안에 고정한다. 아직 실행 plan/열린 예산은 없다. 모델 함수·같은 held/control이므로 V13저장 baseline을 참조하고 초기 GPU 평가를 다시 하지 않는다.
4. 새 fullstate/importer/finite cursor/call separation에 필요한 CPU delta와 실제 worker preflight·독립 OSvalidate만 하고 ROOT가 새 SHA/clock/caps/resources를 확인한 뒤 ONE실행한다. 원래하드웨어승인은 유효하다. 새 source schema는V13이므로 V12-only importer의 허용 schema를 억지로 속이거나 예전 sealed code를 수정하지 않는다. 별도 V14 파일에서 정확히 처리한다.
5. 결과는 teacher byte 성능, 자유 복원, state교환·제거, 앵커 소속도를 함께 기록한다. 자유 byte 편집 오류는 V13에서 처음 계산한 사후 보조지표이며 semantic 정답으로 세지 않는다. 가능하면 기존 저장 raw를 CPU로 비교하고 추가 GPU trace를 자동 조건으로 붙이지 않는다. 의미 동치/관계·부정 능력과 추론/출력 의미 디노이징은 별도 후속 연구다.

새 실행의 fullstate는 import0/mid2048/final4096의 세 지점과 필요 시 emergency를 우선 검토한다. 모든 Adam/scaler/RNG/history/cursor는 각각 완전 보존하며 기존 checkpoint를 삭제·교체하지 않는다. 반복 연구의 디스크 소비를 줄이되 실제 고정 저장 주기는 새 plan 봉인 전에 확정한다.

현재 V14 importer/worker/plan/seal/GPU는 **아직 없다.** 준비를 실제 학습으로 보고하지 않는다. ROOTONLY/noagents/모든원래weights·Adam/scaler/RNG·raw·실패 보존/E2final-registry금지/소유 OS감독/terminal후모든종료확인/상시C20GiB보호/실행중sealedcode불변/자동재시작·oldbudget재사용금지는 유지한다. 타이머는30분/문구 정확히 `재개`다.
