# 다음 한정 목적 — 같은 추출·재복원 목표에서 고유 train 문서 다양성 확대

최신 사람 지시의 목표는 입력 X→원래 퍼지 앵커로 입력 의미 Z(X) 추출→동일 입력 X̂ 복원이다. Reasoning/출력 목적 의미 디노이징은 계속 별도·동결한다. 원문 context·정답 prefix/길이/역할을 추론에 주지 않는다. BOS/자체 prefix를 사용한 상태 조건 복원을 자유창작 benchmark로 부르지 않는다. 원래 μ-union-ρ/value/prior/condition과 현재 구조·손실·정밀도를 유지한다.

실제 최신 source는 V22 `state_8192.pt`, SHA `aa95bc9b1871cb6a3af3cb59876f39df5d29ec5aab8fee475a98ae81a8e67320` / algorithm `fd9b3bf4aa8583966f539f17472905f203f7a9571a6277227ece3fe48356c812` / plan `2dcad61de3d51bdb6c4a06a3b8997d731c98c19c5c09d79e70d103f67e1a09f7`다. 전체 model/63Adamstep39624/scaler131072tracker1339/모든RNG/전체 ancestry/history/orders/exposure를 보존한다. Source local8192attempt8188update/cursor65536, global39636attempt39624update, 과거11AMPskip+1V18failedattempt도 유지한다. V22 미사용1464calls·시간은 CLOSED이며 재사용하지 않는다.

V22에서는 train 온라인 loss와 고정16train 복원의 내용어·도입부가 개선됐고 한 주어-행위 부분 명제가 복원됐다. 반면 관측 dev 보조 정확도63.08→62.82%/NLL1.26921→1.33996은 퇴행했다. 이것만으로 과적합이나 추출기·decoder 중 단독 원인을 단정하지 않는다. 작은 고유 corpus에서 동일 학습량을 더 반복하는 대신 **고유 train 문서4096→8192** 확대를 다음 구체적 변수로 검토한다. 더 나았던 V21 보조 결과와 최신 V22 전체 상태를 모두 보존한다.

CPU 자료 계약부터 진행한다. Same OWTrevision79d93d786212f7344586290adb811d4ae6a1762c/docSHAmod9split/48–128UTF8byte·4선택구간/doc 조건을 유지한다. 기존4096train-doc16384행은 정확한 prefix, 기존64dev-doc256입력/정답/순서는 정확히 보존한다. 새4096train-doc16384행만 붙여 train8192doc32768행 후보를 만든다. 저장 source_pages의 미사용 train 문서를 먼저 사용하고 필요한 추가 fetch는 최대48MiB/180초로 미리 한정한다. 모든 revision·원문/정규화 SHA·문서/선택구간 exactcrosssplit 중복을 확인하고 새 immutable artifact에 보존한다. 새 고유 UTF8bytes와 반복 노출을 분리해 실제 계산한다. source-rights/near-dup/pretrainingUNKNOWN 한계를 유지한다. 기존 held/test/E2final/registry를 열거나 train에 넣지 않는다.

V21–V22에서 약어/원문 단편의 경계 문제가 관측됐다. 새 train 후보에만 보수적인 약어 경계 처리·명백한 단편 배제 규칙을 사전에 정의하고 CPU 예제로 확인한다. 규칙이 완전한 문장/의미의 정답을 보장한다고 하지 않는다. 예전 train/dev의 실제 행을 수정하거나 새 prompt/gold를 train에 옮기지 않는다. 데이터 다양성과 새 행의 품질 처리가 함께 바뀌는 한계를 기록하고 둘의 효과를 분리한 인과 결과로 부르지 않는다. 부족한 후보를 맞추려고 기존 held를 가져오지 않는다.

자료 완성 후 prospective2finiteepoch/batch8/새8192attempt/65536rowselections 후보를 검토한다. Target UTF8bytes는 실제 새 artifact에서 산출하고 old/new4096문서 각각32768행을 노출한다. 동일 cleanCE.5+corruptprefixCE.5/.15mask/strictdeterministic elementwise reduction/현재 AMPguard와 decoder inference를 유지한다. 새 V22schema importer는全63Adam/scaler/RNG/ancestry를 정확히 가져오고 새 local clock/order/callallocation 및 finite cosineLR 구간만 구분한다.

실제3calls/attempt이므로 training24576+finaldev128+devstatecontrolsmax1552+fixedtrainnormalmax520=26776expectedcalls다. V22 실제618.23초/8192attempt를 기준으로 새 max28000/900초/reserve90/output6GiB/start40GiB/GPU12GiB 후보를 실제 코드·자료 비용 뒤 확정한다. Final reserve에 닿으면 현재 상태를 보존하고 닫으며 deadline을 늘리지 않는다. Fullstates import0/mid4096/final8192+STOP을 새 output에 보존한다. 동일 dev/function baseline은 V22 저장 결과를 사용해 초기 GPU 평가를 반복하지 않는다. 기존 고정 train16은 모두 보존되는 old-prefix 내 행이므로 같은 input identity를 유지해 dev와 분리 기록한다.

저장 여유는 삭제 없이 전후 SHA 동일한 파일시스템 압축으로 필수40GiB 기준을 복구했다.22:16:40KST C43,328,319,488B였으나 CPU 자료 준비 후 다시 실측한다. 필요하면 workspace 내부 읽기 위주 런타임/닫힌 생성 파일의 **별도 구체적·한정·전후 SHA 확인 무손실 저장 작업**으로 여유를 확보하고 필수 기준을 낮추지 않는다. 사용자 원본·원문·기존 결과·모델 상태를 삭제하지 않는다. 압축 작업이 GPU 학습이나 새로운 성능 결과는 아니다.

새 data/helper/actualV22schema/finite2epoch노출·정확next복구/ownedOS모든postPopen정리/actualworkerpreflight/독립validate/code-dataSHA/caps/resources/native중복0을 ROOT가 확인한 뒤 ONE 한정 학습을 한다. 이전 broad audit·gradient probe·같은 평가 반복을 무한 gate로 만들지 않고 하드웨어 재승인을 묻지 않는다. **현재 V23 data/code/runner/owner/plan/seal/GPU는 아직 없다.** CPU 자료 준비를 실행으로 보고하지 않는다.

ROOT ONLY/noagents, 모든 source/core/typedpartial9/a877/Phase2/archive/사용자원본/weightsAdamRNG/raw/실패 보존, E2final/registry/browser/mouse/keepalive/유료cloud/외부게시/사용자원본삭제/Windowsos.kill(pid,0) 금지, 자동재시작·olddeadline초기화연장·unusedbudget재사용·실행중sealedcode수정삭제 금지를 유지한다. Timer30분 prompt는 정확히 재개다.
