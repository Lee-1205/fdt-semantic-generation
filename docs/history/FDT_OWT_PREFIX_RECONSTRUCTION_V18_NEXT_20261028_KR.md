# 다음 목적 — 의미 추출 상태를 이용한 prefix 손상 복원 학습

최신 실제 source는 V17 final `runs/owt_fuzzy_reconstruction_20261028_v17/model/checkpoints/state_2048.pt`, SHAd3896e120ff3ed4d684437b541e3d93599b9524a81a005a51af9fb4a1ad56d8e/algorithm2c36b4dc2b05c88b4e0ced59077a8e547ccb6087b9314315a116880da767b485다. 全63Adamstep21200/scaler131072tracker48/모든RNG/전체ancestry를 보존한다. V16FAILED/6864유용updates와 V17COMPLETE2048updates, 각각 닫힌 예산을 유지한다. 최신가중치와 V15의 더 좋은 teacherNLL은 모두 보존하며 자동 승격하지 않는다.

V15→V17은 자유chrF21.52→21.43/teacher정확도58.4177→58.4134%로 정체하고 NLL1.43189→1.66597은 악화했다. 틀린 위치의 평균 확률.4796→.6094,동시 오답8,208위치에서 NLL3.2299→4.0738 증가를 CPU로 확인했다. 자유 출력의 첫 오류는 평균2.75byte,충실한 의미 복원0/16이다. 상태를 바꾸면 다른 출력이 되므로 완전히 무시되는 것은 아니나, 충분히 정확히 읽어 문장화하지 못한다. 추출 완전성과 언어화의 단독 원인은 아직 분리 증명하지 못했다. 이것을 전체 fuzzy 실패·학습 충분으로 단정하지 않고 **새 복원 목표로 계속 학습**한다.

ROOT가 검토할 V18 단일 후보는 원래 fuzzy encoder/anchors/mu/union/rho/모델 parameter 수·추론 함수를 그대로 유지하면서 훈련 objective만 바꾼다. 원문을 추출해 얻은 같은 두 view에 대해 clean teacher CE와 **일부 앞 byte가 가려진 prefix의 복원 CE**를 함께 학습한다. Source corpus2,048train-doc8,192sentences/64dev256sentences는 그대로이며 새 다운로드0/heldtotrain0이다. 출력 의미를 만드는 추론 디노이징은 계속 동결한다. 여기의 prefix 손상 복원을 그 reasoning 학습이나 VP 의미개선이라고 부르지 않는다.

Prospective fixed candidate: cleanCE.5+corruptedPrefixCE.5,eligible previous-byte positions에만 mask확률.15. BOS와 기존EOS/PAD는 유지하고 새 parameter 대신 기존PADid를 훈련 전용 손상 표지로 사용한다. Clean 원문→fuzzy추출은 그대로다. 입력/손상률/손실 가중치는 코드·CPU검토 때 사전고정하며 성능보고 후 조정한 값을 최적이라고 하지 않는다. [BART](https://aclanthology.org/2020.acl-main.703/)의 손상→원문 복원과 [Scheduled Sampling](https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf)의 학습/추론 prefix 차이는 관련 선행연구다. 이 후보는 decoder prefix에 손상을 주므로 BART 그대로의 재현도,효과가 증명된 처방도 아니다.

정답은 기존 teacher training prefix·loss에만 쓰고 추론 feature에 gold내용/길이/역할/regex를 추가하지 않는다. Mask 난수는 고정크기 B×129 전체에 먼저 뽑고 prefix의 허용된 과거 byte위치에 적용해 미래 정답/길이가 mask나 난수 순서로 새지 않도록 한다. Causal test는 동일한 추출 상태·동일 난수에서 미래 target byte를 바꾸어 이전 logits가 그대로인지 확인한다. 실제 runtime은 손상 없이 BOS부터 자기 출력만 사용하고 모든 dev행을 예측 상태만으로 평가한다. Clean branch 함수가 source와 정확히 같음을 CPU확인한다.

한 batch는 extract1회와 clean/corrupt decoder2회로3calls,두loss의 평균을 한optimizerupdate로 적용한다. 필요한 CPUdelta는 gradient가 원래 encoder-tail/fuzzy모듈/decoder에 연결되는지, 미래 누설·BOS/PAD/허용 mask·clean동일성, mask난수 포함 전체 model/Adam/enabledscaler/allRNG/cursor exact-next 복구다. V17의3consecutive/3in16/total8 numericguard와 source경계검사·모든 기존STOP을 유지한다. 기존 광범위 검사를 반복하지 않고 실제workerpreflight/독립OSvalidate/newcode-dataSHA/newplan/rootreview 후에만 ONE실행한다. 하드웨어 재승인은 묻지 않는다.

Prospective trial은 새2유한epoch/2,048attempt/batch8/16,384rowselections/1,457,826targetUTF8bytes다. 두decoderloss pass이므로 decoder supervision exposure는2배이며 새독립문장·2배문서라고 세지 않는다. Train3×2048=6144+최종3조건128+16자유최대1552=7824expectedcalls를 실제코드에서 확인 후 새cap약8500/900secreserve90/output6GiB/start40GiB/GPU12GiB를 고정한다. Whole source moments/RNG/학습경로 보존,fullstatesimport0/mid1024/final2048+emergency,새clock/order/cap이며 모든 이전 미사용시간·call·attempt는 닫힌 채다. 같은 inference/dev이면 V17저장기준을 사용해 초기 GPUbaseline을 다시 돌리지 않는다.

관찰은 정답prefix점수뿐 아니라 자유 exact·chrF·내용어/개체/관계 직접읽기·오류시작위치·정상/교환/제거·오답확신을 함께 기록한다. 작은 관측dev를 통한 다중설계선택 한계를 밝히고, 성과가 없더라도 자동 연속SGD/승격하지 않는다. 원래 fuzzy 구조를 바꾸거나 reasoning을 섞지 않는다. 최신 source 불확실성을 포함해 증거와 다음단계를 firstCURRENT에 남긴다.

현재 V18 actualcandidate/runner/plan/seal/GPU는 **아직 없다**. CPU 준비를 실행·개선으로 보고하지 않는다. ROOTONLY/noagents/모든source·weightsAdamRNG·raw·실패 보존/E2final·registry금지/소유OS감독/모든terminalproxy종료확인/상시C20GiB/실행중sealed불변/자동재시작·olddeadline재사용금지를 유지한다. Timer30분 prompt는 정확히 재개다.
