# 다음 한정 학습 — 넓힌 OWT에서 추출·복원 학습량 확대

Source는 V15 final `runs/owt_fuzzy_reconstruction_20261028_v15/model/checkpoints/state_4096.pt`, SHA e699f7bfed40095476914d2462eca5bb18d37399866c83c9fed5d13c68ac0c7c, algorithm33a85db62c5d1a9b6a8a42858d170f5bead8a15269956b1b2a186cc6550a382a다. 실제4,096추가/AE누적12,288업데이트 COMPLETE/allownedgone/9,624call 예산CLOSED이며 남은876call·시간은 재사용하지 않는다.

최근 문서 확대 후 teacher 정확도52.91→58.42%,NLL1.66688→1.43189,자유chrF20.79→21.52,자기원문식별8→9/16으로 개선됐다. 자유 faithful paraphrase/exact0/16과 첫오류 평균3.125byte는 여전히 부족하다. 원래 입력 의미 추출→재복원 FIRST/원래 fuzzy anchors/추론 별도 동결을 유지하며 학습량을 더 주는 것이 다음 구체적 목적이다. 고유 문장0.729MB/누적AE target노출8.735MB로 전체 모델 실패나 충분한 언어학습이라고 단정하지 않는다.

ROOT는 새 V16 파일/실제 V15schema importer에서63Adamstep12288/scaler65536tracker288/모든RNG/전체 recursive ancestry를 정확히 보존한다. Sourceglobal=V15local4096+V14local4096+V13local2048+V12local2048=12288. 원래 모델/추론 함수/2,048train문서8,192문장/64dev256문장은 그대로, 새 다운로드0을 우선한다. 초기 GPUbaseline은 같은 함수·패널이므로 V15저장 결과를 사용한다. 관측dev 평가와 자유생성16문서의 다중사용 한계를 유지한다.

Prospective 후보는8유한 epoch/최대8,192새attempt/batch8/65,536행/5,831,304targetUTF8bytes다. 기존V13군1,024doc와 V15새1,024doc 각각32,768행/2,913,424·2,917,880bytes다. 같은 language1e-5/other1e-4 populatedAdam을 유지하고 새 유한 LR horizon을 명시한다. 같은 실제AE코드는 attempt당 encode1+decoder1=2calls이므로 training16,384+최종3조건128+최종자유최대1,552=18,064calls다. 과거257.77초/4,096attempt 실측상 새학습 약2배+최종평가가900초 내에 들어갈 것으로 예상한다. 코드·실제 비용을 확인한 뒤 새 cap약18,800/owner900sec/reserve90/output6GiB/start40GiB/GPU12GiB를 고정한다. 기존 한도를 막연히 늘리거나 olddeadline을 여는 것이 아니다.

Complete fullstates는 import0/mid4096/final8192+필요시 emergency의 새 directory에 보존하며 모든 기존 파일은 삭제·교체·rotation하지 않는다. 새 source-schema/cursor/유한order/새-누적budget·마지막batch/allRNG 복구·ownedOS cleanup에 필요한 CPUdelta와 실제 worker preflight/독립validate만 확인한 뒤 ROOT가 ONE실행한다. 하드웨어 재승인으로 멈추지 않고 검사·baseline을 무한 선행조건으로 반복하지 않는다. 현재 V16runner/importer/plan/seal/GPU는 아직 없다.

자유 출력에는 기존 exact/UTF8/EOS와 함께 같은 chrF/내용어·관계 직접읽기/첫오류/정상-교환-제거 차이를 기록한다. Teacher58%를 자유 의미 성공으로 바꾸어 말하지 않는다. 적은 정상4word반복0이 숫자·형태소 반복을 놓친다는 사실도 유지한다. 의미와 언어화가 둘 다 불완전할 수 있어 decoder만의 단독 원인은 확정하지 않는다. 후속 prefix robustness/복원 bottleneck 개선은 별도 목적의 통제로 검토할 수 있으나 이번 준비에 큰 penalty/anchorreset/새회로/이른 reasoning학습을 자동 혼합하지 않는다.

ROOTONLY/noagents, 모든source/weights/AdamRNG/raw/실패 보존, E2final/registry금지, 상시C20GiB보호, 새로운 고정한도/실제owned감독/native중복0/모든terminal-proxy종료확인/실행중sealedcode불변/자동재시작·unusedbudget재사용금지를 유지한다. 타이머30분 prompt는 정확히 재개다.
