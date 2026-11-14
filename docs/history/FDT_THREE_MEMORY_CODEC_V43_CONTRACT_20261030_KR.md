# V43 마지막 학습률 유지: 원문 복원 안정화 계약

V42는 개발 기호 정확도80.2258%로 학습을 이어갔지만,같은 TRAIN참고64 원문 일치는39→31로 퇴행했다. 새 실행은 기존cosine을 재시작할 때의 약10배 LR 상승을 없애고 마지막 group LR 그대로 두 순회를 지속하는 실험이다. 학습을 중단하거나 전체 모델의 실패로 단정하지 않는다. 원문→입력 의미→동일 원문 복원이며 의미 디노이징은 하지 않는다.

V42 FINAL791069621B SHA `41f0fc13626adb5ebdef9fe2e02842f64a0b78de715172e0d51c2079f04c0ad3`,algorithm `129f2b79eae7af34251d2892c69c60951f845af4eadc48c5efee13d8247d2734`,sourceplan `6211513e03f367ca0be8193525f6c52155382950241f3245b72f3cc2f7c06b4b`를 사용한다. ALL91model·populatedAdam·moments/steps·baseLR/scaler/모든 저장RNG를 정확히 이관한다. 초기 reference도 이 전체 상태와 같으며,새 일정은 별도 continuation에 보존한다.

네 group의 실제 부모 최종 LR1.0000003309042696e-6/1.0000003309042697e-5/3.000000992712809e-7/1.0000003309042697e-5를 전체 새8192attempt 동안 상수로 유지한다. 매 update/AMPskip의 history에 실제4LR을 저장한다. 기존 baseLR을 바꾸거나 moment·step을 초기화하지 않는다. 초기 이관과 복구 시 list길이/양수·유한값/optimizer LR 일치를 확인한다. CPU 정확 next-update 복구 검사에 이 일정/AMPskip/마지막7행도 포함한다. 이전 sealedV33/V42 engine을 수정하지 않고 새V43 driver에 필요한 step만 명시했다.

모델111732306/학습38556357parameter,세 메모리K_P/K_C/K_E와Commit-only 디코더,p50cleanCE.5+corrupted-prefixCE.5,훈련 소유권과 원래 동결 범위는 같다. 원문H/byte 우회·통째 복사·literal transport·정답 길이·외부 역할/regex·denoising0이다. K_P certified Writer/K_C 전체층prefix/K_E literal selector·transport 미구현 범위를 유지한다. 미래 explicit Commit 전문 인용 계약을 삭제하지 않는다.

같은8192 TRAIN문서32768구간의13·14번째 유한순회 seed42001/42002,batch8/8192attempt/65536행선택/5852956targetUTF8bytes다. skip도 시도·행·노출을 소비한다. 새 데이터·held-to-train·E2final/registry 접근0이다. 초기·중간 GPU평가를 추가하지 않고 V42 저장 평가를 기준선으로 재사용한다. 끝에 같은 teacher256/TRAIN참고64/held16 복원과세READ 통제를 저장하고 ROOT가 원문/복원문80쌍을 읽는다. Teacher/BOS 첫 오류 위치가 같은지는 관측치이며 통과조건이 아니다. 유사도를 의미 보존율로 대체하지 않는다.

새2100초/modelcutoff1920/reserve180/max30000호출/계산27348호출/output1GiB/max992MiB/startC30GiB/GPU12GiB를 유지한다. 1639학습+120startup+60저장+60평가+180종료=2059초 계획이다. 이전 예산·미사용 시간/호출은 모두 CLOSED다. 모델/학습행/호출 비용을 늘리지 않는 LR 정책 변경이며 자동 추가실험·상한 확대는 없다.

무손실 gzip initial format42를 재사용하고 물리32MiB/rawCPU64MiB,부모필수/CRC/크기/추가member/잘림/덮어쓰기 거부를 유지한다. Initial `.pt`는 새gziploader와 V42부모가 필요하다. 업데이트 후에는 ordinary self-contained FINAL_OR_EMERGENCY 한 벌뿐이며 midpoint/삭제/rotation/두 번째 fullwrite가 없다. 최대8192미저장 시도 위험은 기존과 같다. 종료/정체/수치/반복AMPskip/소유권/디스크·시계 STOP과 독립 ownedOS 감독을 유지한다.

새 worker focused13CPU 검사를 실행하고,내용이 바뀌지 않은 ownedCPUpreflight 모듈의 기존6PASS는 SHA/동결 manifest로 재사용한다. 같은6검사를 새 실험으로 다시 세지 않는다. 실제 전체 부모 import/reference roundtrip은 별도 새 worker 독립 OS `--validate`로 측정한다. 실제 새 Windows owner fixture/새 code/data SHA·preflight·독립validate/resources를 확인한 뒤 새시계를 한 번 봉인하고 hiddenlaunch한다. 준비/테스트/봉인은 GPU실행·능력 개선의 증거가 아니다.

V42의80/80 첫오류 위치 일치와49teacher오류/33행이라는 계산은 뒤의 반복만으로 모든 손실을 설명할 수 없음을 보여준다. LR 재상승이 퇴행의 단독 원인이라는 증거는 없다. 새 학습도 추가 노출·순서가 바뀌므로 결과 개선을 LR의 독립 인과 효과로 주장하지 않는다. 강한 반복벌점/rollout loss/메모리 구조 변경을 동시에 넣지 않고 이번 안정화 결과를 보존한다. 필요하면 이후 별도 구체적 목적을 판단한다.

ROOT만 수행하며 에이전트가 없다. 실제 training 중 상세 모델 분석은 중단하고 OS 감독만 한다. terminal 뒤 모든 owned/proxy 종료 확인부터 하고 저장값을 검토한다. 모든 core/typed/부모·source/raw/실패·weightsAdamRNG/사용자 원본을 보존한다. 사용량이 적어지면 복구 위치/실제 상태/다음 작업을 저장하되 계정 잔여량을 조회한 것으로 표현하지 않는다.
