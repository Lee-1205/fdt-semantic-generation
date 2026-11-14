# V62 내부 메모리 정렬·개체 결합 학습 결과

2026-11-12 08:39 KST. 사람의 새 ‘재개’ 지시에 따라 ROOT가 연구를 다시 진행했다. 이전 중단 기록과 V61 원시 오류를 먼저 읽었다. 타이머는 별도 재활성화 요청이 없어 PAUSED 그대로다. 에이전트는 쓰지 않았다.

**실제 제한 학습과 독립 검토를 완료했다.** 같은 관측 DEV256에서 전체 사실은29→42개,존재 개체 역할은293→334/608로 개선됐다. 18개 오류 위치를 교정했지만5개는 퇴행했다. 의미 동치와 역할 결합이 안정된 단계는 아니며,일반 문장 의미 형성·언어 왕복 완성으로 부르지 않는다. 초기·제한된 학습 결과로 FDT 전체 설계의 실패를 선언하지도 않는다.

## 수정한 문제

V61의 대조 손실은 실제 역할별 메모리값뿐 아니라 이미 예측한 관계·부정·수량 확률도 연결한 벡터를 사용했다. 이것은 계산 오류는 아니지만,내부 메모리의 의미 정렬을 직접 학습하려는 목적과 구분할 필요가 있다. V62에서는 출력 확률을 빼고,각 역할의 융합 메모리값만 정규화해 동치/대조 손실을 계산했다. CPU 검사에서 해당 손실은 factor logits에 경로가 없고 실제 메모리값으로 미분됨을 확인했다. 정답 의미 ID는 손실에서만 사용한다.

개체 경계 학습을 강화해 최종 손실을 2×joint span NLL +관계 CE +open 부정 BCE +2×contain 수량 CE +.5×공유/역할 내부 BCE +.2×memory-only SupCon으로 정했다. 온도는.2다. [Supervised Contrastive Learning 원 논문](https://arxiv.org/abs/2004.11362)은 같은 표적의 표현을 가까이 학습하는 참고이며,이 가중치나 FDT에서의 최적성을 보장하지 않는다.

추론 구조는 V61과 같다. source 신경 메모리→결합 구간/역할 추출→역할별 K_P/K_C/K_E 융합→BoundMeaningState→factor 읽기 순서를 유지했다. factor 읽기는 원문 H·token pool·정답을 받지 않는다. 원래 source/앵커/K_P prior/K_E/표면 codec은 보존·동결했다. K_P의 인증된 writer/consolidation은 여전히 없으며 기존 prior 읽기만 한다.

활성504885파라미터를 V61에서 모두 정확히 복사했다. 새 목적 때문에 fresh AdamW1e-4/WD.01/clip1과 GradScaler1024를 썼고,V61의 이전 Adam/RNG는 부모 fullstate에 그대로 남겼다. 원래 학습 clock이나 업데이트 잔여분을 이어 쓴 것이 아니다. 기존 MeaningCommit과 표면 decoder의 타입 경계도 유지한다. 의미 디노이징·언어 decoder 학습·원문 전체 복사는 모두0이다.

## 자료·비용·실제 실행

새 문서나 정답 행은 추가하지 않았다. V61의 TRAIN4096 view/1024그룹/2008고유 저술 사실,DEV256행,모든 고정 source cache를 그대로 참조했다. TRAIN 중복 prompt3개와 기존 DEV 관측·표현 가족 분리의 한계도 그대로다. 새8개 유한 order는 각 epoch4096행을 한 번씩,각4행 의미 그룹을 같은16행 batch 안에 보존한다. 실제32768행 선택/2396648 cached 입력 UTF8바이트다. 새4096문서나 BPE 토큰으로 세지 않는다. OWT 의미 학습은0이다.

원문 encoder가 동결돼 있으므로 캐시 재사용은 이 실험의 활성 gradient를 끊지 않는다. 실제 source encode0,학습 binder2048+최종16+세 읽기 통제12+event 교환16=2092호출이었다. 시작 기준은 동일 구조·동일 tensor·동일 cache의 **V61 마지막 저장 raw/metrics 재사용**이며,새 독립 baseline 평가나 두 번째 성공으로 세지 않는다. 새 baseline forward는0이다.

실제1024가 아닌 **2048attempt/2048update/skip0**,worker94.594초로 완료했다. 새900초/120reserve/최대2200call/출력1GiB 계약은 CLOSED다. 미사용108call·시간은 재사용하지 않는다. 모든 이전 V61/V60 예산도 닫힌 상태로 보존했다.

실제 worker의 CPU preflight10.531초에서 source96Adam·모든 부모 tensor·모든 V61 binder tensor·fresh optimizer·V62 schema/목적 서명·초기 전체 상태 복원·CUDAfalse·부모 forward0을 확인했다. CPU populated optimizer/scaler/모든 RNG의 다음 업데이트 복구,강제 overflow와 마지막7행 소비,타입/출처/정답 feature 금지도 별도 실제 새 모듈 fixture로 통과했다. CPU 실행과 모든 owned 종료를 확인하고 ROOT가 SHA·프로세스 중복0·C/GPU 자원을 독립 검토한 뒤 ONE hidden launch했다.

## 독립 재채점 결과

| 같은 관측 DEV256 | V61 저장 기준 | V62 실제 최종 |
|---|---:|---:|
| 모든 역할·관계·부정·수량 일치 |29/256 (11.33%)|42/256 (16.41%)|
| 존재 역할 literal |293/608 (48.19%)|334/608 (54.93%)|
| NULL 역할 |139/160|137/160|
| 관계 |202/256 (78.91%)|210/256 (82.03%)|
| open 실제 부정 판정 |33/64|39/64|
| contain 실제 수량 판정 |33/96|60/96|

관계별 전체 사실은 give7/96→7/96,open16/64→22/64,contain6/96→13/96이다. give 관계는78/96→70/96으로 내려갔고,open 존재 역할도112/128→106/128로 내려갔다. 총점 상승으로 이 퇴행을 숨기지 않는다.

조건부 open 부정 head는43/64→44/64,contain 수량 head는40/96→62/96이다. 위의 실제 판정은 관계 확률과 섞은 기존 정책이므로 다르다. 저장 logits만으로 관계를 먼저 고르는 대안은 전체43/256이었지만 모델 실행·성능 승격이나 원래42/256의 소급 수정으로 세지 않는다.

원시 값으로 다시 비교한 결과,기존 정답29개 중24개 유지/5개 퇴행,기존 오답 중18개 교정이다. 행은 같은 사실의 표현·대조로 상관돼 있어18개의 독립 문제를 풀었다고 하지 않는다.

- `As a gift, Søren handed a red badge to Theo.`는 수신자를 Søren으로 잘못 잡던 것이 Theo로 교정됐다.
- `The recipient of a red badge from Nia was Iris.`는 수신자 NULL이 Iris로 교정됐다.
- `The garden gate was not opened by Nia.`는 기존 긍정 오류가 부정으로 교정됐다.
- 반대로 `As a gift, Émile handed a red badge to Nia.`는 이전의 올바른 주체 Émile이 Nia로 바뀌어 주체·수신자를 같은 구간으로 고르는 오류가 생겼다.

마지막 TRAIN epoch의 joint pointer NLL.03709/관계 CE.00900/부정 BCE.00301/수량 CE.13679/내부 BCE.50558을 확인했다. V61 마지막 pointer.19592/수량.48381보다 낮지만,학습량·학습률·손실 가중치를 함께 바꿨으므로 어느 하나의 독립 효과로 돌리지 않는다.

## 의미 메모리에 대해 어디까지 확인됐는가

세 메모리 읽기 분기를 끄면 factor logits가 각각 변하고,구간 routing logits는 변화0인 경계를 유지했다. 이는 읽기 경로가 살아 있다는 확인이며 각 은행에 올바른 의미가 형성됐다는 증명은 아니다.

정답 확률을 넣기 전 역할 상태의 단순 코사인에서 바꿔쓰기가 같은 그룹의 의미 변경보다 가까운 경우는 give15/24,open9/16,contain13/24였다. 이번 손실과 같은 역할별 단위 정규화를 적용하면11/24,6/16,10/24로,안정된 의미 동치 군집을 인정할 수 없다. 단순 코사인이 모든 분산·비선형 의미 정보를 포착한다고 가정하지도 않는다.

event 역할만 바꾸는 저장 진단에서 원래 source/donor의 해당 부정을 모두 맞힌5짝 중5짝은 donor 부정을 따랐다. 이 중 **두 문장의 전체 사실까지 모두 맞힌 짝은4개이며4개 모두 donor 부정을 따랐다.** 수량은 해당 factor 양쪽 정답1짝 중1짝 donor를 따랐지만,양쪽 전체 사실이 맞은 짝은0개였다. 이는 좁은 상태 읽기의 긍정 관측이며,적은 상관 표본·관측 패널·단일 학습으로 보편적인 의미 인과 전달을 확정하지 않는다.

## 추가 코드·자료 점검에서 확인한 다음 문제

기본 구문 자체의 훈련 공백을 CPU로 세었다. 실제 DEV 행을 새 훈련에 넣지 않았다.

| 표현 | TRAIN4096 | DEV256 |
|---|---:|---:|
| 질문 부호 / Yes·No 응답 |0 /0|21 /21|
| was (not) opened by |0|22|
| false 범위 표현 |0|10|
| count 명사구 |0|32|
| number 명사구 |0|32|
| held by |0|32|

즉 이 DEV는 새 이름뿐 아니라 이러한 기본 표현의 미학습 일반화까지 요구했다. 낮은 점수를 전체 의미 모델의 실패라고 단정하기 전에,독립 저술 훈련에서 기본 표현 범위를 넓혀야 한다. 이때 기존 DEV literal/정답 행을 가져오지 않고,**구문 가족 수준의 훈련 겹침을 새로 만들면 기존 DEV가 더는 그 가족의 held 검사가 아니라는 점을 명시**해야 한다. 기존 보고 점수와 파일은 그대로 보존한다.

또한256행 중87행에서 서로 다른 역할의 최종 참조 구간이 겹쳤다. 저장 NN logits의 역할별 상위8후보+NULL을 비겹침으로 조합하는 CPU 가설 계산에서는 사실 참조 점수가50/256이었으나,퍼지 N/D·역할 메모리를 다시 만들지 않았으므로 **유효한 새 MeaningCommit도 새 모델 성능도 아니다. 실제 성능은42/256**이다. 단순 사후 reference 보정만으로 충분하지 않다. 다음 수정은 역할 간 결합과 메모리의 일관성을 함께 다뤄야 하며,일반 언어의 모든 역할은 반드시 비겹친다는 규칙을 임의로 강제하지 않는다.

다음 실제 연구 범위는 이 구문 학습 공백과 역할 결합을 다루는 별도 계약이다. 현재 그 새 데이터/runner/plan/GPU는 아직 없다. 닫힌 예산의 남은 호출로 자동 재실행하지 않으며,의미 디노이징으로 넘어가지 않는다.

## 보존·종료

- PlanSHA30af6fdf0984c84f8da14f059ecd9a357844110365208374b2039f7ad0d89a33
- Final6681297B SHA1c045fe5ead409189c21c0cab0632076adb0d3d9d5f4edd08fc432a66065a25a
- Algorithm0dae39e1d6c24762fa0309af9d48f4256aa1739695e97797e8854fa700bf5171
- Owner29400→32228/worker26592→1836의 생성·parent·실행 경로·명령·생존을 ROOT가 관측했다. worker-ownerexit0/COMPLETE/allownedgone 후08:31:38 fresh Python0을 확인하고 분석했다.08:38:51 사후 CPU 검토 뒤에도 Python0이다.

독립 CPU 사후 검토로5개 fullstate SHA/safeload/재귀 digest,빈 initial Adam→2048step,모든 저장 RNG/order/cursor/노출/목적 서명·baseline/cache/raw refs와 source/codec/V61 부모 SHA 불변을 확인했다. 실제 역할 N/D와 모든 구간 최고 점수·결합 분할합·점유 확률·factor 확률·사실 점수·교환 판정을 재계산했다. 추가 NN forward0/CUDAinitializedfalse다. 실제 CUDA 중단·재실행 복구 실험을 한 것은 아니다.

첫 사후 검사기는 이전1024회 실험의 scaler1024를 그대로 가정해 실패했다. 실제 GradScaler는 정상적으로2000회 무skip 업데이트 후 scale1024→2048,tracker48이 됐다. 저장 초기 growth interval로 모든 history의 scale을 계산하도록 **사후 검사기만** 고쳤고 통과했다. worker 계산 실패나 재학습은 아니며 봉인된 코드는 수정하지 않았다. 이 기록은 root_review_incident_01.json에 보존했다.

근거: runs/semantic_binding_20261112_v62/{result.json,root_saved_review.json,root_dataset_coverage_review.json,root_role_assignment_hypothesis.json,root_terminal_native.json,ROOT_review_terminal_native.json,code_snapshot},scripts/review_semantic_binding_v62_cpu.py. Root 검토 SHA f926774680de76d999f5fcfff079f9b7f2ee42008ecae45f3fa3f470670230dd.

현재 C:140444651520B≈130.80GiB로100GiB 보호선을 유지했다. 원래 source/core/archive/raw/체크포인트/AdamRNG/실패는 보존했고,E2final·registry 접근/에이전트/브라우저·마우스/외부 게시/새 corpus 다운로드는0이다. 재개 타이머는 PAUSED 그대로이며,현재 실행 중인 모델 프로세스는 없다.
