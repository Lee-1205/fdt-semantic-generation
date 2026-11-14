# 의미 추출·세 메모리 읽기 V59–V60 실제 결과

2026-11-11 19:56 KST, ROOT 직접 수행. 최신 사람 지시인 **의미 기반으로 형성됐는지**를 우선했다. 의미 판단이 메모리 구성 전 특징을 우회하던 연결을 수정하고 새 제한 학습을 완료했다. 다만 **현재 결과로 안정된 의미 형성을 인정할 수는 없다.** 기존 긴 문장 복원 성과와 원래 typed 관계 예측 성과는 보존하며, 이번 초기·소규모 실험으로 FDT 전체 원리의 실패를 단정하지 않는다.

## 무엇을 실제로 바꿨는가

V59는 원래 세 메모리에서 신경망으로 주체·수신자·대상 span을 예측하고 역할별 퍼지 N/D를 구성했다. 그런데 관계·부정·수량 head는 그 결과 대신 메모리 구성 전 token pool을 읽었다. 저장된 통제 결과에서도 K_P/K_C/K_E 읽기를 끄면 의미 벡터는 변했지만 factor logits는 모두 정확히 그대로였다. 따라서 V59의 factor 성과를 새 의미 메모리 활용의 증거로 세면 안 된다.

새 V60은 입력에서 먼저 BoundMeaningState를 만들고, 별도 read_state가 **그 상태의 역할별 세 메모리 융합값만** 읽게 했다. 주체·수신자·대상 외에 술어·부정·수량 단서용 네 번째 역할도 신경 attention으로 추출한다. 원문 H, token pool, 정답 역할, 정답 span, 정답 factor는 이 읽기 함수에 들어가지 않는다. 상태 교환은 진단 표시와 출처를 붙인 경우만 허용한다.

원래 V53 인코더와 K_P prior/64개 앵커, K_C 입력 특징, K_E 특징·출처, V57 표면 복원기는 동결했다. V59의 413355개 학습 파라미터는 그대로 복사하고 전역 역할 질의용 0초기값 여섯 개만 추가했다. 활성 모듈은 413361개이며, 그래프 변경 때문에 새 AdamW/scaler를 사용했다. 이전 Adam·RNG는 부모 파일에 그대로 남겼다. **K_P의 인증된 writer/consolidation은 여전히 미구현**이며, 이번에는 기존 prior 읽기만 쓴다. 세 메모리 경로가 있다는 사실을 모든 장기 기억 기능의 완성으로 부르지 않는다.

MeaningCommit은 예측한 원문 span의 출처 참조만 가진다. 원문 전체 바이트나 CopyPermit은 없으며, 기존 표면 decoder가 이를 대신 받아 복원하지 못하도록 타입 경계를 유지했다. 언어 디코더 학습·호출, 의미 디노이징, 원문 전체 복사는 모두0이다. 이 모듈은 give/open/contain 세 종류의 한 사건 추출 실험이며 일반 OWT 의미 인코더나 완성된 언어-의미-언어 시스템은 아니다.

## 자료와 실제 학습

V59 TRAIN1152행/288그룹에서 시작했다. V60은 별도로 저술한 훈련 표현을 더해 TRAIN2880행/288그룹으로 확장했다. DEV256행/64그룹과 정답·그룹 파일은 V59와 바이트 단위로 동일하며, DEV의 새 이름·표현 가족은 훈련에 넣지 않았다. 훈련 literal/의미와 DEV의 겹침은0이다. TRAIN64행과 DEV4행의 중복은 명시했다. 기존 DEV 결과를 본 뒤 설계를 바꿨으므로 **적응적 개발 실험**이며 새 blind 평가가 아니다.

4개의 유한 epoch,576attempt/576update/skip0,11520행 선택,573568 cached 입력 UTF8바이트를 소비했다. 2880행은288개의 사실 그룹에서 나온 상관 표현이다. 기존에 준비한 OWT733개 짧은·중간·긴 window는 이번 의미 학습에 쓰지 않았으며, 자동으로 의미 정답을 붙이지도 않았다.

실제 V60 worker31.002초, source encode180+bind636=816호출이었다. 새900초/120reserve/950call/3GiB 출력 예산은 **CLOSED**다. 미사용134호출·시간을 재사용하지 않는다. V59도 별도576/576/23.913초/708호출로 닫혔다.

## 원문과 예측을 직접 읽고 다시 계산한 결과

| 같은 관측 DEV256 | V60 학습 전 | V60 학습 후 |
|---|---:|---:|
| 관계 정확도 |71/256 (27.73%)|197/256 (76.95%)|
| 존재하는 역할의 literal/span 정확도 |143/608 (23.52%)|186/608 (30.59%)|
| NULL 역할 정확도 |160/160|135/160|
| 모든 역할·관계·부정·수량이 맞는 사실 |4/256|6/256|

V59 마지막 전체 사실8/256보다 V60의6/256은 낮다. 메모리 경로 수정이 자동으로 모든 성능을 개선한 것은 아니다. V60 학습 전 값은 새 그래프로 바꾼 후의 기준이므로 V59 마지막 출력과 같다고 하지 않는다.

| V60 의미 종류 | 전체 사실 | 존재 역할 | 관계 | 필요한 조건부 factor head |
|---|---:|---:|---:|---:|
| give |0/96|88/288|80/96|부정·수량 해당 없음|
| open |6/64|92/128|31/64|부정 43/64|
| contain |0/96|6/192|86/96|수량 24/96|

최종 공개 점수 방식의 open polarity는32/64,contain quantity는19/96이다. 조건부 head 값과 관계 확률을 섞어 NULL/긍정까지 함께 argmax하는 판정이므로 위 조건부 head 결과와 다르다. 저장 logits만으로 관계를 먼저 선택하는 대안을 계산해도 전체 사실은6/256 그대로였고 contain quantity는20/96이었다. **판정 규칙만 바꾸면 문제가 해결된다는 근거는 없다.** 원래 점수를 소급 수정하지 않았다.

실제 예시:

- `A glass bottle went from Iris to Remy as a gift.`에서 관계 give와 수신자 Remy는 맞았지만 주체를 `Iris to Remy as a gift`,대상을 `glass bottle went from`으로 과하게 잡았다.
- `The glass door was not opened by Iris.`는 contain/긍정/수량7로 잘못 읽었다. 단어가 비슷하다는 이유로 의미 보존으로 인정하지 않는다.
- `The count of orange tokens in the steel basket is eleven.`에서 contain은 맞지만 수량7,주체 `count of orange tokens in the steel basket`로 틀렸다.
- `Regarding the garden gate, it is true that Émile opened it.`는 주체 Émile,대상 garden gate,open/긍정을 모두 맞혔다.

맞은6개 전체 사실은 모두 **긍정 open/garden gate**에 몰렸다. 다른 관계·부정·수량까지 안정됐다는 뜻이 아니다. 마지막 훈련 epoch의 pointer CE.01961/관계 CE.01084/부정 BCE.00396에 비해 held 결과가 낮아, 현재 저술 형식·이름 범위를 넘어서는 역할 경계와 결합의 일반화 공백이 크다. 수량 훈련 CE도.94018로 남아 있다. 구체적 원인은 추가로 분리해야 하며, 이 숫자 하나로 모델 용량이나 전체 퍼지 원리의 실패를 확정하지 않는다.

## 세 메모리와 의미 상태는 실제 읽혔는가

같은 앞64개 DEV행에서 읽기 분기를 하나씩 끈 저장 결과와 비교했다. Span routing logits는 모든 조건에서 변화0이며, 메모리를 읽은 이후의 factor logits는 다음처럼 달라졌다.

| 끈 읽기 분기 | 관계 logits 최대 변화 | 부정 | 수량 |
|---|---:|---:|---:|
| K_P prior |.125|.25|.125|
| K_C 문맥 퍼지 |6.78125|9.734375|7.125|
| K_E exact |4.21875|3.5|2.6875|

이는 V59의 우회 문제를 고쳐 **세 읽기 분기가 factor 계산에 실제로 영향을 주게 됐음**을 확인한다. 각 메모리의 고유 의미 정보가 증명됐다거나, 메모리가 올바른 사실을 보유한다는 증거로 확대하지 않는다. Routing 입력에도 신경 메모리 특징이 있기 때문에 전체 은행 제거 실험과도 다르다.

정답 factor가 붙기 전 역할별 융합값의 코사인에서 바꿔쓰기가 같은 그룹의 의미 변경보다 가까운 경우는 give12/24,open8/16,contain11/24였다. 의미 동치의 안정된 군집이라고 인정할 정도의 결과는 아니다. 코사인만으로 분산 의미 정보의 존재·부재를 완전히 판정하지도 않는다.

네 번째 event 역할만 다른 의미의 상태로 바꾼 진단16회에서 donor 부정에 맞은 경우5/8,donor 수량3/8이었다. 그런데 원래 source와 donor의 해당 factor를 둘 다 맞힌 eligible 짝은 두 축 모두0개였다. 따라서 이 결과는 **올바르게 형성된 의미를 인과적으로 옮겼다는 검증을 충족하지 못한다.** 단순 logit 변화와 정확한 의미 전달을 구별한다.

## 실제 실행·복구 증거

ONE hidden launch19:47:01 KST. owner proxy38324→native38196,worker proxy24600→native47096의 생성·parent·ExecutablePath·command·생존을 ROOT가 관측했다. worker/owner exit0,모든 owned/proxy 종료 후19:51:32 fresh Python0을 확인하고 분석했다.19:56 재확인도 Python0이다.

- Plan SHA:242aef2aee0f062518b109dd09f3f17a10e07b85627315f636909eff2b4fab2b
- Final:5269297B,SHA21dc84c862990fb75fafcdfcca6e473ebfb6c924105edf70b7fc833a340d0043
- Algorithm:f1d568e3ab1a22fdbb0daed40e972e9f8d4b3842c01afd615e9aef1c47d1cd02
- 원래 V53 부모 SHA7ad721ea…6e271b1f,V57 codec a7bbdd7a…46b5e2b1,V59 부모66ec2578…089f3d는 전체 파일 hash 불변을 확인했다.

최초 CPU 준비 중 import 경로 오류,그 후 미봉인 revision1의 worker가 V59 engine을 잘못 import한 문제를 ROOT가 잡았다. 오류 원문·미봉인 plan·해당 코드·CPU 결과를 보존했다. **GPU launch 전에 수정**하고 별도 execution_review_v2에서 실제 V60 schema와 memory-only 그래프를 다시 검사했다. 실제 fullsize CPU preflight7.250초/source96Adam 보존/모든 warm binder tensor 복사/새6zero/fresh optimizer/전체복원/CUDAfalse/부모 forward0/allCPUownedgone PASS 후 봉인했다. 잘못된 revision1 증거로 실행하지 않았다.

독립 CPU 사후 검토는5개 fullstate SHA/safeload/재귀 digest,빈 initial Adam→final576step,scaler1024tracker576/모든 저장 RNG/유한 order·cursor·노출,모든 cache와 raw SHA를 확인했다. 저장 degree·mu·projected value로 네 역할 N/D를 다시 계산했고,모든 순서 있는 start/end 조합의 최고 점수와 저장 span을 비교했다. Factor 확률과 전체 사실 점수·교환 진단16개의 확률 판정도 재계산했다. 추가 신경망 forward0/CUDAinitializedfalse다. 이는 실제 CUDA 중단 후 재시작을 수행했다는 증거는 아니다.

근거:
`runs/semantic_binding_20261111_v60/root_saved_review.json`,
`scripts/review_semantic_binding_v60_cpu.py`,
`raw_final.pt`,`event_interchange.json`,`final_status.json`,
`root_terminal_native.json`,`code_snapshot`.
V59 우회 증거는 해당 run의 `root_saved_review.json`에 별도 보존했다.

## 다음 해결 범위

다음 우선순위는 byte 복원을 더 잘하는 것만으로 의미 형성을 인정하는 일이 아니다. **새 이름·어순·문장 길이에서 개체 경계를 찾는 작업과 의미 역할을 배정하는 작업을 분리**한다. 현재 독립 start/end가 문장 조각을 길게 선택하는 실제 오류를 바탕으로,공유된 신경 span 경계와 역할 조건부 결합 후보를 검토한다. 후보 제한을 정답 이름 사전·GT span·regex 역할로 만들지 않는다. 의미 종류/부정/수량과 개체 identity는 typed state에서 일관되게 읽도록 유지한다.

훈련 자료는 더 많은 독립 사실·다양한 길이·역할 반전·부정·수량을 교차시켜 형식/이름 의존을 줄이는 별도 범위로 작성한다. 이미 본 DEV literal/정답은 훈련에 옮기지 않고,새 자료도 설계 선택 이후의 prospective 자료라는 한계를 밝힌다. 저장된 logits의 조건부 판정 개선은 제안일 뿐 V60 성능을 소급 변경하지 않는다. 준비한 OWT window는 의미 정답이 없는 복원 자료로 유지한다.

새 의미 상태가 언어화되면서 사실을 유지하는 연결은 아직 구현·학습되지 않았다. 먼저 이 역할 결합 공백을 고친 뒤에 의미→언어 연결을 평가한다. 원문 전체 자동 복사로 사실 정답을 우회하거나,부족한 초기 결과를 숨긴 채 의미 디노이징으로 넘어가지 않는다. 다음 GPU 예산은 아직 없으며,닫힌 V59/V60 잔여 시간·호출을 쓰지 않는다.

콘텐츠 기반 결합의 참고로 [Slot Attention 원 논문](https://arxiv.org/abs/2006.15055),동치/대조 목적의 참고로 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)을 검토했다. 해당 연구가 현재 FDT 모듈의 성공이나 특정 손실 가중치의 최적성을 보장하지 않는다. FDT의 원래 세 메모리를 대체하지 않았다.

현재 C:125142388736B≈116.55GiB,100GiB 보호선을 유지한다. 에이전트0,E2final/registry접근0,모든 원래 weights/AdamRNG/source/실패 보존. 현재 모델 프로세스는 없다.

