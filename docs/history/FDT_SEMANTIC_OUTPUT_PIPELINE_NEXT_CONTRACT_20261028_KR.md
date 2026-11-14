# 주 설계 계약: 언어→입력 의미→출력 의미 디노이징→언어화

사람의 최신 명시적 정의를 우선한다.

> 중요한건 언어 - 의미 추출 - 의미 개선 - 언어화 이 구조야
>
> 의미개선은 인풋의 원래 의미로부터 아웃풋이 될 특정 의미로 디노이징

## 목표와 실행 경계

\[
S_{in}=E_\phi(X),\qquad Z_T\sim N(0,I),
\qquad S_{out}^{*}=D_\theta(Z_T\mid S_{in},I,E),
\qquad Y\sim P_\psi(Y\mid S_{out}^{*},E_{lex}).
\]

`I`는 입력에서 예측한 질문/의도, `E`는 허용된 내부 기억이다. `E_lex`는 출력 의미가 지시한 이름·정확 문자열 등 표현에 필요한 기억이다. 질문에 답하기, 사실에서 결론 내기, 여러 내용 중 말할 부분을 고르기 등 **출력할 내용의 결정이 디노이징의 목적**이다. 그 과정이 끝난 후 언어화는 표현·어휘·문장 구조를 선택하며, 사실·관계·부정·불확실성을 다시 결정하는 우회 경로를 갖지 않도록 한다.

입력과 출력의 의미는 서로 다른 감독 대상이다. 입력 사실을 복원하는 extraction 점수가 높다는 사실만으로 질문의 답이 될 의미를 형성한 것으로 세지 않는다. 또한 같은 입력에서 여러 적절한 출력 의미가 가능하다. 특정 표본의 `S_out*`가 정해지면 그 표본의 언어 출력이 그 내용을 충실히 표현하는지를 검사한다. 독립적으로 뽑은 다른 의미 표본에 임의의 정답 문장을 강제하지 않는다.

## 기존 구현에서 재사용할 부분

기존 typed `FDT100.target(features,labels)`는 query의 주체/목적어16성분+출력 relation4성분+상수1성분을21개 고정 basis에 올린다. 출력 관계는 `UNRESOLVED/GREATER/LESS/CONFLICT`다. **이 표적은 이미 입력 사실 전체가 아니라 질문에 대한 출력 의미**라는 점에서 핵심 목표와 맞는다. 기존 좋은 관계 예측을 버리지 않고 연결 자산으로 보존한다.

다만 기존 encode/denoise는 구조화된 fact/query roles와 fuzzy closure 등의 도움을 받는다. 다음 raw 언어 모델에서는 이 역할과 질문 대상을 원문에서 예측해야 한다. Gold 역할/정답 relation/regex 해석을 inference feature로 넣어 높은 관계 점수를 보장하는 방식은 사용하지 않는다. 고정 fuzzy 도움을 유지한 후보라면 그 도움을 명시하며 독립 neural 추론으로 과장하지 않는다. 원래 core/typed weights/fullAdamRNG는 수정하지 않고 adapter 연구를 별도 파일·출력에서 한다.

V3/V4의 learned reader와 semantic factors는 raw 질문 대상을 예측한 이력이다. V4의 기존 형식 ID100%와 wording 저하는 추출 모듈의 표현 coverage 문제를 드러냈다. V5 surface CPU artifact의 train2048과 별도 관측 통제 자료를 재사용할 수 있다. 과거 dev/held literal이나 그 graph·facts를 train으로 옮기지 않는다. A-H 후보 제한과 합성 범위는 명시한다.

V9의100% codec 복원/반대 순서8/8은 **byte 언어화 연결부** 자산이다. V9가 출력 의미를 형성했다는 뜻은 아니다. 의미 memory는 출력 byte index와 일대일이 아니므로, V9의 query-byte index addressing을 추상 의미 slot에 그대로 강제하면 안 된다. 의미 field/역할/대상 binding에 맞는 memory adapter와 learned decoder conditioning이 별도로 필요하다. 고정 answer template로 문장을 조립하지 않는다.

## 다음 구현의 실제 계약

1. **의미 추출:** raw prompt만 입력으로 받는다. 입력의 대상 참조, fact/predicate/polarity, 질의 대상과 의도, 근거 위치·불확실성을 예측한다. `S_in` loss의 정답은 offline 훈련 target에만 있다. 모델이 예측한 `S_in`이 다음 단계로 넘어간다.
2. **출력 의미 디노이징:** 의미 target은 질문에 대응하는 `S_out`다. 조건은 예측 `S_in`, 예측 의도, 허용 기억이다. 추출한 문장을 byte별로 복사하는 target이나 입력 의미 그대로를 복원하는 target으로 대체하지 않는다. 기존 비교 도메인에서는 query-role/답 relation의 semantic basis로 시작할 수 있으나 범용 의미 전체로 부르지 않는다. 학습용 noisy gold는 diffusion 감독에만 쓰고 실제 생성은 독립 noise에서 시작한다.
3. **의미 확정:** 선택된 출력 대상/명제·관계/부정/불확실성과 참조를 언어화 전에 고정한다. 단순히 norm이나 loss가 안정적이라는 사실을 의미 정답의 증거로 삼지 않는다. 입력 사실과 출력 결론을 별도 저장한다.
4. **언어화:** 확정된 `S_out*`와 필요한 정확 문자열 참조만 받는다. 현재 raw-context bypass는 주 경로에서 허용하지 않는 계약을 만든다. 학습 초반의 teacher semantic 조건은 **훈련 전용**이며, 실제 평가는 전체 행의 예측 `S_out*`를 사용한다. 잘못 예측한 의미와 gold 문장을 무조건 짝짓지 않는다. 의미 fidelity 학습용 올바른 표적 조건과 end-to-end 성능은 구분한다.

## 학습·결과의 구분

- 추출: 역할/대상/관계·부정/의도별 정확도와 불확실성. 표면/위치 변경 자료는 원래 train과 통제 split을 보존한다.
- 디노이징: `S_in`에서 질문 답이 될 출력 의미로 가는 정확도, 잘못된 출력 후보의 교정과 새 오류, 예측 extraction 조건의 전체 end-to-end 결과. Gold extraction 조건을 진짜 end-to-end로 세지 않는다.
- 언어화: **실제 확정한** 의미와 문장의 일치. 잘못된 `S_out*`를 문장이 충실히 옮긴 경우는 언어화 충실도와 전체 답 정확도를 따로 기록한다. 역할·방향·부정을 바꾼 의미의 올바른 문장 변화도 본다.
- 문자열 NLL/UTF8/EOS/반복은 언어 품질 보조 지표다. 의미 추출·출력 의미 결정·언어화 충실도를 대신하지 않는다. 원문에서 추론한 의미와 gold 의미를 구분하지 않는 점수도 사용하지 않는다.

불필요한 broad audit와 반복 baseline gate는 만들지 않는다. 기존 자료와 코드 이력을 먼저 활용하고, 의미 representation/실제 inference features/학습 표적/손실 전달의 차이를 구체화한 뒤 필요한 CPU 복구·worker/OS 검사를 하고 한정 학습으로 이어간다. 원래 하드웨어 승인은 유효하며 재승인 질문은 필요 없다.

## 현재 실제 상태와 다음 작업

V9는1636추가 업데이트 완료/예산CLOSED/allownedgone이다. V10 `text_state_alignment_v10.py`는 byte VP→고정 디코더의 gradient를 연결한4CPU candidate일 뿐이며 삭제하지 않고 보조 연구로 보존한다. **새 주 의미 파이프라인의 worker/plan/seal/GPU는 아직 없다.** 기존 byte candidate를 이 새 의미 파이프라인이 구현됐다는 증거로 사용하지 않는다.

다음 ROOT는 먼저 원래 typed 출력 의미 basis·raw learned extractor·언어화 adapter 사이의 실제 tensor/API/훈련 target을 연결한다. 학습을 추상 설계나 gate에 무한 대기시키지 않되, byte 목표를 의미 목표로 이름만 바꾸어 계속 돌리지 않는다. 새 계약과 코드가 구체화되면 고정 cap/새 clock/새 output/전체 state 보존 안에서 실제 학습한다. 에이전트/E2final/registry/외부 게시/원본 삭제는 금지하고30분 타이머 문구는 정확히 `재개`로 유지한다.
