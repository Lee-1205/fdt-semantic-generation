# FDT 생성 언어 연구 재개 — 2026-10-28

사람의 새 지시 “재개 연구 심화 언어모델 생성까지 알아서 진행할것”에 따라 root가 직접 구현·검사·학습·평가를 이어간다. 에이전트는 쓰지 않는다. 로컬 실행 승인은 유효하며 이전 종료된 QueryCount 9/128 실험은 그대로 보존한다. 이번 것은 다른 목적·모듈·자료·계획·출력·고정 한도를 가진 새 생성 분기다.

현재 typed FDT는 관계4종류를 내며 runtime의 realize 함수는 고정 문자열을 조립한다. 그것을 학습된 언어 생성으로 세지 않는다. 새 TextDiffusion은 partial9의 language/time/refiner/norm/velocity를 복사하고, 원문 byte별 문맥 표현·출력 위치·문맥 projection·고정 byte 코드북을 추가한다. 원래 typed 모델/core/checkpoint를 수정하지 않는다. 복사한 language는 이번 pilot에서 동결하고 생성 refiner/projection/위치는 새 목적에 학습한다. 새로운 모델의 정확한 파라미터 수는 실제 specification에 기록하며 원래102,237,197과 같다고 하지 않는다.

VP의 x_t=a*x0+s*noise, velocity target=a*noise-s*x0, DDIM clean=a*x_t-s*v 식을 유지한다. 512차원 orthonormal byte258개(0–255/EOS/PAD)의 고정 코드북과 최종 argmax로 문장을 읽는다. 고정 codec은 학습 중 임베딩 붕괴를 피하는 설계 선택이지만 그 자체로 언어능력은 아니다. 원문으로부터의 문맥과 noise만 generate에 들어가며 gold/개체역할/관계 oracle/fuzzy closure/답변 템플릿/검색은 들어가지 않는다. 문구 템플릿은 offline synthetic 자료 작성에만 쓰인다. UTF8 오류를 숨기거나 정답 문자열로 보정하지 않는다.

연속 잠재공간을 정제해 text embedding으로 읽는 방향은 [Diffusion-LM 원 논문](https://arxiv.org/abs/2205.14217)과 관련된다. 다른 가능한 연구 분기인 [MDLM 원 논문](https://arxiv.org/abs/2406.07524)은 mask 복원 기반이며, 이번 pilot에서 원래 VP를 그것으로 교체하지 않는다. 이 구현은 논문의 재현/동등 성능 주장이 아니다.

첫 범위는 A–H의 한·영 비교·부정 원문에서 짧은 답변 문장을 직접 생성하는 authored synthetic 자료다. Train3008행, train probe16/dev pair32/dev wording32행. Entity pair+language 단위 분할로 같은 이름쌍의 방향·operator·질문순서를 함께 나누며 새 wording은 template4만 쓴다. 이름과 정답 문자열은 train/dev에 겹칠 수 있다. 같은 추상 구조/표현 반복이 있어 독립 대규모 일반언어 corpus로 부르지 않는다. E2 final과 사용자 원본에는 접근하지 않는다.

새 prospective 한도는256attempt/최대256success, batch8, train2048행 이하(3008행 pool의 한 유한 순서, 되감기 없음), owner900초/종료 reserve90초, actual encode+denoise1300회다. 예상1022회=훈련512+baseline170+final170+입력shuffle170이며 실제 method 직전 계측한다. Prompt160UTF8bytes/answer40bytes(EOS 포함), DDIM16step, output12GiB, 시작C36GiB/GPU12GiB, 실행 중C20GiB+임시2GiB 보호. CPU 코드/자료/실제 worker preflight/독립 OS owner validate/정확한 plan schema/전체 post-Popen 종료·예외 경로를 확인한 뒤 새 planSHA/절대시각을 한 번만 봉인한다.

Baseline과 최종의 모든 raw byte/EOS/UTF8/문장 exact match를 저장하고 동일 seed17의 shuffle 입력을 통제 비교한다. Loss나 byte 평균만으로 문장 정답을 선언하지 않는다. 정답률이 낮아도 결과를 보존한다. 실제 학습 중 root는 상세 모델분석을 중단하고 독립 owned OS 감독/progress/native를 확인한다. Full model/AdamW/scaler/모든 사용 RNG/유한 order/cursor/history/노출/callbudget를 시작 및64/128/192/256에 검증 저장하며 삭제·회전하지 않는다. 중단 시 현재 fullstate를 별도로 보존한다. 자동재시작/한도 연장/이전 남은119update 사용은 하지 않는다.

이 pilot의 목적은 실제 신경망 문장 생성 경로와 학습 가능성 확인이다. 다음은 raw 오류가 조건 인식·byte/EOS·부정 scope·unseen 조합 중 어디에 있는지 판단하고, 그 구체적 문제에 맞는 별도 자료/코드/한도를 정한다. 이어서 자연 문장·지식·장문·요약 등으로 corpus와 검증을 확대한다. 이번 짧은 synthetic 성공을 범용 LM 완성으로 포장하지 않는다. 기존32문구 평가 계획은 보존하지만 생성 구현의 무한 선행조건으로 반복하지 않는다.
