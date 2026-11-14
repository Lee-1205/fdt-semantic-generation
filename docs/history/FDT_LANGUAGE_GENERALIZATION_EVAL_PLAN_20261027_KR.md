# 현재 언어능력의 한정 추가 점검 — CPU 준비 범위

사람의 현재 언어능력 질문에 대응하는 다음 구체적 목적은, 정렬된 관계 추론 점수와 새 비교·부정 표현의 의미 인식을 분리하는 것이다. 기존 관계 패널을 다시 평가하거나 종료된 학습을 재개하지 않는다. Root만 직접 수행한다.

현재 상태는 계획만 준비됐고 새 builder/artifact/worker/봉인/GPU 실행은 없다. 기존 언어능력 보고서는 보존된 평가와 실제 코드에 근거하며, 새 표현 성능 점수를 이미 측정했다고 하지 않는다.

## 고정 후보 범위

- 한국어·영어의 짧은 비교·부정 문구32개만 사용한다. 기존 관측·학습 문구8개(각grounding종류2개, 한국어/영어1개씩)와 새 literal24개(각종류6개, 한국어/영어3개씩)를 사전에 고정한다.
- 새 literal은 명시적으로 지정한 train 및 이미 관측한 dev 문구 registry와 중복을 검사한다. E2 final은 registry 생성·비교·모델 평가 어디에도 접근하지 않는다. 모든 역사 자료나 미접근 final에 대한 신규성을 주장하지 않는다.
- GREATER/LESS/NOT_GREATER/NOT_LESS 의미와 부정 범위가 명확한 문구만 쓴다. UTF8 48bytes를 넘으면 잘라 쓰지 않고 준비 실패로 남긴다. 모델 결과를 보고 문구를 골라내거나 바꾸지 않는다.
- 각 행은 사실1개와 고정 구조화 역할0→1·질문0→1을 쓴다. 역할 추출은 제공되므로 자유 문장 전체 이해·NER 검사가 아니다. 같은 typed graph를 반복하는 문구 검사이며32개 독립 관계 graph로 부르지 않는다.

## CPU 계약과 향후 별도 실행 한도

CPU에서 의미 label·원래 oracle/독립BFS 관계 label·입력5feature·padding·UTF8bytes·언어/class support·literal registry·파일SHA·학습/평가 구분을 검사한다. 이 단계는 모델forward0/optimizer0이다.

향후 root가 코드/자료/계획/실제preflight/독립owned감독/native/resources를 검토한 ONE 별도 읽기 전용 실행만 준비할 수 있다. 현재 부분9 modelweights를 사용하고 학습된 phrase/grounding의4종류 판정을 primary로 점수화한다. Diffusion을 호출하지 않아 최근 관계 정렬 효과와 구분한다. Initializer는 보조 관측이며 고정 closure 도움을 명시한다.

제안 한도는 batch4의 encode8회(expected8/max10), denoise0, 보조readout8회, owner90초(기동·CPUpreflight·자료/가중치 검증·자원검사 포함)/reserve20초, output16MiB, optimizer0/checkpoint0다. Root 검토 후 구체적인 새 planSHA/output을 고정하기 전에는 실행하지 않는다. 목표 미달로 결과를 버리거나 자동 재실행하지 않는다. 기존 평가의 미사용call이나 시간을 사용하지 않는다.

문구별 raw4종류logits/argmax/gold와 한국어·영어 및 seen/새literal별 정확도·macroF1를 모두 보존한다. 알려진 문구 인식, 제한된 새literal 전이, 자유 문장 해석, 문장 생성은 서로 구분한다. 이 작은 결과를 범용언어능력 점수로 확대하지 않는다. Source모델/buffer/RNG 불변과 모든owned종료를 확인하며 E2final·기존source/weights/AdamRNG/raw는 보존한다.
