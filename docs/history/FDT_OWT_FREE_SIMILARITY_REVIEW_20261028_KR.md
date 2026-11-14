# 자유 복원 재검토 — 완전 일치 외의 부분 유사도

사용자 요청에 따라 V14 정상 출력16개를 원문과 모두 직접 읽고, 저장된 V12/V13/V14 정상·상태교환·상태제거 출력에 CPU 문자열 계산을 적용했다. 새 모델 forward·추론·optimizer는0이다. 완전 일치0/16을 전체 의미 추출 실패라고 단정하지 않는다.

| 같은16문서, 정상 상태 | V12 | V13 | V14 |
|---|---:|---:|---:|
| 문자 n-gram chrF2 /100 |19.77|20.51|20.79|
| 단어 F1 |9.62%|11.27%|12.30%|
| 내용어 exact F1 |0.00%|0.69%|0.00%|
| 원문16개 중 자기 원문이 chrF 단독1위 |4/16|6/16|8/16|
| 처음부터 맞는 UTF8 byte 평균 |0.375|1.438|2.813|
| byte편집거리 / 긴 쪽 길이 기반 유사도 |27.41%|26.92%|26.61%|

문자 조각·시작 표현·문서별 구별에는 일부 진전이 보인다. V14 정상 chrF20.79는 상태교환15.63보다 높고, 교환 조건의 자기 원문1위는0/16이다. 다만 이 표는 이미 관측한 소표본의 사후 기술 통계이며 독립 검증/새 blind benchmark가 아니다. 내용어 F1은 아래 고정 stop list를 제외한 **정확한 어형 일치**이므로 동의어를 부정하는 의미 판정으로 쓰지 않았다.

원문과 실제 출력을 직접 읽은 판단은 **충실한 의미 동치0/16**이다. 표현만 다른 정상 paraphrase로 보기는 어렵다. 예를 들어 원문의 "I then ... without paying ..."는 "I then ... get pay ..."로 변해, 시작 표현과 pay 흔적은 남지만 무지불 관계를 보존하지 못했다. DUI 검사·기소 문장은 She만 남고 핵심 사건이 없어졌다. 함수와 임시 string/pointer 관계도 이름·관계가 없어졌다. Fans/Fanstile, Janissa/Japansians처럼 철자 일부가 닮은 비단어가 실제 개체 보존을 뜻하지 않는다. 이는 ROOT의 비맹검 정성 판단이며 외부 semantic embedding 점수나 보정된 의미 성공 확률은 아니다.

문자 유사도는 [chrF 원논문](https://aclanthology.org/W15-3049/)을 참고해 case-sensitive/공백 제거/문자1–6gram의 평균 precision·recall과 beta2 F-score를 명시적으로 계산했다. 의미적 평가에 contextual embedding을 사용하는 [BERTScore](https://arxiv.org/abs/1904.09675)도 확인했지만 이번에는 외부 모델을 추가하거나 그 점수를 실행하지 않았다. 단어 F1만으로 의미를 판정하면 상태제거의 동일한 무관 문장이12.83%로 정상12.30%보다 높아지는 실제 반례가 있다. 그래서 주어·행위·부정·개체·관계의 보존을 직접 함께 읽었다.

정답 prefix 기반 정확도52.91%와 자유 생성의 차이는 크다. 자유 생성에서 첫 오류가 평균2.813byte, 중앙2.5byte로 매우 이르게 생기고 이후 비단어·잘못된 단어가 이어진다. 이는 학습 때의 정답 prefix와 추론 때의 자기 출력 prefix 차이에서 오류가 누적되는 가설과 일치한다. 이런 차이는 [Scheduled Sampling 원논문](https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf)의 문제 설정과 관련되지만, 현재 원인이 그것 하나라는 증명도 해당 학습법이 자동 해결책이라는 증거도 아니다.

코드에서는 teacher prefix의 shift/BOS/PAD mask, greedy 생성의 BOS/EOS, 동일한 추출 상태 전달, raw-context 우회 부재를 확인했다. 새 실행 결함은 발견하지 않았다. 현재 문장 어댑터는 두 pooled fuzzy view로 압축하며, 선택된 anchor의 동적 value는 같은 문장 value를 공유한다. 비어 있지 않은 정규화 rho의 지지집합에서는 rho@memory가 그 공통 value로 환원되고, 뷰별 구별은 주로 prior/rho/gamma 쪽에서 남는다. 이는 위치별 사실을 따로 보관하는 전체 원래 structured-fact 메모리와 같은 계약은 아니다. 단독 원인으로 단정하거나 사용자 지시의 원래 fuzzy 회로를 임의 대체하지 않는다. 원래 관계 태스크의 높은 성능도 그대로 보존한다.

다음 학습은 동일 fuzzy AE/추론 동결/상태만 받는 decoder를 유지하고 문서 다양성을 늘린 V15다. 기존 모든 Adam/scaler/RNG를 유지하고 새1,024훈련 문서만 추가했다. 기존 개발64문서256문장과 기존훈련 prefix는 그대로다. 반복 벌점 강화·anchor 초기화·새 디노이징을 동시에 섞지 않는다. 이후에도 exact와 함께 문자·내용 보존·첫 오류·정상/교환/제거 차이를 기록해 부분 진전과 실제 충실도를 구분한다.

전체16개 원문·출력·항목별 판단과 지표는 `fdt_local_102m_20261026/research/artifacts/owt_free_similarity_20261028_cpu/all16_reading.md`, 기계 판독 결과는 같은 폴더의 `result.json`, 계산 코드는 `scripts/review_owt_free_similarity_20261028_cpu.py`에 보존했다. Corpus/평가 원문을 새 훈련에 편입하지 않았고 E2final/registry 접근0이다.
