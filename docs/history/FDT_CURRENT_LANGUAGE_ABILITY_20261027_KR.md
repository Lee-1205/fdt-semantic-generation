# 2026-10-28 추가 — 원문 신경망 문장 생성 연구 재개

Root가에이전트없이새VP생성/문맥decoder/semanticreader의3별도학습을완료했다. 현재연구hybrid는109,778,455parameters이며원래typed102M은보존됐다. 고정답변template/외부정답역할을추론에넣지않고원문에서byte/EOS를신경망으로생성한다. 관측개발80문장UTF8/EOS정상이나dev exact12/32·13/32,긍정exact2/22·3/22로의미정렬목표미달이다. 범용대화/요약/일반언어완성은아니다. 상세와다음연구는FDT_GENERATIVE_PROGRESS_20261028_KR.md/FDT_TEXT_SEMANTIC_ALIGNMENT_RESULT_20261028_KR.md에있다. 아래2026-10-27판단은당시typed경로의역사이며소급수정하지않는다.

# 현재102M 모델의 언어능력 평가

새 모델 실행 없이 현재 코드와 보존된 평가를 확인한 판단이다. 현재는 짧은 한국어·영어 비교/부정 문구를 관계 판정에 연결하는 제한된 관계 추론 모델로 평가한다. 범용 언어모델 수준의 능력은 아직 입증되지 않았다.

| 능력 | 현재 증거와 범위 |
|---|---|
| 비교·부정 문구 인식 | ‘더 크다/작다/크지 않다’, greater/less/not 계열의 제한된 표현 패널에서 grounding100%. 좁은 관계 연산자 의미 대응을 학습했다. |
| 주어진 사실의 관계 판정 | 부분9 가중치 평가에서 Q1/Q4/Q8/Q16 dev와 관측 transfer100%, oldrole99.9452%. 같은 관측 자료·상관 view이며 fixed fuzzy closure와 given entity roles·basis 도움을 포함한다. |
| 새 어휘·표현 일반화 | 좁은 표현 계열 검증은 있지만 폭넓은 일반화 향상은 입증되지 않았다. E2에서 이전37.5→100% 패널의 표현은 학습에 편입됐고, 별도 dev64는 부모부터100%였다. |
| 자연 문장 전체 해석·개체 추출 | 주어/목적어와 질문의 개체 역할은 별도 구조화 입력으로 주어진다. 자유 문장에서 사람·사물과 관계를 스스로 추출하는 능력을 평가한 결과가 아니다. |
| 대화·요약·작문·자유 질의응답 | 현재 경로의 출력은 UNRESOLVED/GREATER/LESS/CONFLICT 네 관계 판정이다. 자유 문장 생성기나 이를 위한 광범위 평가 결과는 없다. |

Language encoder는 byte 입력을 처리하는 학습된 신경망이지만 문구당48UTF8bytes, 최대8entity/12facts/16query의 제한된 인터페이스를 사용한다. 문구별 encoder 뒤에 학습된 phrase routing과 고정 관계 폐쇄가 결합돼 있다. 최근 phase2 실험은 language/grounding/initializer를 동결했고 관계 refiner의 질문 정렬을 개선했다. 따라서 최근100%를 일반 어휘력·문장 이해·문서 이해가 새로100%에 도달했다는 뜻으로 해석하지 않는다.

다음에 일반 언어 확장성을 실제로 평가하려면 학습에 쓰지 않은 비교/부정 표현·어순·부정 scope를 고정한 독립 자료가 필요하다. 주어진 개체 역할을 사용하는 문구 의미 평가와, 원문에서 역할까지 추출하는 end-to-end 평가는 별도 과제다. 이번 질의에 추가GPU/optimizer/E2final 접근은 하지 않았다. 에이전트도 사용하지 않았다.

근거: src/fdt_local/data.py의collate 입력계약, model_masked.py의Language/readout/encode, docs/FDT_E2_AND_ROLE_COVERAGE_RESULT_20261027_KR.md, docs/FDT_QUERYCOUNT_PARTIAL9_RESULT_20261027_KR.md 및 PROJECT_STATE 첫CURRENT.
