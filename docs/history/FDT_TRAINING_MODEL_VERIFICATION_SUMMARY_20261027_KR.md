# 2026-10-28 추가 — 원문 신경망 문장 생성 연구 재개

Root가에이전트없이새VP생성/문맥decoder/semanticreader의3별도학습을완료했다. 현재연구hybrid는109,778,455parameters이며원래typed102M은보존됐다. 고정답변template/외부정답역할을추론에넣지않고원문에서byte/EOS를신경망으로생성한다. 관측개발80문장UTF8/EOS정상이나dev exact12/32·13/32,긍정exact2/22·3/22로의미정렬목표미달이다. 범용대화/요약/일반언어완성은아니다. 상세와다음연구는FDT_GENERATIVE_PROGRESS_20261028_KR.md/FDT_TEXT_SEMANTIC_ALIGNMENT_RESULT_20261028_KR.md에있다. 아래2026-10-27판단은당시typed경로의역사이며소급수정하지않는다.

# FDT 학습 상황·모델 분석 검증 요약

최신 갱신: 사람의 명시적 재시도 지시 후 root가 에이전트 없이 부분9 가중치 평가를 완료·원시 재검증했다. Q1/Q4/Q8/Q16 dev와 관측 transfer는 모두100%, uniform은98.8281→100%, oldrole은99.8355→99.9452%(기존 오류1개 잔여)다. 실제700개 encode+denoise/280readout/추가 optimizer0이며 모두 종료했다. 감독의 종료 시점 조회 경합은 별도 수정하고 CPU7개+실제Windows전체프로세스경로1개 검사를 통과했다. 원래 STOPPED 감독 기록은 보존하며 평가 결과 완료는 독립 root_result로 확인했다. 자세한 최신 결과는 fdt_local_102m_20261026/docs/FDT_QUERYCOUNT_PARTIAL9_RESULT_20261027_KR.md에 있다. 아래19:28 요약은 재시도 전의 역사적 상태다.

2026-10-27 19:28KST 기준. 현재 학습·추론 프로세스는 없다. 질문 결합 정렬의 개선 효과는 확인했지만, 가장 최근9회 학습의 성능 효과는 아직 평가하지 못했다.

## 실제 학습 상황

| 단계 | 실제 상태 | 성능 판단 |
|---|---|---|
| QueryMix | 256/256 성공, AMP skip0, 정상 종료 | 같은 seed17 primary89.5508→92.3828%, +2.8320%p. 사전+3%p/94% 목표는 미달, 기존 성능 보존 통과 |
| Query 결합 mask | 추가 학습0, 같은 가중치 | primary92.3828→99.9023%, paired92.0833→100%. 가중치 학습 이득과 구별 |
| 질문 수 Q1/4/8/16 정렬 학습 | 목표128회 중9성공/9시도, AMP skip0, 중단·미완료 | 감독 코드 오류로 안전 중단.9회 뒤 성능 결과 없음 |
| 부분9 읽기 전용 평가 | 감독이 작업자 시작 전에 실패, 모델 호출0 | 새로운 성능 결과 없음 |

최근 두 실패는 감독 코드의 변수 참조 및 plan 항목명 불일치였다. Root 검토와 CPU mock이 이를 놓쳤다. 모델 발산이나 자원 부족이 원인은 아니다. 모든 관측 프로세스 종료를 확인했으며 원래 학습·평가 시간은 연장하지 않았다.

## 모델에서 확인된 변화와 남은 문제

같은 seed17·같은 가중치에서 cross-attention의 query i가 자기 recalled/init 정보와 유효 facts/null에 대응하도록 결합했을 때 큰 개선이 있었다. 원래 encoder·initializer·query self-attention·관계 집계·모든 파라미터는 보존했다. Gold는 모델 입력에 넣지 않았다.

- Primary는1024개 중 과거78오류가 모두 교정됐지만 새 오류1개가 생겼다. 방향 반전은 이 관측 primary에서 사라졌다.
- 같은 facts의 uniform8 패널은99.8047→98.8281%로0.9766%p 하락했다. 따라서 모든 구성에서 개선되거나 완전히 정렬됐다고 할 수 없다.
- 새 관계 graph16개의 작은 전이 검사에서는 Q8이123/128→128/128, Q16이237/256→256/256으로 개선됐다. Q4는63/64로 누락1개가 남았다.64개 prefix view는16개 graph의 상관 관측이다.
- 남은 Q4 사례는 initializer가 맞아도 첫 denoising clean 추정부터 정답 방향을 잃었다. 같은 질문·사실·slot 잡음의 Q8/Q16은 맞았다. 마지막 반올림만의 문제로 보기 어렵지만 self-attention·관계 집계·시간/잡음 중 단일 원인은 아직 확정하지 않았다.
- 이전 저장 학습 계약에는 Q1/Q4 정제 문맥이 없었다. 이를 보완한 자료를 만들었지만 학습은9회에서 중단돼 질문 수별 학습 분포 가설은 아직 검증되지 않았다.

## 검증된 범위

부분9 전체 체크포인트636586945B의 SHA, CPU safe-load, model/populatedAdamW/scaler/모든 RNG/history/sampler cursor72, 동결 영역 및 실제72행·72고유 graph·522query 노출을 검증했다. 복구 정보가 보존됐다는 증거와 능력 개선 증거는 다르다.

수정 감독은 실제 plan schema를 이용한 CPU 전체 경로 및 만료 시 실행 차단 검사까지 root가 재확인했다. 새 owner-v2의2개 focused CPU 검사와 실제 worker CPU preflight·독립 validate가 통과했다. 실제 GPU 재평가나 복구 학습은 실행하지 않았다.

현재 해석은 질문과 조건 정보의 대응 정렬이 중요한 병목이었다는 것이다. 질문 수·시간/잡음 정렬에는 잔여 문제가 있다. 이 결과는 주어진 역할·고정 fuzzy closure·basis 도움을 포함하는102237197파라미터의 좁은 typed adapter 실험이며 범용 LM/NER 완성이나 독립 neural 추론의 증거는 아니다. E2 최종 holdout에는 접근하지 않았다.

근거: PROJECT_STATE.md 첫 CURRENT, FDT_QUERY_MIX_RESULT_20261027_KR.md, FDT_QUERY_BOUND_ALIGNMENT_RESULT_20261027_KR.md, FDT_ALIGNMENT_TRANSFER_QUERYCOUNT_RESULT_20261027_KR.md, FDT_ALIGNMENT_TRAJECTORY_RESULT_20261027_KR.md 및 두 runtime incident 기록. 부분9의 실제 성능 평가는 미완료로 남긴다.
