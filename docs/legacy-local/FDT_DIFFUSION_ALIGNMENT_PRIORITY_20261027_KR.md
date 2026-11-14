# 현재 개발 우선순위 — 디퓨전 정렬 (2026-10-27)

사용자는 디퓨전이 아직 제대로 정렬되지 않았을 가능성을 우선 조사하고, 정리가 될 때까지 지켜보라고 지시했다. 현재 성능만으로 디퓨전 자체의 가능성에 결론을 내리지 않는다.

실측은 제한된 관계 문항에서 고정 fuzzy closure를 포함한 initializer100%, full diffusion92.38%다. 고정8graph 검사에서 full/reverse117/128, single-query128/128이었으나 이는 여러 경로가 동시에 달라진 작은 관측 개발 검사다. Query별 초기 상태와 역할·방향이 denoiser에 결합되는 정렬 가설의 단서이지, 특정 연산의 원인 확증은 아니다.

따라서 원래 full diffusion과 가중치를 유지한다. Query-local 분리 후보는 CPU검사를 통과한 비교용이며 자동 적용하지 않는다. Neutral-initializer E3 후보도 CPU 준비물과 제안서로 보존하되 지금 실행하지 않는다.

다음은 original initializer/전체 query self-attention/관계 집계를 유지하면서 denoiser cross-attention의 query i가 자기 recalled i/initializer i와 전체 유효 facts를 읽도록 묶는 별도 CPU 후보다. Mask는 입력의 query_mask/fact_mask와 slot identity만 사용한다. 파라미터 추가·state key 변경·gold 입력·core 수정은 하지 않는다. 시간/잡음/방향/mask/gradient 계약을 먼저 확인한다.

이후 필요하면 최대64successful updates/70attempts/900초의 별도 제한 alignment stage를 준비한다. CPU 후보 작성은 학습 실행이 아니며 새 코드·데이터 계약·계획SHA·출력·자원·중단 기준·독립 감독을 root가 검토한 뒤만 실행한다. 과거 run을 다시 열거나 같은 replay를 무한 연장하지 않는다. 이미 목표를 만족하는 초기 상태라면 학습 이득을 꾸미지 않는다. 범용 추론/LM 완성 여부는 여전히 미확정이며 E2 최종 holdout은 접근하지 않는다.
