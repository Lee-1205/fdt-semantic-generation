# 질문 간 간섭 통제 검사 — 2026-10-27

읽기 전용 검사를 한 번 실행하고 정상 종료했다. 독립 OS 감독 총12.352초, optimizer0/checkpoint0, 실제 batched model call26회와 logical graph-forward exposure104회(상한200)를 구분한다. Singleton은 배치64개의 한 질문 view를 공유 가중치로 처리했다. 모델 디렉터리를 만들지 않았고 parent QueryMix fullstate를 수정하지 않았다. 감독·worker·두 proxy가 모두 종료된 것을 root native Windows 조회로 확인했다.

관측 QueryMix dev의 고정 systematic indices[0,8,16,24,32,40,48,56]를 사용했다. 저장된 네 original sampling group에서 각2개이며 결과 오류로 선택하지 않았다. 첫8개를 쓰려던 CPU 초안은 group0 편중을 발견해 실행 전에 폐기·수정했다. 이8개는 작은 관측 개발 표본이며 대표성이나 추가 독립 graph를 주장하지 않는다. E2 최종 holdout에는 접근하지 않았다.

| 방식 | 맞힌 query | 정확도 |
|---|---:|---:|
| Fuzzy initializer |128/128|100%|
| Full16 refiner |117/128|91.406%|
| Reverse16 후 정렬 복구 |117/128|91.406%|
| 같은 encoder 결과를 question별로 slice한 refiner |128/128|100%|

각 원래4행 평가 배치의 seed17+batch_start로 만든 float32 noise를 행별로 재구성하고 full/reverse/single에 같은 slot noise를 썼다. 원래 model deterministic/mathSDPA/TF32off와 fp16 protocol을 유지했다. Root는 full128개 예측이 기존 seed17 step256 raw와 모두 일치하고, 보존한[8,16,512] noise의 행별 hash가 raw와 일치함을 확인했다. Noise artifact SHA906e7e532769db996cf4247a348270f36b140bcac14a35393335faebf90f0158.

Reverse의 argmax 차이는0/128이지만 logits가 bit-exact는 아니다(maxabsdelta0.001209). Full의11개 오류는 모두 GREATER↔LESS 방향 반전이었고 single에서 고쳐졌다. 이는 다른 query의 존재와 관계되는 간섭 가능성을 뒷받침한다. Single 조작은 denoiser의 self-attention·cross-context·query간 관계 집계를 동시에 제거한다. 따라서 하나의 연산이 원인이라고 단정하지 않으며, initializer가 이미100%인 이 표본에서 independent neural reasoning 이득을 증명하지 않는다. 정답은 score 계산에만 사용했고 inference 입력에는 넣지 않았다.

다음은 별도 query-local denoiser CPU 개발 후보를 만든다. 기존 FDT100을 상속하고 denoise만B×Q개의 한 질문 view로 처리해 복구하며 원래 parameter 이름/개수/strict state load, shared gradients, time·mask·noise 정렬을 확인한다. Core와 현재 model/가중치는 그대로 보존하며 자동 사용자 경로 교체나 모델 승격은 하지 않는다. Query 간 learned relation 교환을 제거하는 tradeoff와 full encode initializer가 여전히 fuzzy closure를 쓰는 한계를 명시한다. CPU 계약만 준비하고 새 GPU/optimizer를 자동 승인하지 않는다. 같은 replay 학습을 늘리지 않으며 후속 optimizer에는 구체적인 미해결 목적과 새 고정 계획이 필요하다. E3 독립 이득 검증은 여전히 미완료다.

Run planSHA8fa2435a7923361ce2e603661d6433fed913cb42c9b0f01186fa49591eeb29c6. ResultSHA323cd44b8e6be3cb4aab8e4ff7bfacfbc54eaafba83c724acd0a74990250c975. ParentSHAa877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36. 원문·noise·root_result.json·감독 종료 증거를 보존했다.
