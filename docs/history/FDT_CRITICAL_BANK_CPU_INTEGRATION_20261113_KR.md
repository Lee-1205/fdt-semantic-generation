# 취약 핵심 위치를 보호하는 reference bank — CPU 결합 계약

2026-11-13. V83/V84에서 확인한 기존 TRAIN100 기준의 사각지대를 줄이기 위해, 선택 코드를 실제 bank 생성·초기화·판정·저장복구 구성 요소로 연결했다. 새 모델 학습이나 성능 향상 결과는 아니다.

`research/critical_reference_bank_v85.py`가 수행하는 과정은 다음과 같다.

1. 고정 TRAIN allowlist 순서로 teacher를 한 번 소비한다. source ID/SHA, 부모·embedding·prefix SHA와 전체 FP32 정책이 일치해야 한다.
2. 기존 broad reference source는 전체 patch를 보관하고, 그 밖의 TRAIN source에서도 취약 critical 위치가 있으면 해당 prefix patch를 추가한다. 정답·critical mask·margin은 feature와 별도 packet에 둔다. decoder 입력으로 전달하지 않는다.
3. 실제 첫 batched reference 계산으로 초기 정답·margin을 다시 정한다. rowwise teacher와 정답 flag가 다르거나 margin 차이가 고정 허용치 0.001을 넘으면 중단한다. 허용치를 자동 확대하지 않는다.
4. 기존 정답의 top1과 **선택된 취약 critical 위치의 margin**을 보호한다. 다른 일반 바이트는 정답을 유지하면서 margin이 줄어드는 변화를 허용한다. 이것이 V83에서 확인한 불필요한 일반 텍스트 gate 강화를 피하는 부분이다.
5. 부분 teacher cursor와 생성 중인 bank, 초기화된 reference 상태를 저장·복구한다. source를 되감거나 중복 소비하지 않는다. patch 상한을 넘으면 일부를 버리지 않고 오류를 낸다.

CPU 검사 6개와 기존 selector 검사 3개, 합계 **9개 PASS**다. 별도의 ROOT CPU 검토에서도 부분 상태와 초기화된 상태를 파일로 저장하고 safeload한 뒤 재귀 digest가 같은지 확인했다. 일반 byte의 margin 감소는 허용하고, 취약 critical 위치의 정답 top1이 유지되더라도 margin 하한 위반은 거부하는 차이를 확인했다. source/정밀도/epoch 불일치, DEV 입력, gradient가 연결된 feature, cap 초과와 반복 cursor도 거부한다. CUDA는 초기화하지 않았고 global Torch RNG는 변하지 않았다.

이 검사는 합성 CPU feature/logit으로 **bank의 동작 계약**을 검증한 것이다. 실제 학습된 FDT forward, Adam·enabled scaler·전체 phase/RNG가 포함된 학습 driver 복구 검증은 아니다. 실제 worker preflight나 GPU 실행 plan/seal/launch도 아직 없다. dtype와 정책 metadata만으로 잘못 표기된 BF16→FP32 cast의 출처를 자동 증명할 수는 없다. 다음 실제 worker가 고정된 코드에서 전체 FP32 teacher/prefix 계산을 수행하고 그 provenance를 증명해야 한다.

과거 BF16 selection을 단순 병합했을 때 기존 7,487개 bank patch에 새로 필요한 patch는 **179개**, 합계 **7,666개**다. 이 hidden의 FP32 값은 15,699,968B다. 구성 요소의 고정 8,192-patch cap에서는 hidden 값이 최대 16MiB다. 실제 새 부모/FP32 pass에서 선택 개수는 달라질 수 있으므로 과거 수치를 실행 결과로 재사용하지 않는다. 이 16MiB는 bank hidden만의 비용이며 전체 학습 출력·STOP·checkpoint 임시 여유를 산정한 수치가 아니다.

다음은 이 구성 요소를 새 finite driver의 teacher capture·reference callback·전체 fullstate와 연결하는 작업이다. 실제 GPU device 전송, 새 phase/Adam/scaler/모든 RNG/cursor/노출/call cap과 예외 STOP, 전체 owned OS 감독 및 실제 preflight까지 연결한 뒤에만 원인을 겨냥한 별도 한정 학습을 실행한다. 기존 학습이나 미사용 예산을 재개하지 않는다.

원본·과거 가중치/Adam/RNG·raw·실패·분할은 변경하지 않았다. 새로운 corpus 행도 만들지 않았다. 현재 언어→의미→언어 단계와 Exact NULL/content0/copy0, 디노이징0을 유지한다. 미래 사고 학습은 입력·메모리·결합·출력 codec 전체를 동결한다.

증거: `research/artifacts/critical_reference_bank_20261113_v85_cpu_v1/root_cpu_review.json`, `evidence_SHA.json`, `CPU_synthetic_partial.pt`, `CPU_synthetic_calibrated.pt`, 분리된 CPU feature/target packet. CPU fixture는 실제 연구 가중치 체크포인트가 아니다.
