# V33 3종 메모리 문장 복원 결과 — 2026-10-29

V33은 새 3종 메모리 경로로 **학습한 64개 원문 구간을 64/64 완전히 재복원**했다. ROOT가 초기64·최종64·개발16의 144개 출력을 모두 직접 읽었다. 최종 학습 구간은 이름, 숫자, 날짜, 부정, 문장부호와 원문의 오탈자까지 동일하다. 이는 반복 학습한 자료에 대한 복원 용량이며, 외운 자료를 포함한다. 일반 문장 복원 완성이나 의미 추출의 유일한 정답을 입증한 결과가 아니다.

현재 과제는 언어→입력 의미→동일 원문 복원이다. 답변 의미를 만드는 디노이징은 이번에도 0회다. 기본 퍼지 K_P, 문장별 퍼지 K_C, Exact K_E를 유지했고, 디코더에는 Commit의 의미 벡터만 전달했다. 원문 전체 복사·분절을 이어 붙이는 우회·정답 길이·정답 접두사·정답 span을 추론 입력으로 사용하지 않았다. 실제 literal copy bytes=0이다. 미래의 명시적 의미 Commit에 따른 전문 인용 권한은 별도 V32 계약에 보존했다.

## 실행과 종료

- 실행: `fdt_local_102m_20261026/runs/three_memory_codec_20261029_v33`, ONE launch, owner/worker exit0, COMPLETE.
- 새 학습 2048 attempt/2048 update/AMP skip0, worker316.132초. encode2138+decoder7739=9877호출, denoise0. 새900초/12000호출 예산 CLOSED. 미사용2123호출과 시간을 재사용하지 않는다.
- owner proxy38352→native22744의 생성·parent·command·생존은 ROOT가 실행 시 관측했다. worker proxy36852→native18124의 생성(15:07:33.662KST), parent/command/identity는 독립 OS owner 기록으로 확인했다. **ROOT가 실제 worker 생존을 직접 관측한 기록은 없다.** 이 공백을 소급해 메우거나 관측 증거를 얻으려고 재실행하지 않는다.
- owner가 모든 자식/프록시 종료를 확인했으며, ROOT는 18:11:44KST native Python/compact0을 새로 확인한 뒤 분석했다. 그 시점 C 여유45,133,406,208B. 15:12:55 종료 이후를 학습 중이라고 보고하지 않는다.
- planSHA `b5478bde6f6e78aea92a3970ea0297db607db45d6035ec7c0f8412cbaa136741`.
- 최종 fullstate778,568,885B SHA `44ae5f53a838c5dcf803fed5dfdf11883f9dd4d5982ee8b7dff5baa2fac6f748`, algorithm `7deda978f4a618f04d84f1de3e26028f67bce2e65583eaeb8ee8fb679f084fad`.

## 실제 복원과 개발 결과

| 패널 | V33 초기 | V33 최종 |
|---|---:|---:|
| 동일 학습64, BOS/자기 접두사 원문 일치 | 0/64 | 64/64 |
| 동일 학습64, 정상 UTF8+EOS+특수기호 규칙 | 52/64 | 64/64 |
| 동일 학습64 teacher NLL | 4.730274 | 0.000021373 |
| 동일 학습64 teacher byte 정확도 | 24.2445% | 100% |
| 기존 개발256 teacher NLL | 새 초기 평가 없음 | 3.253194 |
| 기존 개발256 teacher byte 정확도 | 새 초기 평가 없음 | 45.7190% |
| 기존 개발16 BOS 원문 일치/전체 의미 충실 | 새 초기 평가 없음 | 0/16, 0/16 |
| 기존 개발16 정상 UTF8/EOS | 새 초기 평가 없음 | 16/16 |

V31의 기존 개발 NLL2.431911/byte53.4020%와 비교하면 V33 최종 개발 성적은 더 낮다. 새 추출 함수로 바꾼 초기에는 기존 학습64도 크게 흔들렸고, 64문장을256회 반복해 다시 맞췄다. 따라서 원인을 디노이저 실패로 돌릴 근거가 없다. 현재 증거는 **좁은 자료의 새 함수 적응과 외운 문장의 복원은 가능하지만 일반화가 부족함**을 보여준다. 모델 전체의 실패나 학습을 중단할 이유로 단정하지 않는다.

ROOT의 개발 문장 읽기에서도 날짜·사람·국가·함수명·부정과 사건 관계가 대체되거나 누락됐다. 예를 들어 음주 측정/DUI 원문에 학습 문장의 Danil/fake가 들어왔고, 선수와 스카우트 문장에 April Ryan/20-year가 들어왔다. 개발 chrF18.9149, word F1 10.6949%, content-word F1 0.6944%다. 문자열 유사도를 의미 보존 비율이라고 부르지 않는다. 확인 가능한 부분 명제 보존도 이번16개에서는0으로 기록했다.

## 메모리 기여와 검증 범위

동일 개발16의 정상 NLL3.655400과 비교해 K_P 제거 +0.042270, K_C 제거 +1.071677, K_E 제거 +0.712383 nats다. 이는 각 경로를 제거했을 때 현재 출력이 변하고 손실이 커진다는 관측이며, 독립적인 의미 이해·문자열 선택 의도·정확 복사 능력의 증명이 아니다. K_E는 학습된 key/value를 입력 의미에 읽는 경로이고, 원문 바이트를 디코더에 전달하는 copy 경로는 아니다.

저장된16개 full Exact key/query/valid mask에서 top16+NULL의 인덱스와 가중치를 다시 계산했다. token μ/value에서 독립 순차 FP64 누적 방식으로 long/recent N,D를 재구성했고 저장값과 일치했다. gold span이나 모델 추가 forward는 사용하지 않았다. 3개 fullstate의 SHA/safeload/재귀 digest, 원래264 tensor와81개 populated Adam의 정확한 이관, 새10개 초기 Adam 빈 상태, 이후91개 ownership, scaler와 저장된 모든 RNG/유한 순서/커서/노출/평가 파일 SHA/동결 범위를 확인했다. 실제 CUDA 복구 재실행은 하지 않았다. CPU 독립 검증 스크립트는 `scripts/review_three_memory_codec_v33_result_cpu.py`, 근거는 run의 `root_result.json`, `root_reconstruction_similarity.json`, `root_all144_reading.md`다.

구조 범위도 그대로 제한된다. K_P는 기존 learned prior이며 certified persistent Writer는 없다. K_C는 문장 전체 입력을 읽은 token별 장기/최근 스냅샷이며 vNext 전체 층의 prefix read-before-write 구현이 아니다. Exact 의도적 literal selector/transport도 미구현이다. 따라서 3종 읽기 경로 연결을 전체 vNext 완성이라고 부르지 않는다.

## 다음 한정 목적

같은 3종 메모리·동일 원문 목적·91개 Adam/scaler/RNG·디노이징0을 유지하면서, 이미 보존된 OpenWebText TRAIN8192문서/32768구간의 **한 유한 epoch**로 학습 범위를 넓힌다. 기존 개발 문장을 학습에 넣지 않고, 학습64 복원도 참고 지표로 남긴다. 64개를 계속 반복하거나 반복 억제 벌점만 추가하는 것보다 현재 명확한 노출 범위 문제를 먼저 다룬다. 이 후속은 별도 코드/CPU 복구·실제 preflight/독립 owner 검토/새 plan SHA·시계·호출 상한으로 준비하며, V33 예산 연장이나 미사용분 재사용이 아니다. 원본·모든 이전 버전·실패·가중치/Adam/RNG는 보존한다.
