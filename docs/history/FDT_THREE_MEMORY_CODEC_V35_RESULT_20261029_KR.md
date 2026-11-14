# 3종 메모리 문장 복원: 첫 넓은 순회 완료와 실제 원문 대조

기존 OpenWebText TRAIN8192문서/32768구간의 한 유한 순회를 완료했다. V34의27256구간과 새 V35의5512구간은 겹치지 않으며 합집합이 원래32768구간과 정확히 같다. 총4096시도/4095업데이트, AMPskip1722의8행도 되감지 않고 노출에 포함했다. 실제 target 노출2,926,478UTF8bytes/clean·corrupt 두 손실의5,918,492symbol occurrences다. 새 문서나 데이터가 늘었다는 뜻은 아니다.

V35는 별도480초/6000호출 예산에서689/689업데이트·skip0·184.719worker초로 완료했다. encode755+decoder3375=4130호출, denoise0이며 예산은 CLOSED다. 미사용1870호출/시간은 재사용하지 않는다. V34의 기존 FAILED/시간 확보 종료 기록도 보존했다. V35가 V34의 시계나 상한을 연장한 것은 아니다.

## 결과

| 같은 관측 패널 | V33:64개 반복 적응 뒤 | V35:넓은 한 순회 뒤 |
|---|---:|---:|
| 개발256 teacher NLL | 3.253194 | **0.885946** |
| 개발256 teacher byte 정확도 | 45.7190% | **73.7493%** |
| 학습 참고64의 BOS 원문 일치 | 64/64 | 25/64 |
| 개발16의 BOS 원문 일치 | 0/16 | 0/16 |
| 개발16의 전체 명제 충실 복원, ROOT 판독 | 0/16 | 0/16 |
| 학습64/개발16의 UTF8+EOS | 64/64,16/16 | 64/64,16/16 |

보조 지표의 큰 개선은 현재 경로가 다양한 입력에 맞춰 학습된다는 유의미한 진전이다. **teacher byte 정확도는 의미 보존율이나 정답 접두사 없이 완전 복원되는 비율이 아니다.** 첫 넓은 순회로 실제 문장 복원이 완성됐다고 하지 않는다. V33의64개 반복 적응을 더 넓은 자료로 바꾸면서 일부 학습 문장도 잊었다. V33의 기존64/64 복원 버전과 모든 Adam/RNG를 별도로 보존했으며 자동 승격하지 않는다.

ROOT는 학습 참고64+개발16의 원문/복원문80개를 모두 직접 읽었다. 학습64 중25개는 바이트까지 일치한다. 다른14개는 독립적으로 확인 가능한 일부 절을 보존했고,1개는 핵심 명제가 거의 같지만 reports→report 단복수가 달라졌다. 나머지24개는 특정 명제의 충실 복원으로 인정하지 않았다. 이 판독은 원문을 보며 한 질적 평가이며 별도 blind 의미 벤치마크가 아니다.

예를 들어 RE24 재계산/park adjustment 제외는 남았지만 SB/CS는 임의의 문자열로 바뀌었다. Danil 문장에서는 자진 출두는 남았으나 혼인이 가짜라는 핵심 서술이 was found로 바뀌었다. 개발 문장에서는 no explosion이라는 부정과 workplace speech의 인과 관계가 사라졌다. 긴 일치 접두사나 비슷한 단어가 있어도 이 변화를 완전한 의미 보존으로 세지 않았다.

개발16 chrF18.9149→24.6344, word F1 10.6949→13.2815%, content-word F1 0.6944→3.4406%, 정확한 첫 접두사 평균1.5→4.1875bytes다. 학습64는 chrF79.2478/word76.3234%/content72.1091%/접두사48.515625bytes다. 어떤 문자열 지표도 의미 보존의 백분율로 사용하지 않는다.

## 메모리 경로와 검증

같은 개발16 정상 teacher NLL1.169567 대비 기본 퍼지 K_P 제거 +0.060839, 문장별 퍼지 K_C 제거 +3.279347, Exact K_E 제거 +0.236971이다. 문장별 퍼지 메모리가 현재 복원 경로에서 크게 사용되고 Exact 읽기도 기여한다는 통제 관측이다. 올바른 의미가 이미 완성됐거나 Exact의 복사 의도를 학습했다는 증명은 아니다. 제거 통제는 동일 관측 패널이며 단독 원인 확정에 사용하지 않는다.

기존과 같은111732306parameters/38556357trainable,91개 populated Adam/scaler/저장된 모든 RNG를 보존했다. V34의 원래4096배치 cosine에서3408번째부터4096번째 위치를 정확히 사용했으며, 학습률 재시작·모멘트 초기화·소비한 행 반복이 없다. 입력 전체의 per-token long/recent FP64 N,D, prior K_P, 입력에서 만든 key/query/top16+NULL Exact 읽기, Commit만 받는 디코더를 유지했다. 원문 바이트/H/저장소를 디코더로 직접 우회하지 않았고 literal copy bytes=0이다. 사고/refiner/time/velocity/관계 모듈은 동결, 디노이징0이다.

Certified persistent Writer, vNext 전체 층의 prefix read-before-write, 의도적 Exact literal selector/transport는 여전히 미구현이다. 미래의 명시적 Commit 전문 인용 권한은 보존했다. 현재 언어 codec 단계에서 전문 복사 요청을 학습했다고 하지 않는다.

독립 OS owner와 ROOT가22828/42764/16404/4824 전체 종료 및 nativePython0을 먼저 확인한 뒤 분석했다. ROOT는 실행 중 actual native4824의 parent16404/creation18:47:55.939KST/command/생존과118업데이트·598업데이트 진행을 직접 관측했다. owner/worker exit0/COMPLETE다. 마지막 fullstate779648693B SHA `b8c32a8d3517152807ba9630509c903d6adb0f7703ebe3a4dfa188744612fe7f`, algorithm `f2a50d87ade521b3264b51dc4f3a7ecbe2dc64f1d88a2797a02d406c0721feb0`, planSHA `68bd98008d89ea08804d73ac97e2fd756097b3ae028ff6a179c2ecb6054e90f5`.

ROOT가 두 fullstate의 SHA/safeload/재귀 digest/전체91개 Adam/모델·scaler·모든 저장 RNG의 정확한 이관/동결/원래 순서와 학습률 위치/노출/커서/평가 SHA/정확한 AR호출수를 독립 확인했다. 저장된 개발16의 full Exact score/top16+NULL 및 독립 순차 FP64 N,D도 재구성했다. 추가 모델 구성·forward·CUDA 재실행0이다. 실제 CUDA 복구 재실행 증거와는 구분한다. 근거는 `runs/three_memory_codec_20261029_v35/root_result.json`, `root_reconstruction_similarity.json`, `root_all80_reading.md`, `scripts/review_three_memory_codec_v35_result_cpu.py`다.

## 다음 학습 목적

새 3종 메모리 함수의 넓은 자료 노출은 이제 한 순회다. 현재 진전만으로 멈추거나 전체 실패를 단정하지 않는다. **같은 목적/구조로 두 번째 넓은 유한 순회를 학습**하는 것이 다음 우선순위다. 실제 비용이 약5.5업데이트/초였으므로 한900초 실행에4096회와 최종 평가를 모두 넣지 않는다. 다음은 새 고정 순열의 앞2048배치와 뒤2048배치로 한 순회를 사전에 분할하고, 전체 순서·cosine 위치·Adam/RNG를 유지한다. 보조 평가는 순회 끝에서 수행해 불필요한 초기·중간 평가를 반복하지 않는다. 새로운 plan/예산·코드가 확정되기 전에 실행했다고 하지 않는다.

추론 디노이징으로 넘어가지 않고, 의미를 읽어 동일 원문을 재복원하는 언어 codec 학습을 계속한다. 고유명사·숫자·부정과 문장 후반의 누적 오류를 별도 관찰하되, 변경 원인을 단일 encoder나 decoder로 아직 단정하지 않는다. 모든 원본·이전 가중치/Adam/RNG·실패를 보존하고 E2final/registry를 접근하지 않는다.
