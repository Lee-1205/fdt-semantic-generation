# V23 — 문서 확대 후 7,437회 갱신, 시간 여유 종료 및 별도 읽기 전용 복원 평가

현재 과제는 **같은 입력 X→원래 퍼지 앵커의 Z(X)→입력 재복원 X̂**다. 원문 context 우회·정답 prefix/길이/역할을 상태 조건 복원에 주지 않는다. 추론과 출력 목적 의미 디노이징은 별도 동결했다. 기존 `free/generation` 파일명은 자유창작이 아니라 BOS와 자체 prefix로 같은 문장을 복원한 기록이다. 보조 teacher-prefix 지표와 전체 재복원 충실도를 구분한다.

## 실제 학습은 부분 종료, 저장 상태는 검산 완료

새 목표 8,192 attempt 중 **7,440 attempt / 7,437 update / AMP skip3(at2,1908,4623)**를 수행했다. 누적47,076 attempt / **47,061 update**다. Worker의 사전 고정 `final evaluation reserve` 시간 guard가 걸려 종료됐으며 원래 supervisor **FAILED/exit2** 기록을 보존했다. 8,192회 완료로 바꾸지 않았다. 수치 실패를 관측한 것이 아니며, 모델·Adam은 모두 유한했다.

전체 복구 상태 `emergency/checkpoints/state_7440.pt`648,555,953B/SHA `c45b5efb9b57d3cba1bda234bda025295b6e275fe99dab378181d971c9360a69`, algorithm `0635fd824fab9908114c169eef2ac4ff15c78fc41ba09e3ba626c2c7212670c9`를 보존했다. Import0/mid4096 및 이전 원본·상태·실패도 보존했다. ROOT CPU는 source전체 component 동일성, 세 fullstate의 SHA/safeload/재귀 digest, 동결 reasoning-prefix-basis, finite model/Adam, 63Adamstep47061/scaler65536tracker817, 모든 저장 RNG·history·cursor·전체 ancestry를 확인했다. 추가 model forward0이다.

실제59,520행/targetUTF8 **5,315,910 bytes**, old-doc29,789행/new-doc29,731행, mask798,212 bytes, 두 decoder 손실의 byte+EOS10,750,860 symbols를 소비했다. 한 epoch32,768행과 다음 epoch의26,752행이며 전체 두 epoch 완료는 아니다. Online 두 구간의 cleanCE .997675→.824597/corruptCE1.461440→1.344704/mixedloss1.229558→1.084651은 학습 중 평균이고 고정 final train NLL은 아니다.

학습 allocation 실제encode7440+decoder14880+denoise0=**22,320/max28,000 calls**다. 원래900초/reserve90초 예산은 CLOSED이며 남은5,680calls·시간·752attempt를 재사용하지 않는다. 학습run은 `runs/owt_fuzzy_reconstruction_20261028_v23_final`이고, `runs/..._v23`는 실행하지 않은 superseded CPU draft다.

Owner16736→28564/worker29844→42684의 생성·parent·command·생존을 ROOT가 확인했다. 전체 소유 proxy/native 종료 후22:55:01KST Python0을 확인하고 CPU 분석했다. 근거는 학습run의 `incident.json`, `root_partial_state_review.json`, native JSON과 `scripts/review_owt_partial_v23_cpu.py`다. 임시 sleepguard는 종료 시 해제됐다.

## 고유 자료 확대와 CPU 준비 이력

최종 immutable artifact는 `research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final`, manifest SHA `936351c98e8e8f5e614814fff84f2a7903ee07bed3a2cc46a18b6f820c0290f8`다. Train8,192문서/32,768구간으로 늘렸고 기존4,096문서/16,384행은 정확한 prefix, 기존64dev문서/256입력·정답·순서는 그대로다. 고유 선택 train2,926,478UTF8bytes=old1,459,378+new1,467,100이며 dev22,951bytes다. 같은 OWTrevision79d93d786212f7344586290adb811d4ae6a1762c/docSHAmod9split/문서·선택구간 exactcrosssplit 중복0을 확인했다. Near-duplicate·pretraining overlap·원문 권리 한계는 기존대로 남긴다. E2final/registry/기존 held의 train 편입0이다.

새 행만 약어·initialism·월 약어·시작 문자·길이의 보수적 선택 규칙을 적용했다. 원문·기존 행을 교정하거나 의미 정답 역할을 NN feature에 넣지 않았다. 이 규칙은 문법·의미 완전성을 보장하지 않으며 다양성과 새 행 품질 처리가 함께 바뀌므로 두 효과를 분리한 인과 실험은 아니다.

첫 준비는43개 완전 페이지21,537,525B를 저장한 뒤 HTTP 읽기 시간 초과로 닫혔다. 별도 한정 스트리밍 복구는 저장 페이지를 재사용하고6,114,784B/14.11초/global retry0으로 완료됐다. 직접 표본을 읽다가 `Nov.` 경계 문제를 발견해 월 약어를 보완했고, cache-only 재구성에서9문서가 부족했다. 새 최대2pages/2MiB/45초/no-retry 목적에서 한 페이지572,427B/5.219초로 이를 채웠다. Final 새4,096문서 중4,087개는 캐시,9개는 마지막 페이지에서 왔다. 모든 실패·중간 artifact·실행 builder·짧은 CPU 검사 예문 수정본을 보존했다.

기본 CPU 계약7PASS18.97초와 최종 자료·OS delta4PASS13.96초/실제 worker preflight·독립validate를 확인했다. 약어 예제 초기3PASS, 월 약어 검사에서는 한 테스트 문장이 최소48byte보다 짧아 해당 fixture만 고친 뒤 실패 case1PASS.05초였다. Production 길이 기준은 낮추지 않았다. 이는 자료·실행 계약의 준비 증거이며 GPU 성능 결과와 구분한다.

## 누락된 최종 평가를 수집한 별도 읽기 전용 실행

학습 시간 여유 guard로 평가가 시작되지 않았으므로, 바뀐 partial 가중치에 대해 **새 목적·새 output·새120초/reserve30·max2400calls**의 읽기 전용 평가를 한 번 수행했다. 학습을 자동 재개하거나 남은752attempt/5,680calls를 사용하지 않았다. Source 전체 Adam/scaler/RNG 파일은 유지하고 모든 model parameter를 동결해 eval했다. Optimizer update0/checkpoint write0/denoise0이다.

읽기 전용run은 `runs/owt_partial_assessment_20261028_v23_readonly`, plan SHA `e132b221c4616ab7b7bd57d79d9e9f29de84bb77c4af763f0a8a3d0edf6de935`다. CPU4PASS10.92초/actualpreflight/독립OSvalidate 뒤 ONE 실행했다. Owner23896→42836(23:06:41.317/.348KST), worker28860→38040(23:06:52.170/.190KST)의 생성·parent·command·생존을 ROOT가 관측했다. Worker-owner exit0/COMPLETE/allowned gone,23:07:55KST Python0 확인 후 raw 분석했다.

실제23.382543초/encode40+decoder1896=**1936/max2400 calls**. 이 예산도 CLOSED/미사용464calls·시간을 재사용하지 않는다. Source 파일 SHA와 model 전후 digest `9f31aecd0d5bf663e4bd63b5d124af22029bb376a49b364de1510193b7a8bb89`가 같았다. Worker의 모든 사용 RNG 불변 검사 PASS, ROOT는 source/model digest와 저장 logits·argmax·NLL·rawbyte/EOS·call 합계를 CPU로 재구성했다. 추가 model forward0이다. 학습의 FAILED 기록과 읽기 전용 COMPLETE 기록은 서로 다른 실행 결과다.

같은 관측 dev23,207active byte+EOS symbols:

| 상태 조건 | V22 NLL | V23 partial NLL | V22 보조 byte 정확도 | V23 partial 보조 byte 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력의 상태 | 1.339957 | **1.061039** | 62.8173% | **67.5141%** |
| 다른 입력 상태 | 4.798403 | 4.718358 | 29.1205% | 27.1685% |
| 상태 제거 | 3.810289 | 3.743438 | 37.2991% | 35.8254% |

정상 correct14,578→15,668, EOS253→254/256이다. 보조 정확도 **+4.69686pp**, NLL **-.278918**은 관측 개선이다. 이는 정답 앞부분을 제공한 byte 예측이며 의미 보존율67.51%가 아니다. 정상-교환 NLL 차이3.657319/정상-제거2.682399는 현재 입력별 상태 정보 사용을 뒷받침하나 완성된 의미 상태나 독립적인 추론을 증명하지 않는다. 데이터 확대·품질 규칙·추가 학습·순서·부분 종료 효과를 분리하지 않았다.

## 같은 입력을 상태만으로 재복원한32개 직접 판독

Dev16과 기존 고정 train16을 모두 읽었다. 두 표본 모두 UTF8/EOS16/16/exact0/16/충실한 전체 의미 재현0/16이다. 비맹검 ROOT 정성 판독이며 일반화된 의미 정확도가 아니다. V22에서 맞았던 train의 `The tweets were flagged`가 이번에는 `The tweets were go fand ...`로 바뀌어 **그 부분 명제는 보존되지 않았다**. Eduardo→Educations, Craig Charles→Cravia Child도 같은 인물 이름으로 인정하지 않았다.

| 상태 조건 복원 지표 | V22 dev16 | V23 partial dev16 | V22 train16 | V23 partial train16 |
|---|---:|---:|---:|---:|
| chrF | 22.2914 | 22.3332 | 32.4742 | 27.8974 |
| 단어 F1 | 10.8918% | 12.4290% | 29.5695% | 21.9728% |
| 내용어 F1 | 1.0952% | .6250% | 15.7188% | 9.1171% |
| 정답 시작 UTF8 byte | 2.625 | 2.625 | 12.0000 | 7.0625 |
| byte edit 유사도 | 24.9734% | 25.3930% | 39.0765% | 36.6897% |

Dev 자기 원문 unique top1은12→15/16이지만 표면 유사도와 핵심 관계 충실도는 다르다. 일부 주제·도입부가 남아도 행위·대상·부정·관계가 빠졌다. 불완전한 원문 구간에는 존재하지 않는 완전한 정답 명제를 만들지 않았다. 보조 예측의 개선과 이미 본 문장의 상태 조건 복원 퇴행을 모두 남긴다. 방법 전체의 실패나 완성으로 단정하지 않고, 원래 의미 추출·언어화 학습을 계속 분리 관측한다.

전체 판독과 지표는 읽기 전용run의 `root_all32_state_reconstruction_reading.md`, `root_state_reconstruction_similarity.json`, `root_result.json`, `scripts/review_owt_partial_assessment_v23_cpu.py`에 있다. 반복 관측 dev·설계 선택·작은 표본·비맹검·lexical-not-semantic 한계를 유지한다.

## 다음 구체적 개선 근거

학습 종료 후에만 CPU로 resource guard의 파일시스템 비용을 측정했다(modelcalls0). 각32회 cached 측정에서 V23 출력의 `owned_bytes` 평균3.398ms/disk_usage .0168ms, attempt당 네 번의 검사 합계는 약13.661ms다. V22도14.636ms로 비슷하므로 **이것을 V23 속도 저하의 원인으로 단정하지 않는다**. 다만 절대 반복 비용을 줄일 실용 후보다. 근거는 `research/artifacts/owt_v23_guard_io_profile_20261028_cpu/result.json`이다.

다음은 STOP·시간·callcap은 매번 검사하면서 비싼 전체 파일 순회는 최대1초 간격과 큰 저장 직전에 강제하는 CPU 정책 계약을 검토한다. 독립 OS owner의1초 자원 감시는 유지한다. 새 구현·CPU 실패경로·정확 복구·actualpreflight·독립validate 뒤에만 새 유한 학습을 한다. 기존 sealed worker를 수정하거나 시간 한도를 연장하지 않는다. CPU timing을 GPU 속도 향상 증거로 쓰지 않는다. 모든 모델·AdamRNG·원본·raw·이전 실패는 보존한다.
