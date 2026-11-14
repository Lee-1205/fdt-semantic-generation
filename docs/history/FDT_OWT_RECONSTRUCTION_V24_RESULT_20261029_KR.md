# V24 — 같은 입력의 퍼지 의미 추출·재복원 학습 8,189회 추가 갱신

현재 목적은 X→원래 퍼지 앵커의 Z(X)→같은 입력 X̂ 복원이다. 출력 목적 의미 디노이징은 별도 동결했다. 원문 context 우회나 정답 prefix·길이·역할을 상태만의 복원에 주지 않는다. 기존 `generation/free` 파일명은 이 복원 기록이며 자유창작 평가가 아니다.

## 실제 실행과 종료

`fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261029_v24`에서 새 8,192attempt / **8,189update** / AMPskip3(at3042,5083,7534)를 수행했다. 누적55,268attempt / **55,250update**다. Worker549.852884초, owner·workerexit0/COMPLETE, 소유 proxy/native580→18396→23988→42680 모두 종료했다. ROOT가 생성·parent·command·생존을07:32:03KST에 관측했고07:41:28KST Python0을 확인한 뒤 분석했다. 임시 sleepguard도 finally 해제됐다.

새 encode8,232+decoder18,276+denoise0=**26,508/max28,000 calls**였다. 새900초/reserve90초 예산은 CLOSED이며 미사용1,492calls·시간을 재사용하지 않는다. 실제 source누적160,680calls는 이전 별도 읽기 전용1,936calls를 제외한다. 해당 평가를 plan/result에 별도로 연결했으며 합계는162,616calls다. 이전 V23partial의752미시도·미사용5,680calls를 이어 쓴 것이 아니다.

같은8,192train문서/32,768구간을 두 유한 epoch로 학습해65,536행/targetUTF8 **5,852,956bytes**를 소비했다. 새로운 독립 문서나 BPE token 수가 아니다. 두 decoder 손실의 UTF8감독11,705,912/byte+EOS11,836,984symbols, past-prefix mask877,779/5,852,956=.1499719다. 기존4096문서 prefix와 원래64dev문서/256입력·정답·순서는 유지했다. 새 다운로드0/E2final·registry접근0이다.

## 작은 실행 비용 개선

새 `scripts/owt_resource_budget_v24.py`는 STOP·clock·callcap을 매번 검사하고 전체 디스크/owned-file 순회만 monotonic1초 간격으로 제한한다. 시작·복구·큰 저장 전 강제 검사, projected-write 여유, emergency의 STOP 독립 자원 검사, 독립 OSowner1초 감시는 유지했다. 모델·원래 퍼지 구조·손실·AMP·엄격 결정성·학습 자료·추론 경로는 바꾸지 않았다. 이전 sealed 코드는 수정하지 않았다.

CPU7검사21.39초/actualworker CPUpreflight/독립OSvalidate를 통과한 뒤 ONE 실행했다. 실제 자원 전체 검사는528회였고 원래 attempt마다4회라면 학습만32,768회다. V23partial709.13초/7440attempt와 달리 이번에는 전체학습·평가·저장까지549.85초 안에 마쳤다. 학습순서·당시 OS/GPU 부하 등이 달라 **I/O 변경만의 고립된 속도 효과로 단정하지 않는다**.

## 보조 예측과 상태만의 복원 결과

같은 관측 dev23,207byte+EOS symbols에서:

| 조건 | V23partial NLL | V24 NLL | V23partial 보조 byte 정확도 | V24 보조 byte 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력 상태 | 1.061039 | **1.015962** | 67.5141% | **68.9447%** |
| 다른 입력 상태 | 4.718358 | 5.019234 | 27.1685% | 26.1516% |
| 상태 제거 | 3.743438 | 4.024280 | 35.8254% | 34.5973% |

정상 correct15,668→16,000, **+1.43060pp/NLL-.045077**다. EOS보조예측254→252/256도 함께 기록한다. 정상-교환 NLL차이4.003272/정상-제거3.008318은 입력별 상태의 정보 사용을 뒷받침하지만 완성된 의미 공간·독립 추론을 증명하지 않는다. 정답 앞부분을 제공한 보조 정확도68.94%를 의미 보존율로 부르지 않는다.

ROOT는 dev16과 고정 train16 복원문을 모두 읽었다. 둘 다 UTF8/EOS16/16/exact0/전체 의미 충실복원0/확인된 부분명제0이다. `They will probably have ... some opinions`처럼 더 긴 도입부가 남아도 추가된 `no`로 극성이 불명확하고 `bike`가 `baby`로 바뀌었다. `Craig Charlean`은 `Craig Charles`로 인정하지 않았다. `The tweets were got fand`도 flagged 명제를 복구하지 못했다. 원문 자체가 불완전한 구간에는 완전한 정답 명제를 만들어 붙이지 않았다.

| 상태만의 복원 지표 | V23partial dev16 | V24 dev16 | V23partial train16 | V24 train16 |
|---|---:|---:|---:|---:|
| chrF | 22.3332 | 22.7484 | 27.8974 | 29.2000 |
| 단어 F1 | 12.4290% | 11.5348% | 21.9728% | 21.8352% |
| 내용어 F1 | .6250% | 1.0870% | 9.1171% | 13.2230% |
| 정답 시작 UTF8byte | 2.6250 | 3.0625 | 7.0625 | 9.1875 |
| byte edit 유사도 | 25.3930% | 26.1214% | 36.6897% | 37.0443% |

Dev 자기 원문 unique top1은15→12/16으로 내려갔다. 모든 지표가 좋아졌다고 보고하지 않는다. 반복 사용한 dev/작은 표본/비맹검 판독/표면유사도 한계를 유지한다. 초기 절대 학습량과 손실 추세를 고려해 전체 FDT·퍼지 방식의 실패나 extractor/decoder 단독 원인을 단정하지 않는다. 기존 typed 관계 예측의 강한 관측 결과도 별도 역사로 보존한다.

## 저장 상태 검산과 다음 작업

최종 fullstate650,370,993B SHA **e3d4ffc26bd380b8fa88aa2549aff3d1953ee0dcdedf04502d26ff29d5b0e312**, algorithm **77f224ccb321fb5f39dbc80a68d50b1923f3b7d2c030c8b4a1d5a1a09a1e84d8**다. PlanSHA842cbab181ecc161253d38327df01aa402839b6d087932b6ffc0b551464b6ec1. ROOT CPU는 import0/mid4096/final8192의 SHA/safeload/재귀digest, source전체component동일성d32d96e4d4de49b35fbc15c66e433718e186d85e6bee2feef18c28331be1a0cd, 63Adamstep55250/scaler65536tracker658/allRNG/history/cursor/전체ancestry/동결 reasoning-prefix-basis/finite model·Adam을 검산했다. 저장logit23,207위치×3의 argmax/NLL와 실제call수도 독립 재구성했다. 추가 model forward0이며 실제 CUDA 복구 재실행 증거는 아니다.

근거: run의 `root_result.json`, `root_state_reconstruction_similarity.json`, `root_all32_state_reconstruction_reading.md`, `code_snapshot/root_terminal_native_observation.json`, `scripts/review_owt_state_reconstruction_v24_cpu.py`.

Online epoch cleanCE .885142→.730218/corruptedCE1.375318→1.272739/mixedloss1.130230→1.001479는 학습 중 평균으로 고정 final trainNLL이 아니다. 손실과 보조 예측은 계속 낮아지지만 BOS부터의 완전복원은 뒤처진다. 다음은 현재 구조·노출 계약을 유지한 새 유한 학습에서 이 학습곡선과 복원 충실도를 이어 관측한다. 검증만 반복하거나 teacher 정확도를 완료 게이트로 삼지 않는다. 현재 새V25worker/plan/seal/GPU는 없고 모델 프로세스도 없다. 다음 새 학습은 최신 fullAdamRNG를 정확히 불러와 새 목적·유한 order/clock/callcap을 고정한 뒤 진행하며 기존 예산을 연장하지 않는다.

모든 원본·core·typedpartial9/a877·Phase2/archive·이전 weights/AdamRNG/raw/실패는 보존했다. E2final/registry/브라우저·마우스/keepalive/유료cloud/외부게시/사용자원본삭제는 하지 않았다. ROOT직접/에이전트 없음이다.
