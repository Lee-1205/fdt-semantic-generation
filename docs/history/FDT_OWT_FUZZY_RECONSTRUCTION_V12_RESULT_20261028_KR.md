# OpenWebText 퍼지 의미 추출·문장 복원 — 첫 실제 학습

최신 사람 지시대로 **문장→기존 퍼지 앵커 기반 추출→추출 상태만을 이용한 문장 복원**을 먼저 학습했다. 출력 의미를 만드는 추론/디노이징은 이번 학습에서 사용하거나 갱신하지 않았다. ROOT가 에이전트 없이 코드·자료·학습·OS감독·결과 재구성을 직접 수행했다.

## 실제 학습과 결과

2,048시도/2,048업데이트/AMP skip0,8유한 epoch을 완료했다. 훈련512문서에서 뽑은2,048문장을8회 사용해16,384행/정답 UTF8 **1,449,624bytes**를 노출했다. 같은 문장 반복을 새 문서나 LM token 수로 부르지 않는다.

검증은 훈련과 문서가 다른64문서/256문장/23,207byte·문장경계 기호다. 다음 정확도는 정답 prefix를 주는 teacher-forced 바이트 정확도이며 자유 문장 정확도가 아니다.

| 추출 상태 조건 | 초기 NLL | 최종 NLL | 초기 byte 정확도 | 최종 byte 정확도 |
|---|---:|---:|---:|---:|
| 해당 문장의 정상 상태 |5.176580|**1.976292**|8.6224%|**43.1465%**|
| 다른 문장 상태로 교환 |5.171753|2.366906|8.7043%|34.4465%|
| 추출 상태 제거 |5.127145|2.231898|8.2044%|36.6269%|

최종 정상 상태는 교환 상태보다 NLL이0.390614nats 낮고 정확도가8.7000pp 높다. 제거 상태보다도 NLL이0.255606 낮다. 초기에는 이 차이가 거의 없었다. 복원기가 추출 상태의 문장별 정보를 더 이용하게 됐다는 좁은 근거다. 이 통제에서 교환은 같은 batch의 서로 다른 문서 사이에서 이루어졌고, 전체 검증 표본은64문서에 묶여 있으므로256독립 문서 실험이 아니다.

서로 다른 검증16문서에서 한 문장씩 골라 자유 복원한 결과는 정상/교환/제거 모두 UTF8·EOS16/16, **원문 완전 일치0/16**이다. 정상·교환 출력은 각각16개로 달랐고 제거 출력은1개로 같았다. 정상 출력에도 단어 비슷한 문자열과 잘못된 문장·내용이 남아 있다. 반복4word 비율0이나 정상 UTF8만으로 복원이 완료됐다고 하지 않는다.

이는 초기 학습에서 정보 보존과 상태 사용이 개선된 결과다. 현재 작은 훈련량과 범위를 고려해 전체 모델·퍼지 방식의 실패로 단언하지 않는다. 다음 학습도 이 추출·복원 단계에서 이어간다.

## 퍼지 경로와 분리 경계

원래 typed `FDT100`의 `Router`/anchors/prior/value/condition 모듈을 복사해 사용했다. `mu`는 top-k sigmoid의 **비배타적 퍼지 소속도**, `rho`는 검색을 위한 정규화 가중치다. 두 의미 뷰의 합성은 기존 probabilistic-sum `1−∏(1−mu)`를 사용한다. 이를 softmax 클래스 확률과 동일시하지 않는다.

자연 문장은 기존 language encoder에서 두 뷰로 모으고, 원래 fuzzy routing/anchor prior 및 동적 값 결합을 거쳐 추출한 두 fuzzy view만 디코더에 전달한다. 원문 byte/context를 디코더에 직접 주는 경로는 없다. 역할을 수동 정답으로 넣지 않고 해당 adapter의 role 입력은0으로 둔다. 원래 structured-fact encoder의 전체 회로를 자연 문장에 그대로 실행한 것과는 구분한다.

Source는 typed partial9와 V9 language·기존34decoder tensor다. V9의 byte-index address prior와 extra memory-position table은 추상 퍼지 memory에 사용하지 않는다. 원래 source 파일과 전체 Adam/scaler/RNG는 그대로 보존했고, 새로 결합한 AE에는 fresh AdamW/scaler/RNG를 사용했다.

전체111,485,711parameter 중 **24,150,914개**를 학습했다. Language 뒤2block+pool, fuzzy routing/anchor/value/condition, 복원 디코더가 대상이다. 나머지 원래 추론/디노이징·초기 language prefix는 동결됐고 저장 tensor 비교에서도 불변이다. 실제 denoise 호출은0이다.

검증 표본에서 양의 소속도를 가진 앵커는64개 중30→15개로 좁아졌다. 뷰당 active top-k는8개, 소속도 합의 평균은1.5569→2.1715다. 의미가 붕괴했다고 단정할 수는 없지만, 앵커 사용이 집중된 관측은 보존한다. 단순히 앵커를 균등하게 쓰게 하면 의미가 좋아진다는 보장도 없으므로, 다음 문서 확장 학습에서 복원·상태 의존성과 함께 추적한다.

## 자료와 출처

사용자가 선택한 [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/)와 [공식 HF dataset](https://huggingface.co/datasets/Skylion007/openwebtext)을 확인했다. Revision은 `79d93d786212f7344586290adb811d4ae6a1762c`로 고정했고, rows API의 `x-revision`도 각 page에서 일치했다. 로컬 OWT가 없어 **4,244,513bytes/7.203초**의 제한된 원문·메타데이터를 받았다. 전체 corpus 다운로드는 하지 않았다.

정규화 문서 SHA의 modulo9로 prospective train/dev를 나누고, 각 문서에서48–128UTF8byte의 문장4개를 hash 순으로 선택했다. 결과는 train512/dev64문서, 문장2048/256, 물리 문장 UTF8181,203/22,951bytes다. 문서 hash와 선택 문장 exact hash의 cross-split 중복은0이다. Source의 중복 제거 외에 추가 near-duplicate 전체 검사는 하지 않았고 과거 모델 pretraining과의 겹침은 UNKNOWN이다. EOS는 선택된 문장 경계이며 원문 문서 EOF라고 주장하지 않는다.

문장 분할·공백 정규화·길이 선택·문서 분할과 원문을 모두 보존했다. 저자는 packaging을 CC0로 배포하면서 underlying text의 소유권을 주장하지 않는다고 밝힌다. 해당 README·고지·출처를 저장했고 사용자 지시에 따른 로컬 연구로 진행했으며, underlying web text의 일괄 권리 확보나 외부 게시 승인을 주장하지 않는다. E2final/registry/기존 held 자료는 열지 않았다.

## 검증·실제 실행 근거

- 8focusedCPU 검사: 원래 fuzzy membership/소속도 합성, state-only decoder, 정확한 model·Adam·enabled scaler·모든 RNG·다음 update 복구, 마지막7행, AMP skip, CPU autocast, 실제 새 schema의 postPopen 자연·예외·UNCONFIRMED 처리. 실제 worker CPU preflight와 독립 OS validate도 통과했다.
- ONE 실행: owner29640→32624 생성15:34:15.267/.274KST, worker30248→35396 생성15:34:21.892/.907KST. ROOT가 생성/parent/command/생존을 직접 확인했다. Worker/owner exit0/COMPLETE/allowned/proxygone을 확인하고15:36:58KST Python0을 확인한 뒤 분석했다. 임시 sleepguard는 worker finally에서 해제된다.
- 실제142.7528초, encode2116+decoder3534+denoise0=**5650calls/max6500**. 새900초/reserve90/output6GiB 예산은CLOSED이며 남은850call/시간은 재사용하지 않는다. 종료 후 Cfree약62.9GB다.
- 최종639,626,417B SHA `1fa94def8635f40e5aac1f12b82176c39118cd1abde65f801323f38438d585da`, algorithm `0475e4e8b36b5d14b292d71e96889a830e7de6de2e12d01cea511d0eebdf35dd`.
- Plan SHA `4c37cdb5d78e46c3c5b258453b2586252f8cfdfa81802213a5f2d6d84d74d0cd`.
- Root가 초기/최종 저장 logits의 모든23,207기호×3조건×2시점, 원문 bytes·EOS·PAD, fuzzy degree, raw 생성, 실제 batch 종료 호출 수와 최종 SHA/safeload/digest/동결 tensor/8epoch 순서/63Adam state step2048/scaler2048 tracker48을 CPU 재구성했다. 저장 FP16 logits의 argmax도 이번에는 worker 집계와 모두 일치했다. 추가 모델 forward0이다.

근거는 `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261028_v12/root_result.json`과 native 관측, raw logits/생성/fullstates다. 원래 core·typed9/128·V1–V9·모든 이전 AdamRNG/원문/실패는 보존했다. 이번 사람의 단계 분리·OWT 지시 전 준비했던 V11 비교 도메인3단계 초안도 보존했으며, **봉인·GPU 실행은 없었다.**

## 의미 추출 정답에 대한 이번 계약

보편적 의미 공간에 하나의 정답 좌표를 주는 방식은 사용하지 않았다. 이 사이클에는 입력 문장 자체라는 명확한 복원 target이 있다. 추출 상태만으로 얼마나 복원되는지, 상태를 교환·제거하면 무엇이 변하는지를 측정한다. 이것은 의미 추출의 정보 보존과 사용을 검증하는 출발점이다. 표현·어순의 기억만으로도 일부 복원이 가능하므로 의미 동치·관계·부정의 정확성이 자동으로 증명되지는 않는다. 원문 의미를 조건으로 출력할 다른 의미를 만드는 학습은 **별도 후속 디노이징 단계**로 유지한다.
