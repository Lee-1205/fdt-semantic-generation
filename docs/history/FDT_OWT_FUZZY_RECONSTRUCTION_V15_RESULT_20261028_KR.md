# OWT 퍼지 추출·복원 V15 — 문서 확대 학습과 자유 출력 유사도

사용자 지시대로 학습을 재개하고 자유 복원을 직접 읽어 부분 유사도도 계산했다. 원래 퍼지 앵커를 이용하는 문장→추출 상태→복원을 유지했다. 추론/출력 의미 디노이징은 동결이며 decoder는 추출한 두 fuzzy view만 받는다. 원문 context 우회·새 parameter·anchor 초기화·반복/균등화 벌점은 추가하지 않았다.

V15 실제4,096attempt/4,096update/AMPskip0, AE누적12,288회다. 같은111,485,711parameter/24,150,914trainable 모델로 훈련 문서를1,024→2,048개, 문장을4,096→8,192개로 늘렸다. 기존 개발64문서256문장과 훈련 prefix는 그대로 유지했다.

| 같은 관측 개발 자료 | V14 | V15 |
|---|---:|---:|
| 정답 prefix 기반 byte 정확도 |52.9065%|**58.4177%**|
| 정상 NLL |1.666882|**1.431893**|
| 상태 교환 NLL |3.933796|4.589970|
| 상태 제거 NLL |2.920832|3.263999|
| 자유 정상 chrF2 /100 |20.7868|**21.5155**|
| 자유 정상 단어 F1 |12.2966%|13.4077%|
| 자유 정상 내용어 exact F1 |0.0000%|0.5208%|
| 자유 정상 byte편집 유사도 / 긴 쪽 길이 기준 |26.6058%|27.5626%|
| 자기 원문 chrF 단독1위 /16원문 |8/16|9/16|
| 처음부터 맞는 byte 평균 |2.8125|3.1250|
| 자유 완전 일치·ROOT 충실한 paraphrase 판단 |0/16·0/16|0/16·0/16|
| 자유 UTF8/EOS |16/16|16/16|

정상 teacher 정확도는5.5113pp 증가했고 NLL은.234989 감소했다. 상태교환 비용은2.266913→3.158077nats, 제거 비용은1.253949→1.832106이다. 문장별 추출 정보 사용은 커졌으나 잘못된 상태에서 점수가 내려간 것을 전체 조건의 품질 향상으로 해석하지 않는다.

자유 정상 chrF는 교환16.0301·제거17.5956보다 높다. 자기 원문 식별과 표면 조각은 조금 개선됐다. 하지만 ROOT가 새 정상16개도 전부 읽은 결과, 충실한 의미 동치로 볼 수 있는 문장은 없었다. "He didnocknowly ... U.S."처럼 부정·acknowledge의 철자 및 미국 주제 흔적, "I then ...", "We can asked ..." 같은 시작 흔적이 있으나 원래 행위·관계·개체·부정이 복원되지 않았다. 숫자1999999/2014 반복도 남는다. 단순4word 반복 지표0은 이런 숫자/부분어 반복과 비문을 포착하지 못한다.

chrF/단어F1/편집거리/원문랭킹은 사후 문자열 지표이며 의미 동치 확률이 아니다. 내용어 exactF1은 동의어를 인정하는 지표가 아니므로 직접 읽은 명제 보존 판단을 함께 기록했다. 외부 BERTScore/embedding judge는 실행하지 않았다. 이미 관측한16문서, 비맹검 ROOT 판단, 같은 dev의 반복 사용 한계가 있다. 완전 일치0점만으로 학습을 무효화하지 않고 실제 부분 진전과 남은 충실도 문제를 함께 보존한다.

처음 약3byte에서 오류가 생기고 뒤에 비단어·잘못된 단어가 누적된다. 정답 prefix 학습과 자기 출력 prefix 추론의 차이는 유력한 개선 후보지만, 단독 원인 증명은 아니다. 현재 문장 어댑터의 두 pooled view 및 anchor 사이 공통 동적 value도 표현 압축의 제약이다. 추출 정보의 완전성과 언어화 오류를 아직 완전히 분리해 증명하지 못했으므로 decoder만의 실패나 의미 추출 전체 실패로 단정하지 않는다. 원래 구조화 관계 태스크의 높은 성능은 그대로 보존한다.

## 자료·실제 실행·보존

새 immutable artifact는 `research/artifacts/owt_fuzzy_reconstruction_20261028_v15`다. 새1,024훈련 문서는 저장 source_pages의 미사용37개와 새987개다. 공식 OWTrevision79d93d786212f7344586290adb811d4ae6a1762c/모든 X-revision 일치를 확인했고 새 fetch7,334,876B/11.828초는16MiB/180초 내였다. Train sentence UTF8 bytes728,913=기존364,178+새364,735,dev22,951불변이다. 문서 hash modulo9 분할·exact normalizeddoc/선택sentence 교차 중복0을 유지했다. 근접중복/이전pretraining 겹침은UNKNOWN, 원문 권리·출처 한계도 유지한다. EOS는 선택 문장 경계이며 원문 문서EOF를 주장하지 않는다. E2final/registry 접근0이다.

4유한 epoch231–234,32,768행 노출,2,915,652targetUTF8bytes=기존1,456,712+새1,458,940이다. 기존·새 문서 각각16,384행을 노출했다. V12–V15 AE 전체 누적 target 노출은8,735,412bytes로, BPEtoken이나8.7MB의 서로 다른 원문이 아니다. 최종 고유 선택 문장728,913bytes라는 작은 범위이므로 학습량 충분/일반LM 완성을 주장하지 않는다.

V14 전체 모델/63populatedAdam/scaler/CPU·CUDA·Python·NumPy RNG/모든 nested ancestry·history·order·exposure를 정확히 가져왔다. Import component SHA231e93dbc8716a03c41b9fa588a440b8f9bbe1ece760edc86b7bc5066f2f961d. 최종63Adam은 모두step12288, scaler65536/tracker288이다. reasoning/diffusion/prefix/basis 불변을 확인했다. 온라인 epoch batchCE평균1.3954→1.2518→1.1214→1.0261이며 고정 최종 모델 trainNLL과는 다르다.

6focusedCPU delta9.28초 PASS/실제 worker CPUpreflight·독립 OSvalidate PASS 후 ONE실행했다. 새900초/reserve90/callcap10,500/output6GiB 안에서 실제257.770초,encode4,132+decoder5,492+denoise0=9,624calls였다. 예산은CLOSED, 남은876call/시간은 재사용하지 않는다. 새로운 allocation과 기존 누적 호출을 구분했다. 저장된 같은 V14기준 결과를 재사용해 추가 초기 GPUbaseline은0이다.

Owner8744→25908 생성16:46:40.759/.766KST, worker10048→19024 생성16:46:49.063/.079를 ROOT가16:46:59에 parent/command/생존과 함께 확인했다. Worker-owner exit0/COMPLETE,allowned/proxygone,16:51:12KST nativePython0/Cfree55,483,494,400B 확인 후 분석했다. 임시 sleepguard는 종료시 해제됐다. 현재 학습/추론 프로세스는 없다.

Plan SHA `d2dc77692882b2ddc94c12315fbfcac705a7397fc68df698e96709393b71cc07`. Final641,253,169B SHA `e699f7bfed40095476914d2462eca5bb18d37399866c83c9fed5d13c68ac0c7c`, algorithm `33a85db62c5d1a9b6a8a42858d170f5bead8a15269956b1b2a186cc6550a382a`다. Import0/mid2048/final4096 fullstates를 모두 보존했다. Root가 전체 source-import component, final SHA/safeload/digest·동결 tensor·63Adam/scaler·4epoch cursor/order/노출·호출을 CPU 재구성했다.23,207기호×3조건 저장 logits의 NLL/argmax는 worker와 일치했다. 추가 model forward0이다.

근거는 run의 `root_result.json`, `root_free_similarity.json`, `root_all16_free_reading.md`, native JSON 및 `scripts/review_owt_fuzzy_reconstruction_v15_20261028_cpu.py`다. 전체16개 V14판단 및 선행연구는 `outputs/FDT_OWT_FREE_SIMILARITY_REVIEW_20261028_KR.md`에 있다. 다음도 추출·복원 학습을 이어가며 의미 디노이징은 분리한다.
