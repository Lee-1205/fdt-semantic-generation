# V5 CPU 자료·언어학습 후보 준비 — GPU 학습 결과 아님

Root가 직접 두 자료 계약과 실제 언어층을 학습할 후보를 작성했다. 새 GPU 학습·추론, 실제 optimizer trial, 실행 worker/OS owner, 봉인 plan은 아직 없다. 실제 학습된 최신 모델은 V4이며 기존 문장 22/32, 새 wording 11/32 결과와 퇴행을 그대로 보존한다. 모든 이전 terminal/예산은 닫힌 상태다.

## 로컬 문서 출처와 분할

지정된 세 역사 감사와 실제 manifest를 좁게 확인했다. TinyStories 기존 원문 train1,911/dev222 문서와 기존 분할·원문 응답·revision 근거가 남아 있었다. 반면 옛 SQuAD QA1,753개 중 개별 의미 검증이 끝난 것은96개뿐이며 나머지1,657개를 검증 정답으로 취급할 수 없다. 이번 corpus는 그 QA/모델 보고서/평가 정답을 포함하지 않는다. FineWeb 및 기타 지식 소스는 개별 문서 검토가 끝나지 않아 이번 후보에 넣지 않았다.

TinyStories 고정 revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`의 [공식 dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md)는 CDLA-Sharing-1.0을 표시한다. [공식 라이선스](https://cdla.dev/sharing-1-0/)의 Use 허용과 publication 조건을 확인하고, 원문·출처·revision·attribution·변경 표시·라이선스 원문/링크를 보존하는 로컬 계산 연구 범위로 검토했다. 외부 게시 허가나 다른 자료의 포괄적 권리 확인으로 확대하지 않는다. README와 라이선스 metadata156,998bytes만 추가 수집했고 새 corpus 다운로드는0이다.

새 immutable artifact는 `fdt_local_102m_20261026/research/artifacts/text_story_continuation_20261028_v5_cpu`다. 기존 train에서512편, 기존 dev에서64편을 정규화 hash 순으로 선택했다. 기존 과거 crop과16토큰 겹침이 보고된 train38/dev5문서와 보수적인 mojibake 표지가 있는 train114/dev13문서를 제외했다. 표지 검사는 완전한 품질 검증이 아니다. 새 선택 문서 간 exact/정규화 중복0, train-dev 8단어 shingle Jaccard≥.8쌍0이다. 과거 모델 전체 사전학습 노출 여부는 UNKNOWN이며 기존 dev는 이미 관측한 자료다.

| 구분 | train | development |
|---|---:|---:|
| 선택 원문 문서 | 512 | 64 |
| 물리적 원문 UTF8 bytes | 417,443 | 57,182 |
| 이어쓰기 row | 3,271 | 444 |
| 반복 입력 문맥 bytes | 706,304 | 97,280 |
| 감독 target bytes | 384,675 | 53,086 |
| 실제 문서 끝 EOS | 512 | 64 |

입력은 직전 최대256bytes, target은 최대128bytes와 실제 문서 끝 EOS 하나다. 최초 최대64bytes는 첫 문맥으로 사용한다. 한 문서의 target byte 구간은 서로 겹치지 않고 문서 끝까지 이어지며, 인위적인 crop 끝에는 EOS를 넣지 않는다. 한국어를 포함한 CPU fixture에서 UTF8 codepoint 경계를 검사했다. 특징 파일은 id/prompt만 담고 target/offset/문서 정보는 분리했다. Byte symbol 수를 기존24K BPE token 수나 독립 문서 수로 부르지 않는다. 이 자료는 영어 합성 짧은 이야기이며 사실 QA·한영 범용 corpus가 아니다.

기존 `sealed_test_records.jsonl`/shard는 열지 않았다. 원문 동일성 검사에 사용한 과거 validation API 응답 페이지에는 미선택 역사 test 행도 함께 들어 있다. 그 행을 새 자료로 내보내거나 학습·평가에 쓰지 않았다. E2 final 및 그 registry에는 접근하지 않았다.

## 질문 표현·위치 통제 자료

별도 immutable artifact `research/artifacts/text_surface_alignment_20261028_v5_cpu`는 train2,048행/64base facts/1,024역전짝과 dev_pair/dev_surface/dev_position 각각64행이다. 같은 사실·언어·이름·답변을 공유하는 상관 view이며2,048개 독립 graph가 아니다. 훈련 질문 가족0–3, 표현 통제 가족4–5를 분리하고 neutral C/H distractor와 앞뒤 배치·padding 길이를 바꿨다. 최대 prompt152bytes이며 query 주어 위치0–111bytes에 걸친다.

기존 pair+language train/dev 경계를 유지했고 기존 관측 dev의 실제 prompt가 새 train에 들어간 수는0이다. 표현/위치 통제는 훈련 사실8개를 의도적으로 공유하며 새로운 독립 blind benchmark가 아니다. 각 행의 방향·부정은 독립 유한 수치 모델에서 equality까지 열거해 검증했다. Gold span/역할/가족/답변은 target에만 있으며 NN feature는 id/prompt뿐이다. 한 유한 epoch2,048행 순서/역전짝의 batch8 유지/JSON cursor nextbatch 복원도 검사했다. 아직 이 자료를 모델에 학습하지 않았다.

## 실제 언어층을 학습할 CPU 후보

새 `research/text_language_continuation_v5.py`는 원문 byte context→VP16step 자유 제안→causal byte decoder 경로다. 언어 encoder의9개 block 중 마지막2개(index7/8,14,158,848parameters)를 autograd에 실제 연결하며 앞7개/embedding은 동결한다. Prompt projection, VP time/refiner/velocity/output positions와 decoder도 학습 대상으로 둔다. 후보100,928,002parameters/학습 대상51,173,122다. 이것은 아직 학습 전 후보의 수치이며 현재 V4 능력 향상이 아니다.

손실은 VP vMSE+.25cleanMSE+.5byteCE+AR byteCE다. VP 감독 손실에만 noisy gold target을 사용한다. AR의 VP 조건은 별도 noise에서 모델이 자유 rollout한 값이며 target/factor/span을 받지 않는다. AR loss는 원문 context를 통해 언어 마지막2개 층으로도 전달된다. 자유 rollout 자체는 detach하여 메모리와 gradient 계약을 분명히 했다. Decoder teacher forcing은 이전 target bytes만 사용하며 미래 byte 접근이 금지된다. Padding은 감독에서 제외한다.

실제 V4 최종 fullstate592,230,431B를 CPU safeload한 뒤 VP177keys와 decoder34keys를 복사했다. 길이40→129의 두 position table만 명시적 linear interpolation을 사용했다. 다른 모든 복사 tensor는 부모와 정확히 같고 부모 model digest는 전후 동일했다. 옛 semantic reader나 Adam trajectory는 가져오지 않는다. 실제 부모 forward0/CUDA initialized false다. 기존 core SHA도 일치했다.

자료10CPU검사와 tiny 후보5CPU검사가 통과했다. 마지막 언어층 gradient, 동결 prefix 무gradient, gold 변경 시 자유 VP 제안 불변, causal prefix, 호출 수19/microbatch, 엄격 복사와 tiny model+Adam의 정확한 다음 update를 확인했다. 이는 전체 worker/scaler/모든RNG/cursor 복구나 실제 CUDA 복구 시험을 대신하지 않는다. Root가 저장2,240표현행·3,715이어쓰기행과576원문을 독립 재구성했다. 실제 모델 능력 평가나 GPU 향상 주장은0이다.

## 근거와 다음 작업

- 표현 manifest FILE SHA `f65237022718a00cac2ea34f7f368055a1cc58c2261800eac0541cd81a2b4977`
- 문서 manifest FILE SHA `5b0ff59fd3985ab825085d0f3e4f5f7312baf5a22b137b62ec4b0b23f66f3f6e`
- 후보 code SHA `78dcacd8b5f09d4e8d32da855a67824d72dac0b523a2e8bfa036568b5602117f`
- Root review FILE SHA `2d9a7287ec138a1d8adf0ec7dc2b0e35281827899a92399745bffa190df16dc0`
- Root review: `research/artifacts/text_language_candidate_20261028_v5_cpu/root_cpu_review.json`
- 독립 재구성: `scripts/review_text_data_and_candidate_v5_20261028_cpu.py`

다음은 `FDT_LANGUAGE_CONTINUATION_NEXT_CONTRACT_20261028_KR.md`의 실제 학습 driver/전체 상태 복구/독립 OS owner를 Root가 구현·검토하는 일이다. CPU 자료/후보를 다시 만들거나 옛 V4 예산을 쓰지 않는다. 하드웨어 승인은 이미 유효하다. 현재 GPU 실행 승인 질문이 필요한 상태가 아니라 실제 worker와 복구·비용 검토가 남은 상태다.
