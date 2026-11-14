# FDT 문서 데이터 효용 감사 및 다음 소형 비교

## 현재 판단

**학습 데이터와 감독 방식에는 실제 편향이 있고, 아직 이상적인 Base corpus는 없다.** 직전 prompt0/0.35 비교는 prompt 보존 NLL만 개선하고 정답·자유 생성 개선을 보이지 않았다. 이번 후보는 완결 문서에 대한 기본 언어 학습 가능성을 분리해서 확인할 진단용이다. TinyStories만으로 지식·근거 QA·코드·장문 성능을 충족했다고 판단하지 않는다.

## 코드와 원자료에서 확인한 편향

- `work/fdt_phase2/scripts/prepare_fdt_v4_1_retention_t4.py`는 `prepare_fdt_v21_knowledge_scale`의 builder를 재사용한다. 후자의 `add_natural_language` 함수 224–230행은 `len(ids)<560` 문서를 버리고, 뒤쪽192토큰을 답으로 만든다. 자연어 quota는 Wikipedia9,000/FineWeb9,000/TinyStories2,000행이다. quota가 감독 토큰·유일 문서·손실 기여 비율을 보장하지 않는다.
- 같은24K tokenizer로 새 TinyStories 학습 표본2,000개를 재계수하면 **1,954개(97.7%)가560토큰 미만**, 중앙값193이다. 검증 표본500개는464개(92.8%)가560미만이다. 이는 분산된 연속 블록 표본의 결과이며 전 모집단의 정확한 비율은 아니다. 기존 builder는 whitespace 정규화를 먼저 하므로 수치 비교는 원문 길이 기준 진단이다. 짧고 끝이 있는 텍스트를 대부분 제외하는 필터의 문제는 코드와 표본 모두에서 드러난다.
- 현재 자연어 후보19,490행의 supervised EOS0, B 실제 자연어338행의 뒤192토큰만 감독한 사실은 이전 감사에서 검증됐다. EOS를 배우지 않는 중간 crop의 NLL만으로 완결된 문장·문서를 생성할 능력을 판단할 수 없다.

## 새 후보의 실측 계약

원문을 자르지 않고64–510토큰 문서에 실제 source-row 종료를 나타내는 EOS 하나를 붙였다. 첫 토큰과 padding만 손실에서 제외한다. 문장 종결 부호는 단순 필터이며 이야기의 의미적 완결성을 보증하지 않는다.

| split | 유일 문서 | active | supervised | supervised EOS |
|---|---:|---:|---:|---:|
| train | 1,911 | 403,095 | 401,184 | 1,911 |
| development | 222 | 53,383 | 53,161 | 222 |
| sealed test | 229 | 59,292 | 59,063 | 229 |

- 원문2,500개 중 길이 조건111개, 종결부호 조건27개 제외. 실제 학습은 아직 없으며 source epoch/replay0이다. source epoch는 다음 실행의 소비량으로 계산한다.
- 정규화 exact 문서 및8단어 shingle Jaccard≥.8 중복을 split 간 검사했다. 이 기준으로 추가 중복0. 의미적 유사성과 공유 주제까지 제거한 것은 아니다.
- 이전 자연어 crop19,490행과16연속토큰이 겹치는 후보57개를 발견했다. 흔한 문구의 겹침도 포함하므로57개 모두를 누출로 확정하지 않는다. 다음 pilot의 train/dev 선택에서는 이57개를 모두 제외했다. 전체 과거 checkpoint 학습 계보의 겹침은 **UNKNOWN**이다.
- 모든2,362개 저장 행을 원문에서 독립 재토큰화해 IDs/labels/mask/padding/EOS/hash/split exact 중복을 검증: **CPU_PREFLIGHT_PASS**. 원본 prepared_data는 변경하지 않았다.
- 공개 Dataset Viewer 응답25개와 응답 SHA를 보존했다. 모든 응답의 `x-revision`은 `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; 최초 페이지는 동일본문SHA 재조회로 헤더를 보완했다. 초기 manifest의 viewer revision `UNKNOWN`은 `revision_binding_audit.json`이 보완한다. 과거 모델 학습 계보 UNKNOWN은 그대로다.

## 의미 품질 직접 점검

실제 선택된 첫6개 문서를 읽었다. 쿠키를 허락 없이 먹은 아이, TV를 끄지 않는 아이 등의 사건·대화·결과는 이어진다. 반면 `train:1300036`은 새가 준 patch에서 인쇄 이야기로 넘어간 뒤 친구가 더는 patch를 줄 수 없다는 이유 없는 결말로 바뀐다. `train:1800087`은 성장하는 아이가 같은 생일옷을 여러 해 입는 등 현실적 일관성이 약하다. 같은 이름과 반복 문장, 교훈형 결말도 많다. 이는 합성 쉬운 영어의 장단점이며6개 표본만으로 전체 품질률을 추정하지 않는다. 마법 설정 자체는 오류로 처리하지 않는다.

**이 후보는 종료·기본 문맥 학습 진단에는 유용하지만 이상적인 Base 학습 데이터는 아니다.** 장기 curriculum에는 서로 다른 출처의 완결 자연어, evidence가 실제 보이는 QA, 반사실 값 변화, 다양한 복사 표현, 검증된 코드·구조 데이터가 더 필요하다. 이번 실험에 그것을 한꺼번에 섞어 원인 판단을 흐리지 않는다.

## 사전 고정한 다음 실험

동일 parent α=.20, 동일256개 문서·순서·32update·active87,636, fresh Adam, LR1e−6, EOS weight2, prompt 추가손실0, weighted-token-mean. 기존 상수와 모델 구조는 유지한다.

| 조건 | 감독 범위 | supervised | EOS | loss weight mass |
|---|---|---:|---:|---:|
| masked | 문서 마지막192토큰 | 49,152 | 256 | 49,408 |
| full | 첫 토큰 이후 문서 전체 | 87,380 | 256 | 87,636 |

총136,532supervised/최대1,800초(30분) 로컬 GPU. 각 문서1회, replay0; 전체 후보1,911개 중256개(13.40%)를 선택하며256토큰 이상 문서를 택하는 길이 편향을 명시한다. **같은 문서 노출량을 통제하며 supervised token 수는 같지 않다.** 이는 감독 범위 변경의 효과를 묻는 비교다. 이전 pilot과는 문서·LR도 다르므로 이전 결과와의 차이를 이 요인 하나의 효과로 주장하지 않는다.

동일 development32문서 전체 NLL, 기존 fixed52 row/token 보존을16/32step에서 검사한다. full의 dev token NLL이 masked보다0.02 이상 낮고, 보존 gate를 통과하며, 사후 원문 생성의 관련성·일관성·종료가 악화하지 않을 때만 후속 작은 혼합 curriculum 후보로 진행한다. 이 임계값은 작은 진단의 실용적 선별 기준이며 유의성 검정이나 완성 기준이 아니다. 봉인 test229개는 아직 모델 평가에 쓰지 않는다.

P0 안전: 독립 외부 감독, step1/8Ksupervised/50분 이내 recovery, 고정시간·토큰 상한, parent와V20 대비 fixed52 row/token+1%를 재확인하면 전체 실험 중단. clean stop 이외에는 재시작하지 않고 보존한다. 자동 승격 없음. 저비용 에이전트가 실행과 감독을 맡고 본 에이전트는 학습 뒤 판단한다.

## 증거 위치

- `outputs/astra_document_data_candidate_20260926`: 원문 API 응답, 길이/중복 제외, shard, 독립 preflight, revision 증거.
- `outputs/astra_document_objective_pilot_20260926`: 선택 문서, 두 label shard, 사전 plan·승인·preflight.
- `outputs/ASTRA_PROMPT_OBJECTIVE_RESULT_20260926_KR.md`: 직전 비교 결과와 생성 원문 판정.

원천: [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). 공개 metadata상 영어 합성 이야기, CDLA-Sharing1.0. 원본 source 파일과 evidence는 모두 보존한다.
