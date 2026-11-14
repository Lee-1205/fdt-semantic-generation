# FDT Codex 로컬 실행·연구 인계 프롬프트

작성·자료 재확인: 2026-10-26. 이 파일 전체를 로컬 Codex에 전달한다. 동봉 자료가 있으면 같은 작업 폴더에서 시작한다. 이 문서는 작업 지시와 근거 인계이며, 로컬 이식이나 새 GPU 학습이 이미 완료됐다는 보고서가 아니다.

---

너는 내 FDT 연구의 로컬 구현·검증·실험 준비를 맡은 Codex다. 이 프롬프트, 동봉 원본, 접근 가능한 현재 채팅과 관련 대화를 읽고 실제 실행 가능한 프로젝트를 완성해라. 한국어로 설명한다. 계획 설명에서 멈추지 말고 가능한 자료 확인, 코드 이식, 수정, 검사, 짧은 GPU 동작 검증, 문서화를 끝까지 진행한다.

## 1. 이번 요청의 의미와 자율 진행 범위

- **Colab을 실행 기반에서 제거한다.** 사용자 PC의 로컬 NVIDIA CUDA GPU에서 터미널 명령으로 실행한다. Colab 계정·노트북·Drive 마운트·클라우드 런타임이 없어도 새 프로젝트의 설치, 학습, 평가, 저장, 재개가 가능해야 한다.
- “코랩 내용을 지운다”는 새 실행 경로에서 Colab 전용 의존성과 안내를 제거하라는 뜻이다. 원본 notebook, Drive 파일, 기존 코드, 가중치, 데이터, 보고서, 이전 실험은 보존한다. 별도 보존 디렉터리와 새 로컬 버전에서 작업한다. 실제 자료의 영구 삭제는 이번 작업에 포함하지 않는다.
- 과거의 “T4/Colab 전용” 실행 제한은 **새 로컬 분기에서 변경**한다. 기존 T4 실험의 계약은 그대로 보존한다. 로컬 GPU를 T4로 가장하거나 기존 체크포인트의 GPU·환경 fingerprint를 고쳐 strict resume를 통과시키지 않는다.
- 일상적인 구현 선택, 파일 분리, 가상환경 구성, 의존성 설치, 버그 수정, 검사 재실행은 자율적으로 한다. 기존 시스템 Python·드라이버를 제거하거나 관리자 설정을 변경하지 않는다. 별도 `.venv`를 쓴다.
- 이 프롬프트가 허용하는 실제 GPU 실행은 설치 확인용 tensor 검사와 별도 smoke run이다. **모든 smoke run 합계 optimizer 시도 최대 5회, GPU 실행 시간 합계 최대 10분**으로 제한하고 실패 시 무한 재시도하지 않는다. 성공 update와 AMP skip을 구분한다. 작은 smoke 모델과 실제 102M 모델을 구분해 기록한다. 실제 102M 검증은 최소 한 번 포함하되 메모리·CUDA 조건이 안 되면 구체적인 차단 원인을 남긴다.
- 장기 학습과 연구 실험 전체 실행은 준비·재로드 검증을 끝낸 뒤 내가 “학습 시작”을 지시하면 수행한다. 과거의 장기 학습 승인 조건을 이번 “프롬프트 작성/로컬 실행 준비” 요청만으로 해제하지 않는다. 이미 같은 세션에서 내가 명시적으로 시작을 승인했다면 다시 묻지 않는다.
- CPU는 코드 점검, 작은 대수·프로토콜·직렬화 검사, 데이터 검증에 사용할 수 있다. CPU로 모델 학습을 대신하지 않는다. CUDA가 없으면 CPU fallback으로 넘어가지 않는다. 체크포인트의 CPU 로딩·메타데이터 검사는 CPU 학습과 다르다.
- 새 유료 구독·클라우드 GPU·외부 서버·공개 저장소 업로드·다른 사람과의 공유는 만들지 않는다. 인증 우회와 서비스 제한 회피를 하지 않는다. 자료를 외부 검색어에 넣지 않는다.
- 중단된 타이머·주기 점검을 다시 켜거나 새 자동화를 만들지 않는다. 실행 중인 GPU 프로세스나 상시 감시가 있다고 추정하지 않는다.

## 2. 먼저 읽을 자료와 접근 한계

우선 동봉 `sources/`와 `original_source_bundle/`를 읽고, 필요하면 연결된 파일 검색·Drive·대화 검색을 사용한다. 같은 계정이라는 이유로 Codex가 모든 ChatGPT 대화·Drive·Library에 접근할 수 있다고 가정하지 않는다. 실제 읽은 대화 제목·문서 버전·범위만 기록한다. 현재 채팅과 “모델 구조 설명”, “FDT 상세분석”, FDT 손실함수 관련 대화가 접근되면 확인한다. 접근하지 못한 대화는 읽었다고 쓰지 않는다. 동봉 자료로 가능한 작업은 계속하고 빠진 자료만 정확히 남긴다. 비밀번호·쿠키·토큰을 요구하지 않는다.

| 우선 자료 | 용도 |
|---|---|
| `FDT_Codex_Local_Prompt_2026_10_26.md` | 이번 로컬 전환 지시. 이전 인계와 충돌할 때 이번 범위를 우선 |
| `sources/FDT_vNext_Formal_Architecture_Spec_KR.md` | Library 저장 버전 6, 본문 설계 v1.1, 2026-10-03. 65개 수식·접근 권한·개발 단계 |
| `sources/FDT_Integrated_Research_Paper_2026-10-06_READABLE.pdf` | 개정 통합 연구논문. 2026-10-06까지의 근거 구분과 연구 계보 |
| `sources/FDT_Code_Audit_2026_10_06_v2.md` | 운영 결함 수정, 29개 fixture 검사, 미검증 항목 |
| `sources/FDT100_training_status.txt` | 확장자가 txt여도 내용은 JSON. 과거 상태 원문 |
| `sources/FDT100_experiment_summary.json` | 과거 모델별 평가·Macro F1·confusion·직접/전이 세부값 |
| `sources/FDT_Codex_Handover_2026_10_07.md` | 이전 인계. 이 문서의 Colab/T4 지시는 이번 로컬 전환 범위에 맞춰 대체 |
| `sources/fdt_data_mix_evidence_20261001.json` | 다른 장기 언어모델 계열의 과거 노출 근거와 미실행 제안 |
| `sources/fdt_data_mix_revision_20261001.yaml` | 90% 새 corpus/10% replay 등의 별도 미실행 계획. 102M 관계 과제와 혼합 금지 |
| `original_source_bundle/FDT_T4_100M_Durable_Recovery_v2.zip` | 원본 ZIP. 압축을 안전하게 풀고 원본을 실행 전에 읽기 |
| `SHA256SUMS.json` | 이 인계 묶음의 파일별 실제 바이트 해시. 모델 체크포인트 완료 증거와는 다름 |

원본 위치:

- 연구 폴더: https://drive.google.com/drive/folders/1cH9Ha58Uc3HTHhp9BV9yJpvgE84-ABil
- recovery v2 ZIP: https://drive.google.com/file/d/1Frrd8kinJ2518nTLrjxG6cy3V3st2Vbt/view
- 감사 보고서: https://drive.google.com/file/d/1xo0Xd-xEbSCzJTN9_Ypk5XRi26o3KPiW/view
- 상태 txt 기존 ID: `1yhpmWHTs5nTbBc_Ef79cjVUAaZfvb5r5`
- 상태 json 기존 ID: `1v245W9TUVPNZLkhWc5Lq2hHCzxehJmer`
- 실험 요약 기존 ID: `1-ewSIYs5sIG-KHZABMo1PP7qRe7c55V8`
- 기존 실행 notebook ID: `1fgS9vEpKk9lfo2ktgK1CtKiXjBzHKZMj`. 보존 자료이며 새 실행에 Colab을 요구하지 않는다.
- 설계 명세 원본: https://chatgpt.com/library?file_id=libfile_2fcf34cff64c81919fb1cd24c6abaa00&account_id=personal
- 논문 원본: https://chatgpt.com/library?file_id=libfile_e950e8ef08548191b0131a99cafcb79d&account_id=personal

Drive에 접근할 때는 폴더 안 파일명·ID·수정 시간·크기를 먼저 확인하고 중복 이름·교체된 ID를 처리한다. 기존 ID 하나만 영구적으로 신뢰하지 않는다. ZIP 경로 탈출과 예상 밖 실행 파일을 검사한 뒤 새 디렉터리에 푼다. notebook 전체를 실행하지 말고 필요한 Python 소스를 정적으로 추출한다.

2026-10-26 받은 v2 ZIP의 SHA-256은 `57e854be4e946e966ab2165df350c81538a9f322f7d4c14329422f83883d09ba`다. ZIP 해시와 아래 원본 source 묶음 해시는 다른 값이다. 각각 별도로 검사한다.

## 3. 확정된 FDT의 중심

FDT의 목표는 자연어를 학습된 퍼지 앵커의 수치적 소속도와 문맥 값으로 표현하고, 역할·방향을 가진 관계 연산을 수행하며, 의미 조건을 이용한 잠재 diffusion으로 내부 의미 상태를 반복 정제한 뒤 자연어를 표현하는 것이다. 앵커는 사람이 개념 이름을 하나씩 붙인 사전이 아니다.

처리 경로는 다음과 같다.

1. 외부 입력이 있으면 언어 인코더·local attention으로 의미 조건과 문맥 기억을 구성한다.
2. 외부 입력이 없으면 학습 semantic prior, 허용된 기억, 기존 목표 또는 NULL 목표로 초기화한다.
3. 두 경로는 Persistent Fuzzy Memory / Contextual Fuzzy Memory / Exact Memory의 고정 snapshot을 읽어 같은 Semantic Working State로 합류한다.
4. 관계 연산과 Fuzzy-Guided latent Diffusion을 수행하고 근거·모순·불확실성을 검사한다.
5. 불변 Semantic Commit을 만든 뒤 Language Realizer가 읽기 전용으로 자연어를 생성한다.

유지할 계약:

- 내부 추론·디노이징 동안 자연어 토큰을 생성하지 않는다. 자연어 실현은 Commit 이후다.
- 소속도 `μ`는 겹치는 의미 활성도다. 진실 확률, 근거 인증, 개체 identity가 아니다. `ρ`는 혼합 주소, `γ`는 활성도이며 서로 구분한다.
- 의미 유사성과 exact identity를 분리한다. entity/role/time/scope binding과 관계 방향을 보존한다. 같은 이름·같은 앵커 활성만으로 동일 개체로 합치지 않는다.
- 기억은 읽기 권한과 수명을 가진다. 내부 가설을 검증 없이 영구 사실로 저장하지 않는다. Writer의 변경은 다음 episode에만 반영한다.
- Realizer는 Commit만 읽고 원래 입력 hidden state나 live memory를 우회 조회하지 않는다. 출력 스타일·seed 변경이 Commit의 사실·근거를 바꾸면 안 된다.
- `stop_gradient`나 frozen dataclass만으로 deep immutability를 증명하지 않는다. 중첩 컨테이너·tensor alias·공유 소유자까지 검사한다.
- HardVerifier는 명시된 출처·typed rule 범위의 검증이다. 열린 자연어 전체의 진실 oracle라고 부르지 않는다.
- 원문 숫자·식별자·UTF-8 bytes와 provenance를 보존한다. 허용된 span만 복사하고 정규화로 원문을 바꾸지 않는다.
- 선행연구와 겹친다는 이유만으로 중심 구조를 폐기하지 않는다. 신규성·효율·성능 우위는 비교 실험으로 판단한다.

## 4. vNext 설계와 현재 실행 코드의 구분

현재 코드의 약 102M 관계 adapter를 vNext 전체 구현이라고 보고하지 않는다.

| 항목 | vNext v1.1 설계 계약 | 현재 102M adapter 원본 |
|---|---|---|
| 언어 모듈 | 20층, 폭 1216, 19 heads, FFN 3264, local window 64 | 9층, 폭 768, 12 heads, FFN 2048, window 48 |
| 앵커 | 256개, key 256, Top-8 | 64개, key 128, Top-8, 의미 view 2개 |
| 의미 상태 | 16×512 | 최대 query 슬롯 16×512 |
| 관계 | 잠재 관계 8개+NULL, factor rank 32 | 관계 8개, rank 8 |
| Refiner | 공유 4블록, 8 heads, FFN 1408 | 6블록, 8 heads, FFN 1536 |
| 기억·근거 | 문서 bank 최대8, Exact 활성16, seen160, binding128, claim64 | 최대 사실12·개체8의 제한된 관계 과제 |
| 출력 | 불변 Commit, Claim/Proof, copy permit, tokenizer/KV 계약 | 제한된 구조적 결과·템플릿 runtime. 범용 생성 모델 아님 |

vNext에서 확정한 추가 내용도 보존한다.

- canonical anchor 주소는 공유하되 조회 view는 다를 수 있다. 좌표 공유가 학습 없이 의미 정렬을 보장하지 않는다.
- 기억 numerator/mass의 기준은 float64, 모델 인터페이스는 float32다. 현재 adapter의 FP32 구현을 동일 구현이라고 부르지 않는다.
- 전체 checkpoint의 representation epoch를 추적한다. anchor 주소가 같아도 value·Claim feature가 달라졌으면 이전 aggregate와 cached embedding을 새 좌표에서 그대로 쓰지 않는다. 원문·인증 이력은 보존하고 재인코딩한다.
- source identity128과 episode-local ConceptRef16을 구분해 결과 참조 최대144를 만든다. source 없는 개념도 표현할 수 있지만 verified가 되지는 않는다.
- Writer의 certificate 확인, CAS, dedup, aggregate 변경은 한 transaction 안에 둔다.
- Refiner context 최대97행, Claim context 최대320행, 결과 claim 최대64·증명 근거 최대8이라는 budget을 검사한다.
- DDIM 기본4단계와 조건부 추가4단계 최대1회, Realizer 총3회 시도와 같은 Commit, fallback에도 동일 token·byte·UTF-8 budget을 적용한다.
- 입력·출력 기본16K 토큰·출력64KiB는 설계 예산이다. 현재 byte 입력 adapter가 실제16K 문서 학습을 했다는 뜻이 아니다.
- 정답 OutcomeCodec은 subject/predicate/object, polarity, time/scope, evidence, status를 구조화하는 향후 인터페이스다. 실제 신경망의 정보 보존과 denoiser 학습 완료는 별도 검증한다.

첫 목표는 현재 adapter의 로컬 실행이다. vNext 전체로 갑자기 확장하거나 다른 수억 파라미터 모델로 대체하지 않는다. 명세 전체 구현은 기능별 새 분기로 계획한다.

## 5. 과거 실행·저장 상태: 최신 조회와 한계

2026-10-26 지정 Drive 폴더 루트를 다시 조회했다. 12개 항목이 보였고 상태·실험 요약은 2026-10-06 수정 정보 그대로였다. 루트 목록에서 새 실험 보고서, recovery 실행 디렉터리, 가중치 완료 marker는 확인되지 않았다. 이를 모든 하위 경로·다른 로컬 PC에도 가중치가 없다는 증명으로 확대하지 않는다.

| 항목 | 확인된 기록 |
|---|---|
| 상태 txt 수정 시각 | 2026-10-06 03:36:09.196 UTC, 2,225 bytes |
| 실험 요약 수정 시각 | 2026-10-06 03:29:27.625 UTC, 55,058 bytes |
| 원본 모델 source SHA-256 | `de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79` |
| 원본 data fingerprint | `e04ca6238eb0774ba51985ed81d254a976ca01b627b6bc469cfb8fc0b48677ee` |
| 파라미터 | 102,237,197 |
| 장치 | Tesla T4, CUDA |
| 기록된 환경 | `torch 2.11.0+cu130`. 과거 파일 값이며 로컬 설치 권장 버전이 아님 |
| 선택 경로 누적 | 2,240 optimizer updates라는 보고 |
| 마지막 실행 | 640 updates / 640 attempts, phase complete, 179.088609338초 |
| 마지막 입력량 | 342,056 UTF-8 bytes. tokenizer token 수로 환산하지 않는다 |
| 과거 peak allocated | 0.8661828041 GiB. 이 실행의 보고값이며 전체·reserved·로컬 요구량이 아님 |
| Drive 가중치 보존 | `weights_saved_to_drive=false` |

ZIP의 `original_source/model.py`, `data.py`, `train.py`, `runtime.py`에 대해 원본 `source_digest()`의 순서대로 파일명 bytes+파일 bytes를 SHA-256에 넣으면 위 source 해시와 일치한다. 2026-10-26 실제 ZIP bytes로 이 일치를 확인했다. 파일 하나의 해시나 ZIP 해시로 대체하지 않는다.

원본 `role_experiment.py`는 이전 source `11e77fa0dd7b2cc9339f19e53378a77f775a6d9efcb8a608aa22ee9beb1e9a8a`의 완료1440회 parent를 확인하고 optimizer를 새로 만드는 명시적 migration 경로를 갖는다. 이 경로는 strict resume가 아니다. 상태의 누적2240이라는 숫자만으로 각 분기·parent의 계보를 산술 추정하지 말고 실제 manifest·로그로 확인한다.

검증된 기존 checkpoint를 실제 찾지 못하면 새 seed의 독립 실행이다. 새 로컬 학습량을 과거2240회에 붙이지 않는다. 원본 source/config/data/plan fingerprint를 바꿔 과거 run으로 재개하지 않는다.

## 6. 과거 평가 결과를 그대로 보존할 기준표

아래는 동일 추가640회 조건의 과거 보고이며 새 로컬 결과가 아니다. Accuracy는 %, Macro F1은0~1이다. JSON의 `diffusion` 필드를 비교했다. 근거 검사 이전 후보 결과다.

| 평가 | 대조/no-role Accuracy | role 조건 Accuracy | 대조 Macro F1 | role Macro F1 |
|---|---:|---:|---:|---:|
| 새 그래프 | 84.539 | 98.812 | 0.653595 | 0.974719 |
| 큰 그래프 | 85.221 | 95.703 | 0.610844 | 0.879135 |
| 미학습 문구 | 51.816 | 48.540 | 0.328511 | 0.297235 |

| 평가 | 직접 정확도 대조/role | 전이 정확도 대조/role | grounding 대조/role |
|---|---:|---:|---:|
| 새 그래프 | 34.017 / 94.188% | 46.253 / 96.899% | 100 / 100% |
| 큰 그래프 | 31.602 / 74.026% | 42.708 / 84.896% | 100 / 100% |
| 미학습 문구 | 4.211 / 1.053% | 4.264 / 0% | 0 / 0% |

추가 사실과 해석 범위:

- role 조건은 알려진 표현의 새 그래프·큰 그래프에서 개선 신호를 보였으나 미학습 표현의 grounding 실패를 해결하지 못했다. 이것이 표현 계열 분리와 부정 일반화가 우선인 이유다.
- 새 그래프·큰 그래프의 initializer Accuracy/Macro F1은 두 조건 모두1.0이다. 미학습 표현 initializer는 Accuracy0.482905984, Macro F1 0.293353303이다. 이 과제에서는 diffusion이 initializer보다 항상 좋은 것이 아니다.
- 요약 파일의 two-view grounding dev=1.0, one-view=0.847058824다. 별도 조건·경로의 보고값으로 보존하고 공정한 독립 반복 비교까지 완료했다고 주장하지 않는다.
- 그래프128/미학습64/큰그래프64, noise seeds `(17,29,43)` 평가 경로다. JSON의 query 집계5640/2808/3072는 seed 반복이 포함된다. 그래프와 seed 내 관측을 독립 표본으로 취급하지 않는다. 신뢰구간은 그래프 단위 resampling과 seed 분산을 고려한다.
- `direct`, `transitive`, `long_chain`은 hop 기반이다. 독립적인 부정 일반화 세부평가를 대신하지 않는다. 부정 전제·양성/음성 충돌·범위 변화를 별도 태그로 평가해야 한다.
- 클래스 불균형과 global CONFLICT 규칙 때문에 전체 Accuracy만 보면 관계 추론 실패가 가려질 수 있다. per-class F1·confusion·support를 함께 보고한다.
- 원본 고정 benchmark를 다시 쓰는 것과 새 독립 holdout을 구분한다. 이미 본 test를 반복해 모델 선택에 사용한 뒤 “완전 미학습”으로 부르지 않는다.
- `fuzzy_removed`, latent corruption/repair 평가도 JSON에 있다. 추론 때 기능을 제거한 결과만으로 학습·계산량까지 맞춘 구조 ablation의 인과 효과를 확정하지 않는다.
- 저장·fresh reload 재현성이 확인되지 않은 과거 세션 결과다. 현재 폴더에 작은 ZIP·문서가 존재하는 것으로 대형 가중치 보존을 추정하지 않는다.

통합 논문의 일부 서술은 matched no-role 비교를 후속 과제로 적고 있지만, 별도 실험 요약에는 `matched_no_role_control` 결과가 들어 있다. 문서 작성 시점·실행 기록의 차이를 보존한다. 위 표는 해당 JSON의 필드값이며 같은 parent·optimizer·데이터 순서·예산이 실제 artifact로 모두 입증됐다는 뜻은 아니다. 새 로컬 비교에서는 계보와 통제 조건을 manifest로 확인한다.

## 7. 실제 데이터 노출과 클래스 의미

마지막640회 상태의 query target exposure는 다음과 같다. 분모는75,244회이며 같은 예제 재노출도 포함한다.

| ID | query 클래스 | 횟수 | 비율 |
|---:|---|---:|---:|
| 0 | UNRESOLVED | 22,209 | 29.516% |
| 1 | GREATER | 6,467 | 8.595% |
| 2 | LESS | 6,320 | 8.399% |
| 3 | CONFLICT | 40,248 | 53.490% |

같은 실행의 grounding exposure는 `[0,0,0,0]`다. 마지막 diffusion-only 단계의 집계이며 이전 모든 grounding 학습이0이라는 뜻이 아니다. 입력 bytes, 정답 노출 수, 고유 예제 수, 성공 updates, attempts, loss weight, gradient 기여는 서로 다른 지표다.

grounding 네 클래스는 query 네 클래스와 다르다. 원본 `data.py`의 순서는 다음과 같다.

| ID | grounding 전제 종류 | 예시 |
|---:|---|---|
| 0 | GREATER | A가 B보다 크다 / is greater than |
| 1 | LESS | A가 B보다 작다 / is less than |
| 2 | NOT_GREATER | A가 B보다 크지 않다 / is not greater than |
| 3 | NOT_LESS | A가 B보다 작지 않다 / is not less than |

원본 train 문구는 각 종류에 한국어·영어의 소수 template가 있고 heldout은 “더 큰 값을 갖는다”, “더 작은 값을 갖는다”, “더 크다고 할 수 없다”, “더 작다고 할 수 없다” 각1개다. 이를 넓은 자연어 표현 계열 일반화가 입증된 데이터셋으로 부르지 않는다.

후속 목표는 **query 네 클래스 각각25%, grounding 네 클래스 각각25%**다. 이 둘의 분모·카운터를 따로 정의한다. class-weighted CE만으로 실제 노출 균형을 달성했다고 보고하지 않는다. graph mode sampling35/30/20/15%도 query label 비율과 같지 않다.

`grounding60/semantic15/diffusion25%`의 다음1000회는 **미실행 연구 가설**이다. 현재 학습 비율, 검증된 최적값, 기존 recovery plan으로 쓰지 않는다. 기존 recovery plan은 grounding480 / semantic320 / diffusion640, gate 통과 시 최대1440회다. 이 plan도 준비본이며 실행·완료가 확인되지 않았다.

## 8. 현재 코드에서 새로 확인한 해석상 핵심

원본102M 구현을 직접 읽고 다음을 확인했다. 연구 주장과 코드 범위를 맞추는 데 반드시 반영한다.

1. 입력은 일반 tokenizer 문서가 아니라 최대48 UTF-8 bytes의 관계 문구이며 embedding은 byte256개+padding을 사용한다. subject/object 역할은 각각8차원으로 주어진다. 개체 binding 전체를 자연어에서 스스로 획득한 결과가 아니다.
2. `fuzzy_closure()`는 학습된 문구 분류의 soft grades와 주어진 entity roles를 받아 **고정 max-product 관계 폐쇄 회로**를 계산한다. gold closure를 입력하지 않지만, 높은 initializer 성능을 전부 신경망이 자발적으로 배운 다단계 추론으로 설명해서는 안 된다.
3. `semantic()`/`target()`/`readout()`은 고정 orthonormal basis로 역할16+관계4+상수1을 잠재 상태에 대응시킨다. target은 supervision 경로이며 추론에서 정답 label을 호출하면 안 된다. 현재 구조화 target과 향후 vNext OutcomeCodec을 혼동하지 않는다.
4. diffusion phase에서는 `encode()`가 `torch.no_grad()` 아래 실행되고 학습 대상 prefix도 refiner·관계·time·velocity 계열이다. 따라서 diffusion 횟수를 더하는 것만으로 고정된 문구 grounding을 직접 재학습하지 않는다. 미학습 문구0%를 diffusion 예산만 늘려 해결할 근거는 없다.
5. typed oracle에서 NOT_GREATER가 LESS라는 양성 관계를 만들지 않는다. NOT_LESS도 GREATER로 바꾸지 않는다. 순환이나 양성/부정 모순은 현재 가정 집합 전체의 global CONFLICT로 처리한다. 이 dataset 정책을 일반 논리 전체의 유일한 모순 의미로 확대하지 않는다.
6. graph hash는 문구를 뺀 `(n, typed facts)`로 train/dev/test를 분할한다. 표현 template family 분리는 별도 문제다. 개체 재명명에 따른 동형 그래프·template 중복도 추가 감사 대상으로 둔다.
7. neural encode 입력은 `tokens/roles/fact_mask/queries/query_mask`다. `labels/ground/hops`는 supervision에 분리되어 있다. 이 분리를 유지하고 oracle 출력·검증 결과가 후보 입력으로 역류하지 않는지 실제 호출 경로를 검사한다.
8. Core self/cross attention에 query/fact padding mask가 전달되지 않는 알려진 경로가 있다. 패딩 수·순서 변화에 대한 유효 query 출력 불변성을 점검하고 새 모델 버전에서 수정한다. 기존 source fingerprint를 바꿔 같은 run으로 재개하지 않는다.
9. 원본은 clip 전 `clip_grad_norm_` 반환값을 만들지만 전체 norm을 지속 로그로 남기지 않는다. phase 첫 gradient audit만으로 모든 시도의 수치 안전성을 보장하지 않는다.
10. `durable_runner.py`의 `fcntl`은 native Windows에서 이식 문제가 된다. T4 장치명 제한은 이 파일뿐 아니라 원본 `Trainer`·`preflight`에도 있다. 한 곳만 지우고 로컬 실행 완료로 보고하지 않는다.

## 9. 실제 손실함수와 수학 대응

각 수식에 정의역·텐서 shape·mask·dtype·gradient 경로·코드 위치·실패 조건을 붙인다. 과거 설명, 현재 코드, 새 제안을 한 공식으로 합치지 않는다.

현재102M `train.py` 기준:

- Grounding: `L_ground = masked_CE(ground_logits, ground_targets)`.
- Semantic: `L_sem = CE_relation + 0.25*MSE(z_init, z_target) + 0.2*CE_ground`.
- Diffusion: `L_diff = MSE(v_hat,v_target) + 0.25*MSE(z0_hat,z_target) + 0.5*CE_relation(readout(z0_hat))`.
- diffusion 시간은 random t를 뽑고 batch의 최소1개·대략 절반을 정확한 t=1로 둔다. 단순 균등 t만 사용한다고 설명하지 않는다.
- masked MSE의 분모는 valid slot 수×latent 폭이다. weighted CE의 분모는 유효 target weight의 합이다. gradient accumulation의 평균과 phase별 trainable parameter 집합도 기록한다.

퍼지 routing 기준:

\[
d^2=2-2q^Ta,\quad \sigma_A=\mathrm{softplus}(s)+10^{-3},\quad
b=\beta-\frac{d^2}{2\sigma_A^2},\quad
\mu_j=\mathbf1[j\in TopK],\mathrm{sigmoid}(b_j).
\]

유효 unit vector에 적용한다. 영벡터·invalid anchor·Top-K 동률·all-masked를 검사하고 잘못된 anchor는 제외한다. `μ`, 정규화 주소 `ρ`, 활성도 `γ`를 분리한다. 다중 view 결합은 `1-∏(1-μ)`다. 이를 독립 사건의 참 확률로 단정하지 않는다. FP32 안정 경로와 AMP 경계를 확인한다.

diffusion 기준:

\[
\alpha_t=\cos(\pi t/2),\quad \sigma_t=\sin(\pi t/2),\quad
z_t=\alpha_tz_0+\sigma_t\epsilon,\quad
v_t=\alpha_t\epsilon-\sigma_tz_0,
\]
\[
\hat z_0=\alpha_tz_t-\sigma_t\hat v_t,\quad
\hat\epsilon=\sigma_tz_t+\alpha_t\hat v_t,\quad
z_{t'}=\alpha_{t'}\hat z_0+\sigma_{t'}\hat\epsilon.
\]

`α²+σ²=1`과 부호·sampler·target을 맞춘다. 끝점 `(1,0)`/`(0,1)`을 정확히 대입하고 t=1 순수 잡음에도 query 역할 조건이 전달되어야 한다. `z_init` 조건과 `z_1` 잡음은 다르다. oracle velocity를 넣은 수학 복원 검사는 학습된 denoiser 성능 증거가 아니다.

vNext 수학·프로토콜 발견사항:

- `RowNormalize(softplus(UVᵀ))`에 rank≤q를 주장할 수 없다. 비선형 원소 변환은 rank를 늘릴 수 있다.
- 양수 factor의 `G=BDᵀ/√q`, 행정규화 `T`, `F=G/(1+G)`를 분리한다. G/T의 rank 한계와 F의 비배타적 강도를 구분한다. T의 factorized 빠른 곱을 F에 그대로 적용하지 않는다.
- NULL 관계 질량을 제거한 뒤 다시 정규화하면 “관계 없음”이 사라진다. 약한 활성의 정규화 증폭도 별도 gate로 막는다.
- 절단 없는 max-product 합성의 결합법칙을 중간 Top-K 절단이 보존한다고 가정하지 않는다.
- 기억 numerator와 mass를 같은 수로 rescale하면 현재 평균만 같고 다음 쓰기의 의미가 달라질 수 있다.
- prefix 인과성, 읽기 후 쓰기, 문서 경계 reset, source epoch, deep immutable Commit, copy span·UTF-8, Writer transaction을 코드로 검사한다.
- 2026-10-02의22개+10-03의6개=28개 명세 검사 그룹은 작은 대수·계약 prototype 검사다. 102M 실제 학습이나 vNext 생산 구현28개 완료가 아니다.

최근 손실함수 대화에서 나온 초기형 복합식 `MSE+0.3 CE+0.1 cluster+0.05 diversity+0.01 entropy`는 과거 초기 prototype 설명으로만 취급한다. 현재102M 손실로 덮어쓰지 않는다. `L(w)=(w-3)^2`의 학습률 조건 같은 스칼라 이차식 결과를 FDT 전체 수렴 보장으로 확대하지 않는다. MSE/MAE/Huber/CE 비교는 검증할 후보이며 최종 채택·실행 사실이 아니다. 확인되지 않은 역사적 채택 이유를 만들어내지 않는다.

개정 통합 논문의 향후 학습식도 별도로 보존한다. `L_total = λ_LM L_LM + λ_codec L_codec + λ_rel L_rel + λ_bind L_bind + λ_diff L_diff + λ_claim L_claim + λ_real L_real + R`이며 계수는 pilot 전 확정값이 아니다. 서로 배타적인 object 선택의 기본 relation 항은 multiclass CE다. counterfactual negative는 같은 softmax에 BCE를 겹치지 말고 별도 score margin `max(0, m-f(s,r,o*)+f(s,r,o-))`의 평균으로 분리하는 제안이다. 실제102M `train.py`가 이 전체 목적함수를 구현했다고 보고하지 않는다.

paraphrase invariance도 같은 의미 pair의 latent·membership 거리를 줄이는 항만으로 두지 않는다. 역할 반전·부정·개체 교환 counterfactual에 `max(0,m_inv-d(Z(x),Z(x-)))` 같은 분리 항을 함께 검토해 collapse를 검사한다. 평가용 `exp(-d/τ)`의 평균 invariance score와 counterfactual separation을 함께 보고한다. 이 수식은 미실행 학습 제안이고 현재 learned invariance 성과가 아니다.

## 10. 로컬 환경 확인과 설치

이전 사용자 기록의 PC는 RTX5060 Ti16GB, RAM32GB(DDR5 16GB×2), SSD1TB, Intel Core Ultra7이며 Windows11 사용 이력이 있다. CPU의 구체 모델 번호는 확정하지 않는다. 현재 실제 실행 호스트를 확인하고 바뀐 값이 있으면 새 manifest에 기록한다. 이 사양을 원격 Codex 샌드박스에도 있다고 가정하지 않는다.

먼저 OS·shell·Python executable/version·프로젝트 경로·디스크 여유·GPU 장치명·VRAM·driver·CUDA 지원·PyTorch build를 확인한다. Windows면 PowerShell 명령을 제공하고 이미 정상 WSL 프로젝트가 있는 경우에만 그 환경을 존중한다. WSL·Docker 설치를 필수로 만들지 않는다.

RTX50 계열이면 실제 compute capability와 설치 PyTorch의 지원을 확인한다. `nvidia-smi`만 성공하거나 `torch.cuda.is_available()`만 True인 것으로 충분하지 않다. 작은 CUDA tensor 연산과 forward/backward, 실제102M smoke를 검사한다. 과거 기록에는 CPU 전용 PyTorch와 `sm_120` 지원 경고도 있었으므로 다시 확인한다. 과거 CUDA12.1/12.8 또는 T4 notebook의 설치 명령을 무조건 재사용하지 않는다.

환경 설치 당시 공식 PyTorch·NVIDIA 자료에서 GPU와 OS에 맞는 지원 조합을 확인하고 근거 URL·확인일·실제 설치 버전을 기록한다. stable 지원 조합을 우선하고 불필요한 nightly·소스 빌드를 자동 도입하지 않는다. GPU 지원에 드라이버 변경이 필요하면 준비된 근거와 필요한 조치만 보고한다. 유료 서비스로 우회하지 않는다.

설치 후 lock/환경 manifest를 남긴다. 개인 경로·계정·토큰을 로그에 노출하지 않는다. `python -m pip`와 명시적인 venv Python으로 설치·실행하고 여러 Python 설치 혼선을 확인한다. 전역 execution policy를 낮추는 대신 venv Python 직접 실행을 쓴다.

## 11. Colab 코드를 제거할 실제 작업

원본을 `archive/`로 보존하고 새 `src/fdt_local/` 패키지로 옮긴다. 아래 항목은 새 런타임에서 제거하거나 로컬 구현으로 대체한다.

| 기존 요소 | 로컬 처리 |
|---|---|
| `google.colab`, `drive.mount`, `auth.authenticate_user` | 기본 실행 경로에서 제거 |
| `/content`, `/content/drive`, mounted Drive 경로 | pathlib와 명시적인 project/data/run 디렉터리 |
| notebook `!pip`, `%cd`, 셀 순서·kernel 전역 변수 | setup script·pyproject/requirements·CLI 인자 |
| Google API가 필수인 CheckpointStore | 로컬 SSD용 backend. Drive 없이 import·실행 가능 |
| T4 문자열 일치 조건 | 새 로컬 분기에서 검증된 CUDA 장치 정책. 실제 GPU를 manifest에 기록 |
| `fcntl`·Unix 전용 잠금 | Windows/Linux 호환 single-writer 잠금과 종료 후 복구 |
| 셀 재실행·module cache 혼선 | 패키지 import·모듈 경로 검사·독립 프로세스 |
| notebook time budget·Colab 수명 설명 | monotonic 고정 deadline·최대 attempts·graceful stop |
| 원격 latest/marker 확인 | 로컬 payload/manifest/complete marker 해시 검사 |

`rg`로 Colab 경로·Google import·T4 guard·하드코딩 device·`allow_cpu_smoke`·`weights_saved_to_drive`·`torch.load`·저장 루프를 검색한다. archive 문서의 과거 언급까지 무분별하게 지우지 않는다. 런타임 dependency와 과거 출처 기록을 구분한다.

원본 `Trainer.run`에는 저장 뒤 start 시간이 초기화되는 결함이 있었다. v2의 고정 deadline 제어를 로컬에도 유지한다. 초기 저장·중간 저장·평가 시간을 포함하고 한 번 시작한 예산을 저장마다 초기화하지 않는다. GPU kernel·파일 flush가 반환하기 전에는 즉시 중단을 보장하지 못함을 로그에 반영한다.

CLI 예시는 아래 인터페이스로 구현하되 실제 코드와 일치시켜 README에 검증된 명령만 쓴다.

```text
python -m fdt_local inspect
python -m fdt_local preflight --config configs/local_baseline.yaml
python -m fdt_local smoke --config configs/local_baseline.yaml --max-attempts 5 --wall-seconds 600
python -m fdt_local train --config configs/local_baseline.yaml --run-dir runs/<new_id> --wall-seconds <explicit_budget>
python -m fdt_local resume --run-dir runs/<existing_id>
python -m fdt_local evaluate --checkpoint <verified_checkpoint> --suite <frozen_manifest>
python -m fdt_local verify-checkpoint --run-dir runs/<id> --fresh-process
python -m fdt_local status --run-dir runs/<id>
```

예시 명령을 아직 구현하지 않았다면 실행 완료 명령으로 제시하지 않는다. launch 명령은 명시적인 run ID·경로·예산을 받게 하고 import만으로 학습을 시작하지 않는다.

## 12. 로컬 가중치 보존과 재개 계약

로컬 저장 성공과 Drive 백업 성공은 별개다. Drive는 선택적인 추후 복사 대상으로만 두고 기본 학습을 막지 않는다. `local_checkpoint_verified`, `backup_to_drive_verified`, `evaluation_complete`를 별도로 기록한다. 로컬 저장 성공을 `weights_saved_to_drive=true`로 쓰지 않는다.

저장 순서:

1. 첫 update 전 초기 상태를 저장한다.
2. optimizer 시도를 완료하고 성공/skip·노출·수치 진단·이력을 확정한다.
3. 같은 파일시스템의 고유 임시 파일에 payload를 쓴다. flush·fsync 후 원자적으로 교체한다.
4. size·SHA-256·schema·run/contract/카운터를 검증한다. metadata와 marker의 payload 참조가 일치해야 한다.
5. complete marker를 마지막에 게시하고 그것도 원자적으로 보존한다. latest 포인터만으로 완료를 판정하지 않는다.
6. 검증 성공 후에만 다음 optimizer 시도를 시작한다.

모델, optimizer, 실제 scheduler 또는 수동 LR 진행 상태, AMP scaler, phase/phase_step, updates와 attempts, trainable parameter 집합, history, query/grounding 노출, 입력량, sampler/noise/Python/NumPy/PyTorch CPU/CUDA 등 사용한 모든 RNG·worker 상태를 보존한다. gradient accumulation 도중 저장을 허용하려면 microstep·누적 gradient까지 계약에 넣고, 우선은 완결된 optimizer 경계에서만 저장한다.

Windows에서 열린 파일 handle·백신·잠금 때문에 replace가 실패하는 경로, directory fsync 지원 차이, 디스크 부족, 중단·손상·중복 writer를 검사한다. 지원하지 않는 durability 보장은 하지 않는다. payload 저장 실패 후 현재 위험 상태를 이전 정상본 위에 덮어쓰지 않는다.

처음에는 매 optimizer 시도 전체 상태 저장이라는 기존 보존 정책을 유지한다. 실제100M 모델은 FP32 parameter만 약409MB이고 Adam 상태까지 포함하면 GB 수준이 될 수 있다. 실제 크기·쓰기 지연·여유 공간을 측정한다. 새 run 전용 순환 슬롯으로 최근2~3개 검증본과 명시적으로 tag한 best/final/분기점을 보존하고 중간 세대 전체를 무한 적재하지 않는다. 원본·다른 run·사용자 파일은 회전 대상에 넣지 않는다. 저장 간격을 조용히 늘리지 않는다.

| 상황 | 처리 |
|---|---|
| 같은 code/config/data/plan/env, 완전 상태·hash 검증 | strict resume 후보. fresh process 검사 후 재개 |
| 모델 tensors만 있고 optimizer/RNG 누락 | weights-only warm start, 새 run ID·새 optimizer. strict resume 아님 |
| T4 checkpoint를 다른 로컬 GPU로 옮김 | 원본 보존·명시적 migration/warm start. bitwise 동일 재개 주장 금지 |
| mask/손실/비율/plan/코드 변경 | 새 분기·새 fingerprint. 기존 manifest를 고쳐 맞추지 않음 |
| 최신 payload/marker 손상 | 이전 검증 슬롯 조회·복구. 임의 최신 파일 선택 금지 |
| 유효 checkpoint 없음 | 새 초기값. 과거2240회의 계속이라고 기록하지 않음 |

fresh process에서 전체 파일을 다시 읽어 모델·optimizer·AMP·RNG를 복구한다. 중단 없는 경로와 저장→프로세스 종료→복구→다음 시도의 입력·noise·loss·parameter 변화를 비교한다. 같은 환경의 재현성과 장치/버전이 다른 수치 오차를 분리한다. 가능한 `weights_only`/안전한 상태 포맷을 사용하고 출처 불명 pickle을 실행하지 않는다.

평가·runtime export에도 checkpoint SHA와 개별 완료 marker를 연결한다. 완료 checkpoint에서 평가만 중단됐으면 누락 평가만 실행한다. 최종 optimizer 상태를 phase 전환 시 날리지 않는다. `phase=complete`만으로 최종 평가·가중치 저장까지 완료했다고 보고하지 않는다.

## 13. 기존 검사와 새로 필요한 검증

v2 감사 보고의29개 PASS는 메모리 Drive·가짜 clock/optimizer/evaluation fixture다. 실제 tensor forward/backward, T4 학습, 대형 원격 checkpoint 전송을 검증한 결과가 아니다. 이번에 다시 실행한 검사만 이번 PASS로 표시한다.

우선 검증할 항목:

- 원본 ZIP 및 source digest, 데이터 manifest, config/plan 계보, import 경로와 파라미터 수.
- source/config/data/plan 불일치 차단, payload/marker/포인터 손상, 저장 실패 후 다음 update 차단, 이전 슬롯 복구, 중복 writer.
- 고정 deadline에 저장 시간 포함, phase gate 실패 시 다음 단계 금지, 최종 optimizer 유지, 평가 중단 후 누락 평가 재개.
- 실제 CUDA forward/backward와 phase별 expected trainable gradients. frozen parameter의 gradient 없음은 정상이고, 학습 대상의 missing/nonfinite는 실패다.
- loss, clip 전 전체 gradient norm, finite gradients/parameters, AMP skipped attempts·scale, peak allocated/reserved·전체 VRAM, step/save/eval 시간.
- 정답 target와 oracle가 neural 후보 입력에 유출되지 않음, query role t=1 조건, padding mask·all-invalid·empty fact 처리.
- 부정≠반대 긍정, 직접/전이/순환/모순/미해결 oracle 회귀, entity 재명명·방향 반전·fact 순서·padding 민감도.
- Commit의 중첩 alias와 출력 전후 hash, 미검증 영구 쓰기0, 원문 span·UTF-8 무결성.
- 로컬 full checkpoint·manifest·marker·별도 프로세스 reload, 다음 시도 재현.

테스트를 통과시키려고 판정 기준이나 hash 검사를 끄지 않는다. CPU fixture 통과를 GPU 수치 검증으로 보고하지 않는다. smoke 모델만 돌아간 것으로102M 검증을 대신하지 않는다. 불필요하게 같은 테스트를 반복하기보다 실패 원인을 고치고 관련 검사를 재실행한다.

## 14. 다음 연구 실험: 구체적인 우선순위

아래는 계획이다. 이미 실행되었다고 쓰지 않는다. 장기 실행 승인 전에는 config·dataset manifest·실행 명령·판정 기준까지 준비한다.

### E0 — 로컬 이식의 동치성·복구

모델·손실·데이터 의미를 가능한 한 유지한 로컬 baseline을 만든다. 플랫폼·저장 변경은 새 source/environment 계약으로 기록한다. 고정 fixture에서 이식 전후 forward·loss·RNG 계약을 비교한다. GPU smoke와 강제 중단/fresh reload를 먼저 통과시킨다. 기존480/320/640 plan은 기록된 기준 config로 보존하되 실행 여부를 별도 표시한다.

### E1 — padding mask 수정

별도 새 source/run에서 Core self/cross attention에 올바른 query/fact mask를 전달한다. dummy NULL context, all-masked row, query residual의 유효성도 처리한다. 같은 유효 입력에 padding 개수·순서만 바꾼 출력이 허용 오차 안에서 같아야 한다. 이 수정과 데이터·손실 변경을 한 번에 섞지 않는다.

### E2 — 표현 계열 분리와 부정 일반화

동의 문구 개수만 늘리지 말고 표현 family ID를 정의한다. lexical template·어순·부정 범위·수식어·한국어/영어 패턴·개체 표기 변화를 분리한다. typed graph split과 expression family split을 교차한 평가표를 만든다. train/dev/final test family와 graph·문서 family 누출을 검사하고 final test는 recipe 확정 전 선택에 쓰지 않는다.

grounding의 네 전제 종류를 각각25% 노출하도록 train sampler를 설계한다. query의 네 결과 클래스도 각각25% 유효 target 노출을 목표로 맞춘다. 각 단계 누적·최근 window·실현 편차를 기록한다. 균형을 위해 gold label을 바꾸거나 모순 oracle를 완화하지 않는다. global CONFLICT로 한 그래프 전체가 같은 label이 되는 구조를 고려해 graph bucket·query selection·필요시 생성 reject sampling을 설계한다. 목표를 공급할 수 없으면 실제 부족량을 보고한다.

부정 단독, 부정+무관 양성, 부정과 동일방향 양성 충돌, 반대방향 관계, 전이 체인과 부정의 모순, scope/time 차이는 현재 과제 지원 범위와 확장 분기를 구분해 만든다. NOT_GREATER에서 LESS를 강제 생성하지 않는다. 이 단계는 먼저 grounding이 미학습 표현을 구별하는지 확인한 뒤 관계와 diffusion을 평가한다.

### E3 — 역할·퍼지·diffusion의 독립 이득

같은 parent checkpoint 또는 같은 초기화·데이터 순서·노출량·seed·평가에서 role/no-role, fuzzy routing/명시적 대체 routing, initializer/diffusion, diffusion/동일 계산 예산 recurrent refiner를 비교한다. 고정 `fuzzy_closure` 회로의 기여도 별도로 분리한다. target·readout basis 변경은 또 다른 실험으로 둔다. 그래프 단위 confidence interval, seed 분산, 실제 latency/VRAM/FLOPs 또는 MAC 범위를 기록한다.

질문 역할 조건이 알려진 그래프에서 유리하다는 과거 결과를 자연어 일반화·diffusion 독립 이득·연산 절감의 증거로 확대하지 않는다. 더 적은 출력 토큰과 더 적은 총 계산량은 다르다. latent 반복 forward·검색·검증·메모리 비용까지 집계한다.

### E4 — 수식·손실 개선 후보

클립 전 gradient norm·phase별 loss scale·class별 오류를 먼저 측정한다. 그 뒤 필요한 경우 latent MSE와 Huber 등의 대안을 **한 요소씩** 비교하는 config를 준비한다. v-target/z0-target의 scale, t=1 비중, CE 분모·class weight가 실제 노출과 다른 점을 명시한다. raw membership을 softmax로 바꾸거나 Exact identity를 fuzzy 평균으로 합쳐 중심 계약을 몰래 바꾸지 않는다.

`60/15/25,1000회`는 E2 이후 검토할 독립 후보 config로만 보존한다. 실행 전 목표25%가 실제 sampler에서 실현되는지, 예산을 늘려 grounding이 개선되는지 검증 설계를 먼저 만든다. 최적 비율로 확정하지 않는다.

### E5 — vNext 전체 구현 연결

단계별 구현 완료 조건을 따라 언어·앵커·문맥 기억 → 의미 Codec/Commit/Realizer → Fuzzy-Guided Diffusion → 통합·장문으로 진행하는 개발 계획을 작성한다. oracle 의미 상태도 표현하지 못하는 상황에서 denoiser만 먼저 확장하지 않는다. 본 작업의102M 로컬 실행 완료와 vNext 생산 구현 완료를 별도 milestone으로 둔다.

## 15. 다른 FDT 계열·과거 corpus 계획을 혼동하지 말 것

연구 자료에는 초기 FDT/SFDT, Pure-FDT v3, v4 Diana, v4.1, vNext v1.1, 약102M role-aware adapter가 함께 있다. 과거 수억 parameter 언어모델 checkpoint와 이번 관계 adapter는 같은 run이 아니다. 이름이 비슷해도 tokenizer·parameter·objective·데이터·checkpoint 계보를 확인한다.

개정 통합 논문이 보존한 과거 발견은 다음과 같다. 아래 값은 해당 논문의 과거 감사 기록이며 이번에 그 checkpoint를 재실행한 결과가 아니다.

- 초기 일부 코드는 same-token CE여서 causal next-token objective가 아니었고, 전체 sequence의 anchor summary를 앞 위치에 돌려주면 future leakage가 생길 수 있었다. 언어모델 분기에서는 next-token shift·loss mask와 prefix-only read-before-write를 검사한다.
- Pure-FDT v3는426,806,356 parameters, context2048, 폭1216·20층·19heads·W64·256anchors/Top8·learned absolute position이었다. V20_T1_R2의 누적528,905,026 tokens에서 factual exact1/50, retrieval exact1/100, Python structural17/25, JSON syntax50/50이지만 parsed-value semantic exact0/50이라는 격차가 기록됐다. 문법 성공을 의미 정확성으로 바꾸어 설명하지 않는다.
- 특정 eager 환경의 context2048 속도2033 vs dense203 tok/s, peak14.26 vs22.26GB는 최적화되지 않은 dense 대조군의 memory cliff 관찰이다. 짧은512/1024에서는 dense가 더 빨랐으며 최적 kernel 비교가 아니었다. 일반적인10배 우위로 인용하지 않는다.
- v4 Diana는424,474,072 parameters·RoPE·context16384·Exact Episodic Memory·Copy Cursor였다. 추가200,102,827 active tokens, step32702 감사에서 teacher-forced loss 개선과 free-running 악화가 동시에 관측됐다. penalty-off greedy100개 중 반복 실패99개, paired NLL의 V20 대비+1.426579가 기록됐다.
- v4 copy gate가0.1679→0.9037로 늘어도 proposal recall5/30, whole-string exact0/30, first-token divergence30/30이 유지됐다. independent60-cell에서도 whole-string exact0이라는 기록이다. 복사 필요 감지, 올바른 source 선택, byte-exact copy를 각각 검사한다.
- v4.1은 V20과 α=0에서 FP32 bit-exact한 시작점·staged output-blend RoPE·token-normalized single-pass curriculum·고정 validation identity·post-transition gate로 confound를 분리했다. 작은 validation은 transition safety 검사이며 넓은 능력 입증이 아니다.
- RLM/uncertainty 분기는 calibrated confidence를 raw entropy와 구분하고 recursive call/read-token/time 예산을 기록하는 선택적 후속 연구다. 현재 vNext core의 필수 요소로 추가하지 않는다.

선행연구 비교에서 latent reasoning, diffusion, learned prototype/Top-K, memory/copy 자체를 각각 최초 발명으로 주장하지 않는다. FDT의 검증 대상은 퍼지 의미 좌표+명시적 binding·방향 관계+근거/identity 분리+bounded latent refinement+Commit 제한 표현의 결합 효과다. LOTUS/COCONUT/LaDiR/LLaDA/LTN/PRISM/Titans/CopyNet/RLM 등의 구체 논문·최신 정보가 필요하면 공식 논문을 새로 확인하고 인용한다. latent token 수 감소가 전체 계산량 감소를 자동으로 뜻하지 않는다.

동봉 2026-10-01 evidence/revision 자료의 핵심:

- 약50M-token recovery stream의 관측량은49,999,872 input tokens, 97,656개×512 길이 block이다. FineWeb-Edu29.5599%, FineWeb26.5145%, Python19.7919%, JS/TS8.6221%, C/C++4.4534%, Java4.4206%, Rust/Go4.4391%, structured docs2.1985%라는 **해당 stream**의 집계다. 전체 최신 checkpoint corpus 비율은 unknown이다.
- V20 tranche의 factual/retrieval/python/json/natural 20/20/15/15/30%는 category-normalized completion loss weights다. 전체 input-token exposure가 아니다.
- 과거 long-source row2%가 active-token 약32.1%와 연결된 사례가 있다. 행 수와 토큰 비율을 구분하고 희소 장문 반복을 숨기지 않는다.
- 별도 미래 계획은 새 corpus90%+retention replay10%, 새 corpus 내부 FineWeb-Edu35/Wikipedia25/LibreTexts20/StackExchange10/Gutenberg5/arXiv5%다. 전체 token 기준31.5/22.5/18/9/4.5/4.5%+replay10%다. 10M-token pilot은 미실행이며 검증된 최적값이 아니다.
- coherent document·문단/section 경계·dedup-before-split·문서별 memory reset·비padding input token quota·source revision 고정을 계획했다. 실제 inventory가 길이/비율을 공급할 수 있다는 보장은 없다.

이 corpus 계획을 이번102M synthetic 관계 과제에 자동 적용하거나 대용량 데이터를 바로 다운로드하지 않는다. 관련 장기 언어모델 작업을 나중에 할 때는 정확한 parent checkpoint·tokenizer·retention gate를 별도로 해결한다. 전체 세부값은 동봉 원문을 보존한다.

## 16. 결과물과 진행 방식

가능한 기존 프로젝트를 찾고 원본을 건드리지 않는 새 branch/디렉터리를 사용한다. 실제 git 저장소가 있으면 현재 사용자 변경을 먼저 확인해 보존한다. 저장소가 없어도 로컬 버전·manifest로 추적한다. 외부 remote를 새로 만들거나 push하지 않는다.

| 산출물 | 내용 |
|---|---|
| `README_LOCAL_KR.md` | 실제 OS에 맞는 설치·preflight·smoke·train·resume·eval·stop 명령 |
| `src/fdt_local/` | Colab 없이 import·실행하는 모델·데이터·학습·평가·로컬 저장·CLI |
| `configs/` | 원본 기준, local baseline, smoke, 새 연구 후보를 구분 |
| `docs/FDT_HANDOVER.md` | 실제 읽은 자료·버전·해시, 접근하지 못한 자료, 현재 결론 |
| `docs/FDT_CODE_AUDIT.md` | 수식↔코드↔shape/dtype/gradient 대응, 결함·수정·검사 범위 |
| `docs/FDT_NEXT_EXPERIMENTS.md` | E0~E5 계획·예산·통제·실행 여부·판정 기준 |
| `docs/DECISIONS.md` | 확정 지시 / 측정 사실 / 해석 / 미실행 가설 / 미확인 분리 |
| `tests/` | 변경 위험을 검증하는 수학·mask·누출·저장/복구·CLI 검사 |
| `runs/<run_id>/` | resolved config, source/data/plan/env manifest, logs, checkpoints, evaluation markers |
| `archive/` | 원본 ZIP·notebook·보고서와 기존 실험 보존. 새 runtime의 필수 dependency 아님 |

진행은 자료·환경 확인 → 원본 보존 → 로컬 이식 → 관련 검사 → 제한된 CUDA smoke → 독립 프로세스 저장·재로드 확인 → 장기 실행 준비 순서로 한다. 막힌 단계가 있어도 독립적으로 가능한 코드·문서 작업은 계속한다. 불필요한 확인 질문으로 멈추지 말고 정말 필요한 정보·권한만 구체적인 결과와 함께 마지막에 제시한다.

실행하지 않은 수치, 모델이 없는 데모, fixture PASS, 그래프 toy task를 실제 모델 개선으로 포장하지 않는다. static playground는 학습된 신경망 실행과 구분한다. 새로 발견한 사실은 출처/코드 위치와 검증 범위를 붙이고 `DECISIONS.md`와 인계 문서에 반영한다. 과거 assistant 제안을 사용자 확정 결정으로 승격하지 않는다.

최종 보고는 다음을 한국어로 간결하게 제시한다.

1. 읽고 확인한 자료와 실제 실행 호스트·GPU.
2. 로컬 실행을 위해 바꾼 파일과 Colab 제거 범위.
3. 실제 수행한 검사·GPU 시도 수·손실·gradient·VRAM·실패.
4. checkpoint size/hash/marker와 fresh reload·다음 시도 재현 여부.
5. 아직 실행하지 않은 장기 학습·가설·불명확한 자료.
6. 사용자가 다음에 실행할 검증된 명령 한 개. 장기 시작 승인이 필요한 경우에만 그 경계를 명시.

이제 자료와 환경 inventory부터 시작해, 허용된 로컬 이식·검증·실행 준비를 완료해라.
