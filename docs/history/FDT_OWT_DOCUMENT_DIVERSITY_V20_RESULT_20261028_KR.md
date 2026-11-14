# V20 — 동일 입력의 퍼지 상태 기반 문장 복원

최신 사람 지시는 “지금 언어-의미-언어만 확인”이다. 현재 실제 경로는 **입력 문장 X → 원래 퍼지 앵커로 추출한 Z(X) → 같은 문장의 복원 X̂**이다. 추론 및 출력 목적 의미 디노이징은 동결·분리돼 있다. 입력과 무관한 자유창작을 검사한 것이 아니다. 과거 worker의 `free/generation` 파일명은 정답 앞부분을 주지 않는 상태 조건 복원을 뜻했다. 파일명과 봉인된 실행 기록은 보존하고, 이후 설명에서는 “상태 조건 복원”으로 부른다.

실제 `FuzzySentenceAE.extract`는 원래 Router의 비배타적 μ, probabilistic-sum union, 검색용 ρ, anchor/prior/value/condition을 사용한다. Decoder에는 추출된 두 view만 전달한다. 원문 byte/context를 우회 전달하지 않는다. `teacher_logits`는 정답의 앞부분을 받는 보조 측정이고, `reconstruct`는 BOS에서 시작해 같은 입력의 Z(X)와 자신이 복원한 앞부분만 이용한다. 추론 때 정답 byte·정답 길이·정답 역할을 주지 않는다. 따라서 둘 다 같은 복원 과제에 속하지만, teacher-prefix 점수만으로 전체 복원이 검증됐다고 할 수는 없다.

## 실제 학습과 복원 결과

V20은 새 4,096 attempt / 4,095 update / AMP skip 1회(attempt 4026)로 정상 완료됐다. AE 누적 27,348 attempt / 27,341 update다. 4,096개 train 문서의 16,384개 문장을 두 유한 epoch에 노출했다. 선택 행 32,768개, target UTF8 2,918,756 bytes, 두 decoder 손실의 UTF8 감독 횟수 5,837,512, byte+EOS 감독 5,903,048 symbols다. Prefix 손상은 437,107 / 2,918,756 eligible bytes = 14.9758%였다. Skip 행도 노출·순서를 소비했으며 되감지 않았다.

원래 2,048 train 문서/8,192행을 정확히 유지하고 train 문서 2,048개만 추가했다. 동일 revision `79d93d786212f7344586290adb811d4ae6a1762c`, 문서 hash split, 기존 64 dev 문서/256문장/정답/순서는 유지했다. 추가분은 저장 페이지의 미사용 54문서와 한정 fetch의 1,994문서다. Fetch 13,516,589 bytes/22초였다. 고유 선택 train 문장 UTF8 1,459,378 bytes이며 반복 노출량과 다르다. 기존 held를 train에 옮기지 않았고 E2 final/registry에 접근하지 않았다.

동일하게 관측해 온 dev의 23,207 active byte+EOS symbols:

| 조건 | V19 NLL | V20 NLL | V19 정답 byte 정확도 | V20 정답 byte 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력에서 추출한 상태 | 1.643806 | 1.286866 | 57.9567% | 62.2183% |
| 다른 입력의 상태로 교환 | 4.692624 | 4.111317 | 28.9525% | 30.0254% |
| 추출 상태를 0으로 제거 | 3.710206 | 3.205514 | 36.5148% | 38.1092% |

이는 **정답 앞부분을 준 조건의 예측 개선**이다. 의미 보존율 62.22%라는 뜻은 아니다. 정상과 교환·제거 간 차이는 현재 경로가 입력별 상태 정보를 이용한다는 근거다. 완전한 의미 추출이나 추출기·decoder 중 하나의 단독 원인까지 확정하지는 못한다.

같은 입력의 상태에서 BOS부터 복원한 기존 16 dev 문장은 UTF8/EOS 16/16, exact 0/16이다. 기존 V19 대비 chrF 21.2620→21.0753, 단어 F1 11.4854→10.9277%, 내용어 F1 1.0880→0.6250%, byte edit 유사도 24.1832→25.6070%, 자기 원문 unique top1 10→9/16이다. 서로 엇갈리는 표면 지표이므로 전체 복원 개선이라고 부르지 않는다. 정상 chrF 21.0753은 상태 교환 15.9594·제거 16.7548보다 높지만 문장 의미의 충실한 재현을 대신하지 않는다.

## ROOT의 원문-복원문 직접 판독

16개 정상 복원문을 모두 직접 읽고 주어·행위·대상·부정/한정·관계를 기록했다. 이 표본에서 충실한 의역 0/16, 확인 가능한 올바른 부분 명제 0/16이었다. He/She/We나 원문과 비슷한 철자는 일부 남지만 핵심 관계까지 보존한 것으로 세지 않았다. 이는 현재 작은 관측 표본의 복원 상태이며 FDT·퍼지 방식 전체의 실패 판정이 아니다.

예를 들어 원문 `I ... without paying for ANY of the stuff I had taken!`는 **물건은 가져갔지만 지불하지 않았다**는 관계다. 복원문의 `I didn’t taken package`는 비문이며 부정이 가져가는 행위에 붙었을 가능성이 있다. 이는 명확한 형식적 모순으로 단정하지 않고 “부정의 대상이 불명확하고 원래 관계를 보존하지 못함”으로 기록했다. 다른 예에서는 DUI 검사 실패·기소, 익명 논문의 두 번째 검토자, 임시 문자열을 가리키는 포인터 관계가 빠졌다. 문법성·글자 겹침만으로 의미 보존을 인정하지 않았다.

전체 원문·복원·개별 판독은 `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261028_v20/root_all16_state_reconstruction_reading.md`에 저장했다. ROOT 판독은 비맹검·정성 판단이며 자동 의미 정확도나 독립 benchmark가 아니다. 정답 prefix가 없는 **동일 입력 재복원**을 확인하는 용도로만 사용한다. 자유창작·추론·범용 LM 성능을 별도 성공 조건으로 추가하지 않는다.

## 실행·보존 근거와 다음 단계

6개 focused CPU 검사(14.07초), 실제 worker CPU preflight, 독립 OS validate 후 한 번 실행했다. Worker 314.873초, encode 4,132 + decoder 9,676 + denoise 0 = 13,808 calls였다. 새 900초/reserve 90초/max 14,800 calls/output 6GiB 예산은 CLOSED다. 남은 992 calls·시간을 재사용하지 않는다. Owner 12180→30160, worker 24272→19668의 생성·parent·command·생존을 ROOT가 관측했다. Worker/owner exit 0, 모든 소유 proxy/native 종료 후 21:02:30KST Python 0을 확인하고 분석했다. 분석 후 21:15:23KST에도 Python 0, C free 46,235,901,952 bytes였다.

Plan SHA `d6871682b72ad8eac993a871f34d4a99f0ad43823291e3a188d0bced3446a019`. Final `model/checkpoints/state_4096.pt` 644,204,785 bytes, SHA `a6772af5e8be788c006d7566af3c82a71dd6a7ab6dfe4a64b37668bef30d3627`, algorithm SHA `edfdef08d2ac0294caf84d8222314f143ea989dd18a94cc5c9808ec1ebbe07f4`다.

ROOT CPU 분석은 import0/mid2048/final4096 SHA·safeload·재귀 digest, source 전체 component 동일성, 63 Adam tensor step 27341, scaler 131072/tracker70, 모든 저장 RNG·중첩 이력, 동결 reasoning/prefix/basis 불변, 두 epoch 순서·노출·AMP outcome·실제 call 합계와 저장 23,207 logits×3조건의 argmax/NLL을 재구성했다. Scaler는 source tracker1992부터 모든 성공·skip을 재생해 확인했다. 추가 model forward 0이다. 근거는 run의 `root_result.json`, `root_state_reconstruction_similarity.json` 및 `scripts/review_owt_state_reconstruction_v20_cpu.py`다.

다음은 동일 복원 목표의 유한 추가 학습이다. Teacher-prefix 성능과 원문을 보지 않는 상태 조건 복원을 분리 기록하고, 필요하면 이미 학습한 고정 소수 문장의 동일 사이클도 최종에 함께 확인해 훈련 문장도 복원하기 어려운지 구분한다. 추출기만 옳거나 decoder만 틀렸다고 단정하지 않는다. 새 자유창작 과제·출력 의미 디노이징·큰 벌점·anchor 초기화를 혼합하지 않는다. 새 코드·자원·고정 상한을 확인하기 전 V21 실행으로 보고하지 않는다. 모든 이전 fullstate/Adam/RNG/raw/실패/원본과 기존 관계 예측 성과는 보존한다.
