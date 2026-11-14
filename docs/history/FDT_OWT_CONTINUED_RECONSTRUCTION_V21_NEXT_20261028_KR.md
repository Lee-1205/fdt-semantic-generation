# 다음 한정 목적 — 문장→퍼지 입력 의미→동일 문장 재복원 학습

최신 사람 지시 “지금 언어-의미-언어만 확인”을 우선한다. 자유창작을 요구하거나 그 성능을 학습 진입 조건으로 두지 않는다. 기존 `free`는 해당 입력의 추출 상태로 BOS부터 원문을 재복원한 결과였으므로 “상태 조건 복원”으로 명확히 기록한다. 정답 앞부분을 주는 teacher-prefix 점수는 보조 지표이고, 같은 입력에서 추출한 상태만으로 원문을 복원하는 결과를 함께 읽는다. 두 평가는 동일 사이클의 서로 다른 조건이다. 추출기의 의미 상태에 유일한 정답 vector가 있다고 가정하지 않는다.

Source는 완료된 V20 `model/checkpoints/state_4096.pt`, SHA `a6772af5e8be788c006d7566af3c82a71dd6a7ab6dfe4a64b37668bef30d3627` / algorithm `edfdef08d2ac0294caf84d8222314f143ea989dd18a94cc5c9808ec1ebbe07f4`다. 누적 27348attempt/27341update, 全63Adamstep27341/scaler131072tracker70/allCPU-CUDA-Python-NumPyRNG/history/orders/노출/전체 중첩 ancestry를 보존한다. 과거 6 AMP skips와 V18 failed attempt도 유지한다. 새 allocation은 0에서 따로 시작한다. V20 예산과 미사용 992 calls·시간은 CLOSED다.

후속은 동일 V20 4096train문서/16384문장에 대한 2유한 epoch/새4096attempt/batch8 후보로 한다. 32768rowselections/2918756targetUTF8bytes, decoder 2손실 감독 5837512UTF8occurrences/5903048byte+EOSsymbols다. 새 문서 다운로드·평가 panel 확장·held-to-train 편입은 없다. 같은 cleanCE.5+corrupt-prefixCE.5/.15prefix mask/strict deterministic elementwise loss/현재 AMP guard/원래 fuzzy μ-union-ρ와 두 추출 view-only decoder를 유지한다. 현재 111485711total/24150914trainable이며 reasoning/output-specific meaning denoising은 계속 별도·동결한다.

Teacher-prefix 57.96→62.22%와 NLL1.64381→1.28687은 관측 개선이다. 정답 prefix 없는 동일 문장 복원의 직접 의미 판독은 아직 충실한 복원에 못 미쳤다. 초기·소량 학습이라는 한계를 유지하고 방법 전체의 실패로 단정하지 않는다. 훈련 노출을 더 주면서 추출→언어화 사이클의 학습을 이어간다. 현 단계에 무관한 추론·창작·범용 LM gate는 추가하지 않는다.

훈련 문장과 dev 문장의 복원 차이를 확인하려면 최종에만 **고정 16 train 문장**의 정상 상태 조건 재복원을 더할 수 있다. 출력에 따른 선별을 피하기 위해 old-doc쪽 row [0,4,8,12,16,20,24,28]과 new-doc쪽 [8192,8196,8200,8204,8208,8212,8216,8220]로 미리 고정한다. 16개 서로 다른 학습 문서이며 새로운 blind benchmark가 아니다. 같은 X→Z(X)→X̂만 수행하고 추론 시 정답 prefix/길이/역할을 주지 않는다. 원문·복원문을 ROOT가 읽되 dev와 train을 합쳐 일반화 점수로 부르지 않는다. 이것은 다음 유한 학습의 최종 관찰 후보이며 지금 별도 GPU probe를 실행한 것이 아니다.

실제 코드 검토 뒤 예상 call은 training12288 + finaldev128 + 기존 상태 조건 복원 최대1552 + train정상복원 최대520 = 14488이다. 새 cap 후보15300/owner900초/reserve90/output6GiB/start40GiB/GPU12GiB를 사전 확정한다. 실제 비용·C20GiB보호+checkpoint임시여유+8GiB headroom이 충족되는지 다시 확인한다. Import0/mid2048/final4096 및 필요 STOP fullstate를 새 directory에 저장하고 삭제·rotation하지 않는다. 새 run의 clock·order·calls와 과거 cumulative 값을 구분한다.

V20 실제 schema importer 및 delta만 확인한다: 전체 model/Adam/scaler/RNG/ancestry 정확 import, finite 2epoch 순서·마지막 batch·AMP skip 노출·복구, 고정 train probe가 target 없는 Z(X) reconstruction인지, 실제 worker preflight 및 post-Popen 예외 cleanup을 포함한 독립 OS validate. 이미 통과한 광범위 baseline·gradient probe를 무한 반복하지 않는다. 동일 dev/function 기준은 V20 저장 결과를 사용한다. ROOT 검토·새 SHA·고정 상한 이후 ONE bounded 실행이며 하드웨어 재승인을 묻지 않는다.

현재 V21 code/runner/owner/plan/seal/GPU는 **아직 없다**. CPU 계획을 실행 성과로 쓰지 않는다. 모든 원래 source/typed core/관계 결과/weights/AdamRNG/raw/실패를 보존한다. ROOT ONLY, 에이전트 금지, E2final/registry 접근 금지, browser/mouse/keepalive/유료 cloud/외부 게시/사용자 원본 삭제/Windows os.kill(pid,0) 금지, 실행 중 sealed code 수정·old deadline 연장·자동 재시작·unused budget 재사용 금지를 유지한다. Timer 30분 prompt는 정확히 재개다.
