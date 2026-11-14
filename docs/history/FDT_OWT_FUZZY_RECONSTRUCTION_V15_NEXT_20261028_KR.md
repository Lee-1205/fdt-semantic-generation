# 다음 한정 목적 — 같은 퍼지 AE의 문서 다양성 확대

최신 source는 `runs/owt_fuzzy_reconstruction_20261028_v14/model/checkpoints/state_4096.pt`, SHA `a9e2c3819b818cfc3e1d968d6ad515c9f136732786bb1ee39919d852f0f151d6`, algorithm `f66a4b10639db7b6bfbd1e3df0e5e40bf14dbf8e7b3597b6c9c1b574d159b4cb`다. V14는4,096추가/AE누적8,192업데이트 완료/모든 owned-proxy 종료/예산CLOSED다. 미사용816call/시간은 재사용하지 않는다.

추출 상태의 정보 사용과 teacher 정확도49.26→52.91%는 개선됐지만, 같은 자료의 추가8epoch 뒤 개발 NLL 개선은.07725로 작아졌고 자유 exact0/16·편집 오류.78895→.79454는 정체했다. 온라인 train batch CE평균은1.593→1.016으로 계속 낮아졌다. 이는 전체 모델 실패·확정적 과적합 증명이 아니다. 현재1,024훈련 문서/364,178고유 sentence UTF8bytes의 작은 범위에서 **자료 다양성을 늘려 계속 학습**하는 구체적 후속 목적이다.

다음 ROOT는 다음을 구체화한다.

1. 같은 모델/원래64fuzzyanchors/topk8/mu-union-rho/twoextractedviews-onlydecoder/추론동결을 유지한다. 새 회로·큰 반복/균등화 벌점·anchor 재초기화를 자동으로 끼워 넣지 않는다. Source63Adam step8192/scaler16384 tracker192/모든 RNG/전체 nested ancestry를 정확히 새 importer에 보존한다. V14local4096+V13local2048+V12local2048=global8192를 유지하고 이후 local을 더한다.
2. 동일 OWTrevision79d93d...1762c·문서 hash modulo9 규칙으로 train1,024→2,048문서, sentence4,096→8,192개를 검토한다. 기존1,024train문서·4,096행을 prefix로 그대로 보존하고 새1,024train문서·4,096행만 추가한다. 기존64dev문서/256문장·정답·순서는 그대로 둔다. 먼저 V12/V13저장 source_pages의 미사용 문서를 사용하고 필요 시 **최대16MiB/180초** 추가 rows API fetch 내에서 채운다. Revision header·exact normalizeddoc/sentence 중복을 확인한다. 새 immutable artifact를 만들고 source권리/near-dup/pretraining 불확실성을 그대로 기록한다. E2final/기존 held/registry는 접근하지 않는다.
3. 실제 자료가 채워지면4유한 epoch/최대4096새attempt/batch8/32,768행을 검토한다. 기존·새 문서 각각16,384행이며 targetUTF8bytes는 실제 새 문장에서 계산한다. 새 문서와 반복 노출을 구분한다. 새 attempt/cursor/order/clock/call cap이며 source의 닫힌 예산을 열지 않는다. 같은 base LR language1e-5/other1e-4의 유한 schedule과 populated moments를 보존한다.
4. 예상 modelcalls는 train8192+final3조건128+final16자유복원최대1552=9872다. 실제 코드/자료 비용을 확인한 뒤 cap약10500/900초/reserve90/output6GiB/start40GiB/GPU12GiB를 고정한다. 같은 held/함수/control이면 V14저장 결과를 초기 참조로 재사용하고 새 baseline GPU평가를 반복하지 않는다. Complete fullstates는 import0/mid2048/final4096와 필요 시 emergency를 우선 검토하며 기존 파일은 삭제·교체하지 않는다.
5. 실제 V14schema를 받는 새 importer/worker/owner/plan을 별도 파일에서 구현한다. Frozen source model/모든63Adam/scaler/RNG component digest, recursive count, 새 finite order/call separation/STOP에 필요한 CPU delta와 실제 worker preflight·독립 OS validate만 한다. 이전 검사와 broad trace를 무한 gate로 반복하지 않는다. ROOT가 새 SHA/clock/caps/resources를 검토한 뒤 ONE실행하며 하드웨어 재승인은 묻지 않는다.

결과는 teacher byte 점수·자유 복원·상태 교환/제거·소속도/anchor사용을 함께 기록한다. Fuzzy coverage11→13 자체를 개념적 의미 정확도로 세지 않는다. 자유 복원 충실도가 아직 부족하므로 출력할 의미를 만드는 디노이징은 별도 후속 단계로 남긴다. 같은 observed dev의 반복 사용·작은16문서 생성 표본 한계도 명시한다.

현재 V15자료/코드/worker/importer/plan/seal/GPU는 **아직 없다.** 준비를 실행으로 보고하지 않는다. ROOTONLY/noagents/원래weights·모든AdamRNG·raw·실패 보존/상시C20GiB+임시여유/소유 OS감독/terminal후모든종료확인/실행중sealedcode불변/자동재시작·oldbudget재사용금지는 그대로다. 타이머는30분, 문구는 정확히 `재개`다.
