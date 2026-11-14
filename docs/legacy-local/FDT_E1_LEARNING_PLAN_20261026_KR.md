# 승인된 102M 학습 — 2026-10-26

사용자가 “학습 시작해 계속 검사하며 해당 구조의 가능성을 조사하고 이상하다 싶으면 멈추고 수정해”라고 지시했다. 이전 준비 단계의 시작 대기는 해제됐다. 이 문서와 계획만으로 실제 실행 중이라고 판단하지 않는다. 실제 `clock.json`, `worker_progress.json`, 감독 소유 PID, 완료 marker를 확인한다.

첫 실행은 `runs/e1_masked_learning_20261026_first`다. 실제 102,237,197 파라미터이며 새 seed19 초기값에서 시작한다. 과거 2,240회 가중치나 smoke의 두 업데이트를 이어 붙이지 않는다. 알려진 Core padding mask 수정 후보를 사용하고 기존 데이터·목적함수·학습률·phase480/320/640을 유지한다. 새 E2 균형 후보는 학습에 사용하지 않는다. 로컬 matched baseline을 아직 학습하지 않았으므로 mask 수정의 단독 인과 이득은 주장하지 않는다.

최대 성공 업데이트 1,440회, 전체 optimizer 시도 1,500회, 전체 실행 7,200초다. 마지막 300초는 저장·평가 여유로 둔다. 학습 종료·재저장·자동화가 deadline을 갱신하지 않는다. 독립 감독은 소유한 child를 한 번만 시작하고 재시작하지 않는다.

매 시도 뒤 model/Adam/AMP/모든 사용 RNG를 전체 저장·SHA·내부 계약 검증한 뒤 다음 시도를 허용한다. 새 run의 최근 3개 일반 세대와 명시적 initial/phase/final 태그만 보존한다. 기존 원본과 다른 run은 회전 대상이 아니다. C: 여유 20GiB와 다음 임시 checkpoint 크기를 점검한다. 학습 process 동안만 Windows의 자동 idle sleep을 막는 일시 요청을 사용하고 종료 시 해제한다. OS 업데이트·사용자 종료·정전까지 차단하지 않는다.

검사는 다음과 같다.

- 매 시도: loss, clip 전 norm, finite gradient/parameter, gradient 소유권, AMP skip, 실현 query/ground 노출, bytes, VRAM, step/save 시간.
- 고정 단계: 최초와 1/64/128/256/384/480/512/640 step 중 해당 phase 범위에 속한 지점의 개발 평가. Ground 종류별 confusion/support/F1, initializer와 diffusion의 차이를 기록한다.
- 원래 phase gate: grounding accuracy≥0.90, semantic initializer accuracy≥0.80. 실패하면 다음 phase를 실행하지 않는다.
- 최종: 새로운 typed 그래프·기존 heldout 표현·큰 그래프 각16개와 noise seed17/29/43. initializer/diffusion, fuzzy 제거, latent corruption/repair 및 raw 사례를 체크포인트 SHA에 연결한다. 한 그래프 내 query와 seed 반복을 독립 표본으로 세지 않는다.

즉시 또는 반복 조건 중단은 nonfinite loss/parameters, gradient 소유권 오류, 연속 AMP skip3회, 최근20회 중 skip5회 이상, preclip norm1,000 초과3회, 반복 loss 폭증이다. 감독은 디스크 부족, 진행 600초 정체, 정상 저장 3,000초 초과, 고정 deadline에도 STOP을 요청한다. 정상 초기 미학습 성능만으로 오류라고 단정하지 않는다. 중단 뒤에는 가중치와 실패 원문을 남기고 원인을 구분해 새 분기에서 수정한다.

외부 controller는 마지막 성공 업데이트 직후에도 phase gate·최종 저장을 처리한다. 기존 core CLI의 attempt cap 경계에서 final 처리가 다음 호출로 밀릴 수 있는 점을 core source 변경 없이 보완했다. Controller/감독 CPU 계약 총9개 PASS이며 실제 장기 학습의 결과와 구분한다.

동결 plan SHA: `8de5aa1edc7d7529b97702d808d51271fef8677bb07c662719ea6418fd8c5801`.
실행 코드·config·core source SHA와 원본 snapshot은 실행 폴더에 있다. 저비용 감독 에이전트와 독립 OS process가 학습 중 감시하고 root는 종료 후 성능과 실패를 판단한다. 좁은 typed adapter의 학습 가능성 검증이며 범용 언어모델 완성을 뜻하지 않는다.
