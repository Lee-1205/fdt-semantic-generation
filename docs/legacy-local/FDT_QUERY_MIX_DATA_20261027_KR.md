# 혼합 질문 curriculum 자료 준비 — 2026-10-27

`research/artifacts/role8_query_mix_20261027_v1`은 CPU 준비와 저장 자료 재구성을 통과했다. 아직 새 optimizer를 실행하지 않았다. 이전1,024개 train/64개 dev의 사실·추상 graph·분할을 유지하고 query 구성만 바꾼 후보로, 새로운 독립 graph를 추가한 것이 아니다.

Train seed33017/dev seed22017의 고정 RNG로 graph당56개 nonself pair 중16개를 중복 없이 균일 선택했다. 같은 정답8query를 강제하던 행 단위 조건을 제거했다. 원래 전역 CONFLICT 규칙은 그대로이므로 모순 graph에서는 여전히 모든 query가CONFLICT다. 기존 graph 선택의 편향까지 제거한 일반 corpus라고 주장하지 않는다.

| 측정 | Train | Dev |
|---|---:|---:|
| 서로 다른 기존 추상 graph | 1,024 | 64 |
| 실제 활성 query positions | 16,384 | 1,024 |
| UNRESOLVED/GREATER/LESS/CONFLICT query 수 | 4,667/3,820/3,801/4,096 | 322/207/239/256 |
| 실제 premise 입력 UTF8 bytes | 105,337 | 6,601 |

질문 ID는 숫자 역할 tensor로 들어가며, JSON의 ID 문자 수를 언어모델 token이나 입력 문자열 학습량에 더하지 않는다. Query가 두 배가 돼도 premise 문자열 byte 수는 같다. 실제 optimizer 노출은 이후 batch history로 별도 계측해야 한다.

## 검증과 신원

CPU6검사와 root 독립 저장 재구성이 PASS다. 모든1,088행의 원래 Floyd oracle/별도 BFS label·최단 hops,16개 query 중복·mask·입력5key·원래 facts/graph identity/split·metadata 이름을 확인했다. `target_class`는 model 입력에 들어가지 않으며, 기존 선택 이력만 `original_sampling_target_class`로 보존했다.

Dev facts·queries·gold는 이미 관측한 runtime_v3 baseline의 mixed64 raw와 정확히 같다. Raw에는 hops가 없으므로 hops까지 저장돼 있다고 주장하지 않고 두 oracle로 재구성했다. 이는 새로운 blind/final 평가 자료가 아니다. E2 최종 holdout에는 접근하지 않았다. 기존 source/data/core와 V2 후보는 그대로 보존했다.

- 실제 manifest 파일 SHA: `430dcbb4afd52d00e436817279326b559161be294ecfef352de2f2036d02dff5`.
- Builder SHA: `92012dd0ae0aeb5a4ac7943cb19e1d1b28bf0b8a79e63c54c125a8afaa0ef75b`.
- Train SHA: `2a0c4e3f91cc9c41b80a009433b3de41b8f401e9c439b84cb0836727f7cac6c0`.
- Dev SHA: `71a6eb832119a8dc7e6532bfdd10daca328ba8779f46df08f099483ffa1f0522`.

`root_cpu_review.json`은 모델 forward0/optimizer0의 CPU 증거다. 내부 self 제외 payload hash는 실제 파일 전체 hash와 구분했다.

## 후속 계획에 미리 적용할 판단 구분

이전 run의 고정94% 미달 STOPPED는 바꾸지 않는다. 다음 별도 계획에서는 이미94%를 넘은 기존 control들의 `max(.94, baseline-.02)` 보존 조건을 유지한다. 아직94%에 도달하지 않은 mixed primary와 paired에는 자기 부모 baseline 대비 accuracy-.02/macro-F1-.05의 회귀 감시를 적용하고,94%는 성능 qualification 목표로 별도 보고한다. 이는 새 자료·새 계약에서 사전 등록하는 기준이며 과거의 실패를 통과로 바꾸는 것이 아니다.

계획 중인 단 한 번의 QueryMix pilot은 보존된 부모 `cf85d259…ddcaa5f` 가중치만 가져온 fresh Adam/scaler/allRNG seed23, phase2 원래 model/loss/LR3e-5/clip1과 동결 backbone을 사용한다. 고정256success/280attempt/새 별도1,800초 reserve180/6,144 logical forwards/최종150초, 매시도 fullstate와64/128/192/256 milestone 저장을 준비한다. 이전 v3 terminal이나08:47:57.644 종료를 재개·연장하지 않는다.

현재 새 worker/supervisor/config와 CPU 계약을 준비 중이며 새 계획은 아직 봉인·실행되지 않았다. Root의 코드·실제 preflight·독립 validate·계획SHA·출력·자원 검사를 통과한 뒤 ONE 새 실행으로 진행한다. 목표 미달 시 같은 replay를 무한 연장하지 않고 query 조건화의 구조적 원인을 다음 한정 과제로 판단한다. 범용 LM·NER·vNext 완성이나 diffusion의 독립적 이득을 선언하지 않는다.


## 09:06 KST 실행 준비 갱신

새 worker·감독·필수 CPU 계약과 root 검토가 완료됐으며 planSHA `a200b3b43665fc0831220a66fec57d6b18ab9802a2131bd7d5cf966a445ba34e`로 별도 QueryMix 계획을 봉인했다. 위의 준비 중·미봉인 문장은 이전 시점 기록이다. 실제 학습 시작 여부는 새 run의 clock/progress/recovery와 native 프로세스로 확인하며, 봉인·위임을 실행 증거로 보지 않는다. 원래 v3 STOPPED와 절대 종료는 그대로 보존한다.
