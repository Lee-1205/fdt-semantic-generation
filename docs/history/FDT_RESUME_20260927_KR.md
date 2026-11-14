# FDT 재개 지점 — 사용자 요청으로 정지

## 먼저 읽을 것

사용자가 "완벽히 재개할 수 있는 포인트까지 완수하고 저장 후 종료"를 요청했다. 자동화 fdt는 PAUSED이며 사용자의 명시적인 재개 전에는 새로운 학습·자동 실행을 시작하지 않는다. 현재 worker/supervisor는 모두 정상 종료, 저비용 에이전트도 완료 상태다. 종료 요청 이후 추가 optimizer update는 없다.

1. 이 파일과 `PROJECT_STATE.md`의 가장 위 항목을 읽는다.
2. `FDT_PAUSE_CHECKPOINT_20260927.json`의 파일/SHA와 복구 검증 결과를 확인한다.
3. `ASTRA_EXACT_IDENTITY_RESULT_20260927_KR.md`를 읽고 다음의 제한된 성과와 실패를 유지한다.

## 정확한 복원 자산

- 실행 출력: `astra_exact_identity_pilot_20260927` (terminal COMPLETE, 재실행 금지).
- 최종 복구: `execution/recovery_step_0512.pt`, SHA43420dfc44edb3f46183a2702fb7305b46591fbb934b4245f1db064972bc32a0.
- 최종 head: `execution/exact_head_final.pt`, SHA01b6af6f18f22060fd152bba06f95c8e7ec1a0fd3493bb61abf2b82e193dd721.
- immutable Base: `astra_answer_focus_pilot_20260926/execution/mixed/final.pt`, SHA59aeb7f25f67dac1b54a1f59ce5079e5659a716a3bb30949949544526073dd72.
- 복구파일은 독립 full Base가 아니다. 위 Base + Exact head를 조합한다. 모델 구성은 Base 내부 model_config, Base state는 model_state_dict, Exact state는 exact_pointer_state_dict이다.
- AdamW gate weight/bias 2개 slot 각각 step512, torch RNG, 완료 counter, plan hash가 포함되어 있고 CPU 역직렬화 검증 PASS. 복구 head와 최종 head 텐서가 모두 일치한다. 계획/데이터/schedule/두 feature cache/실행 소스 snapshot/SHA 모두 검증했다. 작업 상태를 잃지 않고 다음 실험에서 이 learned head와 필요한 optimizer 상태를 읽을 수 있다.
- 이 실험은 이미 완주했다. 기존 clock/deadline/execution을 다시 열거나 terminal을 resume하지 않는다. 다음 실험에 별도 출력/새 고정 예산을 만들고, 동일 objective의 gate-only 지속학습이면 optimizer/RNG 사용 여부를 명시한다. 목표/parameter set을 바꾸면 optimizer 재사용 가능성을 검토하고 명시한다.
- Python: `<USER_HOME>/AppData/Local/Programs/Python/Python310/python.exe`, `-B -X utf8` 사용. Windows에서 os.kill(pid,0) 금지.

## 저장된 성과와 미해결

- gate-only512updates/16384intent labels/Base0. 실제 정확복제32/32+64/64, negative오복사0/96; 새로운 표현positive0/16, negative오복사0/16.
- source는 결정적 등록 이름 대조, intent만 학습, 기존 Exact cursor가 payload+EOS 운송. 신경망 source retrieval이 해결됐다고 부르지 않는다. 원본을 모두 등록하는 metadata 인터페이스에 한정된다.
- 새로운 표현/식별자 접두사/값 형식을 함께 바꿨으므로 의도 다양성의 단독 인과효과는 UNKNOWN. 새표현의 직접 거절 원인은 gate<.9.
- Base97 fixed52 실패와 기존 신경망 source 실패를 보존한다. 모델 완성/자동 승격 아님. 전체 파일럿 기준 FAIL이어도 현재96개 정확복제 성과를 폐기하지 않는다.

## 재개 후 바로 할 일

작동하는 hybrid head를 보존한 채 다양한 복사/비복사 의도 데이터 후보를 만든다. 명시적 부정·설명 요청·값 조회·짧고 긴 명령·표현 및 위치를 다양화한다. Train/dev 이름·값·문서 그룹 및 일부 표현 계열을 분리하고 보수적 기존 replay를 포함한다. 기존 성공96개는 retention, 실패32개는 이미 관찰한 development로 기록한다. 현재 다음 데이터/계획은 아직 생성하지 않았다.

필요한 최소 CPU token/EOS/중복·분리 검증과 새 계획/출력/SHA/고정 시간·intent 상한/중단 기준/독립 감독을 준비한 후 한 번의 작은 후속 gate 학습으로 이어간다. 광범위 audit나 unrelated baseline을 다시 선행조건으로 만들지 않는다. 임계값 사후 완화/정답source 주입 금지. 이후 Base 자연어 능력 개선도 남아 있다.

학습 중 root는 상세 분석을 멈추고 저비용 document_pilot_supervisor+독립 프로세스에게 감독을 맡긴다. 복구는 매1시간 이내, 목표50분 이내. 종료 후 실제 process가 사라졌는지 확인하고 root가 raw copy/rejection 및 retention을 판정한다.

## 보존 제약

W64/256anchors/top8/router256/T.25 유지, Base안정 전 latent reasoning 추가 금지. 원본source/prepared_data/핵심checkpoint/이전 실패 evidence를 보존한다. 승인 밖 삭제·cloud지출·외부게시 금지. 재개 시 실제 프로세스와 최신 PROJECT_STATE를 먼저 확인하고 중복 실행하지 않는다.
