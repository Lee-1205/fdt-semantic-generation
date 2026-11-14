# FDT 추가 저장공간 후보 — 2026-09-18

읽기 전용 조사이며 삭제는 0개다. 아래 6개는 기존 자동 삭제 승인 범위 밖이므로 별도 승인 전 보존한다.
각 파일은 과거 v4 speed run의 최종 가중치다. 삭제하면 해당 중간 실행을 가중치 수준에서 재현·비교할 수 없다. 로그·설정·평가와 bridge100M/200M, V20, α.20, 520M은 보존한다.

| 경로 | GiB |
|---|---|
| `<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_v4_main_424m_curriculum_speed_20260827_r3\latest.pt` | 1.581 |
| `<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_v4_main_424m_curriculum_speed_20260827_r4\latest.pt` | 1.581 |
| `<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_v4_main_424m_curriculum_speed_20260827_r5\latest.pt` | 1.581 |
| `<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_v4_main_424m_curriculum_speed_20260827_r6_fixed_validation\latest.pt` | 1.581 |
| `<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_v4_main_424m_curriculum_speed_20260827_r7_syncfix\latest.pt` | 1.581 |
| `<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_v4_main_424m_curriculum_speed_20260827_r8_fast_backend\latest.pt` | 1.581 |

합계 9.488 GiB.
현재 여유 29.774 GiB; 현재 학습 runner 시작 기준35GiB.
기존 승인 범위의 남은 일반 checkpoint/recovery 0개. pytest 파일 414개/7.19MiB. 모든 metadata 기본 보존 원칙 적용.
Stage A10M은 latest와 SHA가 달라 중복이라고 단정할 수 없으므로 보존했다.
100GB를 이번 조사에서 확보했다는 주장은 하지 않는다. 다른 프로젝트나 prepared_data를 수정하지 않았다.
