# FDT 로컬 인계 — 2026-10-26

## 최신 결과

같은 날 Role8 보강도128/128회 성공·skip0로 종료됐다. 최종636,931,329B/SHA `0b5066444b9fdf0338219fba167edca8c2d211dbdda0bc7ea202bcc8b2784829`, 모든 동결 tensor·복구 상태·실제 노출 재구성 PASS다. 동일 noise17에서 ID 교체90.83→92.08%, 큰 그래프96.09→97.27%로 올랐지만 +5%p 목표에는 미달했고 dev128은95.34→95.12%로 소폭 하락했다. `FDT_ROLE8_RESULT_20261027_KR.md`가 이 단계의 결과이며, 현재는 새로운8개체 그래프·균형 query 자료를 CPU로 준비한다. 완료한 Role8을 재시작하지 않는다.

2026-10-27 재개 시 E2 표현 보강도 128회 성공/128회 시도로 완료됐으며 실행 프로세스는 종료됐다. 최종 fullstate SHA `938b15e963ad4f9a120aed333653e02f588594c4235482960cc4164bb1d1bac4`를 보존했다. 학습에 편입한 표현은 grounding37.5→100%로 개선됐지만 새 dev64는 부모부터100%여서 미관측 일반화 향상이라고 하지 않는다. ID6·7로만 바꾸는 비교에서 diffusion98.47→90.28% 취약성을 확인해 별도 Role8 phase2 보강을 준비 중이다. 최신 결과와 실행 경계는 `FDT_E2_AND_ROLE_COVERAGE_RESULT_20261027_KR.md` 및 상위 `outputs/PROJECT_STATE.md`의 첫 CURRENT가 우선한다. 사용자의 학습 시작·계속검사/수정 지시로 아래 초기의 추가 학습 대기 조건은 이미 해제됐다.

사용자가 이후 학습 시작을 명시 승인하여 E1 masked102M의480/320/640=1,440회 학습과 평가를 완료했다. Full checkpoint636,785,153B/SHA d57d06ec7cc1ebb4f5ccbf365673554efc8a705043d492dd802fff1b6c7c8edb와 별도프로세스 CPU 재로드가 검증됐다. 원래 heldout4문구/새그래프에서 initializer100%이지만 diffusion96.09~98.61%, 별도 새 표현 grounding37.5% 약점이 남는다. 상세는 `FDT_E1_LEARNING_RESULT_20261026_KR.md`가 우선한다. 아래 초기 이식·smoke 절은 당시 범위의 기록이다. 새 E2 표현 보강은 별도 계획으로 구분하며 기존 final/E1 상태를 덮어쓰지 않는다.

## 완료한 범위

Downloads의 이중 중첩 폴더를 확인하고 manifest12개 파일의 size/SHA를 모두 대조했다. ZIP SHA `57e854be4e946e966ab2165df350c81538a9f322f7d4c14329422f83883d09ba`, 원본 four-file filename+bytes digest `de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79` 일치. 경로 탈출·symlink·실행 파일을 검사한 뒤 archive에 풀었고 notebook 전체를 실행하지 않았다.

원본 model.py는 baseline `src/fdt_local/model.py`에 byte-identical하게 보존했다. 데이터·고정 closure·basis·loss는 유지하고 runtime/CLI/환경/저장을 새 로컬 분기로 만들었다. `model_masked.py`는 별도E1 후보다. 전용venv, CUDA정책, FileLock, atomicpayload/marker, source/config/data/plan/env strict검사,20GiB floor, boundedattempt/deadline, perattempt diagnostics를 구현했다. Google/Colab/fcntl/T4-only runtime 의존성이 기본 경로에 없다.

## 실제 검증

- 원본29 fixture 검사 PASS (UTF8 모드). 첫 실행의 Windows cp949 오류는 `-X utf8`로 해결했고 원본을 바꾸지 않았다.
- 새 로컬27 CPU계약 PASS. 초기 testcollection의 문법 오류와 기존pytest임시경로 권한 문제는 수정/전용basetemp로 해결했다. 이들은 학습 성능 결과가 아니다.
- 실제102M 파라미터수 102,237,197, GPU RTX5060Ti, Python3.10.11/torch2.10.0+cu128. CUDA tensor backward와 phase별 smoke를 수행했다.
- 전체5 optimizerattempts/성공5/skip0, GPUwindow19.562s. 실제102M은2고유groundingupdates와같은nextupdate의freshprocess반복1회다. Tinysemantic/diffusion각1회는 별도모델이다.
- Freshprocess에서 model+populatedAdam+AMP+모든사용RNG/입력/loss의 다음trajectory SHA가 동일했다: `e04d2f56c55964a6dddaee213950f1079a7c11f7dd9711e6e8ee820c949522bc`.
- 최신full checkpoint921,622,281B, SHA `8be9884d9a1eb656cfd63f23e3b62f6c5d1adde7a17d404ab52b2ec90f546a7b`, `runs/smoke_102m/complete_000002.json`. 별도CPU프로세스 verify-checkpoint도 통과했다. 실제size/save시간/peakVRAM은 runs/smoke_report.json에 있다.
- 과거2240회 checkpoint를 찾거나 복구하지 않았다. 동봉 ZIP에는 그런 가중치가 없다. 새seed를 과거누적값에더하지 않는다. capability평가/장기학습/vNext전체 구현은 미실행이다.

## 읽은 자료와 접근 한계

인계프롬프트·SHA목록·EVIDENCE_NOTES·감사v2·이전인계·상태JSON·실험요약의 schema/핵심결과·별도corpus계획을 읽었다. vNext v1.1에서 routing/인과기억/관계/VP/Commit/Realizer/Writer/OutcomeCodec/비용·검사 구간을 읽고설계와코드를대조했다. PDF30페이지의텍스트를추출해초록·계보·102M결과/ablation구간을확인했다. 전체PDF시각렌더링이나모든65수식의새생산구현은하지않았다.

대화도구로 '모델 구조 설명', 'FDT 상세분석', 'FDT 학습·구조 개선 점검' 각최근최대3turn을읽었다. 첫대화의assistant본문은일부content-reference만반환되어전문을읽었다고하지않는다. 손실함수관련대화전체/모든계정Library/Drive하위폴더전체를조회하지않았다. 동봉원본만으로가능한구현은계속했다. 문서의과거assistant제안은현재사용자승인으로승격하지않았다.

## 사용자 최신 지시와 실행 경계

자동화재개명시승인으로 fdt ACTIVE. 처음엔Colab없이로컬을요청했지만이후100MColab선호→마우스제어중단/Google런타임지속제어불가시로컬fallback을명시했다. Browserhelper가URL을검증하지못해중단했고이후마우스조작없음. Googlefiletools는GPU런타임제어API를제공하지않아로컬fallback으로검증했다. 계정로그인·ColabGPU·클라우드공유·유료구매없음. keepalive로제한을우회하지않는다.

사용자는C:항상20GB여유를요청했고더보수적인20GiB를하한으로사용한다. 최종관측free=147,244,347,392B. 충분해원본/다른프로젝트삭제없이진행했다. 이번smoke예산5회는소진됐고자동화도추가optimizer를실행하면안된다. 장기학습은사용자'학습 시작'지시후별도예산/새run으로만한다.

## 다음 검증된 명령

`.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local verify-checkpoint --run-dir runs/smoke_102m --fresh-process`

## 재개 후 추가 준비 — 2026-10-26

별도 `research/` 분기에서 E2 데이터 후보 train128/dev32/final32를 생성했다. Query 네 클래스와 grounding 네 종류의 후보 노출은 각각 25%다. 분할별 표현 family/literal과 개체 재명명 typed 그래프를 분리했으며 개발 교차 패널 4×16개도 준비했다. 관련 CPU 5검사 PASS, 별도 path oracle로 256정답 및 실제 collate 계약을 확인했다. GPU·optimizer 추가 실행0이며 pinned runtime source와 smoke budget은 불변이다. 상세와 한계는 `FDT_E2_DATA_READINESS_20261026_KR.md`에 있다.

작은 합성 데이터 후보이며 학습 runner에는 아직 연결하지 않았다. 남은 장기 실험은 사용자 ‘학습 시작’ 지시 이후 별도 고정 예산과 새 run으로 진행한다. 자동화는 ACTIVE 상태로 유지하되 새 지시나 의미 있는 상태 변화가 없으면 조용히 대기한다.
