# 출처와 이번 확인 범위

확인일: 2026-10-26.

이번에 수행한 것은 지정 Drive 폴더 목록 재조회, 상태·실험 요약·감사 보고서 열람, 원본 recovery ZIP 다운로드와 내부 소스 열람, 원본 source digest 대조, 설계 명세 전체1500줄 읽기, 개정 통합 논문 텍스트 확인, 이전 인계·데이터 혼합 계획 읽기, 관련 대화 검색, 새 프롬프트 작성과 인계 묶음 검증입니다. 현재 PC의 GPU 확인, 코드 이식, 모델 forward/backward, 새 학습, 기존29개 검사 재실행은 수행하지 않았습니다.

## 파일 출처

| 파일 | 출처·확인 정보 |
|---|---|
| recovery v2 ZIP | Drive ID `1Frrd8kinJ2518nTLrjxG6cy3V3st2Vbt`, 78,371 bytes, 수정2026-10-06 11:36:08.717 UTC, 원본 bytes |
| 상태 txt | Drive ID `1yhpmWHTs5nTbBc_Ef79cjVUAaZfvb5r5`, 2,225 bytes, 수정2026-10-06 03:36:09.196 UTC, 원본 bytes |
| 실험 요약 json | Drive ID `1-ewSIYs5sIG-KHZABMo1PP7qRe7c55V8`, 55,058 bytes, 수정2026-10-06 03:29:27.625 UTC, 원본 bytes |
| 코드 감사 v2 | 원본 recovery v2 ZIP 안의 동일 명칭 파일을 추출해 보존. Drive에서도 같은 제목·본문을 확인 |
| 설계 명세 | Library `libfile_2fcf34cff64c81919fb1cd24c6abaa00`, 저장 version6, 본문v1.1, 94,291 bytes. 전체1500줄을 순서대로 재구성했고 등록 크기와 일치 |
| 개정 통합 논문 | Library `libfile_e950e8ef08548191b0131a99cafcb79d`, 347,022 bytes. 원본 PDF, evidence cutoff2026-10-06, revised audit edition v2 |
| 이전 인계 | Library `libfile_9ad6bf95c2bc819199ddefcb2f5d3d2e`, 14,189 bytes. 2026-10-07 자료 |
| mix evidence | Library `libfile_dcd7e104b35c81919473f25a4984ed7d`, 3,761 bytes. 2026-10-01 자료 |
| mix revision | Library `libfile_e845fee1ab088191a1f21663d14fa907`, 6,358 bytes. 2026-10-01 미실행 계획 |

텍스트 문서의 바이트 해시는 이 묶음의 `SHA256SUMS.json`을 기준으로 확인합니다. 원본 source 묶음 해시는 `train.source_digest()`의 네 파일명+파일 bytes 계약으로 별도 검증했습니다. ZIP 전체 해시와 서로 다릅니다.

## 이번 소스 열람에서 보강한 사실

- `original_source/data.py`: grounding 네 종류의 정확한 순서, query 네 종류의 순서, graph hash 분할, 부정과 global CONFLICT 정책.
- `original_source/model.py`: 고정 `fuzzy_closure` 회로, 주어진 role16차원, 고정 orthonormal target/readout basis, Core mask 미전달 경로.
- `original_source/train.py`: phase별 실제 손실, diffusion 단계 encoder no_grad, trainable prefix, t=1 샘플 비중, gradient norm 로그 부재, seed 반복 평가 집계, T4 guard.
- `original_source/role_experiment.py`: parent source1440회 확인 후 optimizer reset하는 migration, 평가 noise seeds17/29/43.
- `durable/durable_runner.py`: 별도의 T4 guard, Unix 전용 fcntl, 저장 wrapper·이력·완료 평가 제어.

이 사실은 새 학습 성능이 아니라 코드 경로 확인입니다. 결함 후보의 실제 수치 영향은 receiving Codex가 새 버전에서 검사해야 합니다.

## 과거 대화 검색의 범위

사용자 PC의 RTX5060 Ti16GB, RAM32GB·SSD1TB·Intel Ultra7 과거 발화와 Windows11 사용 기록이 검색됐습니다. 현재 장치가 동일하다는 실시간 증거는 아니므로 실제 실행 호스트 확인을 지시했습니다. CPU의 구체 모델 번호는 assistant 추정과 구분했습니다.

최근 손실함수 대화에서 확인된 초기 복합 손실 설명·스칼라 이차식 수렴 설명은 현재102M 구현과 구분했습니다. 검색은 모든 과거 대화의 완전한 내보내기가 아닙니다. 접근되지 않은 대화·새 결과를 읽거나 재현했다고 주장하지 않습니다.

## 보존·실행 경계

Drive 원본·가중치·기존 실험은 변경하지 않았습니다. 중단된 타이머를 재개하지 않았습니다. 이 묶음에는 GPU 가중치나 사용자의 인증 정보가 들어 있지 않습니다. 포함된 기존 notebook·원격 저장 코드는 원본 보존 자료입니다. 새 로컬 runtime에서 제거할 Colab 의존성을 식별할 근거로 제공합니다.
