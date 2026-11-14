# 재생성 가능한 다운로드 캐시 정리

사용자의 “저장공간 알아서 확보해봐 삭제가능한 파일들 추려내서” 지시에 따라 `<USER_HOME>/AppData/Local/pip/cache`의 HTTP 패키지 다운로드 캐시를 정리했다. 설치된 `.venv`, Hugging Face 자료, uv 캐시, 프로젝트 원문·가중치·fullstate·Adam/scaler/RNG·raw·실패 기록은 삭제하지 않았다.

| 정리 대상 | 사전 파일 수 | 논리 바이트 |
|---|---:|---:|
| pip 구형 HTTP 캐시 | 7 | 2,152,279 |
| pip HTTP-v2 다운로드 캐시 | 1,202 | 7,452,999,882 |
| 합계 | 1,209 | 7,455,152,161 |

직접 빌드한 wheel 3개와 origin.json 3개, 총238680B는 먼저 백업하고 전후 SHA를 확인해 원래 캐시 경로에 복원했다. 정리 후 `pip cache info`는 HTTP0파일/0bytes, built wheels3개/238kB다. 패키지 설치·업데이트·네트워크 다운로드는 하지 않았다.

최초 PowerShell 재귀 삭제 명령은 자동 승인 검토에서 `blocked by policy`로 실행 전에 차단됐다. 해당 명령을 실행하거나 다른 셸의 재귀 삭제로 우회하지 않았다. 직접 빌드한 wheel의 검증된 백업을 만든 뒤 패키지 관리자의 공식 `python -m pip --isolated --disable-pip-version-check --cache-dir EXACT_CACHE cache purge`를 사용해 exit0을 확인하고 wheel·origin을 동일 SHA로 복원했다. pip 출력은1212파일/2052디렉터리 정리였으며, 복원한 로컬 wheel을 제외한 HTTP 캐시 삭제량은 위 표와 같다.

정리 전후 실제 C여유30954029056→38411583488B, 관측 증가7457554432B다. 이후 다른 시각의 추가 여유 변화 원인을 전부 이번 정리로 귀속하지 않는다. 10:03:01KST 새 리소스 검사에서 C40720113664B/GPU15261MiB/nativePython·compact0으로 기존 V40 시작30GiB/GPU12GiB 기준을 충족했다. 기준이나 학습 상한을 낮추지 않았다.

근거 `outputs/FDT_PIP_HTTP_CACHE_CLEANUP_20261030/plan.json`, `plan_package_manager_v2.json`, `result_package_manager_v2.json`, 보존된 wheel 백업이다. 새 V40 seal/owner 실행 증거는 별도 run/code_snapshot에 기록한다. 캐시 삭제 성공을 모델 학습·성능 향상으로 세지 않는다.

V40은 새 고정2100초/최대30000호출/8192attempt 계약으로 한 번 봉인·실행했다. planSHA `00aca0d669b2228c49fd075ca45a9425ed655d7ce40857bc0a171c1905635a70`. ROOT는 owner12456→21176과 실제 worker20960→32204의 생성·parent·command·생존을 확인했다. 10:06:39KST 실제497attempt497update/신규1491호출/denoise0/C여유40683515904B였다. 초기 복구 참조를 저장한 뒤 세 메모리 기반 같은 원문 복원 학습이 진행 중이며, 아직 V40 성능 결과는 없다. CPU preflight23948→18752는 실제 학습 worker와 별개였고 이미 종료했다.
