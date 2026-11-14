# E1 110회 중단과 복구 — 2026-10-26

첫 실행은 20:50:18(KST), 성공 110회/시도 110회에서 종료됐다. 직접 원인은 `worker_progress.json`의 `os.replace`가 `PermissionError [WinError 5]`를 반환한 것이다. 실패 상태를 같은 파일에 기록하려던 두 번째 교체도 거부됐다. 파일 점유 주체나 백신·다른 앱이 원인인지는 로그만으로 확인되지 않았다.

모델 checkpoint 교체 실패가 아니라 진행 상태 JSON 교체 실패다. 110회 직후 full model/Adam/scaler/모든 RNG는 먼저 정상 저장됐다. `complete_000110.json`, payload 크기 **921,653,193B**, SHA `836815fb470703b33f98273f8cb3e37d06b5e9b09da9df6879b38e7b2d15cdc5`를 독립 감독이 재계산해 확인했다. Supervisor·redirector·실제 GPU worker는 모두 종료됐고 재시작하지 않았다. 첫 실행 폴더·코드·원본은 그대로 보존한다.

마지막 step110의 loss는 0.4090994, gradient는 finite, AMP skip은 없었다. 개발용 같은 16그래프에서 grounding accuracy가 초기31.40%에서64회83.72%, macro F1이0.232에서0.683으로 올라갔다. 다만64회 시점 NOT_GREATER14개는 모두 NOT_LESS로 분류해 해당 클래스 F1은0이다. 아직 phase0이 끝나지 않았고 관계/diffusion 학습 가능성이나 자연어 일반화를 입증한 결과도 아니다. 이 중단을 구조적 수치 발산으로 분류하지 않는다.

사용자의 ‘이상 시 멈추고 수정’ 지시에 따라 별도 v2 운영 분기를 준비한다. 모델·데이터·손실·phase·학습률 규칙은 유지한다. Windows 접근 거부에 한정한 최대3초의 bounded retry를 JSON 교체와 자체 checkpoint 회전에 추가하고, 영구 오류면 이전 정상본을 남긴 채 중단한다. 기존 목적 파일을 지워 원자 교체를 우회하지 않는다. 읽기와 교체 경합 및 영구 실패를 CPU에서 검사한다.

v2는 checkpoint110의 model/Adam/scaler/모든 RNG·phase·LR·history·노출을 가져온 뒤 tensor/state digest를 비교하고, 첫 다음 batch도 저장된 sampler 상태와 대조한다. 코드와 I/O backend가 바뀐 **명시적 운영 migration**이며 옛 전체 실행 계약이 같다고 가장하지 않는다. 새 contract/fingerprint/run을 쓰며110회까지를 재학습하지 않는다.

성공 목표는 총1,440회, 시도 상한은 총1,500회 그대로다. 남은 성공1,330회/시도1,390회만 허용하고 원래 절대 deadline **22:43:46.215(KST)** 및 학습 cutoff22:38:46.215를 유지한다. 옛 terminal 상태나 deadline을 다시 열지 않는다. v2의 실제 시작 여부와 복구 결과는 새 run의 clock/import_verification/first_continued_update/recovery/native process 증거로 별도 확인한다.
