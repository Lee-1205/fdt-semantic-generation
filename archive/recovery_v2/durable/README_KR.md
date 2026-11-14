# FDT T4 복구·지속 저장 v2 — 2026-10-06

이 패키지는 실행 전 준비본입니다. 이번 점검에서 학습·GPU 계산·Colab 접속을 실행하지 않았습니다. Drive 폴더의 파일 목록에는 새 학습 체크포인트와 복구 실행 폴더가 없었습니다. 이전 2,240회 학습 가중치는 복구된 상태가 아니며, 과거 보고서의 정확도를 새 가중치 성능으로 사용하지 않습니다.

## 집에서 실행

1. FDT_T4_100M_Durable_Recovery_v2.ipynb을 Colab에서 엽니다.
2. 런타임 유형에서 T4 GPU를 선택합니다. T4가 없으면 중단합니다.
3. 1번 소스 확인 → 2번 정상 Google 인증 → 3번 학습/재개 셀 순서로 실행합니다. 처음에 초기값 저장까지 기다린 후 DRIVE_COMMIT_VERIFIED를 확인합니다.
4. 전원을 연결하고 자동 절전·최대 절전 진입을 끄며 인터넷을 유지합니다. 실행 중 탭을 열어 둡니다. 화면만 켜 두고 3번 셀을 실행하지 않으면 학습하지 않습니다.
5. 3번 셀이 끝난 뒤 선택적으로 4번 전체 다운로드·SHA256·strict 모델 재로드 검사를 합니다. stop_reason=time_budget이면 다음에 같은 3번 셀을 실행해 검증본을 재개합니다. evaluation_pending이면 다음 실행에서 남은 평가만 진행합니다. gate_failed이면 같은 계획을 무작정 반복하지 않습니다.

한 연구 실행을 두 Colab 런타임에서 동시에 실행하지 마세요. 같은 런타임의 중복 학습은 프로세스 잠금으로 차단하지만, Drive는 전역 분산 잠금을 보장하지 않습니다.

## 30분과 Colab 제한

이전 노트북의 wall_seconds=1800은 코드의 30분 실행 예산입니다. v2의 기본은 MAX_RUN_HOURS=6.0, 즉 21,600초입니다. 3번 셀에서 최대 10시간까지 바꿀 수 있으며 학습 Plan/fingerprint에는 영향을 주지 않습니다. 중간 저장 때 전체 실행시간이 초기화되던 원본 run 루프는 wrapper의 고정 deadline으로 대체했습니다. 복구·초기 저장·소스 보존 시간도 실행 예산에서 차감합니다. 진행 중인 GPU 연산/업로드/평가가 반환한 경계에서 확인하므로 엄격한 초 단위 종료 보장은 아닙니다.

Google 공식 FAQ(2026-10-06 확인)는 무료 노트북을 사용량·가용성에 따라 최대 12시간으로 설명하며, GPU·최대 VM 수명·idle timeout은 동적으로 달라진다고 명시합니다. 6시간은 코드 설정이며 자원 보장이 아닙니다. 화면/탭을 열어 두어도 Colab의 제한이 없어지지 않습니다. 자동 클릭, keepalive, 계정 교체, 인증 우회, 유료 결제는 추가하지 않았습니다.

공식 자료: https://research.google.com/colaboratory/faq.html

## 보존 계약

원본 model.py/data.py/train.py/runtime.py SHA256은 de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79 그대로입니다. 모델은 기존 102,237,197 파라미터 구조입니다. 모델·학습 데이터·손실·Plan은 수정하지 않았습니다. wrapper 세 파일은 별도 fingerprint입니다.

첫 실행은 새 폴더 FDT100_T4_recovery_20261006_v2에서 step 0부터 시작합니다. v1 wrapper의 체크포인트를 fingerprint 수정으로 재개하지 않습니다. v2 검증본이 생긴 뒤에는 source/config/data/plan/PyTorch/GPU가 모두 일치해야 이어 갑니다. Plan은 grounding 480 / semantic 320 / diffusion 640회로 총 1,440회입니다. gate 기준 미달이면 다음 단계로 진행하지 않으므로 총 1,440회 완료를 보장하지 않습니다. 이전 2,240회와 합산하지 않습니다.

60/15/25% 추가 1,000회, query 네 클래스 25%씩, grounding 네 클래스 25%씩은 미실행 후속 연구 목표입니다. 현재 학습된 비율로 보고하지 않습니다. 실제 입력 바이트·query/grounding 노출 카운터를 저장하며 AMP overflow로 건너뛴 시도와 성공 update를 구분합니다.

## 저장·복구

첫 optimizer update 전 초기값을 저장합니다. 매 optimizer 시도 후 모델·optimizer·AMP scaler·진행 단계·노출 카운터·sampler/noise/PyTorch/CUDA RNG와 해당 update의 이력을 저장합니다. 전체 payload 업로드 후 Drive API size/MD5 및 버전을 확인하고 완료 marker를 게시합니다. 재개 시 전체 다운로드 SHA256을 확인합니다. 작은 latest 포인터가 실패해도 슬롯별 완료 marker로 복구합니다.

두 슬롯을 번갈아 교체하여 직전 검증본을 남깁니다. 모든 학습 세대를 영구 보관하는 방식은 아닙니다. 이번 복구 실행 전용 슬롯만 교체하고 과거 실험/ZIP을 삭제하지 않습니다. Drive의 파일 revision 정책에 따라 용량이 증가할 수 있으며 저장 용량·요청·전송 제한에 도달하면 중단합니다. 매 시도 수백 MB~GB를 보내므로 GPU 계산보다 저장이 오래 걸릴 수 있습니다.

실패한 저장이나 미완료 optimizer 연산 상태를 검증본으로 덮어쓰지 않습니다. 다음 optimizer 시도는 저장 성공 뒤 시작합니다. 강제 종료 시 이미 확인된 검증본은 재개 대상으로 남지만, 현재 계산/업로드 중인 한 update는 다시 계산할 수 있습니다. 절대적인 무손실·런타임 지속을 보장하지 않습니다.

완료 후 runtime.pt, runtime.json과 개발/새 그래프/미학습 문구/큰 그래프 평가를 단계별 checksum marker로 보존합니다. 같은 완료 체크포인트에서 이미 검증된 평가는 재실행하지 않고 누락된 평가만 처리합니다. 전체 완료는 evaluation_complete.json과 해당 완료 marker로 구분합니다.

## 검증 범위

원본·wrapper·보조 코드와 생성 notebook 셀을 정적 점검했습니다. PyTorch를 불러오거나 CPU/GPU 학습을 실행하지 않고 메모리 Drive, 가짜 시간·optimizer 동작·평가 fixture로 장애를 주입했습니다. 실제 Google Drive 대용량 전송, 새 T4 학습, 수치적 gradients/loss/VRAM, fresh Colab 모델 재로드는 집에서 실제 실행 후에만 검증할 수 있습니다.

전체 점검 결과와 남은 구조 검증 항목은 동봉 FDT_Code_Audit_2026_10_06_v2.md를 참고하세요.
