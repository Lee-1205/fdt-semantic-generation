# FDT 코드·장시간 실행 점검 v2

점검일: 2026-10-06. 학습을 실행하지 않고 코드, 제어 흐름과 저장 실패 처리만 점검했다. 원본 구현 10개 Python 파일과 지속 저장 구현/검사 7개 Python 파일, 생성 notebook의 코드 셀을 점검했다. 이번 검사를 GPU 학습이나 새 모델 실험으로 보고하지 않는다.

## 실행 준비 결과

- 코드의 실행 예산을 30분에서 기본 6시간(21,600초)으로 늘렸다. notebook 3번 셀의 MAX_RUN_HOURS를 조정할 수 있다.
- T4 CUDA만 허용한다. 다른 GPU/CPU로 자동 전환하거나 새 비용을 만들지 않는다.
- 원본 모델/데이터/손실/Plan 코드는 보존했다. 원본 source SHA256: de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79.
- 신규 저장 wrapper는 별도 fingerprint를 기록하고 새 FDT100_T4_recovery_20261006_v2 실행 폴더를 사용한다. v1 또는 기존 실험의 fingerprint를 수정해 이어 학습하지 않는다.
- 첫 실행은 초기값부터 새로 구성한다. 기존 2,240회 가중치가 복구되었다고 주장하지 않는다. 원래 보고서·ZIP은 보존한다.

## 발견·수정한 운영 문제

| 문제 | 수정과 검증 |
|---|---|
| notebook이 wall_seconds=1800을 고정해 실행 예산이 30분이었다 | 기본 21,600초, notebook에서 시간 설정 제공 |
| 원본 Trainer.run은 중간 저장에서 start 시간을 초기화해 총 실행시간 제한이 제대로 적용되지 않았다 | 원본 파일 수정 없이 고정 deadline 제어 루프를 추가. 저장 시간도 예산에 포함하는 가짜 시간 검사 통과 |
| 저장 wrapper가 이력 기록보다 먼저 저장해 마지막 로그가 같은 checkpoint에 포함되지 않을 수 있었다 | optimizer 시도 → 이력 기록 → 원격 commit → 다음 시도 순서로 수정 |
| 최종 checkpoint 이후 평가가 끊기면 재개 경로가 runtime export만 실행하고 평가를 건너뛰었다 | 완료된 checkpoint에서 누락된 평가 항목만 실행·보존. 가짜 평가를 중단 후 재개해 이미 완료된 항목을 반복하지 않음 확인 |
| 최종 phase 전환에서 optimizer를 초기화해 마지막 optimizer 상태가 사라졌다 | 완료 시 마지막 optimizer/AMP 상태를 유지 |
| 원격 완료 marker의 필수 필드가 빠지면 복구 후보 정렬에서 오류가 났다 | schema/슬롯/파일명/크기/해시/카운터 형식 검사. 손상된 최신 marker의 이전 슬롯 복구 검사 통과 |
| 같은 이름의 marker가 여러 개이면 임의 선택 가능성이 있었다 | 복수 marker는 자동 선택하지 않고 중단 |
| 로컬 JSON·다운로드 파일이 도중 중단으로 부분 파일이 될 수 있었다 | 임시 파일 작성 뒤 원자적 교체 |
| 로컬 training_status가 원격 backup marker 기록을 덮어썼다 | CheckpointStore의 상태/완료 기록을 그대로 유지 |
| 과거 save_to_drive_colab.py는 mounted Drive의 재읽기 hash만으로 저장 성공을 표기했다 | v2는 이 보존 셀을 호출하지 않고 Drive API acknowledgment/원격 metadata를 확인. 옛 파일은 기록으로 보존 |
| 다른 버전의 notebook을 같은 kernel에서 실행하면 import cache가 남을 수 있었다 | v2 전용 경로와 모듈 출처 검사를 추가 |
| 같은 runtime에서 중복 실행 가능성이 있었다 | 프로세스 파일 잠금. 다른 runtime 사이의 전역 분산 잠금 보장은 아님 |

## 모델·데이터·수식 점검

현재 구현은 일반 언어모델 전체가 아니라 비교 관계를 다루는 제한된 수치 의미 연구 adapter다. 8개 개체, 최대 12개 사실, 16개 query 슬롯, 문구 48 UTF-8 bytes가 한계다. 이 범위와 일반 자연어 추론 능력을 구분해야 한다.

퍼지 router는 정규화된 query/anchor 간 거리 d²=2−2qᵀa, 양수 sigma=softplus(s)+0.001, sigmoid(beta−d²/(2sigma²))의 비배타적 소속도를 계산한다. top-k 뒤 raw 소속도 mu와 혼합 정규화 rho를 구분한다. 다중 의미 view는 1−∏(1−mu)로 합치고 contextual memory는 소속도 가중 분자/분모로 계산한다. 관련 연산은 AMP 안에서도 FP32로 유지한다.

잠재 diffusion은 alpha(t)=cos(pi*t/2), sigma(t)=sin(pi*t/2), z_t=alpha*z_0+sigma*epsilon, v=alpha*epsilon−sigma*z_0이며 역변환 z_0=alpha*z_t−sigma*v, epsilon=sigma*z_t+alpha*v의 부호가 일치한다. t=1에서 alpha를 정확히 0으로 두어 순수 잡음 끝점의 수치 잔차를 없앤다. 그 끝점에도 query 역할 조건을 주입한다. 내부 연산에서 자연어 토큰을 생성하지 않는다.

모델 입력 collate에는 tokens/roles/fact_mask/queries/query_mask만 전달하고 label/type/hops는 별도 supervision으로 분리한다. typed verifier의 사실 종류는 외부 입력 검증에 사용하며 신경망 후보 평가 이전에 정답을 공급하는 기능과 구분되어 있다. Commit은 frozen dataclass이며 표현기는 확정된 Commit을 읽는다. 미검증 결과의 영구 사실 쓰기는 없다.

데이터 oracle에서 부정 비교는 반대 방향의 긍정 비교를 만들지 않는다. strict-order cycle이나 긍정/부정의 모순은 해당 가정 집합 전체의 CONFLICT로 처리한다. 표현을 뺀 typed graph hash로 train/dev/test를 분리한다. 신경망 성능은 근거 검사 이전 후보로 측정하고 initializer와 diffusion을 구분한다.

새 수치 검증이 필요한 항목도 남아 있다. Core self/cross attention에는 query/fact padding mask가 전달되지 않으므로 유효 슬롯 수와 패딩에 따른 민감도를 별도 실험해야 한다. 현재 모델/데이터 fingerprint를 바꿔 재개하지 않기 위해 이번 운영 수정에 포함하지 않았다. 클립 전 전체 gradient norm을 별도 기록하는 기능도 아직 원본 학습 루프에 없다. 이번 wrapper의 finite gradient/parameter 검사는 모든 성공 시도 뒤 실행되지만, 실제 GPU 수치 검증은 수행하지 않았다.

## 저장 성공의 판정

매 시도마다 전체 모델·optimizer·AMP·진행 단계·실제 노출 횟수·sampler/noise/PyTorch/CUDA RNG를 저장한다. Drive가 대형 payload 업로드를 완료하고 원격 size/MD5/파일 버전이 일치한 다음 슬롯별 complete marker를 게시한다. 재개 때 전체 다운로드 SHA256을 검사하고 내부 계약/카운터 및 원본 Trainer의 source/config/data/plan/environment 일치를 검사한다.

latest 포인터가 실패해도 완료 marker가 있으면 해당 검증본은 복구할 수 있다. payload 또는 marker 저장이 실패하면 다음 optimizer 시도를 시작하지 않는다. 현재 검증본을 남긴 채 다른 슬롯을 교체한다. 강제 종료 시 진행 중인 미확정 update 하나는 재계산할 수 있다. 저장이 확인되기 전까지 절대 무손실이라고 보장하지 않는다.

runtime.pt/runtime.json과 각 평가 보고서도 독립 완료 marker를 기록하고 최종 checkpoint SHA에 묶는다. evaluations_complete는 네 평가 및 최종 보고서를 확인한 evaluation_complete.json/marker가 있을 때 판정한다. source나 단순 상태 문자열만으로 가중치 저장 완료를 추정하지 않는다.

## 학습 없이 수행한 검사

29개 검사가 통과했다. 메모리 Drive 및 가짜 시간/optimizer/평가 fixture를 사용했다. 실제 tensor forward/backward, CPU 학습, GPU 학습, 원격 대형 checkpoint 전송은 수행하지 않았다.

- payload/marker/포인터 저장 중단, 업로드·다운로드 손상, 이전 슬롯 복구
- source/wrapper 불일치 차단, 기존 미검증 가중치 덮어쓰기 차단
- marker 필수 필드·slot/file 불일치·중복 marker 처리
- 보고서 완료 marker, checkpoint scope 불일치, 보고서 SHA 손상
- 저장 시간 포함 고정 deadline, 이력→저장 순서, 저장 실패 후 다음 update 차단
- 미완료 optimizer 상태 저장 차단, phase 전환 저장, gate 실패 종료
- 완료 checkpoint의 평가 전용 재개, 평가 중단 뒤 누락 항목만 실행
- 원격 runtime metadata가 실제 게시된 binary hash를 참조하는지 확인

집에서 실제 실행한 뒤 확인할 항목: T4 장치와 peak VRAM, loss/finite gradients, 첫 초기값 및 update 저장의 실제 Drive acknowledgment, 별도 fresh runtime에서 optimizer/AMP/RNG strict 재개, 모델 재로드와 후보 성능. 작은 notebook/ZIP의 저장 성공은 대형 가중치 저장 검증을 대신하지 않는다.

## 런타임과 전송의 현실적 한계

Google Colab 공식 FAQ(2026-10-06 확인)는 무료 노트북이 사용량/가용성에 따라 최대 12시간 실행될 수 있고, GPU와 idle/max-lifetime 제한이 동적이라고 설명한다. 코드의 6시간 예산을 늘려도 Google 자원 제한을 늘리는 것은 아니다. 탭/화면을 열어 두고 전원 연결·절전 해제를 하면 노트북 자체의 절전으로 인한 단절을 줄일 수 있지만 Colab이 6시간 유지된다는 보장은 없다.

매 시도 저장은 full model만 약 409 MB(FP32 매개변수 크기 기준)이고 optimizer 상태까지 포함하면 더 커진다. 학습 계산보다 전송 시간이 길어지거나 Drive 용량/작업/전송 제한으로 중단될 수 있다. 두 payload 파일만 존재해도 revision에 따른 용량 증가를 무시해서는 안 된다. 저장 실패 시 이전 검증본을 남기고 중단한다. 세션 제한 우회·유료 구독·다른 사람에게 공유는 추가하지 않는다.

자료: https://research.google.com/colaboratory/faq.html ; https://developers.google.com/workspace/drive/api/guides/manage-uploads ; https://developers.google.com/workspace/drive/api/guides/manage-revisions

## 후속 연구와 현재 실적 구분

이번 읽기에서 FDT100_training_status.txt의 파일 ID와 수정 시간은 기존 기록 그대로이고 weights_saved_to_drive=false, 최근 640회/선택 경로 2,240회였다. FDT100_experiment_summary.json의 수정 시간도 그대로였다. 새 결과나 가중치 완료 marker를 찾지 못했으므로 새 성능 향상·새 학습 비율을 주장하지 않는다.

새 복구 Plan은 480/320/640회이며 성공 gate에 따라 최대 1,440회다. 이 phase update 비율과 데이터 query/grounding 클래스 노출 비율은 별개의 지표다. 단계 비율을 데이터셋 노출 비율이라고 보고하지 않는다.

v2의 graph_test/heldout_phrase_test/larger_graph_test는 원래 고정된 benchmark seed로 재검사한다. 이전 연구에서 한 번도 보지 않은 새 독립 평가 세트라는 주장은 하지 않는다. 과거 선택 경로의 warm start와 이번 초기값부터의 학습 경로도 달라 과거 정확도의 재현을 보장하지 않는다.

표현 계열 분리·부정 일반화, query exposure 네 클래스 25%씩, grounding 네 클래스 25%씩이 후속 우선순위다. grounding60/semantic15/diffusion25% 추가 1,000회는 미실행 가설이다. 이 복구 notebook에 이를 적용하지 않았고 실제 학습된 비율로 표기하지 않는다. 표현별 grounding 및 부정 관련 후보/근거 정확도는 별도 분류 평가가 필요하다. 원본 compute_metrics의 direct/transitive만으로 부정 일반화 전체를 검증할 수 없다.
