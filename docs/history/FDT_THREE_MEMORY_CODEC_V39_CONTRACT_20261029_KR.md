# V39 두 순회 언어 codec·초기 상태 중복 제거 계약

V38전체 모델/91Adam/scaler/저장된 모든 RNG를 이관한다. 동일111732306parameter/38556357trainable/새parameter0/세K_P-K_C-K_E읽기/Commit-only decoder/p50clean.5+corrupt.5/denoise0/copy0다. K_P certified Writer/K_C 전체층prefix/K_E literal selector·transport 미완성 범위도 그대로다. 원문 전체 복사·raw H 우회·gold inference feature를 추가하지 않는다.

같은TRAIN8192문서/32768구간을 seed38001/38002로 각각 한 번씩, 합65536행선택/5852956target UTF8bytes/batch8/8192attempt 두 순회를 학습한다. 서로 다른 새 문서65536개가 아니다. 기존 baseLR 새 단일8192step cosine/Adam moments·step 보존이다. V38저장 결과를 기준선으로 쓰며 초기·중간 GPU평가를 반복하지 않는다. 관측 dev256 teacher/같은80원문복원/세메모리 통제는 마지막에만 저장한다.

새 고정2100초/reserve180/cutoff1920/max30000calls(calculated27348=24576train+2772assessment), 보수적5update/sec에서1639train+90startup+60checkpoint+60assessment+180종료=2029<2100을 계산했다. 모든 V1–V38시계·미사용 예산은 CLOSED다. import0는 아래 참조 checkpoint, mid4096/final8192/emergency는 계속 self-contained fullstate로 보존한다. 실행 중 삭제·회전·수정·자동재시작·상한확대는 없다.

## 새 저장 계약

학습 전 모델/Adam/scaler/RNG는 보존된 V38부모와 완전히 같다. 초기 checkpoint만 그 부모의 절대 경로·파일 SHA·재귀 algorithm SHA와 새 plan/order/cursor/history/ancestry metadata로 저장한다. 따라서 초기 참조 파일만 단독 이동하면 복구되지 않으며 pinned부모가 반드시 필요하다. 기존 부모는 삭제·재직렬화하지 않는다. 변경된 mid/final/emergency는 참조 없이 독립적으로 복구된다. 이는 입력 원문 Exact 복사나 NN메모리 구현 변경과 무관한 checkpoint 저장 방식이다.

새파일3개×850MiB(중간/최종/마지막후오류emergency 최악경로)+초기참조32MiB+평가128MiB+기타64MiB=2774MiB<owned출력3GiB다. 각 fullstate/reference 파일의 실제 크기와 전체 owned출력 상한을 강제한다. 보호20GiB+실제최대출력3GiB+임시1GiB+추가headroom8GiB=새시작32GiB/GPU12GiB다. 보호공간·headroom은 줄이지 않았고 과거35GiB계약을 바꾸지 않았다. 출력 한도를6→3GiB로 실제 줄인 별도 계약이다. Runtime22GiB+upcoming1GiB검사/항상20GiB보호는 동일하다.

ROOT9CPU PASS39.71초:두 유한순서/실제plan·cost·독립storage산술/ALL91정확이관/scaler skip·마지막7행포함전체복구/참조파일 safe roundtrip·다음update정확일치/부모변조·업데이트후참조거부/4종owner종료경로를 검증했다. 실제 full-size V38 CPU safeload→새worker import→초기참조 safe memory serialization→fullstate digest 동일도 독립OS --validate 안에서 PASS다. 실제 full-size CPU forward/CUDA복구 재실행 증거는 아니다. 새OSowner 실제 Windows CPUfixture7432→19672 identity/자연종료/allgone PASS, Torch import0. CPU 준비는 새 GPU 성능이 아니다.

에이전트·새 corpus·E2final/registry·원본삭제·원문 자동복사·유료cloud·외부게시0. 실제 launch는 모든 새 SHA·sealed plan·native중복0·freshresource 확인 뒤 한번만 한다. 실제 training 중 ROOT는 OS감독만 하고 terminal 뒤 owned/proxy 종료 확인 후 저장된80 source/output을 직접 읽어 의미와 유사도를 구분한다.
