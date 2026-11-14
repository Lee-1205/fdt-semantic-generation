# 다음 한정 실행 계약 초안 — 실제 언어층 + 문서 이어쓰기

상태: CPU 자료/후보 준비 완료, actual worker/owner/plan/seal/launch 없음. Root 직접 수행하고 에이전트를 사용하지 않는다. 기존 V1–V4와 typed/partial9 모든 예산·terminal은 닫혀 있으며 재개하지 않는다. 이번 새 후보/자료 설명은 새 GPU 학습 증거가 아니다.

먼저 실제 문서 이어쓰기를 통해 동결됐던 언어 부분의 일부를 학습한다. 별도 표현·위치 자료는 후속 정렬 연구용으로 보존한다. 두 목적을 한 번의 결과에서 단일 원인으로 주장하지 않는다. 현재 고정 원문512train/64dev, 특징3,271/444행, context256/target128bytes+진짜 EOF의 계약을 유지한다.

## 구현할 실제 계약

- NEW worker/독립 OS owner/실제 실행 schema/prepare 코드가 필요하다. 제안 output은 `runs/text_language_continuation_20261028_v5`이며 아직 생성·봉인하지 않았다. 현재 후보 architecture와 source/자료/hash를 직접 읽고 actual 비용을 산정한 뒤 최종 한도를 고정한다.
- 모델은 CPU 준비 후보100,928,002parameters다. V4에서 VP/decoder weight만 복사하고 두 position table40→129만 명시적으로 보간한다. 새 optimizer이며 옛 Adam/시간/성공 횟수의 재개가 아니다. 언어 마지막2block(index7/8,14,158,848params)+projection+VP+decoder를 실제 학습하고 앞7block/embedding/unused pool/codec/frequencies를 동결한다. 부모와 원래 typed 모델은 별도 보존한다.
- 한정 제안: seed61의 한 finite epoch3,271행, batch8/실제 microbatch4, 최대409attempt/409success. 마지막7행도 버리거나 복제하지 않는다. AMPskip은 attempt/선택 row/cursor/byte 노출을 소비하며 되감기·두 번째 epoch를 하지 않는다. Fresh AdamW: 언어 tail LR1e-5, 나머지 LR3e-5,weightdecay.01/clip1/cosine409; enabled GradScaler1024와 모든 RNG를 보존한다. 이는 아직 실행 plan이 아닌 prospective 선택이다.
- Loss는 후보 VP vMSE+.25cleanMSE+.5byteCE+AR byteCE다. Microbatch를 합칠 때 active target symbol 수로 가중해 한 batch의 objective를 정확히 유지한다. 원문 encoder를 학습하는 단계에서 frozen cache를 대신 써 gradient가 끊기지 않게 한다. 매 microbatch마다 실제 encode1+free VP denoise16+supervised denoise1+decoder1=19calls를 계수한다. 자유 proposal에 gold가 섞이지 않으며 AR는 이전 byte만 본다.
- 학습 노출은 한 epoch target384,675UTF8bytes+512EOF/입력 context706,304반복bytes다. 첫64bytes 문맥 감독 제외량과 physical source417,443bytes를 분리한다. 이 수를 BPE token이나3,271개 독립 문서라고 하지 않는다.
- 평가: 같은 원래 dev64문서/444행의 teacher-forced byte NLL·byte accuracy·진짜 EOF 점수를 baseline/final에서 비교하고, 문서 ID로 사전 고정한16개 이어쓰기 prompt에서 raw 자유생성/UTF8/EOS/반복/문맥 관련성을 별도로 본다. Teacher-forced NLL을 자유생성 성공으로 세지 않는다. 정답 suffix는 유일한 유효 자유 답변이 아니다. 출력/오류를 전부 남기며 unknown/상수 답변의 옛 관계태스크 성공률과 섞지 않는다. Zero-VP control은 같은 prompt/noise/weights와 자유 proposal을 재사용해 decoder 경로만 비교하며 단일 실험의 인과 한계를 밝힌다.
- 비용 제안은 training15,542calls, dev NLL baseline/final3,996calls,16prompt normal+zero 자유생성 baseline/final 최대2,200calls=최대21,738top-level calls다. 실제 worker 구현으로 재계산한 뒤 제안 cap23,000/owner900초/모델 cutoff780초/종료 reserve120초/output8GiB를 확정한다. 추가 GPU smoke/gradient probe나 broad baseline을 선행조건으로 반복하지 않는다. 실제 비용 근거 없이 상한을 늘리지 않는다.
- 전체 fullstate는 새 model/frozen digest/optimizer parameter names+Adam/scaler/모든RNG/cursor/order/history/실제 byte노출/call budget/evaluation references/고정 plan SHA를 저장하고 CPU safeload+재귀 hash로 검증한다. 제안 milestone0/1/128/256/384/409와 cooperative STOP 현재 상태를 보존하며 실행 중 삭제·rotation은 하지 않는다. 실제 modelstate/AMPskip/cursor/마지막7행/전후 nextupdate를 포괄한 전체 driver 복구 CPU 검사와 actual worker preflight/독립 owner validate가 필요하다. 현재 tiny5검사를 전체 복구라고 부르지 않는다.
- Numeric/frozen/gradient ownership/반복 AMPskip/실제 loss 폭증/진행·저장 정체/disk/deadline/process STOP과 각 실패의 현재 fullstate 보존을 구현한다. 낮은 목표 점수는 결과 폐기·자동 추가학습 사유가 아니다. 개발 선별 목표는 dev byteNLL이 자기 baseline보다0.1nats 이상 감소하고 raw 생성의 정상 UTF8/EOS·문맥 관련성이 악화하지 않는지다. 실패도 결과로 보존하며 모델 자동 승격·일반 LM 완성을 선언하지 않는다.

새 owner는 실제 plan schema의 전체 post-Popen 자연종료·예외 cleanup·partial-CIM/살아남은 자식 UNCONFIRMED 경로를 CPU/실제 Windows proxy-child에서 검사한다. Outer logs는 supervisor.stdout/stderr만, worker logs는 OS owner의 xb다. Root가 fresh duplicate0/시작C40GiB(20GiB보호+temp+8GiBheadroom+출력여유)/GPU12GiB를 확인한 뒤 한 번 별도로 seal/hidden launch한다. 실제 nativePID 생성/parent/command/생존과 progress/latest/final을 관측하고 모든 owned/proxy 종료부터 확인한다. 자동 재시작/미사용 시간·call 재사용/old deadline 초기화·확대는 금지한다.

현재 새 모델이 자연어 이야기를 잘 쓴다는 증거는 없다. 위 목적·범위와 복구/실행 계약이 완성되면 사람의 기존 자율 연구 승인 안에서 진행한다. E2 final/registry·기존 held문서의 train 혼입·권리 불명확 corpus·모델 평가보고서 corpus·paid cloud·외부 게시·사용자 원본 삭제는 사용하지 않는다.
