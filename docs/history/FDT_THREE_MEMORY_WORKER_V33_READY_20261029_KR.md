# V33 실제 worker/독립 OS 감독 준비와 새 실행 저장 계약

신경 후보·복구 driver의 이전 CPU9검사는 그대로 보존했다. 이번에는 실제 실행 worker와 Torch를 import하지 않는 독립 OS 감독을 연결했다. **이 문서 작성 시 GPU 실행은 아직 0**이다.

새 worker CPU7검사가 통과했다. 실제 plan/data/schema, 새 유한 순열, 저장한 Exact query/key/valid mask로 전체 score·NULL·후보 확률 재구성, per-token μ/value로 FP64 long/recent N/D 재구성, 원문 feature만 받는 BOS 복원 API, 감독 정상종료/예외/불확실 종료/STOP파일 쓰기 실패 때의 owned cleanup, 첫 invoked.py 중복 판별을 검사했다. 마지막 resource 계약 변경에는 관련5검사만 다시 통과시켰다. 반복 횟수를 새 독립 증거로 더하지 않는다.

실제 worker의 CPU preflight를 독립 owner의 `--validate`로 수행해 PASS했다. 처음에는 과거 외부 평가 ledger의 total만 옮겨 run/planSHA/calls provenance가 빠져 거부됐다. 이는 **GPU Popen 전 CPU metadata 오류**이고 모델 실패가 아니다. 원문 plan과 실패 코드/기록을 남겼고 전체 ledger를 정확히 보존한 r2/r3 계획이 통과했다. 실제 source safeload/기존81 Adam/scaler/RNG 이관만 했으며 실제 크기 forward/optimizer/CUDA는 0이다.

최종 owner로 Windows 실제 Popen proxy→native CPU fixture child의 identity/parent/command/자연종료/모든 owned 종료를 확인했다. 정상·예외의 mock CPU검사와 별도로 남겼다. CPU fixture는 실제 모델 worker가 아니며 신경 성능/실제 GPU 복구 성공으로 세지 않는다.

## 실행 전에 명시적으로 변경한 저장 계산

이전 root 기록의 ‘40GiB 시작선은 낮추지 않는다’라는 **예정 정책을 이번 미실행 새 계획 r3에서 변경했다**. 감추거나 기존 실행에 소급 적용하지 않는다. C20GiB 보호, checkpoint 임시 여유, 시작8GiB headroom이라는 실제 자원 조건을 다음과 같이 코드·독립 owner에서 동일하게 계산했다.

`20GiB 보호 + 최대 owned output6GiB + 임시 checkpoint1GiB + 추가 headroom8GiB = 새 시작35GiB`

즉 output이 상한까지 자라고 별도 임시1GiB를 쓰는 보수적 경우에도 보호20GiB 위에8GiB가 남는다. Worker의 C22GiB 중단선/상시 C20GiB 보호/GPU0 free12GiB는 유지한다. 실제 새 fullstate는 source775MB에 새246595parameter와 moments가 붙는 수준이며, 정상3개+긴급1개 및 평가 자료를 모두 포함하는 output 상한은6GiB다. 임시1GiB는 실제 fullstate보다 크다.

이는 **아직 seal/Popen/clock이 없었던 r3 새 실행의 사전 계산**이다. 과거 모든 C40GiB 계획, 종료 상태, deadline/unused budget은 그대로 닫혀 있다. r1/r2 미실행 계획과 검증 기록도 남긴다. 시간/호출/출력 상한을 실행 중 늘리거나 예전 timer를 재개하는 것이 아니다.

별도로 workspace의 아직 압축하지 않았던 Torch DLL14개만 Windows LZX로 처리해 **실제 allocated843,847,344B를 절약**했다. 전후 logical SHA/길이 모두 동일, 삭제0/이동0/모델·GPU호출0/owned compact 모두 종료다. Torch2.10.0+cu128 CPU import가 정상이고 CUDA는 초기화하지 않았다. 전체 C free는 동시에 감소했으므로 다른 OS 사용량/파일시스템 요인이 불확실하며 그 감소를 모델이나 해당 절약량과 혼동하지 않는다. 같은 파일의 압축을 자동 반복하지 않는다.

## 새 한정 학습 목적

보존된 V31 가중치를 새 세 메모리 입력 의미 경로에 적응시킨다. 기존 같은64 TRAIN span/4304physicalUTF8bytes를 새 seed8331–8586의256유한 epoch로 선택한다. 2048attempt/최대2048update,16384rowselection,1,101,824targetUTF8byte노출이며 새 문서가 아니다. 기존81 Adam moments를 유지하고 새 memory head10tensor만 빈 moments로 시작한다. **새 함수의 warmstart이며 동일 trajectory 재개는 아니다.**

새 wall900초/reserve120초/call cap12000/계산상 최대11028/output6GiB로 한 번만 실행한다. 원문→세 메모리→입력 의미 Commit→같은 원문 재복원만 학습하고 denoise/copy bytes는0이다. 시작/끝의64 TRAIN 재복원과 기존256held teacher/고정16held 원문복원, 마지막 held16의 각 memory 제거 조건을 분리한다. Raw 생성은 정답 prefix·길이 없이 같은 입력 의미와 BOS/자기 prefix만 사용한다. 이 반복 TRAIN의 복원 용량을 일반화·완성된 의미 공간·범용LM·3memory vNext 전체 완성으로 해석하지 않는다.

실제 GPU native 생성/parent/command/생존과 durable 상태를 확인한 뒤에만 실행했다고 기록하고, 종료 뒤 모든 owned/proxy가 사라진 것을 먼저 확인한다. 그다음 raw 문장을 직접 읽어 의미 보존과 byte identity를 나눠 판단한다. 원래 모든 원본/weightsAdamRNG/실패/typedcore는 보존하고 E2final/registry에 접근하지 않는다.
