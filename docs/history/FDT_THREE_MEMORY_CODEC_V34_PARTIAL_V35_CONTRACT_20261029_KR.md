# V34 부분 학습 보존과 V35의 별도 잔여 자료 목적

V34는 3407attempt/3406update에서 사전 고정된 최종 평가시간 확보 규칙으로 종료됐다. 실제 supervisor의 `FAILED`와 worker exit2는 그대로 보존한다. 원인은 `TimeoutError('final assessment reserve')`이며, 기록에 수치 오류나 감독 프로세스 오류는 없다. 4096회 완료·한 epoch 완료·성능 향상이라고 보고하지 않는다. 학습 자료가64개인 V33보다 넓은 배치의 실제 처리시간이 길었고, 이전 CURRENT에도 고정 시간 내 전체 완료가 보장되지 않는다고 기록했다. 기존 상한을 늘리지 않았다.

독립 owner와 ROOT가 모든 native/proxy 종료를 확인한 뒤 ROOT가 import0/mid2048/emergency3407의 전체 상태를 CPU에서 검증했다. 모든 모델·Adam 값은 유한하고, 91개 Adam ownership/모멘트·step, scaler, 저장된 모든 RNG, 동결된 원래 추론 모듈/언어 앞부분, 유한 순서/커서/노출을 보존했다. 한 번의 AMPskip은 attempt1722로 행8개를 정상 소비했고 되감지 않았다. 새 encode3407+decoder6814=10221호출, denoise0. V34의900초/17000호출 예산은 CLOSED다. 미사용6779호출·시간을 재사용하지 않는다.

실제 학습 노출은 TRAIN27256개 구간/8184문서/2,433,818 UTF8 target bytes다. 원래 순열32768구간의 앞부분이며 중복0이다. 남은 suffix는689배치/5512구간/492660bytes다. 최종 복원 평가는 실행되지 않았으므로 이 부분 학습의 새 복원 성과는 아직 없다. 온라인 loss가1.48569→1.19279로 줄었지만 다른 배치 구간 평균이므로 고정 평가 성능으로 해석하지 않는다.

V34 emergency fullstate779455221B SHA `f75165174549116d6c731d0bac9a867187120896c64c6aecc628b7910a1c2cfb`, algorithm `e2c596a7ac33d4480fe9a7ba4aa1ec30df92727b81b2d3caf728a297e97cf05e`. 원래 planSHA `7f0014a8dc35a0120f0cc4d663330baa3ea7633e858cee9a7a6fb065b65877e0`. 근거는 run의 `root_result.json`과 `scripts/review_three_memory_codec_v34_partial_cpu.py`다. 추가 모델 forward/CUDA평가0이다.

## 새 V35 목적

**기존 한 유한 순회의 아직 처리하지 않은 suffix5512개 구간만 학습하고, 완성된 노출 뒤의 원문 복원을 평가한다.** 이미 처리한27256개를 다시 시작하지 않는다. V34의 시계·호출 예산은 재개하지 않는다. 별도480초/reserve90/modelcutoff390초,689attempt,6000호출(코드 최대4839),출력6GiB,시작C35GiB/GPU12GiB와 새로운 output/planSHA를 사용한다. 새 파라미터·새 모멘트·데이터·epoch 추가는 없다.

V34의 모든 모델/91개 populated Adam/scaler/저장된 모든 RNG를 정확히 이관한다. 원래4096배치 cosine 학습률의3408번째 위치에서 시작해4096번째 위치까지 진행한다. 학습률을 재시작하거나 optimizer step을 초기화하지 않는다. 부분 원문 순서도 보존한다. CPU의 실제 next update를 중단 없는 같은 순회와 비교해 모델·Adam/scaler/RNG 일치를 확인했고, enabled CPU scaler skip와 마지막7행을 포함한 전체 복구를 검사했다. 새 plan/네 가지 OS exit 경로까지7개 CPU검사 PASS18.99초. 실제 전체 크기 worker의 CPU preflight를 독립 owner --validate로 통과했으며 실제 Windows CPU fixture의 native identity와 전체 종료도 확인했다. 이것은 GPU 실행 증거가 아니다.

3종 메모리·언어→입력 의미→동일 언어 목적과 디코더의 Commit 경계는 그대로 유지한다. K_P는 learned prior이고 certified Writer는 없으며, K_C는 입력 전체의 per-token long/recent N,D, Exact는 입력에서 만든 학습된 index 읽기다. Exact literal selector/transport나 vNext 전체의 층별 prefix 구현이라고 부르지 않는다. 추론 디노이징0·전문 자동 복사0·raw H/원문 바이트의 디코더 우회0이다. 미래의 명시적 Commit 전문 인용 기능은 보존한다.

최종 평가는 기존 V33의 동일 함수 평가를 저장 기준선으로 사용해 초기 GPU 평가를 반복하지 않는다. 학습64와 개발16의 동일 입력 의미 기반 BOS 재복원은 ROOT가 모두 직접 읽고, 문자열 유사도와 실제 명제·부정·수치·이름 보존을 구분한다. teacher byte 지표는 보조로 남긴다. 기존 개발256은 반복 관측 패널이며 새 blind 성능이라고 하지 않는다. 원래 학습·실패·원문·모든 Adam/RNG는 보존한다. E2final/registry 접근은 없다.
