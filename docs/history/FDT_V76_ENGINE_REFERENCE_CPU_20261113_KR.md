# V76 teacher/head engine와 과거 Commit 참조: CPU 계약

2026-11-13 ROOT 직접 작업. 새1,024행 자료와 capped 교정 손실을 실제 teacher/head 상태기계에 연결한 **engine 부품**을 준비했다. 최종 대형 worker·실행 plan·owner·봉인·GPU 학습은 아직 없다. 현재 연구 성능은 V75 그대로다.

`scripts/joint_quantity_engine_v76.py`는 검토된 V75의 loss-only teacher/원문 재인코딩 head/phase·STOP 복구 흐름을 유지한다. 새 stage/schema와 V76 손실을 연결하고 분모 `max(W,C/8)`와 최대 위치별 증폭8을 상태 메타데이터에 기록한다. 모델의 기존 원문→K_P/K_C 퍼지→Commit→언어 경계와 readout7개만 학습하는 정책은 유지한다. 정답이나 teacher 확률을 추론 feature로 넘기지 않는다.

새 tiny CPU fixture에서 다음을 확인했다.

- 부분 teacher 상태와 나머지 teacher 완료 상태가 소유 파일 경로를 제외하고 일치한다.
- teacher 완료→head 경계의 모델/optimizer/scaler/모든 저장 RNG/cursor/order/exposure가 정확히 복원된다.
- populated7Adam과 enabled CPU GradScaler를 포함한 parent-linked 복구 뒤,같은 overflow와 다음 update를 실행한 전체 상태 digest가 일치한다.
- AMPskip도 row/cursor/노출을 소비하고 head에서 실제 source를 다시 인코딩한다. feature cache는 쓰지 않는다.
- teacher/head 도중 STOP의 in-flight 상태를 정확 복원하되 같은 행을 자동 재실행하지 않는다.
- 상류 고정,잘못된 source epoch 거절,실제 history의 새 loss cap 연결을 확인했다.

CPU fixture 성공은1.375초,출력9,429,061바이트였다. 최대60초/32MiB 출력·예정 write 포함/C100GiB 보호를 별도 검사했다. 한 완주 경로는 encode6/readout6이다. 이것은 smoke 크기 모델의 계약 검사이며 실제 V75 대형 부모 forward·학습이나 CUDA 재시작 증거가 아니다. 처음 CPU test wrapper는 시작 시각 변수를 선언하지 못해 첫 encode 이전에 NameError가 났다. 실패 source/fixture를 보존하고 새 fixture에서 선언을 고친 뒤 성공했다. 과거 봉인 engine/예산은 바꾸지 않았다.

코드 이력에서 다음 호환성 문제도 미리 찾았다. V75의 일부 관측 패널 raw는 Commit을 직접 담는 대신 V74 raw를 가리키는 `fuzzy_commit_reference`를 가진다. 기존 `commit_from_cache`는 직접 `packet['fuzzy_commit']`만 읽으므로 다음 driver에서 V75 raw를 그대로 넘기면 KeyError가 발생할 수 있다. **미래 연결에서 발생할 문제를 찾은 것이며,V75 실제 평가가 실패했다는 뜻은 아니다.** 당시 V75는 직접 Commit이 있는 V74를 입력으로 사용했다.

새 `scripts/saved_commit_reference_v76.py`는 프로젝트 내부·plan SHA pin·원문 source id/hash·상류 digest·Exact0·tensor shape/수치와 target 무결성을 확인하고 한 단계의 직접 Commit 참조만 해소한다. 재귀/중첩 참조·틀린 source/상류·pin 누락·target 불일치를 거절한다. Label은 무결성 검사에만 쓰고 반환값은 신경 Commit tensor와 provenance뿐이다. 새 TRAIN/새 평가 source를 과거 다른 행으로 대체하는 기능이나 Exact 원문 회수기가 아니다.

5개 좁은 CPU packet 검사는 직접/한 단계 참조의 tensor 일치,pin/SHA/상류/source/Exact/shape/target 오류 거절,중첩·모호한 representation 거절을 통과했다. 최초 기록 파일의 줄바꿈을 literal escape로 쓴 포맷 오류도 원본을 남기고 새 작은 fixture에서 고쳤다. 최신 파싱 가능한 증거는 `root_commit_reference_contract_v2.json`이다. helper는 아직 최종 assessment driver에 연결하지 않았다.

증거 위치:
- `research/artifacts/joint_quantity_20261113_v76_cpu_v1/root_engine_contract_v1.json`
- `research/artifacts/joint_quantity_20261113_v76_cpu_v1/root_commit_reference_contract_v2.json`
- `outputs/FDT_V76_ENGINE_CPU_TEST_REV1_FAILED_20261113.py`
- `outputs/FDT_V76_REFERENCE_CPU_TEST_REV1_20261113.py` 및 최초 기록 사본

다음은 V75 linked 최종의 모델을 복원하고 저장 base는 기존 standalone V74로 유지하는 실제 대형 initializer,새 평가 명세/자료,정확한 코드 비용·출력·임시·STOP 예산에 맞춘 worker/plan/owner 연결이다. 실제 대형 CPU preflight와 독립 validate가 완료되기 전에는 실행 준비 완료라고 하지 않는다. 이전 최종 체크포인트·Adam/scaler/RNG·원문·실패는 모두 보존한다.

C여유가 약100GiB라 새 GPU의100GiB 보호+8GiB headroom+출력/임시/STOP 조건은 여전히 부족하다. 기존 worker의101GiB+write reserve guard를 낮추거나 과거 시간/calls를 재사용하지 않는다. 새 대형 GPU 실행·성능 변화·whole language cycle 완료·디노이징은0이다. 에이전트도 쓰지 않았다.
