# 04. 실제 학습 결과와 복구 검증

2026-11-14. ROOT 직접, 에이전트0. VERIFIED=이번 실제 코드/실행·독립 재계산, OBSERVED=기존 저장 관측, INFERRED=해석, PROPOSED=미실행 제안.

## VERIFIED

실행은 정상 완료됐다. **268attempt/268update/AMPskip0**,2,144행,37,088답변 UTF8byte다. 실제 호출은40,032thought+224D=40,256이며 encoder0/훈련decoder0/훈련언어loss0이다. worker 시간은536.539초다. owner/worker exit0과 모든 owned/proxy 종료를 확인했다.

| 실제 학습 loss,8개 update 평균 | 처음8 | 마지막8 |
|---|---:|---:|
| body MSE |1.873389|1.094052|
| tail MSE |0.920715|0.537094|
| VPaux |1.335723|1.181045|
| 합계 loss |2.052997|1.239012|
| clipping 전 norm |0.884699|2.346127|

gradient가 끊기거나 모든 update가 skip된 실험은 아니다.64개 θ tensor가 실제 변했고 E/D 전체 buffer/parameter 경계는 같았다. 실제 로드 구조와 최종 저장 가중치 모두 velocity512×520이며, LayerNorm 이후 유효 rank512를 확인했다. NaN/Inf 없이 끝났다.

| checkpoint | byte | 파일 SHA |
|---|---:|---|
|0|38522912|`bef27266041f1ad72c3051b4519675fdca81a45cc6d74f69631a426fd885915b`|
|134|115291709|`e9848b9a0e36542b840626754429ae9e37fbf320e549d8e9551219a45e407522`|
|268|115324221|`e905927d288dff3f91dcb0c4eb53257503fd92929bbd50248b371db183fb7f76`|

ROOT는3개 fullstate의 SHA/weights-only safeload/재귀 digest, 초기 warm θ 전부 일치·빈 Adam, 최종 Adam step268/scaler1024 tracker268, 모든2,144개 noise draw의 정확한 재생, 순서·노출·cursor를 독립 재구성했다. 실제 CPU 사전검사에서는 populated READY의6-row 잔여 batch와 다음 update, 중간 gradient와 입력 checkpoint 불변을 검증했다. 최종 사후 평가에서 정답 없이 생성한 실제 의미를 저장했다.

사전검사의 같은 seed 출력은 bitwise 같았고 train/eval16step 최대 차이는0.00000215였다. 최종 의미 분산도 저장 raw에서 계산했다. rank나 분산이 정상이라는 사실을 의미 생성 성공으로 해석하지 않는다.

## OBSERVED / INFERRED

loss는 내려갔지만 의미 변환은 아직 확인되지 않았다. 다음05의 실제 출력이 우선이다. 모델을 자동 승격하지 않았고 다음 SGD를 시작하지 않았다. 원래41,000예산의 미사용744call은 닫혔으며 재사용하지 않는다.

필요한 전체 TRAIN/실제32긍정 역전쌍 검사는 별도 plan으로만 수행했다. optimizer0,134행,2,144thought+134D=2,278calls,14.953초였다. 별도 cap2,300/120초(90+30)/64MiB도 닫혔다. 원래 미사용 budget의 재사용이나 학습 연장이 아니다.

[독립 저장 검토](root_saved_review.json) · [전체 TRAIN 별도 검토](../../docs/LOCAL_ASSETS.md)
