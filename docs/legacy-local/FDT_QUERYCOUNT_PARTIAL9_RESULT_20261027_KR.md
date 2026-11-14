# 부분9 가중치 읽기 전용 평가 — Root 직접 재시도 결과

최신 사람이 “재시도하고 에이전트 쓰지 말고 네 선에서 해결해”라고 지시한 뒤 root가 에이전트 없이 직접 준비·실행·감시·원시 결과 재검산을 완료했다. 추가 optimizer 학습은 없으며 기존9회 가중치를 평가했다. 기존128회 학습 실험은 여전히9/128 미완료다.

## 실제 실행과 종료

새 output은 runs/querycount_partial9_readonly_20261027_user_retry_v1, plan SHA db59967eb25f79d68088d8edb81d8a3fe32ef5d0d448730c8b10679aedf0ae7c다. 고정 owner120초/종료 reserve30초/실제 encode+denoise 최대720회 안에서 계산을 끝냈다. 실제 encode140+denoise560=700회, readout280회, optimizer0/checkpoint0다.14개 이름 중 Q16 dev는 primary와 정확히 같은 결과를 재사용해13개 실제 패널·560행·5392개 상관된 score 위치를 평가했다.

Root가19:32:39.712/19:32:39.743KST owner19788→28148,19:32:45.736/19:32:45.751KST worker20960→26016의 실제 생성·parent·command·생존을 관측했다. Worker는 결과를 저장하고 exit0으로 종료했다. 모든4개 프로세스 종료와 native Python0을 root가 재확인했다.

감독은 종료 순간 CIM이 PID/parent는 주지만 CommandLine/ExecutablePath를 비운 일시적 행을 반환하자 identity mismatch로 잘못 판단했다. 따라서 원래 감독 final_status는 STOPPED로 남겨 보존한다. 이는 모델 계산 실패가 아니다. 저장된 worker 결과 READONLY_PARTIAL9_ASSESSMENT_COMPLETE와 모든 raw를 root가 독립 재검증해 평가 완료를 확인했다. 원래 감독 상태를 소급 COMPLETE로 바꾸지 않는다.

Result5667412B SHA e391d4505acc054b889126c564e9c1b3f7f4ea977828cf1de00fe48e03d0f70c. Root 검증은 run/root_result.json에 있다.

## 같은 seed17·같은 입력의 결과

| 관측 패널 | 학습 전 bound baseline | 부분9회 뒤 | 남은 오류 |
|---|---:|---:|---:|
| Primary / Q16 dev alias | 99.9023% | 100% | 0/1024 |
| Uniform8 | 98.8281% | 100% | 0/512 |
| Old role128 | 99.8355% | 99.9452% | 1/1824 |
| Normal16 | 100% | 100% | 0/240 |
| Large16 | 100% | 100% | 0/256 |
| Paired16 | 100% | 100% | 0/240 |
| Q1 dev | 100% | 100% | 0/64 |
| Q4 dev | 99.2188% | 100% | 0/256 |
| Q8 dev | 100% | 100% | 0/512 |
| Transfer Q1 | 100% | 100% | 0/16 |
| Transfer Q4 | 98.4375% | 100% | 0/64 |
| Transfer Q8 | 100% | 100% | 0/128 |
| Transfer Q16 | 100% | 100% | 0/256 |

Q1/Q4 dev의 같은 비중 평균 오류율은0.390625%→0으로, 오류율 절반 및 각 Q dev99% 기준을 이 단일 seed17 관측에서 만족했다. 사전 보존 기준은 모든 패널 통과다. Grounding/initializer도 모두100%다. 한 번 평가이므로2연속 보존 통과를 주장하지 않는다.

Root는5392개 score 위치의 gold·logit argmax·confusion·class F1/macro F1·기존 오류 교정/새 오류를 독립 재구성했다. 기존 오류13개 위치 중12개가 교정됐고 새 오류는0개다. 겹친 graph/prefix view가 포함돼12개의 독립 사례로 부르지 않는다. 남은 하나는 old_role_dev128의 graph index9, query[6,3], slot8에서 GREATER를 UNRESOLVED로 누락한 기존 오류다.

진단했던 transfer graph8/query[4,6]의 Q4 누락도 교정됐다. 이전 uniform8의6개 오류도 모두 교정됐다. 이전 mask 직후 하락이 있었다는 역사적 사실은 그대로 보존하며 이번9회 뒤 관측에서 회복된 것으로 구분한다.

## 가중치·입력·잡음 검증과 해석 한계

Source fullstate SHA5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675가 그대로임을 root가 재확인했다. Model/buffer digest는 평가 전후 모두ce6ba1b3adf1e0b7ba1d3b8d7b17915e6514429cd6a518057dae1211958614db이며 global Torch RNG 불변 검사도 통과했다. 원래 full AdamW/scaler/모든 RNG/history/cursor를 보존했다. Frozen core SHA도eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352 그대로다. 모든 transfer row의 저장 noise hash는 앞선 transfer의 실제 저장 tensor와 CPU에서 일치했다. Root 검증에 새 모델 forward는 없었다.

이는9회 학습 뒤 관측 개발 성능이 개선됐다는 증거다. 질문 수 문맥·sampler·class weight가 함께 바뀐 계약이므로 단일 요소의 인과 효과로 단정하지 않는다. 같은 관측 dev·상관 graph/prefix이고 transfer16도 이미 관측했다. 최종3noise 평가/새 blind benchmark/128회 실험 완료/자동 승격/범용LM·NER 완성의 증거가 아니다. Given roles·fixed fuzzy closure·target basis 도움과 old-role 잔여 오류를 포함한다. E2 최종 holdout에는 접근하지 않았다.

## 종료 판정 코드 직접 수정

봉인된 worker와 owner-v2는 수정하지 않았다. 별도 supervise_querycount_partial9_20261027_v3.py에서 (1) proxy가 이미 종료하면 stale progress PID의 새 identity 조회를 건너뛰고, (2) 부분 CIM 행은 미확인 생존으로 유지하며 bounded settling 후 실제 부재가 확인될 때만 종료로 판정하고, (3) PowerShell JSON은 UTF8로 고정하며 지역화된 native 진단 출력은 decode 실패로 감독을 깨지 않게 했다. 지속적으로 조회 불가하거나 살아남은 자식은 UNCONFIRMED로 보존한다.

Root가 직접7개 CPU 회귀를 통과시켰고, 별도의 실제 Windows Popen proxy/child·CIM 종료 확인 전체 경로 CPU 검사도 통과했다(총8개, 모델 호출0). V3 SHA467932c1028df88a97a66591129e199d06468b04c9339327e25da6d8a58b2087. GPU 평가를 다시 반복하지 않았다.
