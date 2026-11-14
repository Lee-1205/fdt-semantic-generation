# 문서 언어학습 V5→V6 — 추가260회 완료, 초기 언어학습 신호와 생성 과제

사용자 지적에 따라 해석을 명확히 한다. 아래 FAIL은 해당 한정 실험의 사전 자유생성 조건 미달이며 모델·diffusion 구조 전체의 실패 판정이 아니다. 실제 학습량은512문서/약0.384MB의 SGD 목표로 초기 단계다. 관계 예측의 높은 관측 성과와64/64 개발 문서 NLL 개선은 보존한다. 이력 확인→1차 자료 검색→직접 계산 원칙 및 후속 학습 판단은 `FDT_HISTORY_FIRST_LEARNING_DECISION_20261028_KR.md`에 기록했다. 숫자·원시 결과·과거 미달 판정은 바꾸지 않는다.

사용자의 “그리고 학습계속해”에 따라 Root가 저장 상태를 복원해 새 한정 실행을 마쳤다. 새260attempt/260update/AMPskip0이며 누적409attempt/408update다. 과거149번째 실패 시도를 되감지 않았다. 원래 typed 모델과 V4·모든 실패/가중치/Adam/scaler/RNG/원문은 보존했다. 새600초/14,000호출 예산도 닫혔으며 미사용분은 재사용하지 않는다.

## 검증된 결과

| 항목 | 학습 전 → 누적408회 후 |
|---|---:|
| dev64문서/444행 teacher-forced byte NLL | 10.00424 → 2.43480 |
| 감독53,150symbol의 byte accuracy | 13.2437 → 31.3791% |
| 실제 문서 끝에서 teacher-forced EOS argmax | 52/64 → 0/64 |
| 별도16prompt 자유생성 UTF8 | 16/16 → 16/16 |
| 같은 자유생성 EOS | 16/16 → 0/16 |
| 최종 자유생성 평균 반복4단어 비율 | 60.97% |

Baseline은 V5에서 이미 계산한 동일 초기 모델의 저장 결과를 SHA로 묶어 재사용했다. 148회 중간 능력 평가는 하지 않았으므로 위 개선을 추가260회나 AMP 수정 하나의 단독 효과로 돌리지 않는다. Root가 full logits에서 안정적인 float64 log-sum-exp/argmax/실제 EOF 위치를 독립 재구성했다. Worker NLL과의 차이는 baseline 약4.6e-7/final 약6.8e-8nats다. 추가 모델 forward는0이었다.

최종16개 자유 출력은 모두129byte 상한까지 이어졌고 `the`와 `and` 등을 반복했다. 평균 단어의82.29%가 `the`였으며 문맥에 맞는 이야기 이어쓰기로 인정할 수 없었다. 예를 들어 서로 다른 문맥에 ` and the the the ...`가 반복됐다. Baseline도 이전 비교 태스크의 깨진 조각을 반복한 상태였으므로 유용한 문장 생성은 전후 모두 미입증이다. Zero-VP 출력도 UTF8 16/16/EOS0/16/반복4단어67.34%로 실패했다. 이 관측만으로 diffusion 기여나 반복의 단일 원인을 확정하지 않는다.

실제 EOF의 gold-logit margin 평균은 +9.336→−2.968, 순위 중앙값1→3이었다. 종료 문제는 자유 rollout에만 한정되지 않고 정상 teacher prefix에서도 관측됐다. 원래 훈련 대상384,675bytes에 EOF512개로 감독 symbol 중 EOF 약0.13%라는 희소성은 관련 가설이다. 가중치·loss 기여율만으로 원인을 증명한 것은 아니다. 개발 목표의 NLL 감소 조건은 통과했지만 자유생성 보존 조건을 실패해 **전체 개발 목표는 FAIL**이다.

## 실제 언어층·확산 gradient와 복구

V5 첫 실행은 CUDA 결정론을 유지한 mean CE 경로가 지원되지 않아1attempt/0update로 종료했다. 원본을 변경하지 않고 별도 v2에서 reduction-none/sum으로 같은 CPU 값·gradient를 유지하는 수정을 했다. v2는148update 후 감독 atomic 임시파일이 교체되는 순간 용량 `stat` 검사가 실패해149attempt에서 멈췄다. 해당 fullstate595,445,657B SHA `b8d29965fa01430e47db68669667ba20a397bee0877e3e0ec563f653413a31e8`를 그대로 보존했다.

Root는 실제 Adam state에서144개 학습 대상 parameter 중74개만 state가 생긴 것을 확인했다. 언어 마지막2block의18tensor는 실제로 변했지만 VP velocity matrix는 변하지 않았다. CPU bfloat16 autocast에서 no-grad 자유 proposal이 캐시한 detached cast 때문에 이후 supervised VP 경로의70개 matrix gradient가 없어지는 현상을 재현했다. 이전148회에 모든 학습 대상51,173,122parameters가 갱신됐다는 주장은 성립하지 않는다.

V6는 no-grad proposal 안에서만 autocast cache를 끄고, optimizer 전에 모든 학습 대상 gradient 존재를 검사한다. CPU forward 값은 동일했고 required gradient가 모두 생겼다. 기존 가중치·74개 populated Adam state·scaler1024/tracker148·모든 저장RNG·history·cursor1192와 CUDA RNG를 실제 복원했다. 알고리즘 BEFORE=AFTER digest `0ff5f38ebb27b68fb5b2307013a74164aa4bd4e7527500392065851b0a3c27e1`를 GPU import 직후와 Root CPU 재검증에서 확인했고 첫 새 SGD 전에 import fullstate를 저장했다.

V6 최종 Adam state는144개다. 이전74개는step408, 새로 gradient가 연결된70개는step260을 보존한다. 확산 velocity matrix의 실제 변화와 동결된 prefix/codec의 불변도 확인했다. 앞7개 언어 block/embedding/pool은 계속 동결이고 마지막2개14,158,848parameters와 VP/decoder가 학습 대상이다. 모델은100,928,002parameter hybrid이며 순수 diffusion 생성기나 범용 언어모델 완성으로 부르지 않는다.

## 노출·실행 근거

- 누적 selected3,271행/384,675target bytes/512EOF/512문서, encoder에 실제 진입한3,267행/384,163bytes다. SGD가 실제 반영된3,263행/383,651bytes와 구분한다. 과거 실패149의8행을 다시 학습하지 않았다. 최초 별도 실패 실행의8selected/4processed는 이 누적에 더해진 별도 중복 노출이며 소급 삭제하지 않는다.
- V6 actual calls는 encode635+denoise10,680+decoder1,663=12,978/새상한14,000. v2와V6 누적은21,153calls다. 최초0update 실패의2,568calls도 별도 역사로 남는다.
- Worker202.369초. Native owner43956→22076는12:58:53.143/.172KST, worker42796→4752는12:59:00.570/.586KST에 생성됐고 Root가 parent/command/생존을 관측했다. Worker/owner exit0/COMPLETE,13:02:32KST freshPython0/모든owned·proxy 종료 확인 후 상세 분석했다.
- Import149와 global256/384/409의4개 fullstate를 전체SHA/CPU safeload/재귀 digest로 검증했다. Source fullstate와 정확한 model/Adam/scaler/모든RNG/order/cursor/history/exposure/call/evaluation refs 복구, 이후 실패149 하나만 남고 추가260행 업데이트 기록이 이어짐을 확인했다. 새로운16CPU 검사에는 fullstate import/실제 enabled CPU scaler/AMP skip/마지막7행/atomic race/전체 post-Popen 자연·예외·UNCONFIRMED/실제 Windows proxy-child 종료가 포함됐다.

Plan SHA `6e9d5736ae9949f988411898935f712314b3dd848db8f5a3d30d6d4aa170ab86`. 최종813,930,293B SHA `914a2de69bc98dc9a8b182a08f25daa7c8a09285fc084ea87d1da0d71f5ec013`, algorithm `a5eab1a1945f0838f7e8077b02e44d80dd0b9ce5446c0b19192871779201eceb`, result SHA `598192081260deafb12568f7dce413d3e8fc4442c1f645cd50cc2b734f744f9c`다. 근거는 run/root_result.json과 scripts/review_text_language_v6_20261028_cpu.py다.

다음은 `FDT_GENERATION_ROLLOUT_STOP_NEXT_20261028_KR.md`의 종료·반복 정렬 CPU 계약이다. 같은 epoch를 자동 추가하거나 낮아진 NLL만으로 승격하지 않는다. Timer는 사람 지시대로30분/문구‘재개’만 유지한다. 에이전트/E2final 접근/외부 게시/유료 cloud/원본 삭제는 사용하지 않았다.
