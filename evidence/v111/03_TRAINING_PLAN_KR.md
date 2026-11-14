# 03. 실제 봉인한 유한 학습 계획

2026-11-14. ROOT 직접, 에이전트0. VERIFIED=이번 실제 코드/실행·독립 재계산, OBSERVED=기존 저장 관측, INFERRED=해석, PROPOSED=미실행 제안.

## VERIFIED

실제 plan SHA는 `1a914ad1880afc87e5e9053c08c5c46d787283e654e80f24a66598fe1bd9672a`이며 [봉인 plan](plan.json)에 저장됐다. 준비 파일과 실행 plan을 구별한다.

| 항목 | 실제 설정 |
|---|---:|
| 초기 θ |V110 hidden520,9,576,426개 전체 가중치 warm ONLY|
| optimizer |fresh AdamW, LR1e-4, decay.01, clip1|
| scaler |CPU enabled, 초기1024, growth interval16384|
| TRAIN / epoch / 노출 |134 /16 /2,144|
| 답변 UTF8byte 노출 |37,088|
| batch/micro/attempt |8 /1 /268|
| sampling |target-free16step|
| auxiliary time |각 row가16개 실제 sampler grid를 한 번씩|
| 전체 예정 호출 |40,032thought+224D=40,256; E0|
| 상한 |41,000calls, owner1,200초, modelcut1,080초, reserve120초|
| 저장 |C 연구 run 전체1GiB,0/134/268와STOP·원시trace·임시파일 포함|

58개 실제 역전짝과18개 singleton을 유지했다. singleton 두 개를 일정상 같은2-row block에 넣어도 실제 역전짝이라는 label을 붙이지 않았다. 16epoch를 하나의 stream으로 연결해8행 batch268개를 만들었고, 실제짝은 전부 batch 안에 있었다. 모든134행을 각 epoch에1번씩 사용했고 순서·시간·cursor를 저장했다. **주 학습의 noise는 행마다 독립 draw다. 실제짝은 같은 time/batch였지만 주 학습에서 공통 noise를 사용했다고 주장하지 않는다.** 공통 noise는 별도 사후 paired 평가에서 사용했다.

초기·중간·최종 fullstate는 model/optimizer/실제 scaler/모든 RNG/순서/cursor/누적 gradient/attempt-update/노출/call/phase/codec digest/loss 설정을 저장한다. 마지막 state는 학습 DONE 직후, 최종 평가 전에 저장됐다. 최종224개 평가 reference는 `assessment.json`과 result에 별도로 보존했다. 존재하지 않는 post-evaluation fullstate나 실제 CUDA 재시작을 주장하지 않는다.

사용자 최신 저장 지시에 따라 실행 C 하한은50GiB, OS owner의 조기 정리선은54GiB다. 시작 전에는 `100GiB+실험 전체1GiB+64MiB`를 확인해 우리 산출물을 모두 써도100GiB가 남도록 계획했다. 실제 시작 전 약107.557GiB였다. D에 새 실험을 저장하지 않았다. 중복 Python0과 실제 lock 사용 가능성, RAM8GiB 이상을 확인했다. 실제학습 중 ROOT는 과학 분석을 중단하고 독립 OS 감독·진행 상태만 확인했다.

## OBSERVED

기존 호출 비례 예상은 약637초였고 실제 worker는536.539초였다. 사전검사와 준비 시간은 이 worker 시간에 합산하지 않는다. 기존 예산을 연장하거나 남은 update/call을 사용하지 않았다.

## INFERRED / PROPOSED

16epoch는 충분성 보장이 아니라 첫 유한 작업 단위다. 일반화·기억 인과·Composition은 이 plan의 성공 기준이 아니다. 이후 추가 학습에는 새 목적·새 상한·새 plan이 필요하다.

실제로 실행한 명령은 다음과 같다. **현재 이 봉인 프로필은 종료됐으며 재실행이 차단된다. 기록용 명령을 새 학습 명령으로 재사용하지 않는다.**

```powershell
& 'PROJECT_ROOT/.venv/Scripts/python.exe' -B -X utf8 'PROJECT_ROOT/scripts/supervise_answer_meaning_v111_c.py' --plan 'evidence/v111/plan.json'
```
