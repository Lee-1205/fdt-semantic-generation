# 다음 CPU 목적 — 종료·자유 rollout 반복 정렬

2026-10-28 갱신: 아래 V6 직후 제안은 역사다. 최신 사람 지시에 따라 V7에서 전체 자료 두 epoch/818update를 실제로 추가했고, 경과는 `FDT_LANGUAGE_CONTINUATION_V7_MILESTONE_20261028_KR.md`에 저장했다. EOF-only 대조1024행을 먼저 학습하지 않고 전체 내용 자료+CE EOSweight4를 선택했다. 다음 작업은 V7 최종 fullstate에서 계속하는 초기 언어학습이며, 품질 미달을 전체 모델 실패나 자동 중단 조건으로 삼지 않는다. 현재 V7도 COMPLETE/allownedgone/CLOSED다. 아래 예산이나 옛 V6 worker를 다시 시작하지 않는다. 관련 문제는 이력→1차 연구→직접 계산 순서로 판단하고, 안전·복구에 필요한 검사만 완료해 학습을 이어간다.

현재 V6는 학습·저장·검증이 끝난 CLOSED 실행이다. 새 GPU/runner/plan/seal은 아직 없다. 사용자의 학습 지속 승인과 Root 직접/noagents 지시는 유효하다. 완료된600초/14,000calls나 이전 clock·미사용 step을 이어 쓰지 않는다.

관측 문제는 (1) 정상 teacher prefix에서도 EOF0/64로 떨어진 종료 확률, (2) 자유 생성16개 전부129byte까지 반복하며 EOS0/16이라는 점이다. 문서 NLL 감소와 구별해 다룬다. 원문512문서/같은 train-dev 경계/기존 원본·모델·AdamRNG·모든 실패를 보존한다. 관측 dev를 새 blind 평가로 부르지 않는다.

다음 Root CPU 작업은 구체적인 소규모 감독 계약과 비용 검증이다.

1. 기존 train 안의 실제 EOF512행과 정해진 hash 순의 비EOF 대조행512개를 별도 후보로 선택해 문서별 중복·길이·byte·EOS/비EOS 지원을 명시한다. dev의 실제 prompt/정답을 train에 복사하지 않는다. 후보1,024행은 새 문서1,024개가 아니다. EOF만 학습해 무조건 빨리 끝내게 만들지 않도록 비EOF에서 잘못 종료하는 점수도 함께 보존한다. 아직 이 후보를 작성·학습하지 않았다.
2. 문서/행 단위 정규화와 훈련 전용 EOS 가중치 후보의 정확한 loss mass를 CPU 계산한다. EOS 희소성은 가설이며 독립 원인이라고 단정하지 않는다. 기존 VP v-prediction/latent/시간잡음과 causal decoder 정렬을 유지하며 무조건적인 EOS runtime 삽입이나 정답 길이 강제 종료는 평가에 쓰지 않는다.
3. 반복 개선은 teacher-prefix 점수와 실제 rollout을 구분한다. 필요할 때 모델 자체의 짧은 train-prompt rollout에 반복 unlikelihood 보조손실을 적용하는 후보를 CPU 계약으로 검토한다. 오답 자기 prefix에 gold 답을 무조건 붙여 정답쌍으로 취급하지 않는다. 반복 규칙을 쓰면 훈련용 휴리스틱임을 밝히고, 추론은 원래 NN logits로 생성한다. Dev 출력에서 고른 임의 문장을 train에 주입하지 않는다. 단순히 같은409회를 더 돌리는 범위는 아니다.
4. 고정 dev444행의 NLL/전체 byte accuracy/EOF·비EOF 분리 혼동, 동일16개 자유생성의 UTF8/자연 EOS/반복/문맥 관련성을 보존한다. 추가 baseline을 무한 선행조건으로 반복하지 않고, 같은 출발 checkpoint의 검증된 저장 결과를 명시적으로 alias할 수 있으면 재사용한다. Zero-VP 비교도 이미 있는 자료와 구별한다.
5. 최종 방식은 CPU 자료·모듈을 보고 결정한다. 새 objective/code/data SHA, 실제 plan-schema, 모델·populatedAdam/scaler/모든RNG/cursor의 정확한 복구 또는 명시적 새 optimizer 정책, 전체 OS post-Popen 경로, actual worker preflight/독립validate/고정time-call-row-epoch상한을 Root가 검토한 뒤 새 한정 실행 한 번만 한다. Timer30분 prompt는 정확히‘재개’를 유지한다.

현재 학습된 언어/확산 가중치는 유용한 복구 자산으로 남지만 문서 생성 성공은 미입증이다. 새 성능/학습 준비 완료를 선행해 주장하지 않는다. 하드웨어 승인은 다시 묻지 않는다. C20GiB+임시 여유/시작 headroom/GPU12GiB와 검증된 owned OS 감독, 모든 종료 확인, E2final/registry 접근 금지와 기존 원본 보존을 유지한다.
