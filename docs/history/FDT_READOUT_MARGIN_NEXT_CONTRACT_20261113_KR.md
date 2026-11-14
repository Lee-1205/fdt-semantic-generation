# 다음 범위: TRAIN에서 위치별 경쟁 margin과 보존을 함께 학습

현재 V77은 실제1,920/1,920 완료/모든 owned 종료/저장 검토 완료다. 새 예산은 닫혔고248calls·시간은 재사용하지 않는다. V75가 다음 초기 기준이며 V76/V77 상태·개선·퇴행을 모두 보존한다. 영어 언어→의미→언어의 보존을 먼저 해결하며 의미 디노이징은0이다.

## 이미 해결·확인한 것

- V77 실제 head update는 signed768/unsigned768/OWT384로40/40/20이고 V75 base symbol weight를 그대로 썼다. 비중이 구현되지 않았다는 설명은 배제됐다. 이 변경은 수량36→28/OWT122→110의 부분 회복과 사건번호10→17의 퇴행을 함께 만들었다.
- 상류는 완전 고정, 같은 원문의 bound Commit도 동일했다. 이번 새 출력 변화는 readout 학습에서 생겼다. 이것은 기존 상류 표현의 완벽한 의미성 인증은 아니다.
- 저장 logits의 신규 `+` 오답12위치에서 경쟁 logit 변화 중앙값1.920780에 비해 최종 bias 변화는0.001123이다. `+` 최종 bias만 일괄 수정하거나 추론 단계에서 `+`를 지우는 것은 근거가 부족하고 정상 부호를 손상시킬 수 있다.
- 현재 teacher는 TRAIN p(gold)/correct만 loss 전용으로 저장한다. 위치별 top 경쟁 class/margin은 저장하지 않아 특정 숫자와 부호의 경계 보존을 직접 제약하지 않는다. 더 강한 전체 검증을 반복하기보다 이 학습 계약을 고친다.

## 다음 CPU 구현 목적

1. TRAIN 원문만으로 loss-only teacher packet에 gold 대 strongest non-gold margin과 경쟁 class를 저장한다. 올바른 teacher 위치만 보존 대상으로 삼고 틀린 teacher는 gold CE로 교정한다. teacher 오답 class를 정답으로 증류하지 않는다. 이 값은 decoder 입력에 주지 않는다.
2. 현재 correct-critical/noncritical Bernoulli anchor와 다른 범위의 margin 보존 손실을 설계한다. 절대 logit의 이동보다 gold-경쟁 차이를 사용하고, easy byte의 평균으로 희소 경계가 사라지지 않도록 TRAIN의 낮은 margin 위치를 한정한다. 특정 dev오답/장문 위치를 마스크로 사용하지 않는다. 수량·사건번호·일반 문자의 충돌은 TRAIN 자료로 판단한다.
3. 단순 scalar loss 확대가 아니라 같은 update 안의 signed/unsigned/OWT 누적 또는 명시한 finite order를 사용한다. micro 손실의 정규화·유효 위치 수·clip 적용 순서·누적 파라미터 gradient의 의미를 직접 계산한다. 기존 두 차례 자료 비중을 더 바꿔 보는 일만 반복하지 않는다.
4. 처음에는 기존7개 readout 범위를 유지한다. K_P/K_C 추출과 위치 결합을 함부로 바꾸지 않고 K_E Exact NULL/내용0/복사0,Commit-only decoder/원문 H·byte·정답prefix·GT길이0을 지킨다. 보존 margin으로 해결되지 않으면 그때 역할별 readout의 충돌을 별도 architecture 후보로 검토한다.

실제 모델 순방향을 또 돌리기 전에 작은 수치 예제로 margin 손실의 불변성·경쟁 class 변화·teacher 오답 제외·PAD/EOS·stop/inflight fullstate·populated Adam/enabledscaler/모든 RNG·다음 update 복구를 확인한다. TRAIN target,actual prompt,메모리 feature 경계를 분리한다. 이미 완료한 전체 baseline/넓은 감사/gradientprobe를 무한 선행조건으로 반복하지 않는다.

새 CPU schema에는 `scripts/plan_scope_contract_v78_cpu.py`를 실제 optimizer-owned tensor와 연결한다. `dict(old)`로 계승한 오래된 trainable 수/다중 output 추정은 거부한다. 이 guard의8개 CPU 검사는 끝났지만 V78 worker/plan/seal/실제 모델 학습은 아직 없다. 추정은 단일 component 합과 하드 cap, start C100GiB+outputcap+8GiB로 일관되어야 한다. 새 cost/time/노출/epoch/STOP 저장 상한을 실제 코드에서 계산한 뒤 고정한다.

## 평가와 해석

같은 관측 보존 패널의 결과를 계속 남기되 새 블라인드 사례로 부르지 않는다. 새 문장은 실제 TRAIN exclusion,자연 문법,문서/이름/숫자 중복을 먼저 확인한다. 관계·수량·부정·단위·이름/사건번호 보존을 분리하고 문자열 일치는 보조로 둔다. 현 장문은262KiB의 반복 작성 문법2개이며 input-dependent local Commit이 증가한다는 한계를 유지한다. 자연 장문/새 어휘 범위는 별도 권리·출처·split·용량 계약 후 확장한다.

V77의 END32 문법 수정과 MID32 동일 입력을 섞어 V76과 직접 성능 비교하지 않는다. 보고서는 동일 source 전후로 비교한다. 숫자 중심부가 보이지만 기호가 깨진 경우와 실제 다른 값,순수 표현 차이를 나눈다. 예외 행 제거·정답 대입·regex 추론·Exact 우회·평가 사실 TRAIN 편입을 하지 않는다.

다음 실제 한정 학습에는 새로운 SHA/output/CPU 실제 workerpreflight/ROOT독립검토/native 중복0/resources/ownedOS 감독이 필요하다. 이전900초/7,000call/미사용248call은 닫혔다. 승인된 자율 연구 안에서 ROOT가 진행하며 재승인 질문으로 멈추지 않는다. 학습 중 ROOT 상세 분석은 끄고 OS감독만 하고 terminal 뒤 모든 owned/proxy 종료를 먼저 확인한다.
