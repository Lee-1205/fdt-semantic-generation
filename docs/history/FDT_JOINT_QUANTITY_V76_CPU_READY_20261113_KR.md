# V76 보관량·단위·끝부분: CPU 자료와 제한 교정 손실 준비

2026-11-13 ROOT 직접 준비. V75의 전면 롤백은 보류하고 기존 V73/V74/V75와 모든 Adam/scaler/RNG·원문·raw·실패를 유지한다. 이번 단계는 코드·자료 계약 준비다. **새 GPU 학습,성능 향상,실제 worker 전체 복구·plan/seal/launch는 아직 없다.** 새 실제 학습 모델은 여전히 V75다.

V75 원인 재검토에서 확인한 두 부분만 좁게 다룬다. 첫째,소수 중요 오답의 별도 평균이 위치별 교정 계수를 지나치게 키웠다. 둘째,EOS 위치16개는 덮었어도 단위가 문장 끝에 오는 조합과 독립적인 두 수량 보존을 충분히 덮지 않았다. 디노이징은 계속0이다.

새 손실 후보 `research/critical_retention_objective_v76.py`는 V75의 다른 손실과 계수를 유지하고 중요 오답 CE 분모만 `max(W,C/8)`로 바꾼다. C는 해당 TRAIN행의 전체 중요 위치 수,W는 초기 중요 오답 위치 수다. 오답에 대한 추가 계수는 전체 중요 평균 대비 최대8배가 된다. 오답이 충분히 많을 때는 기존 오답 평균과 같고,오답0이면 유한한0이다. Teacher 확률/정답 여부와 gold는 TRAIN loss 전용이며 추론 feature나 정답 조립기가 아니다. 출력/퍼지 메모리/Exact 경로는 바꾸지 않는다.

7개 focused CPU 검사는 sparse gradient 계수 상한,dense 구간의 기존 평균과 일치,empty mask의0gradient,다른 항과 PADgradient 보존,teacher확률 detach,초기 오답의 보존항 제외와 gold방향 교정,잘못된 subset/EOS critical mask 거절을 확인했다. Tiny logits의 검사이며 실제 모델 능력·전체 phase/scaler/RNG/STOP 복구의 검사가 아니다.

실제 V75의 저장640행 history에서 logits와 손실값을 그대로 두고 계수만 재계산했다. 중요 오답항 합은38.721665→10.441654,전체 재합산 손실에서 차지하는 비율은68.3439%→36.7960%다. 이는 **동일 저장값의 가상 스칼라 재가중**이다. 바뀐 학습 경로의 실제 loss/전체 파라미터 gradient/정확도 예측이 아니며8이 최적이라는 증거도 아니다. 과도한 교정을 줄이면 새 교정 이득도 줄 수 있으므로 다음 실제 결과에서 구분해야 한다.

새 immutable 자료는 `research/artifacts/joint_quantity_20261113_v76_cpu_v1`이다.

| 계약 | 실제 CPU 자료 |
|---|---:|
| TRAIN 전체 |1,024행|
| 새 작성 입력 |896행|
| 독립 수량 A/B의2×2 상관 quartet |224그룹|
| 부호·단위 입력 |768행|
| 부호 없는 개수 입력 |128행|
| 보존한 기존 OWT TRAIN |128행|
| 원문 노출,한 유한 epoch |1,010,943 UTF-8바이트|
| 중요 supervision 구간 |45,460바이트|
| 최대 입력 |16,384바이트|
| 새 평가 행/GPU/modelforward/optimizer |각0|

A만 바뀌면 B는 그대로,B만 바뀌면 A는 그대로인 네 조합을 각 quartet에 명시했다. kg/온도 degrees C/거리 km와 ±변화량,양수 별도 측정값,문장 중간/끝을 교차했다. 부호 없는 개수 문장도 두 독립 수량을 가진다. 총14개 unit×layout×sign cell 각각 EOS의16개 byte 위치를 모두 갖는다. 단위 표현이 문장 끝에 오는 입력은 kg/온도/km 각각128행이다. 숫자를 같게 두고 ±만 바꾸는 한 종류의 짝으로 두 사실 보존을 대신하지 않았다.

ROOT 독립 검토는 builder를 호출해 스스로 확인하는 방식 대신896개 전체 문장을 별도 문법으로 다시 파싱했다. 모든1,024행의 UTF-8 span/정답 위치를 확인하고,224quartet에서 A/B 외의 문자 불변·네 조합·값 변경·order 인접·JSON nextcursor를 검증했다. 기존 OWT128행 feature/target은 V75와 그대로 같다. 한 번의 finite order이며1,024개 독립 문서나 새로운 OWT 수집이 아니다.

Feature는 id/prompt뿐이다. Quartet/숫자 값/행위자/span은 targets 파일에만 있다. 이전 관측 평가의 실제 prompt·gold·이름을 TRAIN에 옮기지 않았다. 기존 V73 short/long,V74DEV,V75DEV는 SHA를 고정해 중복/부분 포함 및 held 이름 제외 확인에만 읽었다. E2최종 holdout/registry는 열지 않았고 새 다운로드도0이다. 새 DEV는 아직 만들지 않았으므로 새 blind/미학습 평가 성과라고 보고하지 않는다. 반복 문법·기존 이름·고장 관측 후 설계한 자료라는 한계도 유지한다.

Manifest FILE SHA:`ae5a0469e09fcd4d4d0b039897da27ec406615c69e336259fa8a4382d2febf43`.
독립 증거:`root_CPU_review.json`.
준비 source:`scripts/prepare_joint_quantity_v76_cpu.py`,`scripts/review_joint_quantity_v76_cpu.py`,`scripts/test_critical_retention_objective_v76_cpu.py`.

다음은 이 부품을 실제 teacher/head driver에 연결하고,V75 parent-required runtime에서 NN을 정확 복원하되 storage base는 기존 standalone V74로 유지하는 계약이다. 기존 helper/OSowner를 다시 만들거나 옛640회·미사용140calls를 재개하지 않는다. 실제 전체 phase·populatedAdam/enabledscaler/모든RNG/order/cursor/in-flight STOP 복구,실제 full-size CPU preflight·독립 validate,새 평가 명세·코드 비용에 따른 fixed clock/call/output/temp/STOP 상한을 준비한 다음 자원 조건을 만족할 때만 새 ONE trial을 한다. 이 문서는 실행 plan이 아니다.

현재 C여유는 약100GiB라100GiB 보호+8GiB headroom+새 output/임시/STOP 여유에 못 미친다. GPU를 시작하거나 보호선을 낮추지 않았다. 지금은 메모리·저장량이 작은 CPU 부품 준비만 진행했다. K_P 기본 퍼지/K_C 문장 퍼지 활성,K_E ExactNULL/복사0,Commit-only 출력/source H·rawbytes·GTprefix·GTlength 우회 금지,에이전트 없음,언어→의미→언어 안정화 이전 디노이징 금지를 유지한다.
