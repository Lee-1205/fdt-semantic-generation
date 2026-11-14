# 실제 TRAIN 보존 기준의 사각지대와 수정 계약 — 2026-11-13

이전 V81/V82 진단에 이어, **보존 기준을 무조건 강하게 만드는 대신 실제 무엇을 놓쳤는지** 조사했다. 새 SGD는 하지 않았다. 결론은 기준 bank의 핵심 취약 위치 coverage를 고치는 것이 먼저이며, 모든 일반 바이트의 마진을 더 엄격하게 제한하는 것만으로는 이번 수치·인명 퇴행을 해결할 근거가 부족하다는 것이다.

## 실제 모델 재측정: V83 읽기 전용 완료

동일한 TRAIN100의 저장 hidden 7,487개에 V75/V80의 마지막 Linear만 연결했다. BF16 최종 투영 2회와 FP32 최종 투영 2회, 총 **4회**다. worker 1.660초, peak GPU 387,794,944B. optimizer/encoder/디노이징/새 가중치 체크포인트는 모두 0이다.

기존 bank hidden은 BF16 prefix에서 만들어졌다. 그러므로 여기서 `last_fp32`는 **기존 BF16 hidden에 대한 마지막 투영만 FP32로 바꾼 구성 요소 진단**이다. V82의 전체 출력기 FP32 경로와 같다고 보고하지 않는다.

| 같은 TRAIN100 | 기존 BF16 투영 | 마지막 투영 FP32 |
|---|---:|---:|
| 유효 byte/EOS 위치 | 117,644 | 117,644 |
| V75 초기 정답 | 117,600 | 117,603 |
| V80 최종 정답 | 117,614 | 117,615 |
| 초기 정답 중 최종 오답 | **0** | 1 |
| 초기 오답 중 교정 | 14 | 13 |
| 제안한 마진 하한 위반 | 1 | 3 |
| 그중 핵심 critical 위치 | **0** | **0** |

BF16 초기·최종 argmax는 과거 native 결과와 전체 bank에서 정확히 일치했다. 따라서 V80의 기존 ‘보호한 정답 손실 0’ 보고는 잘못된 성공 판정이 아니었다.

마진이 감소한 위치는 BF16 46,039개, 증가한 위치는 49,000개다. 마진 변화 중앙값은 0이다. 이 숫자를 모두 의미 손실이나 전면 붕괴로 세면 안 된다. 새 마진 하한의 위반은 동일 OWT 문서 안의 일반 텍스트 위치에서 나왔다. BF16 한 곳은 `aside`의 `e` 정답이 유지되면서 마진만 1.0→0.25로 줄어든 경우다.

FP32 구성 요소 진단의 한 raw argmax flip은 닫는 따옴표의 UTF8 연속 바이트 `0x9D`였다. UTF8 FSM을 통과한 실제 문장이 숫자 `1`로 바뀌었다거나 사실이 손실됐다고 해석하지 않았다. 나머지 두 마진 위반도 일반 텍스트다. **이 bank 안의 핵심 위치 마진 위반은 없는데, bank 밖의 개발·장문 수치 오류는 존재한다.** 따라서 같은 100개에 더 강한 일반 byte 마진 gate만 거는 것은 확인한 퇴행의 직접 해결책이 아니다.

전체 통계의 margin은 저장 gold score와 최고 비정답 score에서 CPU로 독립 재구성했다. 각 조건의 선택한 128개 patch, 합계 8,192개 byte 위치에서는 저장한 258개 전체 클래스 logit으로 argmax/정답/경쟁자 score를 추가 검증했다. 나머지 위치의 전체 클래스 logits는 저장하지 않았으며, 그 전부를 독립 재계산했다고 주장하지 않는다. BF16 두 조건은 과거 전체 argmax와 일치 검사를 별도로 통과했다. 과거 각 학습 step의 logits를 재실행한 결과도 아니다.

실제 native owner 14984→21796, worker12640→26904의 생성·parent·command·exe·생존을 ROOT가 14:49:17 KST에 관측했다. worker와 owner exit0, ROOT가 14:49:35에 allowned/proxy gone 및 Python0을 확인한 뒤 분석했다. 새 120초/90초 cutoff/30초 종료 여유, 6-call/32MiB 상한은 닫았다. 미사용 2call이나 시간을 재사용하지 않는다.

## 실제 자료·teacher coverage: V84 CPU 완료

새 모델 forward 없이 기존 TRAIN 1,024개 teacher packet의 SHA, source ID, 원문 byte/EOS/PAD, critical span을 재구성했다. 이 packet은 V80 학습 전의 V75 BF16 teacher 기록이다.

‘처음 정답이면서 margin≤1인 critical byte’를 모으면:

| 항목 | 실제 수 |
|---|---:|
| TRAIN critical byte 전체 | 44,516 |
| 초기 critical 오답 | 229 |
| 초기 정답이지만 취약한 critical byte | **213** |
| 기존 100개 bank 안에 들어간 취약 위치 | **24** |
| 기존 bank 밖의 취약 위치 | **189** |
| 취약 위치가 있는 TRAIN source | 171 |
| 그중 기존 bank 밖 source | 149 |

즉 기존 bank는 이 정의의 취약 핵심 위치 **11.27%**를 포함했다. 부호·폭 cell을 고르게 택하는 것과 실제 결정 경계 가까이 있는 핵심 위치를 덮는 것은 다르다. 213곳은 독립 문장·독립 사실 213개가 아니라 상관된 TRAIN view 안의 byte 위치다. 또한 이는 **초기 teacher의 취약 위치 coverage**이며, 189곳 전부가 V80에서 실제로 오답이 됐다는 뜻은 아니다.

## 수정한 코드와 아직 하지 않은 작업

`research/train_critical_frontier_v84.py`는 고정 hash100만 보지 않고 **TRAIN 전체의 취약 critical 위치를 모두 선택하는 계약**을 구현한다. 학습 allowlist와 source ID를 확인하고, teacher/reference/runtime 정밀도가 다르면 거부한다. PAD, 일반 비critical byte와 초기 오답은 보호 마스크에서 제외한다. 초기 오답은 별도 학습으로 개선할 수 있다. 상한 초과는 오류로 처리하며 조용히 일부를 버리거나 상한을 늘리지 않는다.

현재 과거 BF16 자료에서는 213개 위치가 201개 patch에 모였다. 추가 frozen prefix feature의 FP32 값만 약 **411,648B**다. 모든 긴 원문 hidden을 중복 저장할 필요 없이 핵심 보존 사각지대를 줄일 수 있는 구체적인 구성이다. gold span과 정답은 TRAIN loss/선택·검사에만 사용하며, decoder의 neural feature로 주지 않는다. 미래 feature packet과 target/margin packet도 분리해야 한다.

CPU 검사 3개가 통과했다: 취약 핵심 위치 전부 선택 및 PAD/비critical/초기 오답 제외, DEV·정밀도 불일치 거부, 상한 초과 시 무단 truncation 방지. 실제 1,024개 packet에서도 독립 재구성 결과와 일치했다.

**아직 새 GPU 학습·optimizer driver 통합·실행 plan/seal은 없다.** 저장한 historical BF16 selection은 진단 artifact이며, 다음 전체 FP32 학습에 그대로 넘기는 실행 입력이 아니다. 실제 새 부모/정밀도에서 다시 teacher와 prefix를 계산해 selection을 만들어야 한다. 이전 100개 broad reference의 top1 보존에 전체 TRAIN의 취약 critical 위치와 적절한 마진 보존을 추가하는 방향이며, 모든 일반 텍스트의 마진을 무조건 고정하는 방향은 피한다. 이 유한 TRAIN 보존이 미학습 장문 전체를 보장하지는 않는다.

이 결과로 다음 학습의 수정 목적이 더 좁아졌다. 핵심은 ‘마진 gate 강도만 올리기’가 아니라 **어떤 핵심 위치를 보호하고 어떤 정밀도로 검증하는지 일치시키는 것**이다. 기존 V75/V80의 개선과 실패, 모든 가중치·Adam/RNG·자료·원시 기록은 보존했다. Exact는 NULL/content0/copy0이고 언어→의미→언어 단계, 디노이징0을 유지한다. 미래 사고 학습은 전체 codec을 고정한다.

증거: `runs/margin_boundary_diagnostic_20261113_v83/root_saved_review.json`, `raw_projection_statistics.pt`, `ROOT_terminal_native.json`; `research/artifacts/critical_frontier_20261113_v84_cpu_v1/root_cpu_review.json`, `historical_BF16_TRAIN_selection.json`, `source_pins.json`. V83 plan SHA: `55e5225402cb8214171d451d03603760381da6db30c15a77f2ea0264e453e2b2`.
