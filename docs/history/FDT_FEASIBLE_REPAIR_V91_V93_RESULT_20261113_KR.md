# V91 학습·V93 평가 복구 결과

방향 보정으로 **실제6개 Adam 업데이트를 수락**했고, 원래 TRAIN 보호 기준을 지켰다. 그러나 문장 출력의 개선은 혼합적이다. 짧은192문장 중1개의 이름 대문자를 복구했고, 약524KB의 장문에서는3개 byte 오탈자가 추가됐다. 새 가중치를 기본 모델로 승격하지 않는다.

## 실제 실행과 보존

V91은 새96attempt 한도에서70attempt/6수락/64거부/AMP skip0을 수행했다. 마지막 수락은38회였고, 이후 사전 고정한32회 연속 미수락 기준으로 학습 단계를 닫았다. 여섯 수락 모두 원래 Adam 방향은 거부됐으나 방향 보정 뒤 **전체 유한 TRAIN bank**를 통과한 실제 업데이트다. 원래119,769개 top1과152개 취약 critical floor의 최종 위반은0이다. 이것은 관측하지 않은 문장 전체의 의미 보존 보증이 아니다.

입력·기본/문장 memory·binding·출력 prefix는 동결했으며 마지막 Linear의2개 tensor만 학습했다. K_E는 NULL/content0/copy0, denoise0다. 원래 teacher1024행은 정적 훈련 target으로 재사용했고 새 teacher encoder 노출은0이다. 실제 head 노출은350행/591,127UTF8 byte이며 독립350개 새 문서라는 뜻은 아니다. 기존과 새 가중치·populate된 Adam6step·scaler tracker70·모든 저장 RNG와 실패 원문을 보존했다.

V91은 최종 평가 중 `state.json.*.tmp`가 atomic rename되면서 공간 집계의 `is_file→stat` 사이에서 사라져 중단됐다. 제가 과거 V88 worker의 안전하지 않은 공간 집계 함수를 재사용한 통합 누락이었다. supervisor에는 이미 과거 V52의 안전한 집계가 있었으나 worker에는 적용되지 않았다. **V91의 FAILED/exit2 기록을 그대로 유지한다.** 긴급 fullstate29,239,795byte를 독립 safeload·재구성하고 마지막2개 외 모든 tensor의 일치를 확인했다.

이후 V92 보충 평가에서는 실행 객체에 `device`를 빠뜨려 모델 호출0에서 중단됐다. 원본 코드와 실패를 보존하고, 새 V93에서 공통 실행 객체 생성 함수와 **실제 평가 경로의 CPU forward1회**로 이를 확인했다. 공간 집계도 새 코드에서 기존 V52 함수를 사용한다. 원래 경합은 CPU에서 재현하고 수정된 집계가 통과하는 것을 확인했다.

V93은 별도 read-only73call/180초 한도로 **미생성73개만** 평가했다. 실제73readout/5.464worker초/optimizer0/encoder0/denoise0/checkpoint0, owner·worker exit0/COMPLETE다. V91의 완료된121개를 GPU에서 반복하지 않았다. V91과 V93 모두 모든 owned/proxy 종료 및 Python0을 ROOT가 확인한 뒤 분석했다. V92는 빠른 종료 전 ROOT가 네 프로세스 모두의 생존을 포착하지 못했으므로 owner의 actual CIM 기록과 ROOT 종료 확인을 구분한다.

모든 예산은 닫혔다. V91의956call과 미사용444call·26attempt를 재사용하지 않았다. V92의0call 실패와 V93의 별도73call도 각각 닫혔다.

## 실제 출력 비교

동일한 V88 가중치의 저장 baseline과 V91의 긴급 저장 모델을 비교했다. 기존에 관측한 개발192문장+고정 Commit의 장문2개이므로 새 blind benchmark나 새 장문 encoder 실행이 아니다.

| 패널 | V88→V91 출력 변화 | 해석 |
|---|---:|---|
| new64 | 0문장 | 기존 B 의미값52/64와 잔여 오류가 그대로다 |
| observed75 64 | 0문장 | 출력 변화 없음 |
| observed74 64 | 1문장·1byte 교정 | `kyrene→Kyrene` 이름 대문자 복구 |
| 262,042+262,029byte 장문 | 오류49→52 | 기존 오류를 유지하면서3개 오탈자 추가 |
| UTF8/EOS | 194/194 | 정상 형식이 의미 정확도를 보증하지는 않는다 |

추가된 장문 오탈자는 `drawer→draber`, `episode→epitode`, `did not remove→did not remoxe`다. ROOT가 이3곳과 대문자 교정 문맥, 장문의 시작·중간·끝6개 구간을 직접 읽었다. 문맥상 사람이 일부 오탈자를 추론할 수 있지만, 손상 없는 언어화로 인정하지 않는다. 전체524,071byte를 사람이 일일이 읽었다는 주장도 하지 않는다.

원문에 연결한 quantity13/2,718, reserve2/2,718, actor2/2,718, item7/2,718, `moved`5/2,718 오류는 변하지 않았다. checker와 제한된 temporal-order 필드는0을 유지했다. location noun 오류는9→10/10,872로 증가했다. `did not remove`도 별도로 검사해 새 동사 손상을 드러냈다. 이는 반복 합성 장문의 정해진 필드 검사이며 범용 NLU 인증이 아니다.

장문에서 추가된3byte는 전체의 약0.000572%다. 큰 붕괴로 과장할 규모는 아니지만, 위치·동사 언어화의 퇴행을 무시할 수도 없다. 장문 NLL은0.000803969→0.000800671로 낮아졌으나 실제 오류는 늘었다. NLL 감소만으로 의미 개선을 선언하지 않는다.

## 다음 원인을 좁힌 결과

64개 거부 중57개는 선형 보정 후 전체 FP32 판정에서 거부됐고,7개는 고정2회 sweep에서 수렴하지 않았다. 이번에는 실제 source/byte/gold/rival 진단을 저장했으므로,57개 거부의156개 기록이 **처음 보정한 동일 class 쌍**에 해당함을 확인했다. 기록된 잔여 부족량은 최대5.7220459e-5로 보정 여유1e-6보다 컸고, 정확한 top1 동점도 있었다.

이는 선형 계산에서 맞춘 경계와 실제 FP32 행렬 연산 판정 사이의 수치 여유가 부족하다는 가설을 강하게 지지한다. 새 문맥의 의미가 대거 무너졌다는 증거로 해석하지 않는다. 다만 거부 candidate의 전체 weight/logit을 저장하지 않았으므로 부동소수점 세부 기전을 유일한 원인으로 확정하지 않는다. 원래 floor와 전체 판정을 낮추는 해결책도 택하지 않는다.

다음은 TRAIN의 원래 anchor를 유지하면서 실제 FP32 연산의 수치 여유를 고려한 보정 계약이다. 임의로 같은 SGD를 더 돌리거나 넓은 baseline을 다시 만들지 않는다. 현 비교에서는 V91을 승격하지 않고 V75 balanced·V80 FP32 기준 및 V88/V91 복구본을 보존한다. 다음 보정의 warm 후보는 V91보다 V88을 우선 검토한다. 새 학습은 별도 코드·CPU·고정 비용·실제 preflight·OS 감독 검토 뒤 진행한다.

원본이나 가중치를 삭제하지 않았다. 닫힌 V79/V78 raw4개만 투명 NTFS 압축해201,805,956byte의 물리 공간을 확보했고, 논리 SHA·크기와 CPU safeload는 전후 동일했다. C:100GiB 보호를 유지했다.

언어→의미→언어는 아직 완성되지 않았다. 사고 diffusion을 학습할 때 입력·모든 memory·binding·출력 codec과 buffer 전체를 고정하는 원칙은 그대로다.

증거: `runs/feasible_repair_20261113_v91/root_saved_review.json`, `root_combined_assessment_review.json`, `root_manual_semantic_and_numeric_review.json`, `root_combined_evidence_SHA.json`; `runs/missing_assessment_20261113_v93/result.json`, `ROOT_terminal_native.json`.
