# V76 제한 학습 준비 완료

2026-11-13 ROOT. 기존 V75를 정확 복원하고 상류 전체를 고정한 채 최종 출력기7개 tensor/2,908,192파라미터만 학습한다. 전면 롤백이나 Exact 복사 경로 추가는 없다. K_P prior/K_C dynamic fuzzy→Commit→언어,K_E NULL/복사0,디노이징0을 유지한다.

새 TRAIN은1,024행/896개 작성 입력/224개의 독립 수량A/B 2×2 상관 quartet와 기존 OWT TRAIN128행 그대로다. 한 유한 epoch 원문1,010,943 UTF-8바이트/중요구간45,460바이트/최대16,384바이트다. kg·온도·거리,±,중간/끝,EOS16개 위치를 결합했고 오답 중요 CE 분모를max(W,C/8)로 제한한다. 다른 V75 손실·보존항은 유지한다. 새 평가64개는 새로운 문장 형식/16상관 quartet이며 held 이름4개는 과거 관측됐으나 TRAIN에는 없다. 새 자연/블라인드 corpus로 부르지 않는다.

실제 worker CPU preflight18.218초:대형 V75 linked runtime 전체 재구성→새7Adam/scaler→원문 추출2/readout2/only7gradient/initial·partialteacher linked 전체 상태 복원/long wrapper strict copy/실제 직접·간접 Commit 참조를 확인했다. optimizer0/CUDAfalse/모든 owned/proxy 종료 후 ROOT가982pins와 두 실제 상태를 helper와 별도로 재구성·검토했다. 전체 teacher/head의 populatedAdam/enabledCPUscaler/모든RNG/skip/in-flight STOP 계약은 별도 tiny engine에서 통과했다. 실제 CUDA 재시작 증거는 아니다.

처음 대형 CPU preflight는 V74 direct raw에 source_id/source_SHA/embedding_sha가 없다는 실제 포맷 차이를 잡아 종료했다. 모델 계산이나 GPU 실패가 아니다. 모든 CPU owned/proxy 종료를 확인한 뒤 실패 plan·source snapshot·stdout/stderr/복구본을 보존했다. 새 resolver v2는 기존 metadata를 없애는 것이 아니라,엄격한 metadata가 있는 새 referencer·정확 SHA pin·동일 target·알려진6개-key 구형 schema·Exact휴면 정책을 통해서만 구형 direct packet의 provenance를 확인한다. 부분 metadata/정책 불일치/중첩은 거절하며7CPU검사를 통과했다. 새 raw는 직접 tensor까지 참조를 정규화하므로 참조가 재귀적으로 늘지 않는다. 별도 execution_review_v2에서 실제 preflight와 독립 validate를 새로 완료했다.

새 계획의 실제 예상 비용은encode2,180+readout2,522+streamchunk258=4,960calls,cap5,200/denoise0/longblock2,322다. teacher2,048calls/head2,048calls/새64 baseline128+final128/관측340 readout/두262KiB 실제 source와 native-stream parity가 포함된다. 900초 wall/780 modelcutoff/120 reserve,TRAIN1,024attempt 최대/AMPskip도row·cursor·노출을 소비한다. 기존 미사용140calls나 시간을 재사용하지 않는다.

출력 예상1,092,548,960바이트는모든 raw/teacher/initial·teachercomplete·final·추가STOP/runtime임시 및32MiB slack을 포함한다. 새 fixed hardcap은1.25GiB다. 종료한 V75 cap을 확대하는 것이 아니라 다른1,024행 목적의 별도 계획이다. 시작 C100GiB보호+1.25GiB최대소유출력+8GiBheadroom=109.25GiB,GPU12GiB free를 요구한다. worker101GiB+예정write/OSowner104GiBstop의 여유를 낮추지 않았다.

07:36 독립 자원 확인은C121,279,660,032바이트/약112.95GiB,GPU15,290MiB free,RAM19,761,455,104바이트/Python0였다. 수 시간 전 부족하던 C여유가 회복됐으나 원인을 특정 앱/정리 효과로 단정하지 않는다. 이로써 자원 조건은 충족됐으며 아직 이 준비 문서를 GPU 학습이나 성능 개선이라고 부르지 않는다.

봉인 뒤 새시계에서 ONE 숨김 실행만 한다. ROOT가 실제 native 생성/parent/command/생존과 durable 상태를 관측하고 실제 training 중에는 상세 분석을 중단/OS감독만 한다. terminal후 모든 owned/proxy 종료를 먼저 확인한 다음 raw의 실제 수량·단위·부호·이름·장문 보존과 새 손실 효과를 독립 검토한다. 자동 재시작·같은 평가 반복·과거 예산 재사용·실행중 봉인 코드 수정/삭제·게임/유용한 가중치 삭제는 없다.
