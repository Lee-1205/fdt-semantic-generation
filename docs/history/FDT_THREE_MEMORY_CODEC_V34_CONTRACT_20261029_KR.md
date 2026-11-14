# V34: 같은 3종 메모리 문장 복원의 학습 범위 확대

V33은 새 K_P/K_C/K_E 경로로 반복 학습한64문장을 모두 복원했지만 기존 개발 문장에서는 원문의 명제를 충분히 보존하지 못했다. 다음 목적은 구조를 바꾸거나 디노이징을 시작하는 것이 아니라, **같은 언어→입력 의미→같은 언어 경로에 다양한 기존 학습 문장을 노출하는 것**이다.

새 V34는 V33 최종 모델의 모든 tensor와91개 populated Adam, enabled scaler, 저장된 모든 Torch/CUDA/Python/NumPy RNG를 정확히 가져온다. 새 파라미터·모멘트 초기화는 없다. 원래 V31→V33 구조 변경 기록은 보존하며 이번 V33→V34는 신경 함수가 같다. 동일한4개 그룹의 base LR(1e-5,1e-4,3e-6,1e-4)을 새 유한 cosine 일정에 사용한다. 학습률 일정은 새로 시작하지만 모멘트/step은 초기화하지 않는다. 원래 사고/refiner/time/velocity/관계 모듈은 계속 동결한다. Exact 원문 바이트를 디코더에 직접 보내거나 전문 복사를 수행하지 않는다.

자료는 기존 불변 OpenWebText artifact의 TRAIN8192문서/32768구간, 2,926,478 target UTF8 bytes다. seed34001의 한 유한 순열, batch8,4096attempt가 상한이다. AMP skip도 행/attempt를 소비하며 되감기·추가 epoch를 하지 않는다. 기존 개발64문서/256구간과 학습64 참고 구간은 분리한다. 개발 자료 편입·새 다운로드·E2final/registry 접근은 없다. 과거 학습량과 관측 개발 반복 사용의 한계는 계속 명시한다.

별도 신규900초, model cutoff780초, 종료 reserve120초,17000호출 상한(코드 최대계산15060), 출력6GiB. 시작C35GiB는 이미 명시한20보호+6최대출력+1체크포인트 임시+8추가여유의 합이다. 상시C20GiB 보호/workerC22GiB stop/GPU0free12GiB. import0/mid2048/final4096+필요한 emergency fullstate를 보존하며 삭제·회전하지 않는다. V33 및 과거 모든 예산은 CLOSED이며 재사용하지 않는다.

V33 최종은 신경 함수가 같은 저장 기준선이다. 같은 초기 GPU 평가를 반복하지 않는다. 최종 학습64/개발256 teacher 보조지표, 학습64/개발16의 입력 의미 기반 BOS 재복원, 개발16의 세 메모리 제거 통제를 저장한다. 지표가 높아도 원문과 실제 출력을 ROOT가 직접 읽어 명제/부정/이름/숫자가 보존됐는지 확인한다. 무조건적 자유 창작 출력은 평가하지 않는다.

신규 driver import/복구/마지막7행/enabled CPU scaler skip/정확한 다음 update, 실제 one-epoch plan, OS owner 자연종료·post-Popen예외·STOP 기록 실패·종료 불확실을 다룬 **6개 필요한 CPU 검사 PASS(13.77초)**. 실제 worker의 전체 모델 CPU safeload/import를 독립 owner --validate로 통과했다. 원래 의미 경로 자체의 CPU 검사를 다시 반복하지 않았다. 신규 owner의 실제 Windows CPU fixture child도 native parent/command/identity와 전체 종료를 확인했다. 이것은 GPU 학습이나 성능 개선 증거가 아니다. 별도 seal/실제 생성된 native worker/진행 기록을 확인한 뒤에만 실행했다고 기록한다.

코드는 `scripts/three_memory_codec_engine_v34.py`, `scripts/run_three_memory_codec_20261029_v34.py`, `scripts/supervise_three_memory_codec_20261029_v34.py`; CPU 근거는 `research/artifacts/three_memory_codec_20261029_v34_worker_cpu`다. ROOT 직접 처리하며 agent를 사용하지 않는다.
