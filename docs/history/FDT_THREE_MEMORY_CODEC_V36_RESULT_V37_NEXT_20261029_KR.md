# V36 원문 복원 결과와 세 번째 넓은 순회

V36은 두 번째 넓은 OpenWebText 순회를4096attempt/4092update로 완료했다. AMPskip은10/289/2400/3159의4회이며 행/노출을 소비하고 되감지 않았다. Worker706.857초, encode4162+decoder10176=14338호출, denoise0. 새1200초/17000호출 예산은 CLOSED이고 미사용2662호출/시간을 재사용하지 않는다. 모든 owner/proxy/worker32464·21532·41620·43100 종료를 독립 owner와 ROOT가 확인한 뒤 분석했다.

최종 fullstate780688437B SHA `53a4486871a0e94df78cf8426dd786c5fdb86536c3b06d07469db742c8d40adc`, algorithm `800ff9820a44d9f88bda4d894963b778ebebf084113fa34adeab014fcd7dbb36`, planSHA `d63383e4d8b4bcaf4ed0427d7fa14fa0c08b478e16bdd9c224bd8e44cd25c639`. ROOT가 import0/mid2048/final4096 세 fullstate의 SHA/safeload/재귀digest/모델·91개 Adam/scaler·저장된 모든 RNG의 정확한 이관/수치 유한성/동결/커서·순서·노출·평가 SHA·호출수를 확인했다. 새 모델 forward·GPU 재실행0이다. Scaler262144/tracker374→32768/tracker937의 변화도4skip·성장 일정과 일치했다.

## 실제 출력

| 동일 관측 패널 | V35 | V36 |
|---|---:|---:|
| 개발256 teacher NLL | 0.885946 | 0.828950 |
| 개발256 teacher byte 정확도 | 73.7493% | 75.2747% |
| 개발256 teacher EOS | 254/256 | 255/256 |
| 학습 참고64 BOS 원문 일치 | 25/64 | 23/64 |
| 개발16 BOS 원문 일치/전체 의미 충실 | 0/16,0/16 | 0/16,0/16 |
| 학습64/개발16 UTF8+EOS | 64/64,16/16 | 64/64,16/16 |

ROOT가 원문/복원80쌍을 모두 직접 읽었다. 참고 학습64는23개가 바이트까지 일치하고,9개는 확인 가능한 일부 명제를 보존했으며32개는 충실한 명제 복원으로 인정하지 않았다. V35 대비7개가 새로 정확해졌고9개가 정확 복원에서 벗어났다. 예를 들어 WordPress/plugin과Swansea/passes는 회복됐지만1948→1988 날짜 변경, 피해 사건·부정·고유명사 및 후반 절의 손실이 남았다. 개발16도 사건 관계를 완전히 보존하지 못했다. 이 수치로 전체 FDT 실패를 단정하거나 의미 추출이 완성됐다고 하지 않는다.

개발16 chrF24.6344→25.9781, word F1 13.2815→16.4007%, content-word F1 3.4406→5.4515%다. 학습 참고64는 chrF79.2478→71.3377, word76.3234→68.4454%, content72.1091→62.6167%로 하락했다. 문자열 유사도는 의미 보존율이 아니다. V33의64/64 용량 버전과 V35/V36의 서로 다른 결과를 모두 보존하며 자동 승격하지 않는다.

## 저장된 예측으로 확인한 첫 오류

동일80행에서 정답 접두사를 사용하는 teacher 예측과 BOS/자기 접두사 재복원의 **첫 오류 위치가 전부 같았다**. 참고 학습64의 teacher 전체행 정답도23개로 BOS 정확23개와 일치했고, 개발16은 둘 다0개다. 추가 forward 없이 저장된 logits와 token IDs만 비교했다.

이는 이 표본에서 별도의 복원 실행 모드가 첫 오류를 새로 만들었다는 증거가 없다는 뜻이다. 후속 오류의 누적 영향이나 encoder 대 decoder의 단독 원인을 확정하지는 않는다. 올바른 접두사에서도 틀리는 예측이 남아 있으므로75% teacher byte 정확도를 완전 복원 직전이라고 해석하지 않고 언어 codec 학습을 이어간다.

## 3종 메모리의 현재 범위

같은 개발16 정상 NLL1.125439791에서 K_P 제거+0.048252813/K_C 제거+3.500159604/K_E 제거+0.287040376이었다. 저장된 full Exact key/query/valid mask/top16+NULL와 per-token FP64 long/recent N,D를 독립 재구성했다. 세 읽기 경로가 현재 출력에 사용된다는 근거다.

기본 퍼지 K_P는 학습된 prior이며 certified persistent Writer는 미완성이다. K_C는 문장 입력 전체를 처리한 token별 장기/최근 메모리이며 vNext 전체 층의 prefix read-before-write 구현은 아니다. Exact는 입력 원문 색인/학습된 선택 읽기이며 명시적 literal selector/transport는 미완성이다. 원문 전체 자동 복사·분절 연결 우회·raw H/바이트의 디코더 직접 전달은0이다. 디코더는 이 메모리들을 반영한 Commit만 받는다. 미래 명시적 Commit 전문 인용 계약과 원래 typed core/이전 버전은 보존했다.

근거는 `runs/three_memory_codec_20261029_v36/root_result.json`, `root_all80_reading.md`, `root_reconstruction_similarity.json`, `root_three_memory_readouts_review.json`, `scripts/review_three_memory_codec_v36_result_cpu.py`다. 관측 개발 자료 반복 사용·참고64의 과거 반복 적응·4개 구간/문서의 상관 한계도 유지한다.

## V37의 별도 목적

동일한3종 읽기 경로와 언어→입력 의미→같은 원문 목표로 **세 번째 넓은 유한 순회**를 수행한다. 기존TRAIN8192문서/32768구간, seed36001/batch8/4096attempt/2926478target UTF8bytes. 구조·trainability·p50prefix corruption·clean.5+corrupt.5 손실을 바꾸지 않는다. V36의 전체 모델/91개 populated Adam/scaler/저장된 모든 RNG를 정확히 이관하고, 기존 base LR의 새 유한 cosine 주기만 적용한다. Adam 누적 step/모멘트는 초기화하지 않는다.

새로운1200초/reserve120/cutoff1080/17000호출(최대계산15060)/출력6GiB/C시작35GiB/GPU12GiB를 사전 고정한다. 실측5.46updates/초보다 보수적5.0으로820초 학습+시작60+checkpoint60+평가60+종료120=1120초를 계산한 같은 정책이다. 모든 이전 시계와 예산은 닫힌 채로 둔다. import0/mid2048/final4096+필요시emergency를 보존하고 실행 중 코드/상한 변경·회전·자동 재시작을 하지 않는다. V36 저장 결과를 기준선으로 써 초기 GPU 평가를 반복하지 않는다.

후속도 ROOT 직접 처리하며 에이전트·새 corpus 다운로드·E2final/registry 접근·원본 삭제를 하지 않는다. 새 원문 복원 성과는 실제 학습 종료와 저장된 출력 대조 뒤에만 보고한다. 디노이징은 계속0이다.
