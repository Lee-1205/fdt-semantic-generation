# V37 세 메모리 언어 codec 결과와 다음 학습

V37은 세 번째 넓은 유한 순회를 완료했다. 4096attempt/4095update/AMPskip1(1886), worker646.736923초, encode4162+decoder10194=14356호출, denoise0이다. 1200초/17000호출 예산은 CLOSED이며 미사용2644호출·시간을 재사용하지 않는다. 모든 owner/proxy/worker8532·24428·42088·8888 종료를 확인하고 ROOT20:49:12 nativePython/compact0을 확인한 뒤 분석했다.

최종 fullstate781728117B SHA `9ae9837e4d6c2324473e8a69653d16786dea70b671c9f402725538c8aedd2ab9`, algorithm `bac7db67fc3e53e1621df3b429456a306ea0ec8f0e74a4c1ae516c3ce3b427bf`, planSHA `2da0351ed2de70511b0c0dca3044dab6bb28f1df0811a3f248e17945b5f707eb`다. ROOT가 import0/mid2048/final4096의 SHA/safeload/재귀 digest/91개 Adam·scaler·저장된 모든 RNG의 정확 이관/동결/순서·커서·노출·호출·평가 SHA를 확인했다. 추가 모델 construction/forward/CUDA replay0이다. CPU reviewer 마지막 버전 라벨이 V36으로 남은 오류는 원본/실패를 보존한 뒤 V37로 고쳤다. 학습 코드나 결과를 바꾼 오류가 아니다.

| 같은 관측 패널 | V36 | V37 |
|---|---:|---:|
| 개발256 teacher NLL | .828950 | .808698 |
| 개발256 teacher byte 정확도 | 75.2747% | 76.1279% |
| 개발256 teacher EOS | 255/256 | 255/256 |
| 학습 참고64 BOS 원문 일치 | 23/64 | 21/64 |
| 개발16 BOS 원문 일치/전체 의미 충실 | 0/16,0/16 | 0/16,0/16 |
| 학습64/개발16 UTF8+EOS | 64/64,16/16 | 64/64,16/16 |

ROOT가 원문/복원80쌍을 모두 읽었다. 학습 참고64는21정확/9일부 명제 보존/34충실하지 않음이다. V36 대비7개 정확 회복/9개 정확 복원 이탈이다. 개발16은 완전 의미 보존이나 확인 가능한 일부 명제 복원으로 인정하지 않았다. 고유명사·부정·사건 관계·날짜·후반 절의 손실이 남는다. 초기 넓은 학습의 보조 정확도 개선은 관측됐지만 전체 문장 복원은 아직 부족하다. 전체 FDT의 실패나 의미 추출 완성을 선언하지 않는다.

학습64 chrF69.2140/wordF1 65.4896%/contentF1 60.3470%, 개발16 chrF25.1075/wordF1 12.5621%/contentF1 1.1685%다. V36 개발 chrF25.9781/word16.4007%/content5.4515%보다 문자열 지표는 하락했다. 문자열 유사도는 의미 보존율이 아니다. 저장된 동일80행에서 teacher 첫 오류와 BOS 자기접두사 복원의 첫 오류 위치가 모두 같았고, teacher 전체행 정확21/0도 BOS21/0과 일치했다. 추가 forward0이며 encoder/decoder 단독 원인이나 모든 입력의 등가성을 증명하지 않는다.

같은 개발16 정상NLL1.109799341에서 K_P 제거+.042479817/K_C 제거+3.872050014/K_E 제거+.319167924다. full Exact key/query/valid/top16+NULL 및 입력 UTF8 색인, 독립 FP64 long/recent N,D를 저장값으로 재구성했다. 세 읽기 경로가 실제 사용되나 K_P certified persistent Writer, K_C 전체 층 prefix read-before-write, K_E 명시적 literal selector/transport는 미완성이다. 현재 원문 자동 복사/분절 연결 우회/literal transport0, 디코더는 Commit만 받는다. 미래 명시적 Commit 전문 인용 계약과 원래 typed core는 보존했다.

이번 노출은 같은TRAIN8192문서/32768구간/2926478target UTF8bytes의 한 순회다. 새 메모리 함수로 넓은3회98304rowselections/8779434target bytes이며 새 문서 수가 아니다. 참고64 과거 반복 적응/관측 개발 반복 사용/문서당4구간 상관 한계를 유지한다. V33의 학습64/64 용량 분기와 모든 이전 결과·AdamRNG·실패를 보존한다.

근거: runs/three_memory_codec_20261029_v37/root_result.json, root_all80_reading.md, root_reconstruction_similarity.json, root_three_memory_readouts_review.json 및 code_snapshot/root_terminal_native_observation.json.

## 저장공간 복구

별도 고정44파일/34GiB/480초 상한의 lossless NTFS 작업은 목표 도달로36파일/189.688초에 종료했다. allocated2379898698B 절약, 실제 Cfree35518427136→37900746752B, 모든 파일 논리 SHA·길이 동일/모든 owned compact 종료/삭제0/이동0/직렬화 변경0/모델 forward0이다. V37 최종 source는 제외했다. OS의 다른 디스크 사용도 있을 수 있어 실제 Cfree 변화를 전부 압축 인과로 단정하지 않는다. 나머지7파일/미사용 시간을 같은 작업으로 자동 재사용하지 않는다. 근거 research/artifacts/owt_checkpoint_lossless_space_20261029_v6/result.json.

다음은 동일 세 메모리 언어→입력 의미→같은 원문 목표의 네 번째 넓은 유한 순회다. 원래 의미 개선 denoising은 계속 동결한다. V37 전체 모델/91Adam/scaler/저장된 모든 RNG를 이관하고 기존 baseLR 새 유한 cosine 주기만 사용한다. 새 목적·plan·고정 한도·CPU 복구/실제 preflight·독립 OS검토 후 한번 실행한다. 과거 예산을 다시 열지 않는다.
