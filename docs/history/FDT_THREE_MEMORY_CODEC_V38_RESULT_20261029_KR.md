# V38 세 메모리 문장 복원 결과

V38은 같은 원문을 복원하는 네 번째 넓은 순회를 완료했다. 4096attempt/4094update/AMPskip2(1717,3872), worker653.785314초, encode4162+decoder10185=14347호출/denoise0이다. 새1200초/17000호출은 CLOSED/미사용2653호출·시간 재사용0. owner/proxy/worker26316·40536·15172·42704 모두 exit0/종료, ROOT21:22:15 fresh nativePython·compact0 뒤 분석했다.

최종782767797B SHA `06f26d363c4ade4f0c70f043bc2506da4024f06644ff156ded0aabc29da74137`, algorithm `c4351d80c8c5cce0cdda11f6f64d3266feaf24c02a55e69ecd261464fbb412e0`, planSHA `0fdf36a73f9f81dda3b97c06dbbe7c92f4303cc104a65be490e1b5d4d8d5e4a6`. ROOT가3fullstate/importALL91Adam-model-scalers-ALLstoredRNG/모든SHA·재귀digest·동결·노출·순서·커서·호출·평가파일을 검증했다. 추가 model construction/forward/CUDA replay0이다. Scaler65536/tracker210→32768/tracker224는2skip과성장일정에 일치한다.

| 같은 관측 패널 | V37 | V38 |
|---|---:|---:|
| 개발256 teacher NLL | .808698 | .792027 |
| 개발256 teacher byte 정확도 | 76.1279% | 76.8475% |
| 개발256 teacher EOS | 255/256 | 254/256 |
| 학습 참고64 BOS 원문 일치 | 21/64 | 20/64 |
| 개발16 BOS 전체 일치/전체 의미 충실 | 0/16,0/16 | 0/16,0/16 |
| 학습64/개발16 UTF8+EOS | 64/64,16/16 | 64/64,16/16 |

ROOT가 원문/복원80쌍을 전부 읽었다. 학습64는20정확/6일부 명제 보존/38충실하지 않음이다. 6개 정확 회복/7개 정확 이탈을 모두 보존했다. 예컨대 movement/Chandler Parsons/수백만 선택지/재생산품을 싫어한 여성/Danil과가짜결혼/Lync 설명은 정확해졌다. Robert Oliver 사진·생존 사건은 남았지만1899가1990으로 변해 사실 오류가 생겼다. teams12/35-37 기록은 남아도one-score게임 맥락이 손실됐다. 개발16은 핵심 관계·이름·부정·수치가 여전히 소실되어 완전 의미 보존으로 인정하지 않았다. teacher 개선을 완전 복원이나 FDT 전체 성공/실패로 단정하지 않는다.

학습chrF68.2925/wordF1 65.1264%/content58.8053%, 개발chrF24.9148/word11.9218%/content1.2829%다. 개발 chrF/word는V37보다 소폭 하락했고content는1.1685→1.2829%로 소폭 상승했다. 문자열 유사도는 의미 정확도가 아니다. 저장된80행 teacher 첫 오류와 BOS 복원 첫 오류가 모두 같았고 전체 teacher 정답20/0도 BOS20/0과 같았다. 별도 inference 첫 오류·encoder 대decoder 단독원인을 증명하지 않는다.

같은 개발16 정상NLL1.091859049에서 K_P제거+.050030332/K_C제거+4.153124299/K_E제거+.314467118이다. full Exact key/query/valid/top16+NULL·원문UTF8색인과 sequentialFP64N,D를 저장값으로 독립 재구성했다. 세 읽기 경로는 작동하지만 K_P certified persistent Writer/K_C 전체층prefix/K_E literal selector·transport는 미완성이다. 현재 whole-source-copy·fragment-tiling·literal transport0/Commit-only decoder/no raw H-byte bypass/denoise0를 지킨다. 미래 명시적 Commit 전문 인용 계약을 보존한다.

현재 새 메모리 함수의 넓은4회131072행선택/11705912target UTF8bytes는 같은8192문서/32768구간 반복이다. 새 corpus/download/held-to-train/E2final/registry0. 관측dev 반복 사용·TRAIN64 과거 반복 적응·문서당4구간 상관 한계를 유지한다. 모든 이전 원본·fullAdamRNG·raw·실패·용량64/64분기는 보존한다.

근거 runs/three_memory_codec_20261029_v38/root_result.json, root_reconstruction_similarity.json, root_all80_reading.md, root_three_memory_readouts_review.json, code_snapshot/root_terminal_native_observation.json. global120804attempt120759update/누적384330calls+별도닫힌1936=386266이다.

다음은 삭제 없이 학습 지속 공간을 줄이는 초기 fullstate 중복 제거다. 초기 모델·91Adam·scaler·RNG는 이미 보존된 부모와 완전히 같으므로 pinned parent+새 cursor/order/ancestry metadata로 손실 없이 재구성할 수 있는지 검증한다. 변경된 mid/final/emergency는 계속 self-contained fullstate다. 보호20GiB/추가headroom8GiB를 유지하면서 실제 출력 상한을 줄이는 새 저장 계약이며 과거 floor·시계·예산 변경이 아니다. 새 계약 CPU복구·실제preflight·독립OS검사 후 다음 별도 유한 학습을 수행한다.
