# V40 세 메모리 원문복원 결과

V40은8192attempt/8189update/AMPskip3(547,2819,5597),worker1223.181173초에 완료됐다. encode8258+decoder18355=26613호출/denoise0이다. 새2100초/30000호출은 CLOSED이며 미사용3387호출·시간을 재사용하지 않는다. owner12456→21176,worker20960→32204 모두 exit0/종료했고 ROOT10:39:21KST nativePython·compact0을 확인한 뒤 저장값만 분석했다.

최종 독립 fullstate786918709B SHA `fbaf29eca2c5c9e1df12e3ea71702aeee40c0bda5c7f9ec73cba339acd4cd677`, algorithm `3262ad6fd5e6f9f38552c4ee1207735e76f42a9f61cc8c005d01179b65aa4199`, planSHA `00aca0d669b2228c49fd075ca45a9425ed655d7ce40857bc0a171c1905635a70`다. 초기29508525B 참조는 보존된V39부모와 합쳐 정확히 복구했다. V40에는 중간 fullstate가 없고 최종 한 벌만 있다. ROOT는 두 저장점의 SHA/safeload/재귀digest,ALL91model·Adam/scaler·모든 저장RNG 이관,동결·수치·Adamstep·scaler skip/growth·두epoch seed39001/39002 실제순서·cursor65536·노출·호출·평가SHA를 검증했다. 추가 model construction/forward0/CUDA복구 replay0이다.

| 같은 관측 자료 | V39 | V40 |
|---|---:|---:|
| 개발256 teacher NLL | .770554 | .757894 |
| 개발256 teacher byte/EOS 기호 정확도 | 78.3083% | 79.0796% |
| 개발256 teacher EOS | 255/256 | 255/256 |
| 학습 참고64 원문 완전 일치 | 28/64 | 31/64 |
| 개발16 원문 완전 일치/전체 의미 충실 | 0/16,0/16 | 0/16,0/16 |
| 학습64/개발16 UTF8+EOS | 80/80 | 80/80 |

worker의 `byte_accuracy`는 PAD를 제외한 byte와 EOS를 함께 센 값이다. EOS는 이번48–128B 원문 구간의 종료 기호이며 문서 전체의 실제 EOF 복원 능력을 뜻하지 않는다.

ROOT가80원문·복원쌍을 모두 읽었다. 학습 참고64는31정확/1비정확이지만전체명제보존/11부분명제보존/21충실하지 않음이다. 비정확 전체명제1개는 자수와 Danil과의 결혼이 가짜라는 진술이 모두 남고 쉼표 위치만 바뀐57번이다. 이는 수동 정성 판정이며 byte exact에 합치지 않는다. 정확 회복14개/이탈11개다.1948·1899 날짜,automaton,질문표,WordPress 설치 설명,Oakland/NFL 거부와Krugman 진술 등이 회복됐다. 반대로 hive frames→farmers,2008/17→2017/7,여성의 remanufactured model 거부 소실,baby shower·Bluetooth speaker 내용 소실 같은 퇴행도 남겼다.

개발16은 전체 의미 충실0이다. V39에서 알아볼 수 있었던 no-explosion 핵심도 V40에서는 식별할 수 없었다. 개발chrF26.7362→26.9117,wordF1 14.1254→16.0923%,content3.5565→4.3501%,prefix4.5→5.5625bytes는 문자열 지표의 관측 개선이며 의미 보존율이 아니다. 학습chrF77.5905/word75.0016%/content70.9086%다. 모든80행의 저장 teacher 첫오류 위치와 BOS 자기접두사 복원의 첫오류 위치가 같았다. 이후의 자기접두사 오류 누적만으로 모든 문제를 설명할 수 없지만 encoder 또는 decoder 단독 원인을 확정하지도 않는다.

같은 개발16 정상NLL1.071808154에서 K_P제거+.061701576/K_C제거+5.165900967/K_E제거+.227925628이다. 저장된 Exact key/query/valid/top16+NULL/원문UTF8 색인과 순차FP64 long/recent N,D를 독립 재구성했다. 세 읽기 경로는 실제 사용된다. 그러나 K_P certified Writer/K_C 전체층prefix/K_E literal selector·transport 미완성 범위는 그대로다. Commit-only/no rawH-byte bypass/전문자동복사0/literalcopy0/denoise0를 유지했다.

이번은 같은8192TRAIN문서/32768구간의 두 유한 순회65536행선택5852956target UTF8bytes다. 새 메모리 함수의 넓은 학습은 누적8순회262144행선택23411824target UTF8bytes이며 새 문서262144개가 아니다. global137188attempt137138update/누적437572calls+별도닫힌1936=439508이다. 관측dev 반복/참고64 과거 적응/문서당4구간 상관 한계를 유지한다. 새자료·held-to-train·E2final·registry0/모든 과거 원본·실패·weightsAdamRNG 보존이다.

넓은 자료의 teacher NLL과 문자열 지표는 계속 학습되고 있지만 전체 원문복원은 미완성이다. FDT 전체 실패나 완전한 의미 추출을 선언하지 않는다. 다음은 같은 구조·세 메모리·모든AdamRNG를 유지한 별도 고정 한도의9·10번째 TRAIN 순회다. V40 저장값을 기준선으로 재사용하고 초기·중간 GPU평가를 반복하지 않는다. 새 code/plan/실제복구·CPUpreflight·독립OS 검증과 리소스 조건을 확인한 뒤 한 번만 실행한다. 현재 단계는 언어 인코더/디코더이며 의미 디노이징으로 넘어가지 않는다.

근거 run/root_result.json,root_reconstruction_similarity.json,root_all80_reading.md,root_three_memory_readouts_review.json,code_snapshot/root_terminal_native_observation.json 및 scripts/review_three_memory_codec_v40_result_cpu.py,review_v40_three_memory_readouts_cpu.py다.
