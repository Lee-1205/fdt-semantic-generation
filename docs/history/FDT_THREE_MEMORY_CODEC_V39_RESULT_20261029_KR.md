# V39 언어–입력 의미–원문 복원 결과

두 순회가8192attempt/8190update/AMPskip2(2140,4565),1245.453475초에 완료됐다. encode8258+decoder18371=26629호출/denoise0. 새2100초/30000호출은 CLOSED/미사용3371호출·시간 재사용0이다. 실제 owner32836→33832/worker31036→12392 모두 exit0/종료했고, ROOT23:19:14 nativePython·compact0을 확인한 뒤 분석했다.

최종 fullstate784843253B SHA `1b698b47e9ad386f2713b5532b5f604166f448ec9f8d561c97b830a9462ce3a3`, algorithm `fbc825d7e678b61c516019dd8323c704a857767a5601488463ddd6f413ea0bf9`, planSHA `3d9563a081a80ed6c9be52e4c221bccd7eb9de3d83f9094818f9fd5a2183a68b`이다. 초기 참조 checkpoint는 보존된 V38부모와 합쳐 정확한 전체 상태로 복원했고, mid4096/final8192는 독립 fullstate로 확인했다. 세 지점 SHA·재귀digest/safeload/ALL91model·Adam/scaler·모든 저장RNG/동결/순서·커서·노출·호출·평가SHA를 검증했다. 추가 model construction/forward/CUDA replay0이다. 초기 참조 파일만 단독으로는 복구되지 않는다는 제약을 유지한다.

| 같은 관측 패널 | V38 | V39 |
|---|---:|---:|
| 개발256 teacher NLL | .792027 | .770554 |
| 개발256 teacher byte 정확도 | 76.8475% | 78.3083% |
| 개발256 teacher EOS | 254/256 | 255/256 |
| 학습 참고64 BOS 원문 일치 | 20/64 | 28/64 |
| 개발16 BOS 원문 일치/전체 의미 충실 | 0/16,0/16 | 0/16,0/16 |
| 학습64/개발16 UTF8+EOS | 64/64,16/16 | 64/64,16/16 |

ROOT가80원문·복원쌍을 모두 읽었다. 학습 참고64는28정확/8일부 명제 보존/28충실하지 않음이며14정확 회복/6정확 이탈이 있다. April2015 벌통 배치·Swansea 패스·의식 설명·금융 문제와 tea-garden 노동자·baby shower 선물·7년지연·FDA not rare·Bluetooth 스피커는 회복됐다. 반면1948→1984,1899→1999,automaton→automation,질문→진술 및Wilson wins→부정처럼 실제 내용 오류가 남았다. 개발16은 전체 의미 보존0이며,8번에서 misspelling exploions에도 no-explosion 핵심은 읽을 수 있었지만 직장 발언제약의 인과 관계는 사라졌다. 이는 정성적 일부 보존1개이며 전체 의미 성공으로 세지 않는다.

학습chrF75.7763/wordF1 73.4248%/content67.8436%, 개발chrF26.7362/word14.1254%/content3.5565%다. 개발 문자열 지표와 teacher 정확도는 관측 개선됐지만, 문자열 유사도를 의미 보존율로 바꾸지 않는다. 동일80행 teacher 첫 오류 위치와 BOS 자기접두사 복원 첫 오류 위치는 모두 같았다. 저장값만 비교했으며 encoder 또는 decoder의 단독 원인을 확정하지 않는다. 초기 넓은 학습에서 관측된 개선으로 기록하고 FDT 전체 성공/실패나 완전한 의미 추출을 선언하지 않는다.

같은 개발16 정상NLL1.091582466에서 K_P제거+.070511937/K_C제거+4.695897636/K_E제거+.308868702다. 저장된 Exact key/query/valid/top16+NULL·원문UTF8 색인과FP64 long/recent N,D를 독립 재구성했다. 세 읽기 경로는 사용되지만 K_P certified persistent Writer/K_C 전체층prefix/K_E literal selector·transport는 미완성이다. Commit-only decoder/no raw H-byte bypass/whole-source-copy0/literal transport0/denoise0를 지킨다. 미래 명시적 Commit 전문 인용 계약과 원래 typed core를 보존한다.

새 메모리 함수의 넓은6회196608행선택/17558868target UTF8bytes는 같은8192문서/32768구간 반복이다. 새 corpus·held-to-train·E2final·registry0. 관측dev 반복 사용/참고64 과거 적응/문서당4구간 상관 한계를 유지한다. global128996attempt128949update/누적410959calls+별도닫힌1936=412895다. 모든 이전 원본·fullAdamRNG·raw·실패·용량64/64분기를 보존한다.

근거 runs/three_memory_codec_20261029_v39/root_result.json, root_all80_reading.md, root_reconstruction_similarity.json, root_three_memory_readouts_review.json, code_snapshot/root_terminal_native_observation.json.

## 저장공간과 후속

별도lossless checkpoint v7은13파일/79.141초,allocated922449321B절약/전체논리SHA·길이동일/삭제·이동·재직렬화·모델forward0/모든compact종료로 닫혔다. 당시 실제Cfree31623684096→31584968704B로 오히려 감소했다. 뒤의 별도raw-logit v8은 시작시Cfree32524042240B로 이미 목표를 넘어0파일 처리 후 종료했다. 두 작업의 미사용 예산은 재사용하지 않는다. Cfree변동의 원인을 전부 압축으로 귀속하지 않는다. 현재V39최종 source는 제외했다.

V40은 같은 구조·AdamRNG로 두 순회를 더 학습할 준비다. 중간 fullstate 저장을 생략하고 정상 최종 또는 중단 상태 한 벌을 보존해 새 출력 한도를1GiB로 줄인다. 초기 참조와 이전 부모는 보존한다. 정상 최종 뒤 오류는 전체 상태가 동일함을 확인한 경우에만 기존 최종을 복구점으로 사용하고, fullwrite 자체 실패는 후보를 보존해UNCONFIRMED로 기록하며 두 번째 큰 파일을 쓰지 않는다. 이 정책은 중간 내구 checkpoint가 없어 최대8192attempt의 진행이 초기 참조 이후 미확정일 수 있다는 대가가 있다. 보호20GiB/임시1GiB/추가headroom8GiB를 유지해 새시작30GiB를 계산한다. 과거32/35GiB계약은 바꾸지 않는다. 새CPU·실제preflight·독립OS검토·freshresource 후 한 번 실행한다.
