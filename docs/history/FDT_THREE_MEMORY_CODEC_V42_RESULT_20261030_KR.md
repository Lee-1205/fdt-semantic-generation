# V42 원문 복원 결과: teacher 개선과 완전 복원 퇴행

V42는8192attempt/8189update/AMPskip3(2397,6436,7147),worker1235.567516초로 완료했다. encode8258+decoder18422=26680호출/denoise0다. 새2100초/30000호출 예산은 CLOSED이고 미사용3320호출·시간을 재사용하지 않는다. ROOT13:39:21·13:39:41KST에 nativePython/compact0,owner22660·40880/worker3620·29008 종료를 먼저 확인했다. 이후 저장값만 검토했고 추가 모델 construction/forward/GPU0다.

최종791069621B SHA `41f0fc13626adb5ebdef9fe2e02842f64a0b78de715172e0d51c2079f04c0ad3`,algorithm `129f2b79eae7af34251d2892c69c60951f845af4eadc48c5efee13d8247d2734`,plan `6211513e03f367ca0be8193525f6c52155382950241f3245b72f3cc2f7c06b4b`다. 초기 gzip12917561B는 부모V41 FINAL을 사용해 정확 복구했다. Raw metadata33659437B/물리32MiB·raw64MiB 상한을 지켰다. 최종은 일반 self-contained fullstate 하나이고 midpoint/emergency는 없다. ROOT는2저장점 SHA/safeload/재귀digest,ALL91Adam/model/scaler/모든 저장RNG 정확 이관,동결·수치·Adamstep·skip/growth·seed41001/41002 순서/cursor65536·노출·호출·평가SHA를 확인했다. Scaler65536/tracker367→65536/tracker1045다. 별도 owned CPU 검토66.922초 내 자연종료/allgone이며 CUDA 복구 replay가 아니다.

| 같은 관측 자료 | V41 | V42 |
|---|---:|---:|
| 개발256 teacher NLL | .747215 | .745030 |
| 개발256 teacher byte+EOS 정확도 | 79.8078% | 80.2258% |
| 개발256 teacher EOS | 255/256 | 255/256 |
| 학습 참고64 자기 접두사 원문 완전 일치 | 39/64 | 31/64 |
| 학습 참고64 전체 명제 보존 | 41/64 | 33/64 |
| 개발16 원문 일치/전체 명제 보존 | 0/16,0/16 | 0/16,0/16 |
| 80복원 UTF8/EOS | 80/80,80/80 | 80/80,79/80 |

ROOT가80원문·복원쌍을 모두 읽었다. TRAIN참고64는31정확/2비정확이지만전체명제보존/8부분명제/23충실하지 않음이다.45번 linker options→linkers option과57번 turned herself in의 쉼표 이동은 내용 보존으로만 분리했다. 부분명제는7,17,20,22,27,31,36,59번이다. 날짜1899→1989,기관NFL→NBA,부정·이름·역할 소실을 문자열 유사도 때문에 정답으로 인정하지 않았다.

정확 회복4개(10,12,24,47),이탈12개(9,13,16,17,27,29,36,37,43,46,53,63)다.12팀/35-37/one-score games,일부 정의의 movement/일부는 아님,노조의 역할,Belichick의 칭찬은 회복했다. Gameplay,Wall Street의 century,David Cameron의 big society,NFL,일부 조건문 등은 퇴행했다.0번은 Mr_Ogunjobi/Panthers 희망을 잃고 f 반복으로129symbol 상한까지 가서 EOS가 없었다. 이는 같은 원문을 복원하는 출력의 오류이며 별도 자유 주제 생성 결과가 아니다.

학습chrF83.0620→78.9581,wordF1 81.2789→76.1773%,content77.6091→72.1631%,올바른 초기byte50.5→47.015625다. 개발16은chrF25.6977→27.7626,word14.0162→15.5469%,content3.8730→4.6200%,prefix4.5625→5.5625B로 상승했다. 개발16의 전체 의미 충실은 여전히0이다. 유사도 개선·teacher 개선과 완전 복원 퇴행을 모두 남긴다. Teacher byte 지표는 구간EOS 포함이며48–128B 원문 구간의 복원을 문서 전체EOF/일반 의미 완성으로 확대하지 않는다.

같은 개발16 정상NLL1.077394740에서K_P제거+.063719153/K_C제거+6.127640676/K_E제거+.159427415다. 저장 Exact top16+NULL/UTF8 index와 sequentialFP64 long/recent N,D를 재구성했다. 세 READ 경로를 유지했다. K_P certified Writer/K_C 전체층 prefix/K_E literal selector·transport는 미구현이며,Commit-only/no rawH-byte bypass/wholecopy0/denoise0다.

## 저장 오류 위치와 후속 판단

V41은43개 teacher 기호 오류가25행에, V42는49개가33행에 분포했다. V42의80행 모두 첫 teacher 오류 위치와 자기 접두사 복원 첫 오류 위치가 같았다. 따라서 뒤의 반복 누적만으로 모든 오류를 설명할 수 없다. 반면 이 일치가 encoder·decoder 중 단독 원인을 증명하지도 않는다. 이전V41의79/80일치는 역사로 그대로 보존하며 같은지를 통과 조건으로 강제하지 않았다.

완전 복원 이탈12개 중10개는 첫 오류의 저장 FP16 gold-vs-best-other 로짓 차이 절댓값이1이하,중앙값.3671875였다. 이 값은 전체 파라미터 gradient나 실제AR logits 측정이 아니다. 기존8192-step cosine의 최종 factor는.10000003309042696이고 같은 baseLR 일정 재시작은 약9.99999669배 상승을 만든다. 이 산술과 참고64의 출렁임은 낮은 학습률 지속을 시험할 이유가 되지만, 자료 순서·동시 학습이 달라 인과 증명은 아니다.

다음 목적은 마지막4개 group LR을 그대로 유지한 유한 두 순회로 재상승을 없애는 안정화 실험이다. 후보LR은1.0000003309042696e-6/1.0000003309042697e-5/3.000000992712809e-7/1.0000003309042697e-5다. ALL91 moments/step/scaler/RNG와 세 메모리·언어-의미-언어 구조를 보존한다. 같은8192회 더 돌리기만 하는 기존 cosine 재시작과의 차이는 이 LR 정책 하나다. 아직 새 실행/효과를 검증한 것은 아니다.

학습/추론 접두사 차이는 [Scheduled Sampling 원논문](https://arxiv.org/abs/1506.03099),반복을 직접 낮추는 손실은 [Unlikelihood Training 원논문](https://arxiv.org/abs/1908.04319)을 다시 확인했다. 두 연구는 후속 후보의 근거이며 현재 오류의 단독 원인이나 byte 반복 벌점 최적값을 정해 주지 않는다. 현재 자료에서는 첫 오류가 teacher에도 있으므로 이번 후속에 rollout·강한 반복벌점·메모리 구조 변경을 동시에 넣지 않는다. 낮은 LR 결과를 보존한 뒤 필요한 변경을 판단한다.

새 메모리 함수의 넓은 학습은12순회393216행선택35117736target UTF8bytes/SAME8192문서32768구간이다. Global153572attempt153515update/누적490868calls+별도닫힌1936=492804다. 관측dev 반복·TRAIN64 과거 적응·문서당4구간 상관의 한계를 유지한다. 범용 완전 복원/전체 FDT 실패/독립 추론을 선언하지 않는다. 원래core/typed/모든 부모·실패·raw·weightsAdamRNG와 사용자 원본을 보존했으며,E2final/registry0다.

근거: run/root_result.json,root_all80_reading.md,root_reconstruction_similarity.json,root_three_memory_readouts_review.json,root_saved_error_margins_and_LR.json,root_saved_review_cpu/final_status.json,code_snapshot/root_terminal_native_before_analysis.json. 이후 준비와 새GPU 실행은 별도 증거로 기록한다.
