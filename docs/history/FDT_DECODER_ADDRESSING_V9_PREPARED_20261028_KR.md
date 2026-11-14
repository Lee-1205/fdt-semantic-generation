# V9 구현·CPU/실제 preflight 완료, 실행 봉인은 자동 검토에서 차단

최신 사람 지시 “해봐”에 따라 V8의 연결부 학습을 이어갈 V9를 구현했다. 실제 GPU 학습은 아직 없다.

변경은 decoder query의 index i가 latent memory의 같은 index i를 더 명확하게 참조하는 soft cross-attention prior다. 가중치는 `log(유효 memory 수−1)`이며, 내용 score가 같을 때 해당 한 칸의 attention 질량은1/2가 된다. 예를 들어 전체386칸에서는 기존 균등 prior1/386≈0.00259 대신0.5다. 나머지 memory를 차단하지 않고, 실제 입력으로 알 수 없는 정답 길이·역할·답변 template를 사용하지 않는다. 일반적인 상대 위치 연구([Shaw 등 원문](https://arxiv.org/abs/1803.02155))를 확인했지만 이 특정 주소 prior는 별도 설계이며 논문 재현을 주장하지 않는다.

V8의 모든100994050개 가중치·145개 populated Adam state·scaler·모든 저장 RNG·이미 학습한 memory position을 그대로 가져온다. 새 parameter나 위치 초기화는 없다. VP 전체 동결/β=0/paired clean-noisy codec/원문 context dropout.5/EOS CE4는 V8과 같다. 새 순열81–84로3271행을 네 번, 최대1636attempt/13084행/1538700positive UTF8 target bytes 학습하는 별도 목적이다. 같은512문서 재노출이며 새로운 문서가 아니다.

예산은 새900초/reserve90/actual model call11000상한/계산 최대10182/output6GiB로 준비했다. 초기·최종의 작은 clean/noisy codec 및 opposite-order oracle 인터페이스 진단과, 최종 실제 예측 VP dev NLL/자유생성을 분리한다. 주소 prior는 초기 함수 자체를 바꾸므로 V8의 저장 점수를 새 V9의 정확한 초기 baseline이라고 하지 않는다. 기존 V8 예산의 미사용 시간·호출을 재사용하지 않는다.

관련 CPU 검사는8개와 CPU autocast(float attention mask) 검사1개, 총9개를 통과했다. 유효 memory mask/같은 index prior, 순서에 대한 민감도와 causal prefix, 모든 기존 가중치·Adam/scaler/RNG의 복원,4epoch 마지막7행과 다음 update의 정확한 복구, 실제 scaler skip, 새 실제 schema의 post-Popen 자연·예외·UNCONFIRMED 처리를 포함한다. 실제 worker CPU preflight를 포함한 독립 OS validate도 `INDEPENDENT_VALIDATE_PASS`로 완료했다. 새 GPU/optimizer 호출은0이다.

새 코드:

- `fdt_local_102m_20261026/research/text_decoder_addressing_v9.py`
- `fdt_local_102m_20261026/scripts/text_decoder_addressing_engine_v9.py`
- `fdt_local_102m_20261026/scripts/run_text_decoder_addressing_20261028_v9.py`
- `fdt_local_102m_20261026/scripts/supervise_text_decoder_addressing_20261028_v9.py`
- `fdt_local_102m_20261026/scripts/prepare_text_decoder_addressing_20261028_v9.py`
- `fdt_local_102m_20261026/tests/test_decoder_addressing_runtime_v9.py`

실제 draft는 `runs/text_decoder_addressing_20261028_v9/plan.json`, SHA `ec87f12a4277c8d32c3263c771067f5935f1cdc86f07d9027d64157b737075b3`다. Source는 V8 최종 `9b7f5bf310922c6d7261ef75b30091b1252dc51c66bf61d92ed440534dd61258`이며 불변이다.

실행 전 자동 승인 검토가 명령을 두 번 차단했다. 첫 번째는 계획 봉인과 `Start-Process` 실행을 묶은 명령이었고, 두 번째는 더 안전한 도구 세션 내 foreground 감독 실행을 준비하기 위해 **봉인만 분리한 명령**이었다. 두 번째도 실행 전에 `rejected: blocked by policy`로 거부됐으며 구체적인 차단 사유는 반환되지 않았다. 이 결과로부터 특정 코드나 자원 요청이 원인이라고 추측하지 않는다. 다른 경로로 같은 봉인을 우회하지 않는다.

두 명령 모두 프로세스 생성 전에 거부됐다. 실제 read-only 재확인에서 `authorized:false`, `allocation_start/deadline/cutoff:null`, `readiness.json` 없음, `progress.json` 없음, Python 프로세스0이었다. 따라서 **봉인·학습·GPU 실행·새 fullstate는 없으며 V9 성능 개선을 주장하지 않는다.** 기존 모든 모델·Adam/scaler/RNG·원문·실패는 보존했다.

다음은 자동 승인 검토의 차단이 해소된 뒤 이 구체적인 계획을 정상 봉인·실행하는 단계다. 같은 봉인 명령을 자동으로 무한 재시도하거나 오래된 deadline을 재개하지 않는다. 사람의 기존 하드웨어 승인은 유효하며, 이번 중단 사유는 새 하드웨어 승인을 기다리는 것이 아니다. Timer는30분/문구‘재개’를 유지하고, 바뀌지 않은 차단 상태를 반복 정상 진행으로 보고하지 않는다.
