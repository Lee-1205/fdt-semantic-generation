# V33 — 실제 3메모리 언어 Codec 후보와 복구 driver의 CPU 연결 완료

현재 목적은 **문장 → 입력 의미 → 같은 문장 복원**이다. 디노이징·답변 의미 생성은 계속 학습하지 않는다. 기존 V31의 학습 결과와 모든 원래 가중치/Adam/RNG는 보존했다. 아래는 새 구조의 CPU 구현·복구 계약 결과이며 **새 GPU 학습이나 복원 성능 개선 결과가 아니다.**

## 이번에 실제로 연결한 경로

`입력 bytes → 기존 언어 encoder H → 분리된 K_P/K_C/K_E → 입력 의미 Commit → 기존 causal decoder`

- **K_P**: 원래 anchors/prior와 μ/ρ/γ 읽기를 재사용한다. 인증된 장기 N/D Writer가 완성됐다는 뜻은 아니다. 현재는 학습된 prior의 snapshot이다.
- **K_C**: 기존의 ‘문장평균 value 하나를 활성 anchor 전체에 공유’하는 구현을 바꿨다. 각 입력 token의 문맥 H에서 별도 μ와 value를 계산해 anchor별 long/recent N/D를 FP64로 누적한다. Anchor별 recent decay와 query별 long/recent gate를 학습 가능하게 연결했다. 빈 memory는 0이며 prior로 채우지 않는다. Padding은 recent memory를 노화시키지 않고 서로 다른 입력 행의 기억을 합치지 않는다.
- **K_E**: 원문 UTF8 bytes/sourceID/version/hash/byte span을 독립 보존한다. 정답 annotation이나 regex 대신 입력의 모든 UTF8 codepoint 경계에 learned key/value index를 만든다. 입력 의미 query가 최대 16후보+NULL을 읽고, NULL 질량을 버리지 않은 읽기 결과가 입력 의미에 들어간다. 모델 update 뒤 index를 재사용하지 않도록 embedding epoch를 묶었다. 이 새 key/query/gate는 **아직 훈련되지 않았다**.
- **Commit**: 훈련용은 독립 tensor clone을 통해 encoder까지 gradient를 유지한다. 추론용은 detach한 bytes로 봉인하고 reader마다 복사한다. Realizer의 공개 context 인터페이스는 Commit만 받고 live memory/원문 H/episode/dict는 거부한다. 이는 Python API 경계이며 악의적 내부 접근까지 막는 보안 sandbox라는 뜻은 아니다.
- **현재 복사량 0**: Exact 원문 bytes나 정답 span을 decoder에 주지 않는다. 현재 신경 literal selector/copy 의도 학습은 미완료다. 임의 암호의 완전 복원이나 실제 Exact 조각 선택 성공을 주장하지 않는다. 향후 명시적인 semantic Commit으로 전체 원문을 운송할 수 있는 V32 permit 기능은 보존했고, 현재 자동 전체복사·조각 타일링으로 점수를 올리는 정책은 켜지 않았다.

K_C는 **입력 전체를 관측한 뒤의 문장별 snapshot**이다. 원래 vNext의 모든 layer에서 prefix read-before-write를 수행하는 큰 backbone/지속 Writer/인증 증거 구조 전체를 이번에 구현한 것은 아니다. 이후 의미 디노이징이 같은 세 memory bundle을 읽을 수 있는 타입 경계를 만들었으며 지금 denoise 호출은 0이다.

## 필요한 CPU 결과

서로 다른 focused 검사 **7개 + 새 driver 복구 검사 2개 = 9개 PASS**다. 기존 검사를 반복한 횟수를 독립 증거 수에 더하지 않았다.

1. 기존 Language.forward의 pooling과 새 per-token 노출 경로가 같은 tiny 모델에서 정확히 일치했다. 입력 행 분리·padding 불변을 검사했다.
2. 새 벡터화 FP64 long/recent N/D가 V32의 순차 기준 연산과 일치했다. 서로 다른 token value와 decay gradient를 확인했다.
3. K_P/K_C/K_E를 각각 끄면 Commit이 달라졌고, decoder context에는 의미 view만 남았다. 이는 기능 경로 연결 확인이며 의미 이해·성능·기억 기여의 실험적 증명이 아니다.
4. UTF8 경계/정규화하지 않은 identity/동일 source revision의 충돌/오래된 Exact index 거부/NULL 확률 보존을 검사했다.
5. Commit reader를 바꿔도 원래 Commit은 바뀌지 않았고, 훈련 Commit의 gradient와 추론 봉인 값이 맞았다.
6. 같은 입력 재복원 loss에서 encoder·새 memory head까지 gradient가 연결됐다. 원래 reasoning/refiner/time/velocity는 동결·호출 0이었다. 출력 정답은 source와 같은 bytes+EOS인지 별도로 강제했다.
7. 기존 모델 tensor key/shape/dtype를 엄격 복사하고 새 key만 초기화하는 계약을 검사했다.
8. 원래 3개 populated Adam group을 보존하고 새 4번째 memory group만 빈 moments로 추가했다.
9. Enabled CPU GradScaler의 AMP skip, 모든 저장 RNG, phase/worker stage, 평가 참조, 유한 cursor/order, 마지막 7행까지 저장·복구한 뒤 다음 update와 전체 algorithm digest가 정확히 일치했다. Stale epoch의 복구 거부는 기존 driver 상태도 바꾸지 않는다. **실제 CUDA 복구 재실행 증거는 아니다.**

첫 후보 검사에서 argsort의 dim을 positional로 준 오류를 keyword로 수정했다. 실패한 코드와 XML을 보존했다. Driver 최초 검사는 Windows 기본 pytest 임시 폴더 ACL 때문에 setup에서 막혔고, 새 artifact 내부의 별도 tmp 경로로 해결했다. 기존 임시 자료를 삭제하거나 권한을 변경하지 않았다.

CPU manifest 작성 중에는 package 전체의 SHA를 단일 model 파일 SHA와 비교한 metadata 검사 오류도 수정했다. 원래 helper대로 `src/fdt_local` package SHA `eb68...f352`와 V31 plan의 `model_masked.py` 파일 SHA `ef0d...d4d7`를 각각 확인한다. 실제 원래 core가 바뀐 사건은 아니며 실패 기록을 남겼다.

## 실제 V31 source를 대상으로 한 CPU 이관

Tiny 검사를 실제 모델 성능으로 바꾸어 말하지 않기 위해 source 이관도 별도로 검사했다.

| 항목 | 실제 값 |
|---|---:|
| 새 후보 전체 parameter | 111,732,306 |
| 학습 대상 parameter | 38,556,357 |
| 추가 memory parameter | 246,595 / 10 tensor |
| 기존 source model tensor 그대로 복사 | 264 / 264 |
| 기존 populated Adam tensor 그대로 이관 | 81 / 81 |
| optimizer group tensor 수 | 19 / 44 / 18 / 새10 |
| 새 group의 초기 Adam moments | 0 |
| 실제 크기 source/candidate forward | 0 / 0 |
| 실제 크기 optimizer step / 새 CUDA 호출 | 0 / 0 |

V31 fullstate SHA `1d02d0d52c44683a582e70273e3e951e84384ecb406700470af620e21e8c252f`, algorithm `66316e12eeeb9ae3d6c3515f5983deca0363f9d54f2166c1bc6210a0c604bf19`가 전후 동일했다. 기존 model/Adam/scaler/frozen scope/Torch·CUDA저장·Python·NumPy RNG 및 source call ledger를 보존했다. 신규 allocation counter는 0이다. 새 구조의 함수가 달라지고 parameter도 추가되므로 **엄격한 동일 학습 trajectory 재개라고 부르지 않는다**. 원래 optimizer moments를 보존한 새 branch의 warmstart다.

실제 source를 검사한 첫 기록은 ‘driver 미구현’ 시점의 inventory이고, 이후 `root_driver_source_review_r2.json`이 최종 driver의 실제 source 이관 결과다. 앞선 기록을 소급 수정하지 않았다. 실제 크기 새 fullstate를 disk에 저장했다는 주장은 하지 않으며, serialization/next-update 검사는 tiny driver에서 수행했다.

## 현재 상태와 다음 작업

새 코드는 `research/three_memory_codec_v33.py`, `scripts/three_memory_codec_engine_v33.py`다. 증거는 `research/artifacts/three_memory_codec_20261029_v33_cpu`에 고정한다. 이전 V32 reference와 V31/V30/모든 이전 모델·실패·source를 바꾸지 않았다. 현재 실제 훈련 성과는 여전히 V31의 **훈련64/64 정확복원, 기존 held 퇴행**이며 새 구조의 성능은 미측정이다.

다음은 이 driver에 맞는 별도 worker/실제 plan schema/CPU preflight/독립 owned OS 감독을 연결하고, 새 자료·시간·call·노출·output cap을 정한 한정 **언어 Codec 학습**이다. 이전 GPU allocation/unused budget을 재사용하지 않는다. 현재 C 여유 `39,462,854,656B`는 strict 시작 기준 `42,949,672,960B`에 못 미친다. 새 GPU 전에 원본 삭제 없이 별도의 저장 여유도 해결한다. 재승인 질문으로 멈추지 않는다.

그다음 실제 재복원 결과를 직접 읽어 주어/대상/관계/부정/수량/시간과 exact identity를 구분해 평가한다. 현재 순수한 입력 의미는 관측되지 않는 정답 잠재 공간으로 선언하지 않는다. 언어 Codec이 충분히 안정된 뒤에만 **입력 의미 + 동일 세 memory → 답변 의미 디노이징 → 같은 realizer**로 넘어간다.
