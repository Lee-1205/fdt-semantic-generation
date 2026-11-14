# V18·V19 prefix 복원 학습 — 결정성 수정과 실제 결과

원래 퍼지 앵커 기반 문장→입력 의미 추출→재복원을 유지하며, 정답 prefix 일부가 없어도 추출 상태로 복원하도록 훈련 목적만 조정했다. CleanCE.5+손상prefixCE.5,과거 byte의15%를 기존 decoderPAD257로 가리는 단일 후보였다. BOS/EOS/PAD는 손상 대상에서 제외한다. Full B×129 난수를 먼저 뽑아 미래 정답·길이가 과거 mask의 난수에 새지 않도록 했다. 추론 함수는 그대로이며 손상·정답 prefix·gold역할을 runtime에 제공하지 않는다. 추론/출력 의미 디노이징은 계속 동결했고 새 parameter·rawcontext 우회는 없다.

## V18 실제 실패와 수정

CPU 자료/실제 source import/동일 추론/gradient/causal mask·난수/전체복구/ownedOS 계약은 통과했으나, 첫 GPUattempt에서 새 CE의 `reduction='sum'`가 엄격한 결정성 설정의 CUDA nll_loss2d 경로에서 지원되지 않아 중단됐다. V18은1attempt/**0update**/encode1+decoder2=3calls로 닫혔다. SupervisorFAILED/workerexit2/원문 traceback/봉인 코드·plan·fullstates를 그대로 보존했다. 성능이나 학습 성공으로 세지 않았다.

Root가 allowned/proxy 종료와17:36:53nativePython0을 확인한 뒤 CPU로 비교했다. 모든 model tensor와 populatedAdam moments·step,scaler는 V17과 동일하고 gradient는 비어 있다. 새 LR schedule의 지정과 실제 소비된1행묶음/702targetbytes/mask난수/실패 history는 emergency에 보존했다. Source model이 손상되거나 optimizer가 잘못 갱신된 사건은 아니다.

새 V19파일에서 **기존 V12가 실제 사용하던 reduction='none' 후 명시적.sum/count**로 수정했다. Strict determinism을 끄거나 warn_only를 쓰지 않았고 수학적인 loss는 동일하다.9focusedCPU계약12.36초 PASS,추가 reduction값·gradient동치/엄격결정성 유지/실제source schema·mask난수복구/gradient·causal/ownedOS를 확인했다. 실제 workerCPUpreflight·독립OSvalidate도 통과한 뒤 별도 새 plan·SHA·출력·한도의ONE실행으로 진행했다. V18의 미사용8,497call·시간·2,047attempt를 재사용하지 않았다.

이력과 실제 traceback을 먼저 확인한 뒤 [설치 버전과 맞는 PyTorch2.10 원본](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/native/cuda/NLLLoss2d.cu)도 확인했다.213–225행은 sum/mean 경로의 atomic 연산 때문에 결정성 경고를 내고,234행부터 none 경로를 별도로 처리한다. 따라서 엄격 설정을 완화할 이유가 없으며, 수정 방향이 해당 버전 구현 및 V19 실제 CUDA 완료와 모두 일치한다.

처음 V18 CPU 검사에서 manual tiny test가 encoderPAD256과 decoderPAD257을 혼동한1건도 있었다. 실제pack_prompts나 모델코드 문제는 아니며 test 입력만 고쳤고, 나머지7검사는 그대로 두고 실패한 검사만 재실행했다. 수정 전 test/draft와 원인을 보존했다. 이후 V19의9개 검사는 모두 통과했다. CPU 준비를 CUDA 실행 증거로 대체하지 않는다.

## V19 실제 학습

V19는2유한epoch271–272/**2,048attempt·2,046update·2AMPskip**를 완료했다. Skip은1·56회이며 source scaler131,072를 보존한 상태에서 자동 보정됐다. 기존3consecutive/3in16/새allocation총8의 수치 중단 규칙과 모든 비유한loss·gradient·ownership·disk/time/call/process STOP은 유지했다.

AE누적은23,252attempt/23,246update다. 차이6개는 역사적AMPskip5회와 V18의실패attempt1회다. 모델111,485,711parameter/trainable24,150,914와 train2,048문서8,192문장/dev64문서256문장/OWTrevision·분할은 그대로다. 새 다운로드0/heldtotrain0/E2final·registry접근0이다.

이번16,384행/1,457,826targetUTF8bytes를 한 번씩 추출해 clean·손상prefix 두decoderloss에 사용했다. Mask218,800/eligible1,457,826=15.00865%였다. Decoder supervision은2,915,652UTF8byte occurrences/2,948,420byte+EOS symbols로2배지만 새로운 독립 문서나 문장이 아니다. AMPskip도 행·노출·호출을 소비했다. 온라인 epoch평균은 cleanCE.73038→.55746,손상CE1.58177→1.35601,혼합loss1.15607→.95673이다. 고정 최종 모델 trainNLL과 구분한다.

## 같은 관측 개발 자료 결과

| 지표 | V17 | V19 |
|---|---:|---:|
| 정답 prefix 기반 byte 정확도 |58.4134%|57.9567%|
| 정상 NLL |1.665974|1.643806|
| 오답 위치 평균 최대확률 |60.9357%|58.6501%|
| 사후10bin ECE |.170595|.157334|
| 정상 자유 chrF2 /100 |21.4314|21.2620|
| 정상 자유 wordF1 |9.6185%|11.4854%|
| 정상 자유 내용어 exactF1 |1.1117%|1.0880%|
| 자기원문 chrF 단독1위 /16 |11/16|10/16|
| 처음부터 맞는 byte 평균 |2.750|2.3125|
| 자유 완전 일치 / ROOT 충실한 의미동치 |0/16·0/16|0/16·0/16|
| 자유 UTF8/EOS |16/16|16/16|

확률 집중은 조금 줄었고 정상NLL도.02217낮아졌지만 정확도는.45676pp 내려갔다. V15NLL1.431893보다도 여전히 나쁘다. **자유 복원이 개선됐다고 할 결과는 아니다.** 새 learning과 mask objective를 같은 출발점의 별도clean-SGD대조군으로 비교한 것이 아니므로 차이를 mask 하나의 인과 효과로 단정하지 않는다.

ROOT가 새 정상16개도 원문과 모두 읽었다. Syria와 Robby가 남는 등 일부 개체 흔적은 있으나 Israel·Assad·ceasefire,scouting·시점 같은 관계는 사라졌다. 행위·부정·개체 연결과 비단어 문제는 여전하다. 상태교환 chrF15.7491/제거15.8770보다 정상21.2620이 높고 정상·교환16종/제거1종이므로 추출 정보를 전혀 쓰지 않는 것은 아니다. 다만 상태교환·제거 NLL은4.692624/3.710206,정상 대비3.048818/2.066400으로 이 역시 완전한 의미 보존 증거는 아니다.

모든 chrF/F1/편집·확신 점수는 같은 관측 dev의 사후 지표이며 calibrated 의미 성공률이 아니다. 직접 읽은 판단도 비맹검16개 소표본이다. 원래 관계 예측 성공은 보존했고 이 작은 자료·초기학습으로 전체 FDT나 퍼지 방식이 실패했다고 단정하지 않는다. 그렇지만 이 prefix목표를 같은 자료에서 자동 반복해 성과가 난다고 주장하지도 않는다. 다음은 목표·구조를 고정한 채 문서 다양성을 확대하는 학습이다.

## 실제 실행·보존

- V18:planf02b2e8e023337151e69d93d942ed4c5ca0e65693b6727b21545630772d95613/emergency642,939,057B SHAe64642280fa53562d796ae34055e0c46bc69a57f2ad43dbcb0cee1de88f8936d/algorithm75701cf038f95088abea169da34a9a67507d20816387f04dca897ec3c4c94a3d. 0update/3calls/max8500budgetCLOSED,원문실패보존.
- V19:plan d970339cf54fdb32cb31f4c4a288e5fb1268876c57ac5159fbed43a0bb1bd2f4. 새900초/reserve90/max8500calls/output6GiB에서 실제169.976초/encode2084+decoder5492+denoise0=7,576calls,COMPLETE. 예산CLOSED,미사용924call·시간재사용금지. 최종643,342,833B SHA0942bc1017eed24c90432f9be5d0a93939d17feb5b32dd5c96fe401b3f3c3679/algorithm5063771d8d35cb1fa347d5b80125edbf7823aeb0338bbf9c007aa67fad503ac8.
- 全63Adamstep23246/scaler32768tracker1992/모든CPU·CUDA·Python·NumPy RNG/전체nestedancestry·실패·history·order·cursor·노출보존. Source-import component1f6f7fe3cd057a5af73ddc7b7910222f6b093e0c83a233dc04785438abe05c0e 동일. 모든model·Adam floating tensor 유한/동결 reasoning·prefix·basis 불변 확인. Completeimport0/mid1024/final2048를 삭제·rotation없이 보존했다.
- ONE owner34592→31404생성17:42:30.326/.338KST,worker11948→25536생성17:42:39.455/.470. ROOT17:42:47생존·parent·command 확인. Worker-ownerexit0/COMPLETE/allownedproxygone,17:45:47nativePython0/Cfree49,035,304,960B확인 후 분석했다. Sleepguard는 종료시 해제됐다.
- Root가 전체 finalSHA/safeload/digest/source-import model·Adam·scaler·모든RNG/동결tensor/finite/63Adam/2epochcursor-order/두loss의혼합·mask수·노출·호출을 CPU재구성했다. 전체23,207기호×3조건 저장logits의 NLL/argmax는 worker와 일치했다. 추가forward0,동일16문서의유사도·직접판독·확신계산만 수행했다.

근거는 V18`root_stop_review.json`,V19`root_result.json`/`root_free_similarity.json`/`root_all16_free_reading.md`/nativeJSON와 `scripts/review_owt_prefix_reconstruction_v19_cpu.py`다. 모든과거source/core/typedpartial9/a877/weightsAdamRNG/raw/실패/immutablecorpus를 보존했다. 범용LM·독립추론·NER·최종완성이나자동승격을 주장하지 않는다.
