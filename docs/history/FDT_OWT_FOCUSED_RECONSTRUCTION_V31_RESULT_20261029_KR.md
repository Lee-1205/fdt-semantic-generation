# V31 — 집중 학습64구간의 원문 복원4→64, 관측 개발 성능 퇴행

동일한 퍼지 입력 의미→언어 복원 분기에서 기존 train64문서의 짧은 구간을 집중 학습했다. 원문에서 추출한 상태+BOS+자기 prefix만으로 복원한64구간이 최종 **64/64 원문 UTF8bytes·EOS exact**였다. ROOT는 학습 전64출력·학습 후64출력·별도 개발16출력을 모두 직접 읽었다. 이름·숫자·날짜·부정·문장 순서와 원문의 오타/불완전 문법까지 같은 구간을 복원했다. 자동 원문 복사·Exact pointer·정답 길이는 이 분기에 없다.

이는 현 경로가 적어도 이64훈련구간을 학습해 담고 복원할 수 있다는 증거다. **범용 의미 추출·전체3메모리구조 보존·독립 추론·완벽한 언어 복원 장치 완성의 증거는 아니다.** 특히 아래의 새 구조 점검으로 이 legacy분기에 ExactStore가 미연결이고 문맥 메모리가 단순화됐음을 확인했다. 과거의 ‘원래 fuzzy 유지’ 표현은 기존 μ/ρ/anchors/prior/value/condition 재사용 범위이며 전체 vNext3메모리 구현을 뜻하지 않는다.

## 실제 학습·종료

- Run `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261029_v31` ONE COMPLETE. 2048attempt2048update/AMPskip0/261.223초. Owner11620→36528,worker8808→22104의 생성/parent/command/생존과 독립identity를 ROOT가 관측했고13:31:18KST에 모든owned/proxy종료·freshPython0부터 확인한 뒤 분석했다.
- 신규 encode2132+decoder7474+denoise0=9606calls/max12000. 새900초/reserve90/output6GiB 예산은 CLOSED이며 미사용2394calls·시간은 재사용하지 않는다.
- 같은111485711total/38309762trainable/last4languageblocks/퍼지·decoder·p50·추론 경로를 유지했다. 관계 전이·디노이징은 동결·호출0이다. 실제 sourceV30의 model/81populatedAdam3groups/scaler/allRNG/frozen/전체ancestry를 exact로 가져왔고 새 parameter·moment·추가 unfreeze는 없다.
- 새로운4096TRAIN문서 절반에서 문서SHA순 첫64개와 각 가장 짧은 기존span1개를 선정했다. 선택은 출력에 의존하지 않았다. 물리적64구간/4304targetUTF8bytes를256finiteepochs,16384행/1101824targetUTF8byte노출로 사용했다. 신규독립문서16384개가 아니다. 원래 broadtrain32768/dev256과 모든 원문을 보존했고 held→train0,다운로드0,E2final/registry접근0이다.
- 필요한CPU5검사49.55초, actualworkerCPUpreflight/독립OSvalidate 통과. 저장 여유는 CLOSED구생성분기V1–V6의29checkpoint에 대한 새 제한LZX패스로 회복했다:100.282초/할당1225989127B절약/전후논리SHA·길이 전부동일/삭제0·이동0/전체compact종료. V9/typed/최신AE27–30/원문/raw는 제외했다. 전체 Cfree증가에는 다른OS디스크변동도 포함돼 전부 압축 성과로 세지 않는다. Artifact `research/artifacts/owt_checkpoint_lossless_space_20261029_v4`. 시작40GiB보호선을 낮추지 않았다.

## 같은64훈련구간의 복원

| 항목 | V30 source baseline | V31 final |
|---|---:|---:|
| 상태만 받는 전체구간 exact |4/64|64/64|
| UTF8/EOS |64/64|64/64|
| chrF |43.1680|100|
| word/content F1 |39.6664/27.0792%|100/100%|
| 처음부터 맞은 UTF8bytes 평균 |18.4219|67.25|
| teacher NLL |.261359|.0000856|
| teacher byte정확도 |92.2619%|100%|

‘2008년에 범죄가 일어났고 당시17세’의17이 baseline10에서 정확히 복원됐고, FDA의 `not rare`, `Some definitions require movement, and some do not`, `20-year veteran ... does not have an agenda`, 인명·지명·날짜·UTF8결합문자 등을 원문대로 유지했다. 원문 `youshould` 같은 오타도 그대로이므로 문법 교정·새 사실 생성의 성과가 아니다.64번 반복 기억이 포함된 훈련 복원이다.

최종 train64 teacher의 swapNLL5.683545/zero3.800849는 정상.0000856보다 높았다. 적어도 입력별 상태에 의존하는 조건부 복원이며, 모든 입력에 같은 문장을 출력한 결과는 아니다. 다만 이 통제만으로 추상적 의미 표상·유일한 latent정답·학습된 원문 복사 의도 판단까지 입증하지 않는다.

## 별도 관측 개발 문서는 악화

원래 dev256행23207symbols의 teacher NLL1.016412→2.431911,byte정확도71.5689→53.4020%,정답EOS위치253→252/256으로 퇴행했다. 별도16개 상태 기반 원문 복원은 exact0/16/전체충실0/확실부분명제0이며 UTF8/EOS16/16이었다. 일부 출력에 집중훈련구간의 어휘가 섞였다. 기존 사건·개체·관계·부정은 유지되지 않았다.

따라서 V31은 작은집합 **복원 가능성·집중학습 분기**로 보존하며 broad언어모델에 자동승격하지 않는다. V30과 모든 이전 가중치/AdamRNG도 보존했다. 좁은 집중학습 뒤 관측 퇴행을 확인한 것이며 복구불가능한 실패나 FDT원리의 한계를 단언하는 결과가 아니다.

## 저장·독립 재구성

PlanSHA `6e0be6c55d882a3ab2a00fa764b03a11333bda10d4cbfa50c694cf4b137f2375`, start1793247977.482099/deadline1793248877.482099/cutoff1793248787.482099. Source와 같은64의 baseline평가 후 import0을 저장했으므로 import0에는 **새 baseline1423calls(encode24/decoder1399)와 그 파일SHA**가 들어 있다. 이 호출을 source학습이나 실제optimizerupdate로 세지 않는다. Model/Adam/scaler/모든RNG는 baseline 뒤에도 source와 exact였다. Source 원래 호출 장부는 ancestry에 그대로 보존됐다.

- Import0 774685765B SHA `d6375f373944dc0bae080181e56b7a861912e94129adf3b90424bd4fc470be55`, algorithm `1a3c8d0b6b71fe901f956cb958632a296ede6f5bce75091432f001837106e6fa`.
- Mid1024 774877509B SHA `4555a206db28676465cbe974da4cff19f18697e510e3d984f681e1cc5ff513ed`.
- Final2048 775070341B SHA `1d02d0d52c44683a582e70273e3e951e84384ecb406700470af620e21e8c252f`, algorithm `66316e12eeeb9ae3d6c3515f5983deca0363f9d54f2166c1bc6210a0c604bf19`.

ROOT saved-only reviewer `scripts/review_owt_focused_reconstruction_v31_cpu.py`가3fullstates SHA/safeload/재귀digest,전체81Adam(최종기존63step102335·추가18step14330),scaler65536tracker269,모든저장RNG·history·order·cursor·baseline/final평가참조·중첩ancestry·frozenreasoning/basis·finite를 검증했다. 추가modelconstruction/forward0,실제CUDA복구재실행0이다. 새train교사4368symbols×2×3조건과held23207×3조건의 저장logits/NLL/argmax·EOS,모든144원시출력bytes·EOS·실제batch마지막EOS 기반calls를 재구성했다.

AE누적102372attempt102335update,장부317061calls에 별도V23readonly1936을 합치면318997이다. 과거36AMPskip+V18실패attempt1은 그대로다. 원시판독 `root_all144_before_after_and_held_reading.md`,수치 `root_result.json`/`root_reconstruction_similarity.json`을 보존했다.

## 최신 사람 지시가 후속 계획을 변경

사람이 인코더·디코더 복원 안정성을 먼저 확보하고 **PersistentFuzzy/ContextualFuzzy/Exact 세메모리**를 반드시 유지하라고 재확정했다. 지금 디노이징을 하지 않는 것은 올바른 단계 분리다. 의미 변형/답변 의미 생성은 이후이며 현재 추가하지 않는다.

현재 legacy102M natural-language AE는 이3메모리 전체를 구현하지 않았다. Exact가 빠져 있고 문장별 메모리는 한 문장 평균value를 여러 앵커에 공유한다. 따라서 단순64→256확대나 pure-AE SGD자동추가보다 이 인터페이스를 먼저 맞추기로 했다. 새 CPU기준모듈 `research/three_memory_contract_v32.py`와6검사만 준비했으며 **실제 신경encoder/decoder연결·Exact선택기학습·새GPU는 아직 없다**. 상세계약은 `outputs/FDT_THREE_MEMORY_CODEC_REQUIREMENTS_20261029_KR.md`다. 원래 vNext의 큰모델·인증Writer전체가 구현됐다고 보고하지 않는다.
