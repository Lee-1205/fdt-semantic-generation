# 최신 사람 확정 — 3메모리를 보존하는 언어↔의미 Codec부터

현재 사람 지시의 의미는 **언어 인코더·디코더를 안정화한 다음**, 그 입출력 형식 위에서 입력 의미를 목표 답변 의미로 디노이징하도록 학습하는 것이다. 지금 디노이징하지 않는 것이 맞다. 메모리 기반 추론을 일반LM의 언어생성과 뒤섞거나, 복원 성과를 위해 핵심 메모리를 없애는 방향은 허용되지 않는다.

우선 지킬 연결은 다음과 같다.

`입력 → 인코더 → K_P·K_C·K_E와 입력 의미 → 입력 의미를 바꾸지 않는 Codec Commit → 디코더 → 동일 입력 복원`

이 안정화 뒤에만 가운데의 의미 변형기를 학습한다.

`입력 의미 + 동일3메모리 snapshot → 의미관계 연산/디노이징 → 답변 의미 Commit → 같은 디코더`

추론·디노이징 중 자연어 토큰을 내지 않는다. 현재 Codec 단계에서 refiner/time/velocity/관계전이 parameter를 학습하거나 denoise호출을 추가하지 않는다. K_P는학습된 기본/지속 퍼지 기억,K_C는입력별 문맥 퍼지 기억,K_E는원문/식별자/숫자/근거의 정확한 identity를 보존한다. Fuzzy membership은 truth확률이나 exactidentity가 아니다.

## 원래 설계와 현 구현의 차이 — 확인된 사실

읽은 원본은 `archive/handoff/sources/FDT_vNext_Formal_Architecture_Spec_KR.md`의E01/E09–20/E24/E50–55,원래handover,현재 `docs/FDT_CODE_AUDIT.md`,기존 `work/fdt_prompt_objective/src/fdt_rlm/lexical_pointer.py`,현재 `src/fdt_local/model_masked.py`와 `research/owt_fuzzy_reconstruction_v12.py`다. 기존 source/archive/core는 수정하지 않았다.

| 계약 | Legacy V12–V31 natural-language AE의 실제 상태 |
|---|---|
| K_P 기본 퍼지 | anchors/prior와μ/ρ/gamma읽기는 있음. 인증된장기N/D·Writer전체는 없음 |
| K_C 문장별 퍼지 | 문장평균value 하나를 모든 활성앵커에 공유하는 단순화. 앵커별 서로 다른 token/section값 및long/recent N/D전체를 구현하지 않음 |
| K_E Exact | local102M AE에는ExactStore/선택기/copypermit 경로가 없음 |
| 미래 메모리 기반 디노이징 | originaltypedcore의관계/refinerweights는 보존됐지만 현재AE호출0·동결. 지금단계에서는 의도적으로맞음 |
| Commit/Realizer접근권한 | legacyAE는extracted2views만 받는decoder. vNext의3메모리snapshot/immutabletypedCommit/permit전체는 아님 |

과거부터102M adapter감사에는 ExactStore 등 미구현이 명시돼 있었다. 새 최근 실행에서3메모리를 전부 연결한 뒤 제거한 것으로 설명하지 않는다. 기존 ‘원래퍼지유지’는 μ/ρ/anchors/prior/value/condition재사용 범위이며 **전체3메모리구조 보존을 확인한 표현으로는 과도했다**. 새분기는 이 간극을 분명히 보완해야 한다. V31의64/64정확복원은 legacy경로의훈련표현용량증거로남기고3메모리검증으로소급승격하지 않는다.

## Exact의 현재·이후 역할

- 지금은 전체문장을Exact에서자동조회·그대로운송해 재복원점수만 올리는 경로를 사용하지 않는다. 여러조각 permit으로원문전체를덮는우회도 차단한다. Exact만으로정답문장전체를copy한결과를 의미Codec성공으로세지않는다.
- 원문/암호/숫자/식별자/긴문구의byte정체성과sourceID/version/span/hash보존은필요하다. 퍼지공간이비슷한뜻의다른단어를선택해도 정확문자열은구별돼야한다. 단편운송·주소정합을 실제언어복원과따로측정한다.
- **추후에는 전체원문을그대로꺼내야하는요청도지원한다.** 전체복사기능을영구삭제하지않는다. 그때는의미처리단계에서선택된source/span을Commit의명시적permit으로확정한뒤운송한다. 현재Codec학습에서는그정책을켜지않는다.
- 지금은의미변형/답변의도처리가없으므로copy요청이자동으로이해됐다고주장하지않는다. CPU fixture에source/span을명시한검사는기계적운송계약이지신경선택성공이아니다. 실제selector가학습되지않았다는상태를유지한다.
- 디코더는live K_P/K_C/K_E나원문H를직접조회하지않고 Commit의의미feature·허가된Exact조각만받는다. 원문bytes의snapshot복제와허가를통과한copy는범용rawcontext우회와구분한다. 모든출력의CopySpan기여·범위·source선택은별도기록해야한다.

## 새 CPU 기준 계약 — 구현됐지만 신경모델 연결 전

`research/three_memory_contract_v32.py`/`tests/test_three_memory_contract_v32.py`,artifact `research/artifacts/three_memory_contract_20261029_v32_cpu`를 만들었다. 기존 파일/학습 checkpoint는 수정하지 않았다.

1. Persistent prior와Contextual bank를분리했다. 비배타μ와정규화ρ를구별하고,문맥빈mass는0으로유지해prior로채우지않는다. 인증된지속Writer는이reference에없고해당사용은미구현오류로남긴다.
2. 문장별long/recent N/D를float64로갱신하는미분가능기준연산을만들었다. 이전prefix상태를바꾸지않는functional쓰기·문서source/version·validmask를검사한다. Padding은recent기억을노화시키지않는다. 서로다른앵커값을같은평균value로강제하지않는다.
3. ExactSource는원문UTF8bytes와source/version/hash를그대로보존한다. Unicode정규화나숫자재표기를하지않는다. 조각경계와sourcebinding을검사한다.
4. Codec기본정책에서전체source복사와조각타일링우회를차단했다. 미래post-semantic-Commit의명시적전체span허용정책은별도로남겼다. 이후기능의CPU기계적fixture는현재모델의copy의도학습이아니다.
5. 세메모리bundle의epoch/source정합과inference용byte-sealedCommit을만들었다. reader에독립tensor복사본을줘writablealias를막고realizer에는livebank전달을거부한다. 이seal은inference용으로detach하므로 **end-to-end훈련용미분Commit wrapper가완료됐다는주장은하지않는다**.
6. Codec단계에서denoise호출과학습가능reasoning/refiner/관계parameter를거부하는guard를추가했다.

확장된focusedCPU6검사PASS/1.38초다. 앞선5검사기록과추가padding/타일링/phase검사기록은덮어쓰지않고남겼다. ManifestSHA `d2c370bbad21a45890d67642c97003a092adaf13bfe2a457046574f2b3833c12`. Tiny tensor미분/운송fixture이며 실제모델construction-forward0/CUDA학습0이다. **현재학습모델에3메모리가이미연결됐다거나복원성능이개선됐다는증거가아니다.**

## 바로 다음 구현 목적

현재legacyAE를무조건확대학습하는계획보다이3메모리정합복구를우선한다. 다음에는보존된언어encoder/decoder가중치를명시적새분기로가져오고,입력별서로다른token/sectionvalue에서 K_C를구성하며 K_P와K_E를분리한실제신경encoder인터페이스를연결한다. Exact의학습된주소/binding·Commit표현을구분하고,디코더입력은Commit으로제한한다. 새tensor/module이필요하면이름/수/초기값/기존Adamownership/모든RNG를새계약으로보존하며strict재개라하지않는다. 큰vNextbackbone/인증Writer/근거증명전체를이번작업완료로부르지않는다.

훈련gradient가살아있는memory/Commit어댑터와inference봉인을분리하고,사전정답span·role·출력길이·원문전체복사로성공을만들지않는다. 암호/정확문자열의기계적운송과그내용을어디서골랐는지,일반문장내용복원을나눠기록한다. Exact를끄면임의암호까지항상원문일치해야한다는모순된요구를두지않는다.

코드·소스·자료SHA/필요CPUdelta/전체모델과각optimizer/scaler/모든RNG/phase-cursor-order·평가참조복구/실제preflight·독립OSvalidate/새목적·고정예산·output/native0/resources를ROOT가확인한뒤새한정언어Codec학습만한다. 디노이징은현재계속0이다. 현재Cfree약39.47GB는strict40GiB시작기준미달이라새GPU전에별도저장여유도해결해야한다. 원본/실패/fullAdamRNG/raw삭제·보호선인하·olddeadline연장·unusedbudget재사용은금지한다.

ROOTONLY/noagents/하드웨어재승인질문없음/timer30분prompt재개. E2final/registry/Browser·마우스/keepalive/유료cloud/외부게시/사용자원본삭제/os.kill(pid,0)금지. 새로운진행·수정·불확실성은PROJECT_STATE첫CURRENT에남긴다.
