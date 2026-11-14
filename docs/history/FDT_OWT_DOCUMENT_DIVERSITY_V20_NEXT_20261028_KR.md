# 다음 한정 목적 — 같은 퍼지 추출·복원에서 문서 다양성 확대

최신 실제 source는 V19 `runs/owt_fuzzy_reconstruction_20261028_v19/model/checkpoints/state_2048.pt`, SHA0942bc1017eed24c90432f9be5d0a93939d17feb5b32dd5c96fe401b3f3c3679/algorithm5063771d8d35cb1fa347d5b80125edbf7823aeb0338bbf9c007aa67fad503ac8다. 全63Adamstep23246/scaler32768tracker1992/모든RNG/전체ancestry를 보존한다. V18FAILED0update와 V19COMPLETE2046update/2skip,모든이전예산은CLOSED다. V19미사용924call/시간·V18미사용8497call/2047attempt는 재사용하지 않는다.

V19의 train-only prefix 손상 목표는 오답확신.60936→.58650/NLL1.66597→1.64381로 조금 완화됐지만 자유chrF21.43→21.26/exact·직접의미복원0/16으로 개선을 입증하지 못했다. 목표를 더 바꾸거나 큰 벌점·anchor초기화·reasoning을 혼합하기보다, **같은 목표·구조에서 문서 다양성만 확대**하는 다음 한정 학습을 ROOT가 구체화한다. 원래의 문장→퍼지 입력 의미 추출→복원 FIRST/추론·출력 의미 디노이징 별도동결을 유지한다.

현재 고유 선택 문장은728,913UTF8bytes/2,048train문서로 작다. 다음 prospective pool은4,096train문서/16,384문장이다. 기존2,048train문서/8,192행을 정확한 prefix로 보존하고 새2,048train문서/8,192행만 붙인다. 기존64dev문서/256문장·정답·순서는 그대로다. 같은 OWTrevision79d93d786212f7344586290adb811d4ae6a1762c/normalizeddocSHAmod9 split/EOS선택문장경계/4hash선택48–128byte문장 조건을 유지한다.

먼저 V12/V13/V15의 저장 source_pages 미사용train문서를 사용하고 필요시 최대24MiB/180초의 추가rowsAPI fetch 안에서 채운다. 모든X-revision과원문SHA/exactdoc·문장 교차split중복0을 검증하고 새immutableartifact로 보존한다. 새원문/반복노출을 구분한다. 권리·원문출처/near-dup·이전pretraining겹침UNKNOWN의 한계는 유지한다. 사용자원본·과거held·E2final·registry를 열거나train에 넣지 않는다. 대규모전체다운로드·유료cloud·외부게시는 하지 않는다.

Prospective trial은2유한epoch/batch8/새4096attempt/32,768rowselections이다. TargetUTF8bytes는 실제 새자료에서 계산한다. 기존·새2,048doc 각각16,384행을 노출한다. Originalfuzzy/동일runtime/동일cleanCE.5+corruptprefixCE.5/.15mask/strictdeterminism/elementwiseCE:none_then_sum/V17persistentAMPguard를 그대로 두어 추가objective·정밀도·architecture변경을 혼합하지 않는다. 새source-V19schema importer에서全model/63Adam/scaler/allRNG/과거실패와누적23252attempt23246update를 정확히 가져오고 새finiteLR/order/clock/callallocation만 분리한다.

현재실제3calls/attempt이므로 training12288+최종3조건128+최종자유최대1552=13968expectedcalls다. V19실제169.976초/2048attempt를 기준으로 새4096trial은900초내에 충분히 들어갈 것으로 예상한다. 실제자료·코드비용을 확인해 새cap약14800/owner900secreserve90/output6GiB/start40GiB/GPU12GiB를 사전고정한다. Fullstatesimport0/mid2048/final4096+emergency를 새directory에 보존하며 기존파일은 삭제·교체·rotation하지 않는다. Cfree약49GB이므로 시작40GiB와임시 checkpoint+8GiB/상시20GiB보호를 실제값으로 다시 확인한다.

필요한CPUdelta는 실제V19schema전체state·RNG/mask난수복구/새자료의oldprefix-held불변/2finiteepoch순서와노출cap/전체postPopencleanup와실제workerpreflight·독립OSvalidate다. 이전대규모검사·GPUbaseline을 무한반복하지 않는다. 같은 inference/dev이면 V19저장결과를 기준으로 사용하고 ROOT의새SHA/clock/caps/resources 검토 뒤 ONE실행한다. 하드웨어재승인으로 멈추지 않는다. 준비를GPU성능으로 보고하지 않는다.

관찰은 teacherNLL/정확도·오답확신·freeexact/chrF/내용어·관계의직접판독·정상/교환/제거를 함께 기록한다. 같은 observeddev의 반복사용/설계선택/작은16문서 표본 한계를 밝힌다. 추출완전성·decoder단독원인·일반LM성공을 단정하지 않는다. V15의더좋은teacherNLL과 최신V19가중치를 모두 보존하고 자동승격하지 않는다. 현재 V20자료/runner/plan/seal/GPU는 **아직 없다**.

ROOTONLY/noagents/모든source·weightsfullAdamRNG·raw·실패보존/E2final-registry금지/소유OS감독/native중복0/terminal후모든proxy종료확인/실행중sealed불변/자동재시작·olddeadline·unusedbudget재사용금지를 유지한다. Timer30분prompt는 정확히 재개다.
