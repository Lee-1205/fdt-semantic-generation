# V28 다음 목적 — p50 고정 적응 구간 한 번, ramp 재시작 금지

최신 source는 `runs/owt_fuzzy_reconstruction_20261029_v27/model/checkpoints/state_8192.pt`, SHAbe85ab3b6bdd21b70ce9942d40a704a0a04a157245e3336446bde401f5da4648 / algorithm626462998312ab4634db9ecaff5eb5ec676cbb94ec17cc5cfe87eba283b52fb1다. ActualV27schema/local8192attempt8190updatecursor65536/global79844attempt79817update/63Adam/scaler65536tracker842/모든 RNG/history/order/전체 ancestry/외부readonly1936ledger/source15%objective와新curriculum 계약을 보존한다. V27미사용1,501calls·시간과 모든 이전 예산은 CLOSED다.

V27은 p.15→.50 첫4096attempt와 p.50 고정4096attempt였다. Teacher70.6339%/NLL.987024는 소폭 개선됐지만 BOS상태 복원은 dev/train전체0/16, 여러 train유사도와25-year-old 부분이 퇴행했다. Stronger mask 성공·최적비율·자동승격으로 부르지 않는다. p.50 고정 구간의 후반 onlineclean/corruptloss가 아직 낮아진다는 저장 증거를 바탕으로 **추가 고정50% finite구간을 한 번만** 관측한다. 같은 출발점의15%control이 없어 고립된 인과 실험은 아니다.

변경은 새 allocation의 train past-byte masking probability=.50 고정뿐이다. Ramp를15%로 되감거나.50을 더 높이지 않는다. CleanCE.5+corruptCE.5/원래 퍼지 μ·ρ·anchors/value-prior-condition/2views-onlywriter/원문 context우회 없음/BOS자체prefixinference/별도 reasoning동결을 유지한다. FullB×129 uniform draw수·순서와 BOS/EOS/PAD 보호·과거 byte만 손상 계약은 동일하다. Gold 역할·정답 의미 벡터·출력 template·추론 시 정답 prefix/길이를 추가하지 않는다.

새 module/engine은 oldV27의 끝난 curriculum을 실제 source 계약으로 검증하고 새 constant50% objective를 분리해 저장한다. 모든 source tensor·63Adam/scaler/RNG/ancestry 초기 import는 정확히 같아야 한다. History에는 actualp=.50과 eligible/masked 수를 기록하고 새 fullstate에 constant objective를 저장해 정확 복구한다. 필요한 CPU delta만 source/schema·초기p50/고정 RNG·mask과거인과성·enabledscaler AMPskip/fullstate exact-next/마지막7행·OS post-Popen cleanup으로 검토한다. 변경하지 않은 broad gradient/resource guard와 초기 GPU baseline을 다시 반복하지 않는다.

후보는 같은 immutableV23final8192train문서32768행/원래64dev256unchanged/두 finite epoch8192attempt/batch8/65536행/targetUTF8 5,852,956bytes/3NNcalls perattempt다. Train24576+dev128+dev상태controls1552max+고정train520max=26776expected/max28000/새900초reserve90/output6GiB/start40GiB/GPU12GiB를 코드 비용 확인 뒤 고정한다. SourceV27외부readonly1936calls는 sourcecumulative240202와 별도로 계속 연결한다. Initial GPUbaseline은 V27저장평가로 대체하고 final같은 panel을 ROOT가 모두 읽는다.

이번 고정50% 후속에서도 whole재복원 개선이 확인되지 않으면 같은 목적의 추가 run을 무한 반복하거나 p를 자동 확대하지 않는다. Whole/부분 명제·행위/대상·인명/나이·부정/관계와 graded lexical지표, 이전에 맞은 부분의 손실을 분리해 다음 구체적 변경을 판단한다. 작은 학습과 관측dev로 전체 FDT/퍼지 실패나 extractor/decoder 단독 원인을 확정하지 않는다.

현재 V28 code/module/runner/owner/plan/seal/GPU는 **없다**. 필요한 CPU·actualworkerpreflight·독립OSvalidate·새SHA/output·native0·자원 확인 후 ROOT가 ONE 새 한정 학습만 한다. 준비를 GPU 실행/능력 개선이라고 하지 않는다. 하드웨어 재승인 질문 없음/ROOT ONLY/noagents. 학습 중 상세 분석 중단·독립OS감독,terminal 뒤 allowned/proxy 종료부터 확인한다. 모든 원본/core/typedpartial9/a877/Phase2/archive/weightsAdamRNG/raw/실패 보존. E2final/registry/browser/mouse/keepalive/유료cloud/외부게시/사용자원본삭제/Windowsos.kill(pid,0)/자동재시작/olddeadline연장·unusedbudget재사용/실행중sealedcode수정삭제 금지. Timer30분 prompt는 정확히 재개다.
