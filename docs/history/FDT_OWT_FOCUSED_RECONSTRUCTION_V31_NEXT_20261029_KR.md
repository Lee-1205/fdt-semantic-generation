# V31 다음 목적 — 같은 퍼지 의미 경로의 작은 훈련 집합 집중 복원

실제 최신 source는 V30 `model/checkpoints/state_8192.pt`, SHA96480189836964c58153d7d002a27b25b279057af1555cd38950c754d724e504/algorithm1ce01437db667d46adebca638fc295e07d793d9268d8158e76094f4c3112ec81/planfce72e7a7e91773ae3ce8bdfbfd45cbd4f5c194e11287eca32bcf93670a12fbe다. Local8192attempt8188updatecursor65536/global100324attempt100287update/81populatedAdam3groups/기존63step100287·추가18step12282/scaler32768tracker221/전체 RNG/history/finiteorders/중첩ancestry와 V29최초 층 해제 provenance/외부readonly1936call을 보존한다. V30남은1482calls·시간 및 모든 이전 예산은 CLOSED다.

두 번의 실제 학습으로 총12,282갱신을 더 했다. Dev정답-prefix 정확도는71.57%지만 NLL은1.0164로 악화됐고 전체 충실 복원은 여전히 dev0/16·train0/16이다. Train의 `일부 사람이 위독한 상태`라는 부분 서술1건은 실제로 보존됐다고 ROOT가 인정했다. 이 긍정 결과와 퇴행을 함께 남기며 FDT전체나 특정 모듈의 실패를 단언하지 않는다.

같은 broad학습/검증을 무한 반복하기보다 원래 퍼지 입력 의미 경로가 **작고 선언된 훈련 문장 집합을 집중 학습해 재복원할 수 있는지** 확인하는 다음 목적을 정했다. 새로운 의미 정답 벡터를 가정하지 않는다. 성공해도 train문장 memorization/표현 용량 증거이며 일반 의미 추출·추론·blind LM 완성이 아니다. 실패해도 정해진 학습량과 optimization에 제한된 결과다. 다른 광범위 baseline/audit/gradienttrace를 자동 선행조건으로 더 붙이지 않는다.

이미 ROOT가 `scripts/prepare_owt_focused_reconstruction_v31_cpu.py`로 새 immutable CPU artifact `research/artifacts/owt_focused_reconstruction_20261029_v31_cpu`를 만들었다. 다시 만들거나 덮어쓰지 않는다.

- 기존 V23final의 보수적으로 선정된 **새 train4096문서 절반 안에서만** normalized문서SHA오름차순 첫64문서를 골랐다. 각 문서의 기존4span 중 UTF8bytes가 가장 짧은1개를 택하고 동률은spanSHA로 정했다. 모델 출력을 보고 잘 맞는 행을 고른 것이 아니다. Held문서·dev행 편입0/새 다운로드0이다.
- Feature는 id/prompt만, 정답byte/EOS는 targets파일에 따로 있다. 실제 source row index/문서ID·normalizedSHA/세파일SHA/정확한 finiteorder를 manifest에 보존했다. 원문/자료/원래 전체 broadtrain분할은 그대로다.
- 물리적으로64기존 문서/64span/4,304targetUTF8bytes다. 제안256finiteepochs(seed391…646)/batch8/2048attempt는16,384rowselections/1,101,824targetUTF8byte노출이며 새로운16,384문서나 독립 graph가 아니다. 매epoch64행을 정확히 한 번씩 선택한다.
- 이는 **CPU자료만**이고 실제 model construction/forward/optimizer/CUDA0이다. 새 fullstate driver·실행plan·실제 workerpreflight·독립OSvalidate·seal·GPU는 아직 없다. CPU자료 완성을 능력 개선이라고 말하지 않는다.

후속 구현은 SAME111485711total/38309762trainable/last4languageblocks/원래μ·ρ·anchors-value-prior-condition/2views-onlydecoder/clean.5+corrupt.5 p50/같은BOSselfprefix추론을 유지한다. 출력 목적 의미 디노이징/이유추론부는 동결한다. Runtime에 정답vector/span/길이/roles/정답prefix/template/retrieval/rawcontext우회를 제공하지 않는다. Source의81Adam·세group과 RNG를 그대로 import하고 새 moment나 parameter를 만들지 않는다. Phase localstep/cursor/order/LR/callbudget만 별도 한정한다.

예상 목적은2048attempt/max2048success/skip도노출·cursor소비/되감기·자동epoch추가금지, import0/mid1024/final2048+STOP 전체 state보존이다. Baseline/final은 이 새 train64의 teachercontrols와 **같은 입력 의미에서 시작한64행 복원**을 나눠 비교한다. 새로운64행 비교이므로 source reference가 필요하되 과거 dev의 광범위 초기baseline은 재실행하지 않는다. 유지 평가로 원래 dev256teachercontrols와 고정dev16normal재복원을 최종에만 두는 후보를 검토한다. 새 train성공과 held일반화를 섞지 않는다. ROOT가 실제 문장/정답을 읽어 주체·관계·부정·수량·시간을 판독한다.

현재 코드 비용 상한 후보는 train6144 +train64 baseline/finalteacher64 +train64 baseline/finalBOS복원4160max +기존held최종teacher128 +고정dev16normal복원520max =11016calls/max12000이다. 실제 새 driver 코드로 다시 확인 후 고정한다. NEW900초/reserve90/output6GiB/start40GiB/GPU12GiB 후보, olddeadline 연장·reset·unusedbudget재사용 금지다. 필요CPU delta/실제planschema/전체source3group import와 enabledscaler·모든RNG·finitecursor exact복구/실제workerpreflight/독립OSowner의 postPopencleanup/newSHA/output/native0/resources를 ROOT가 확인한 뒤 ONEhiddenlaunch만 한다. 하드웨어 재승인 질문은 하지 않는다.

V30후 Cfree약40.91GB는 strict시작40GiB=42,949,672,960B 미달이다. 기존 무손실 파일시스템 압축 이력과 결과를 먼저 보고 별도 좁은 저장 여유 목적을 정한다. 이미 압축한 CLOSEDV20–V26의21파일을 같은 예산으로 반복하지 않는다. 어떤 저장 변경도 최신source·원본·실패·fullAdamRNG·raw의 논리SHA/내용을 보존하고 삭제·이동·lossy직렬화·보호선 인하로 우회하지 않는다. 자원 충족 전 GPU를 시작하지 않는다. 권한 질문 대신 가능한 CPU 구현을 계속할 수 있다.

ROOTONLY/noagents/timerACTIVE30minexact재개. Allsource/core/typedpartial9/a877/Phase2/archive/useroriginals/weightsAdamRNG/raw/실패 보존. E2final/registry/Browser·마우스/keepalive/유료cloud/외부게시/useroriginal삭제/Windowsos.kill(pid,0)금지. 실행중sealedcode수정삭제/자동재시작 금지. Terminal뒤allowned/proxy종료부터 확인하고 증거·성과·한계·다음단계를 첫 CURRENT에 저장한다.
