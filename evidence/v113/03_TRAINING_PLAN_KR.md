2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.

# 03. 실행한 유한 계획과 사전검사

**VERIFIED** 두run각각134TRAIN×16finiteepoch=2144행/37088정답UTF8byte, batch8/원자단위2행/268attempt, LR1e-4/weightdecay.01/clip1/같은cosine를사용했다. 58실제짝+18singleton을보존했고각행매epoch정확1회,1216독립noise draw다. 노출행을새독립자료로세지않는다. DEV편입0/E2final·registry접근0.

V112예산41000call/expected40358, V11343000/42502. 각각owner1200초/cut1080/reserve120/C전체1GiB. C실행하한50GiB/owner54GiB조기정리/시작101GiB+64MiB로종료100GiB보호. output3fullstates와STOP임시여유까지포함했다. 이전미사용예산재사용0/자동재시작0/상한확대0/실행중봉인코드변경0/체크포인트회전삭제0.

V112 focusedCPU7PASS,실제fullsizepreflight388calls/fixture·재생4optimizer실행, V113 focusedCPU3PASS/fullsize424calls/fixture·재생4실행이다. 각각populated Adam/scaler/ALL RNG/중간scaledgradient/MID와6행READY다음update정확,입력checkpoint불변/codec고정/이전TRAIN·DEV2개출력bitwise검증했다. 본연구536update와fixture8회는별도다. 실제V113고정D CE→θ gradient norm6.230322>0을확인했다. 이는한행진단이며최적계수의증거가아니다. CUDA복구검사라고하지않는다.

기존230baseline출력을재사용해반복baselineforward0이었다. 최종230행은동일row/ε/decoder조건으로실행했고V113훈련D2144call이추가됐다. D는학습가중치가아닌고정미분경로였다. 독립Windowsowner는Torch를불러오지않고모든postPopen예외·자연종료의verifiedowned cleanup을보장하는기존코드를유지했다.
