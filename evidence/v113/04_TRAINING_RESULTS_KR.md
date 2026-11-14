2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.

# 04. V113 학습과 저장 검증

**VERIFIED** 268attempt268update/AMPskip0/2144행/37088UTF8byte, 학습thought36448+D2144/평가thought3680+D230, 전체42502calls/worker657.889초다. 이전V112는40358calls/642.193초였다. V113목적에서고정D forward를썼다는점과D가중치를학습했다는것을혼동하지않는다.

|V113 checkpoint update|bytes|SHA256|
|---|---:|---|
|0|38916128|`a525da786c7cc104cbd6148ae7627b702b76cd96e02360fced375f5c6b8125c8`|
|134|115615485|`91c0473142612de7018d313472906ee098fd205b3a71bddead5c2b40974779a8`|
|268|115653629|`a01c4f1170ab8a2428a61ac239858436db67229293a2056dfc417fc9bbe386e3`|

3fullstates의파일SHA/크기/weights-onlysafeload/재귀digest, 초기θ전부부모일치와빈Adam,최종64θtensor변경/Adamstep268/scaler1024tracker268/gradient정리/모든134행16회/순서·노출/RNG1216draws정확재생을검증했다. 고정codec전체modelSHA는 `42a2032db3137ffb9f876585cf26979854d2555e19639fa5c2f1e8d61201b96a`, 전후같다. frozen endpoint digest도3상태에서같다.

|학습8update평균|처음|마지막|
|---|---:|---:|
|bodyMSE|0.781003|0.567552|
|contentCE|1.113463|0.102771|
|combinedloss|1.676814|0.844941|
|짝차이MSE|0.886087|0.462378|

추가NN/optimizer0인독립사후검토에서460전후raw의파일SHA/UTF8·EOSdecode/NLL/body-tailMSE와모든정답byte/EOS위치full258logit을검증했다. 긴생성tail전체258는미저장이다. 저장goldLP의gather−logsumexp는460행전부bitwise재현됐고FP64 log_softmax기준NLL최대오차2.5814e-6였다. 초기사후검사의고정2e-6허용오차실패는FP32연산형태차이여서실제식bitwise+FP64오차경계검사로교정했다. 데이터·점수·모델을바꾼것이아니다.

최종checkpoint는학습DONE후/최종평가전상태이며230최종평가참조는별도assessment/result에보존했다. 모든새trial은CLOSED다.
