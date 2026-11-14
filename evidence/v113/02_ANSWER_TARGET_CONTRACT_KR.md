2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.

# 02. 수정 수식과 누수 경계

**VERIFIED** source E(X)+고정K_P/K_C+K_E NULL+독립ε → 실제16step Rθ → 확정 Zhat → 고정D이다. 정답문장·E(Y)·정답길이는추론조건이아니다. target은여전히해당행E(Y), FP32(1,35,512)=global2+local33이며body/tail mask는loss에만쓴다. labels는(34,16)의byte/EOS256/PAD257이다.

V112에서실제58역전짝에만공통독립ε를주고 .25*||(Zhat_a−Zhat_b)−(Zy_a−Zy_b)||²의body평균을추가했다. 같은body mask에서행별endpoint평균은 ||e_mean||²+.25||e_a−e_b||²이므로차이계수가.25→.5가된다. UNKNOWN26짝의target차이는0,18singleton은독립ε·차이loss0이다. 서로역전한답문이같은canonical사실을표현할수있음을보존한다.

V113 L=bodyMSE+.05tailMSE+.1VPaux+.25실제짝차이+.5contentbyteCE다. CE는실제target-freeZ를고정D에넣은뒤계산하며정답prefix로decoder를조건화하지않는다. EOSaux0/새AR0/template0/NN검색0/Exactcopy0. E/D가중치는동결되고고정D입력에대한미분만θ에전달된다. 따라서여러모델을공동학습한것이아니다. 생성중의언어피드백은없고훈련손실의피드백만있다.

추가CE는MSE에충분히반영되지않은개체·관계의D결정경계를감독하려는후보이다. 계수.25/.5가최적이라는증거는없다. V112의공통잡음+차이손실, V113의추가학습량/freshAdam 영향은각각의인과실험으로분리되지않았다. VP는16논리단계가아니라16노이즈복원step이다.
