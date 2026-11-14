2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.

# 06. 고정 출력기 경계

**OBSERVED** 같은고정D는실제27종E(Y)직접입력에서정답prefix27/27, whole0/27,뒤1–12byte였다. V111실제검사와hash가같아재호출하지않았다. 따라서정답내용읽기능력과종료문제를구분한다.

V112에서조건구별은좋아졌지만 lmaleer 등혼합철자가남았다. V113에서고정D의contentCE를θ로역전파하자TRAIN 관계word정확3→74/80, 주체9→46/목적어29→63/80, 정상관계core34개가생겼다. 출력기자체는변하지않았다. 따라서생성잠재가고정D의결정경계에맞도록조정되는것이내용충실도에중요하다는관측이다. CE만의단독인과효과는별도통제하지않았다.

TRAIN noEOS5→2, lateEOS114/134, positivewholeexact0/80다. UTF8는모든raw가유효하지만고정된neuralUTF8결정trace의성질이며정상문장·의미정확성증거가아니다. 정답길이crop으로성적을올리지않았고EOS보조loss는이번에추가하지않았다.

새DEV에서문법이정상인데주체가틀리는사례는남은실패전체가EOS만의문제가아님을보여준다. 또한TRAINcore성공을고정E의추상의미완성/긴입력사고/범용LM으로승격하지않는다. 현재D는비AR Commit→MLP Byte/EOS, V7의context우회·ARprior반복진단을여기에자동적용하지않는다.

후속종료충실도는별도θ-onlyEOS실험후보이지만현재최우선은새사실·표현에서올바른개체·관계를생성하는조건일반화다. 메모리필수counterfactual episode와장문Composition은아직미실행이다.
