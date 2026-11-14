# 현재 FDT 연구 구조와 성과 범위

언어입력→고정E→입력잠재표현, 입력표현과K_P기본퍼지prior/K_C맥락snapshot/현재NULL인K_E Exact를조건으로독립잡음에서16step디노이징→확정답변잠재표현→고정D언어화의구조다. 현재과제는짧은A–H관계Transformation이며Composition·장문사고·범용LM은아직아니다.

최신사고모듈hidden520/출력512/9576426parameters, 고정codec123497080parameters+전체buffer다. 옛typed102237197adapter/426M모델/V7 ARdecoder사양을최신사고모델성적으로합산하지않는다. E(Y)는훈련loss표적이며추론의초기값·정답조건이아니다.

V112는실제역전짝의공통독립잡음과차이MSE를추가했다. 같은body mask에서행별오차평균은 MSE(mean error)+.25MSE(error difference)이며 .25차이항추가로차이성분을2배강조한다. V113은고정DcontentbyteCE를추가해gradient만사고가중치로보냈다. D학습/새AR/Exactcopy/정답길이crop은하지않았다.

V113 TRAIN에서는최소35/80의읽을수있는긍정관계core와8/32역전짝bothcorrect를관측했다. 전체긍정문장exact0/80이며뒤잡음이남는다. DEV긍정0/48, NLL악화이므로일반화는미해결이다. 다음주목적은TRAIN전용개체조합·질문표현·배치범위를넓히는자료분기이며아직실행되지않았다.

메모리경로의존재는개별메모리의인과기여증거가아니다. source가복원가능하다는것도독립추상의미공간의증거가아니다. 실제기록의 VERIFIED/OBSERVED/INFERRED/PROPOSED를구분한다. 자세한계약·평가·원시출력예시는 evidence/v113의01~07보고서를참조한다.
