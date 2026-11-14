# V65 문자 균형 언어 왕복 학습 결과

512/512 attempt/update, AMP skip 0, 229.899초, encode528+readout532=1060회로 완료했다. Exact0/copy0/denoise0이다.

이전 V64 최종 가중치를 그대로 warm start하고 fresh Adam/scaler로 학습했다. TRAIN 기반 문자 가중치는 대문자3.405832/소문자.738918/숫자4/기타1.382325이며 expected per-window byte scale1로 정규화했다. 기존 V56/V57의 TRAIN 균형 이력을 확인한 뒤 실제 TRAIN 기여를 계산했다. 제한 없는 반복 벌점이나 추론 시 문자열 치환은 추가하지 않았다.

기존 관측 DEV16: 바이트EOS 99.721269→99.782529%, NLL .011874291→.010847702, 오류182→142, 완전복원3/16 유지. 대문자52→26, 숫자10→4, 소문자26→30 퇴행, 기타94→82. 교정70위치/새오류30. loss와 더 넓은 TRAIN 노출이 함께 바뀌어 가중치 단독 효과로 보지 않는다.

TRAIN512행/505원문/단일 유한 epoch/3724282UTF8바이트, 39~16384B. 새 개발 평가를 반복하지 않고 동일 V64 최종 raw를 baseline으로 재사용했다. actualCPU preflight9.781초, 독립272pins 확인 후 ONE hidden 실행. owner13260→18308/worker9900→1284 모두 종료09:41:39/Python0 확인 후 독립4fullstates/30Adam512/scaler1024tracker512/모든RNG/cursor/order/노출/모든동결tensor/활성인코더와디코더변경/32raw 재구성을 통과했다.

최종635671623B SHA3351bae2358ed660effbb67c642650b3401b9142f6b9b09e19d823310f9cc7a4, algorithmbdc7409c0da8fa43fe8c314cab742e9c50c41ef611706897231883c55485dceb. GPU최대할당3441015808B. 새1200초/1100call/4GiB예산CLOSED,미사용40call재사용금지.

문자 클래스 재가중치의 일반 연구 맥락으로 [Class-Balanced Loss (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)를 확인했다. 논문은 effective-number 방식이며 이번 bounded inverse-sqrt 규칙과 동일하지 않다. 이 논문이 현재 가중치의 최적성이나 FDT 성능을 입증하지는 않는다. [Unlikelihood Training](https://arxiv.org/abs/1908.04319)은 반복 저감 연구 맥락으로 확인했으나 이번 동일입력 복원 목적에는 추가하지 않았다.

사람의 새 요청에 따라 별도 미학습 문장 V66 읽기전용 평가를 완료했다. [상세보고](FDT_UNTRAINED_RECONSTRUCTION_V66_DETAIL_20261112_KR.md)에 언어·길이별 수치와 실제 의미 손상을 기록했다.
