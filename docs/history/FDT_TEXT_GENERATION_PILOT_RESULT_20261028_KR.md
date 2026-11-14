# 첫 VP 문장 생성 pilot 결과

새 생성 분기는 91,633,920 parameters/학습27,720,192개다. 원래 typed102,237,197 모델을 그대로 문장 생성기로 평가한 것이 아니다. 256attempt/256update/AMPskip0, 실제2048고유행, prompt116,275UTF8bytes/target40,603bytes를 읽었다. LM BPE token 수로 환산하지 않는다.

Worker36.046초/owner는07:34:46–07:35:31KST, 286encode+736denoise=1022호출로 새900초/1300호출 안에서 COMPLETE였다. Root가 실제native7712→16072→34336→41324의 생성/parent/command를 관측했고 supervisor exit0/모든owned종료·freshPython0를 확인했다. PlanSHA24d233db295378b253d60f67fd50468070ab10b2a058ea4ecdd3c4bd6d72fcb3. 미사용 시간·278call은 닫혔고 재사용하지 않는다.

| 패널 | baseline→final 문장 exact | 최종 유효UTF8 | 최종EOS | byte정확도(EOS포함) |
|---|---:|---:|---:|---:|
| train candidate probe | 0/16→0/16 | 8/16 | 16/16 | 42.90% |
| 미관측 entity pair+language | 0/32→0/32 | 17/32 | 32/32 | 42.13% |
| 미학습 prompt wording | 0/32→0/32 | 15/32 | 31/32 | 43.15% |

실제 생성은 `Itiss unlnown.a  ..` 같은 문자 혼합과 깨진 한글이었다. EOS와 일부 byte 패턴을 배웠지만 **문장 생성 성공은 아니다**. 원문 shuffle시 dev byte정확도는22.16%/21.57%로 낮아졌으나, 언어/문장 길이까지 함께 바뀌는 통제라 정교한 의미 이해의 독립 증명은 아니다. 모든 raw token IDs/hex/UTF8오류를 보존했다.

Root가240raw문장/baseline·final·shuffle의EOS/정확/UTF8/byte수를 CPU에서 재구성했고0/64/128/192/256의5fullstate SHA·safeload·전체재귀digest·동결language를 검증했다. Root 추가 모델호출0. 최종589,023,727B SHA0e3947eea37d5141ef299044c7c24369aeb79a18048f283b526cb29c82ce2163, algorithmSHAa41aafad5d4fbe2b188f63743645aac4c7f3c8f0c9d87cd5ab0a10f541223015. resultSHA09862b27991aafa39f8067ffeb88eb2e13b03b6b05ba5c2e4d8c7670e1c991b3. 원래partial9fullstate/core/source/이전실패/AdamRNG/E2final불변이다.

다음 구체적 가설은 byte별독립rounding이 문맥·UTF8 공동의존을 회복하기 어렵다는 것이다. 조건 인식 부족도 함께 남아 있으므로 단일원인으로 확정하지 않는다. [CoDAR 원 논문](https://arxiv.org/abs/2603.02547)의 contextual autoregressive discretizer 방향을 참고해 **확산제안은 동결·보존하고 별도 신경망 문맥 디코더를 학습하는 V2**를 준비한다. 논문 재현이나 동등 성능 주장은 아니다.
