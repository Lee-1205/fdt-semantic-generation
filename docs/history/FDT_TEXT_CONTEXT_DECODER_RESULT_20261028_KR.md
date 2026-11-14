# V2 문맥 디코더 결과 — 형식 회복, 의미 정렬 미완료

Root 직접512attempt/512update/AMPskip0, 59.504worker초, actual138encode+2208denoise+1397decoder=3743calls로 COMPLETE. Native39308→35248→41056→36696 생성07:42:58/.097·07:43:01.978/.993KST의 parent/command를 root가 관측했고 종료 후 모든owned/proxy 부재·owner exit0를 확인했다. 고정900초/5000call은 닫혔고 미사용분은 재사용하지 않는다.

원래91,633,920 VP모델을 동결하고9,202,946 decoder parameters만 학습했다. 원문 문맥과 확산 latent를 모두 읽는 hybrid이며 pure diffusion 생성은 아니다. 고정1024고유행을4개유한순서4096행으로 읽었고target80,632UTF8bytes를학습했다. 같은관측synthetic개발패널이다.

| 패널 | V2 초기→최종 exact | 정상UTF8/EOS | 의미동치 별도판정 | ‘불명’ 상수 기준선 |
|---|---:|---:|---:|---:|
| 기존train probe16(이번pool포함4개) | 0→5/16 | 16/16 | 6/16 | 5/16 |
| 미관측pair+language | 0→10/32 | 32/32 | 10/32 | 10/32 |
| 미학습wording | 0→13/32 | 32/32 | 14/32 | 10/32 |

문장 형식·한글 공동byte·EOS는80/80정상으로 회복됐다. 그러나 dev_pairs의 긍정관계22개는 모두 개체/방향이틀렸다. `C is smaller than H.`, `E는 E보다 작습니다.`처럼 그럴듯하지만 사실과다른문장도생성했다. 이 패널10개정답은모두불명응답이라단순기준선을넘지못했다. 미리정한형식80%/exact25%목표는넘었어도 의미생성 완성으로판정하지않는다.

Zero-VPlatent는원문context를유지했고exact5/16·10/32·12/32였다. 정상latent의미학습wording13/32와작은차이가있으나확산의독립의미기여를증명하지않는다. Decoder가원문을직접읽는다는제약도명시한다.

Root가240raw의정확/EOS/UTF8/byte/개체방향동치를CPU재구성하고5fullstateSHA·전체digest·동결VP동일·cache227MBtensors/sha를검증했다. Root추가모델forward0. 최종477691465B SHAb843d90b71f69419402fe3a6064522224c85058ab046b6f6fd1536348ca7d063, algorithmSHAcbac8d5fbe10ad7ffbc4bc74b0f6a189216ff8a34c2b41627bcccb7697f436f3. resultSHAf028a71e07bc0d66a8f5851636b9568ad32e3974f3482a212a300399d6fd43b1/cacheSHA c0fe127015ddd17319e46d352be526264ffec0693990679585916051bca3fce9. 모든부모/원래typed/core/과거실패/E2final을보존했다.

다음구체적문제는원문질문의두대상과비교방향을출력에정렬하는것이다. V3는원문context로부터4종류질문정보(주어8ID/목적어8ID/3관계/언어)를예측하는learnedreader와softfactor조건을추가한다. Gold는훈련손실·사후score만쓰고추론에는제공하지않는다. 역할외부제공·regex추론·고정answer조립으로바꾸지않는다. 이finite A–H결합도광범위NER/자유언어완성으로과장하지않는다.
