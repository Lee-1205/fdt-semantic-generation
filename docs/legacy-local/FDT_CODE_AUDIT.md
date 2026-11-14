# 수식·코드·운영 감사

## 설계와 실제 adapter

| 항목 | 실제102M코드/shape·정밀도·gradient | 한계·실패 조건 |
|---|---|---|
| μ/ρ/γ, E02~04유사 | model.Router: q[B,F,V,128],anchors[64,128],FP32,stableTop8,invalidzero;query/view/sigma/beta학습 | μ는진실확률/identity아님;float64기억의vNext전체구현아님 |
| Local attention E05~08유사 | Language9층/768/12heads/FF2048/W48,byte257embedding;FP16autocast/RMSFP32 | tokenizer16K문서경로아님,문구48UTF8bytes |
| 문맥memory | encode: mu[B,F,64],values256,FP32num/mass;한episode안factsaggregate | vNext prefix-read-before-write/float64bank/CASWriter미구현 |
| 관계E26~31유사 | rel_left/right[8,64,8],softpluspositiveG/T/F구분,NULL질량재정규화하지않음 | G/T저랭크를F에일반화불가;명시적typedproof가아닌학습메시지 |
| initializer | fixedmax-productclosure와주어진role16을latent512에매핑,Core2블록 | 스스로학습한개방자연어추론전체로해석불가 |
| VP E36~40 | objectives.diffusion_loss 및 model.rollout: t[B],Z[B,16,512],αε−σZ0,역변환αZ−σv,정확t=1;refiner6블록 | 원래시간의최소1개/대략절반을t=1;oraclevelocity대수PASS≠denoiser성능 |
| groundingloss | maskedCE,분모Σvalid*classweight,grounding4종 | query4종과label의미다름;classweight≠노출25% |
| semanticloss | CErelation+.25maskedMSE+.2groundCE,분모slot수*latent폭 | rowmean/accumulation1/K는기존목적유지,미니배치분모에따라가변slot가중이생길수있음 |
| diffusionloss | MSEv+.25MSEz0+.5CEreadout,encode는no_grad | diffusion을늘려frozenphrasegrounding학습했다고할수없음 |
| gradient | phase별원본trainableprefix유지,매시도missing/frozen/nonfinite,clip전norm/scale/skip기록 | smoke에서102Mgrounding만실제,다른두phase는tiny로검사 |
| Commit/Realizer | runtime.Commit는int/stringtuple복제·scalarfinite검사,표현기는Commit만참조 | vNexttensor-backedCommit/Claim/Permit/Writer전체미구현,범용생성기아님 |

65개 vNext수식 전체는 archive 명세에 보존되어 있으며 위 표는 **현재adapter에실제대응하는부분**이다. ExactStore/인증epoch/Codec/ClaimProof/RealizerKV·copypermit등은해당명세의미구현인터페이스다. 이번준비에서이를생산구현완료로보고하지않는다.

## E0/E1 분리

baseline model.py는원본과byte-identical하며Corepaddingmask미전달경로도그대로다. E0가중치의변경없는CPUforward비교가bit-exact였다. 별도model_masked.py는queryvalid/factvalid/contextNULLmask를self/crossattention에전달하고paddedresidual을차단했다. 같은validquery에paddingfact/query수를바꾼CPUtinyencode불변성을검사했다. E1CUDA성능·실제학습은하지않았으며baseline가중치에몰래적용하지않는다.

## 저장·실행 계약

- FileLock은Windows/LinuxOS잠금이다. 다른PC의분산잠금보장은아니다. import는학습을시작하지않고CUDA없음은오류다.
- model/optimizer/scaler/모든사용RNG/phase/성공update/시도/노출/history/trainable목록/manualcosineLR상태를완결시도경계에저장한다. 성공과AMPskip을구분한다.
- 유일한임시payload→flush/fsync→replace→전체safe-load/size/SHA/내부계약확인→complete marker→latest순서다. Windowsdirectoryfsync는지원보장안함. 완전전원상실시OS/하드웨어durability를절대보장하지않는다.
- 최신payload손상/marker필드오류/계약불일치/Windowsreplace실패/중복writer/디스크하한/미완료평가재개를fixture로검사했다. 완료marker가없으면진행저장성공으로세지않는다.
- 고정deadline은초기저장/중간저장시간을포함한다. kernel/flush가반환하기전즉시중단보장은없다. 매시도저장이50분간격보다촘촘하다.
- C:20GiB하한과임시payload예상을매시도/저장전에검사한다. 다른프로세스가갑자기디스크를채우는상황까지예방하는시스템관리서비스는아니다.
- 평가결과는checkpointSHA+suiteSHA로묶고부분완료시누락항목만이어간다. 학습phase완료와평가완료는별개다. full plan1440회와마지막phase전환은fixture만검사했고실제장기실행아님.

## 알려진 개선 지점

1. E0의Corepadding민감도와고정closure의기여를학습성능과구분해야한다.
2. 기존graphhash는entityrename동형그래프를완전히묶지않는다. 표현family분리는별도E2개발이다.
3. class균형sampler는아직실제trainer에적용하지않았다. 25%목표와60/15/25비율을현재학습노출로기록하면안된다.
4. 새로운evalsuite는아직GPU실행안했다. 이번5attemptsmoke는성능수준을측정하는학습이아니다.
