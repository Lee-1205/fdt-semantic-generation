# V79 실제 결과 — 마지막 출력층만 학습해도 보존 문제 잔여 (2026-11-13 10:13 KST)

V79는 한 번의 한정 실행을 완료했다. V75의 전체 입력·퍼지메모리·Commit 바인딩 및 출력 terminal/LayerNorm/첫 Linear 5개를 고정하고 **마지막 Linear weight/bias 2개(2,117,664 parameters)**만 학습했다. V78의 loss, weak teacher margin<=3, 계수, 384개5행 그룹, 1,920행 노출, LR1e-5와 seed를 유지해 trainability 범위만 비교했다. Clipping/Adam 부분공간도 달라지므로 개별 prefix 부품의 인과증명은 아니다.

결정은 **V75를 다음 연구 초기화로 유지**다. 짧은 두 수량 패널의 일부 회복은 있었지만 장문 숫자 보존 개선으로 이어지지 않았다. V79와 이전 모든 실제 가중치/Adam/scaler/RNG/raw/실패 자료를 보존했으며 기존 모델/registry를 덮어쓰지 않았다.

| 같은 관측 지표 | V75 | V78(7텐서) | V79(2텐서) |
|---|---:|---:|---:|
| 두 수량 모두 읽히는 행 /64 |54|56|56|
| 첫 수량 /64 |64|64|64|
| B kg /16 |10|12|12|
| B 온도 /16 |14|16|16|
| B km /16 |14|14|14|
| B items /16 |16|14|14|
| 다른 사실을 바꿔도 유지된 값 /64 |59|60|60|
| 기존 V75 패널 stock 엄격 보존 /32 |14|20|14|
| 기존 양의 kg 엄격 보존 /72 |59|60|59|
| 장문 수량 오류 /5436 필드 |16|28|29|
| 장문 episode fixed-span 오류 |10|25|22|
| 장문 인명 오류 |2|2|1|
| 장문 item 오류 |6|6|8|
| 장문 decoded byte 차이 /524071 bytes |58|76|77|

장문은 실제262042/262029UTF8 bytes의 두 입력으로 최대 TRAIN16384보다 약16배 길다. 2718개의 상관 event이며 5436개의 수량 필드다. V75→V79 수량은 교정0/신규13/기존16; V78→V79는 교정2/신규3/잔여26이다. Episode V75→V79는 교정1/신규13/잔여9다. 제한된 before/after 관계·색상 위치·부정 필드는 계속0변화였고 UTF8/EOS 전체 길이는 유지했다. 드문 오류가 실제로 있으나 문서 전체 의미가 붕괴했다고 과장하지 않는다. 반대로 이를 자연어 의미가 완벽하다는 인증으로 쓰지도 않는다.

직접 읽은 실제 오류: +50.25kg→450.25kg는 값 변화, 75 washers→7; washers는 수량 훼손, 22 ribbons→+2 ribbons와17→16은 사실 변화다. `Reviet`, `drawer→draber`, Cyril→Ckril, silver→s lver도 남았다. 긴 행1007에서는 원문 `28 silver`가 `2Üsilver`가 되어 정렬된 수량 substring만의 판별보다 위험하다. 숫자 뒤 경계도 확인해 review7109;→71090 및145+/169+를 별도 위험으로 기록했다. V77/V78의1089→10893,53→533 병합은 V79에서 관측되지 않았다. 기존 fixed-span 점수를 소급 변경하지 않았다.

ROOT는 새 패널 비문자일치22개, 기존V75 패널 비문자일치20개, **장문 변경 event72개 전체 문맥**과 시작·중간·끝6개를 원문과 대조했다. 변경 field pattern42개다. 전체524071byte를 사람이 하나씩 의미 채점하거나 모든 OWT/old128 문장을 수동 인증한 것은 아니다. 관측 개발 자료를 여러 번 재사용했으며 새로운 blind benchmark가 아니다.

새64 패널 NLL은 .00668718→.00636117로 낮아졌지만 byte/EOS accuracy는99.8588%→99.8460%였고 auxiliary literal은44→42/64였다. NLL·byte accuracy·수량 보존은 서로 다르다. OWT decoded 오류108→117, 기존128은54→56, 표현패널8→11, retained English3→3이었다. 기존V75 primary64개는 유지했지만 stock의 unit 누락2개/값과unit동시훼손2개도 남아 숫자 core만으로 성공 처리하지 않았다.

전체 NN과 buffer를 독립 재구성해 upstream과 출력 prefix5개가 bitwise 동일하고 **정말 마지막2개만 달라졌음**을 확인했다. Δweight L2=.3988028/max=.00244912, Δbias L2=.0170553/max=.00107134였다. 같은 새64 Commit도 bitwise 같았다. 따라서 이번 출력 변화는 같은 Commit 아래 마지막 출력층 갱신에서 발생했다. 이것은 모든 입력 의미가 완벽하다는 증거나 모든 역사적 오류가 출력층만의 문제라는 증거가 아니다.

TRAIN teacher에서 correct critical45063 중 weak1943, correct noncritical964665 중 weak2995만 margin 대상으로 선택됐다. 실제 margin 항은 적용됐고 critical84/noncritical416 반복 위치 위반, 293/384그룹 clipping, 평균 margin항 .00339214/.00181836이었다. 미관측 사실 보존 보장은 아니다. wrong-plus는8→33(V78 36), digit→plus8→30(V78 34)였다. prefix 고정만으로 sign/숫자 간 출력 간섭을 해결하지 못했다. 실제 task gradient 충돌을 계측했다고 주장하지 않는다.

실행:384attempt/384update/AMPskip0, 317.803424worker초, encode3076+readout3418+chunks258=6752calls, longblock2322, denoise0. Teacher1024행1010943UTF8B, head1920행2720189UTF8B/86108critical positions. 새7000calls/900wall780cut120reserve/1.25GiB budget은 **CLOSED**, 미사용248calls/시간 재사용·자동 재학습은 없다. GPU peak 3324810752B/teacher3264918528B.

Owner508→19884(10:01:14.859/.884), worker17124→2136(10:01:23.073/.087)의 생성·parent·command·exe·생존을 ROOT10:01:43.881에 관측했다. Worker/owner exit0 후 ROOT10:07:12.864에 모든 owned/proxy 종료와 freshPython0을 확인한 뒤 분석했다. 실제 학습 중 ROOT는 상세 분석을 중단하고 OS 감시만 했다. 마지막 C117120512000B=109.08GiB로100GiB 보호 유지/현재 모델 프로세스0.

ROOT는 1035pins/3fullstates, populated2Adam step384/scaler tracker384/모든 저장RNG/finiteorder·cursor·384그룹1920micro history·노출·teacher1024packets와470raw+parity1을 독립 재구성했다. CPU3focused/tiny실제엔진 복구 및 actualfullsizeCPU2encode2readout/optimizer0/CUDAfalse도 구분해 통과했다. Teacher full logits는 미저장이므로 teacher전체argmax/margin을 raw에서 재계산했다는 주장은 하지 않는다. 추가 postrun NNforward0/실제CUDA 복구 재실행0.

저장 수정도 검증했다. V75의 prefix5개는 standaloneV74와 다르므로 파일에는7개 delta를 저장하되 optimizer는2개만 소유한다. 2개만 저장하면 V75 prefix가 사라지는 사례를 CPU에서 거절했다. Direct standaloneV74 base를 참조하며 recursive model link/Exact 문장 검색이 아니다. 최종29674803B SHA `3ec7e21ec02a8f6de3db64a6477b08be18cfccf608a29ef56cf2ab62e0c4ad43`, algorithm `1718aa6710d5e76e4c921a7913f68f8d8578d0ee9489e945f6aafaa580b1dc69`; plan `3c712e707f5c5070c217395b1002a64afbda6f7eab49b1a3171369abecf97965`. Mandatory base `734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7`를 보존한다.

K_P learned prior/K_C input long/recent fuzzy는 유지, K_E는 invalidNULL/content0/copy0다. K_P certified writer는 아직 없다. 현재 언어 복원 수리 단계이고 디퓨전 호출0이다. 미래 사고학습은 전체 input-memory-binding-output codec을 고정하고 별도 사고 모듈만 학습한다. Whole language→meaning→language 완료/독립 abstract 의미 인증/자동 승격은 하지 않는다.

다음 구체적 계약은 [출력 갱신 보존·TRAIN coverage](FDT_READOUT_UPDATE_RETENTION_NEXT_CONTRACT_20261113_KR.md)에 저장했다. 동일2개를 같은 목적·같은384회로 자동 반복하지 않는다. GEM/PCGrad 원 논문은 그 문서의 출처 링크로 확인했으며, raw gradient 제약이 Adam 실제 update에 그대로 성립하지 않는 수치 반례도 계산했다. 새 GPU/생성 능력 개선 증거는 아니다.
