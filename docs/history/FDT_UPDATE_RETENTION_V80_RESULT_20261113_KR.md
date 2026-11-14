# V80 실제 결과 — 장문 회복과 짧은 표현 잔여를 함께 보존 (2026-11-13 14:07 KST)

한 번의 V80 한정 실행과 독립 검토를 완료했다. **384attempt 중247update를 채택하고137개 제안은 거절, AMPskip0**이었다. 1920source micros는 모두 소비했고 거절분을 재생하지 않았다. 입력·퍼지메모리·Commit binding·출력 prefix5개는 고정, 마지막 Linear2개2117664parameters만 바뀌었다. 디퓨전은 호출0이다.

다음 언어 수리의 연구 후보로 **V80의 유용한 진전을 이어가고 V75는 균형 비교·복구 참조로 보존**한다. 전역 모델 승격이나 language↔meaning 완료가 아니다. 모든 이전 모델/Adam/scaler/RNG/raw/실패도 보존했고 registry를 바꾸지 않았다.

| 동일 관측 패널 | V75 | V79 | V80 |
|---|---:|---:|---:|
| 두 수량이 모두 명확한 행 /64 |54|56|51|
| 첫 수량 /64 |64|64|64|
| B kg /16 |10|12|12|
| B 온도 /16 |14|16|12|
| B km /16 |14|14|12|
| B items /16 |16|14|15|
| 다른 수치를 바꿔도 정답 유지 /64 checks |59|60|57|
| 기존V75 primary /64 |64|64|64|
| 기존V75 stock 엄격 보존 /32 |14|14|16|
| 기존 양의 kg 엄격 보존 /72 |59|59|61|
| 장문 수량 오류 /5436필드 |16|29|15|
| 장문 episode fixed-span 오류 /2718 |10|22|8|
| 장문 인명 오류 |2|1|2|
| 장문 item 오류 |6|8|7|
| 장문 action moved→roved |8|0|5|
| 장문 decoded byte 차이 /524071 |58|77|52|

장문 두 입력은262042/262029UTF8 bytes로 TRAIN 최대16384보다 약16배 길다. 2718개의 상관 authored event이며 독립 자연 문서2718개가 아니다. V75→V80 수량은4교정/3신규/12잔여, episode2교정/0신규/8잔여다. V79→V80 수량은15교정/1신규, episode15교정/1신규였다. Restricted order/location-color/negation 필드는0변화이고 전체UTF8/EOS·길이는 유지됐다. 이것이 모든 동작/의미가 정확하다는 뜻은 아니다.

실제 원문과 출력의 **새64 비문자일치21개, 기존V75 비문자일치18개, 장문 변경 event45개 전체 문맥 및 시작·중간·끝6개**를 읽었다. 장문21개 field pattern이 남았다. 전체524071byte를 사람이 개별 의미 채점하거나 모든 OWT/old128 의미를 인증한 것은 아니다.

짧은 두 수량의51/64를 전체 문장 의미정확도 또는13개 확정 의미 오류로 해석하지 않는다. B51개는 값·단위가 명확하고,8개는 `#64.25 degrees C`/`(71.25 km`처럼 숫자 core·unit은 유지했지만 앞기호가 비표준이라 미확정이다. 네 개는 +50.25→450.25 유형의 명확한 값 변화이고, 한 개는75→7:의 digit손상이다. V75는54명확/6비표준core보존/4값변화였다. 즉 strict3개 감소 전체를3개 새로운 확정 사실오류로 부풀리지 않되, 추가 digit손상1개는 실제 위험으로 남긴다.

장문에서는26→+6,21→+1,7→&,Fiora→2iora가 새 위험이고, Cyril→Ckril/silver→s lver/drawer→draber가 남았다. `moved→roved`는 단순 의미동치로 용서하지 않았다. 이전 이력을 확인하니 V75에서8개/V79에서0개/V80에서5개여서 전부 V80 신규라 하지 않는다. 기존 fixed episode span 밖의1089→10893/53→533/145+/169+ 추가 문제는 이번 V80에서 없었다. 짧은 review7109;→71090,Review7110→Review7 10은 수량 slot과 별도로 남았다. 과거 strict 점수를 소급 변경하거나 출력 교정을 적용하지 않았다.

참조 TRAIN100개 source의7487hidden patch를 별도 frozen-prefix feature로 만들고 labels·초기 argmax/correct mask는 loss-only target으로 분리했다. 임의 goldrole/GTlength/raw byte가 decoder feature로 들어가지 않았다. 초기 reference가 맞춘117600개의 nonPAD 위치(critical4380/noncriticalEOS113220)는 최종 손실0, 초기오답44개 중14개를 고쳤다(critical8/noncriticalEOS6). 이들은 byte/EOS위치이며 독립 사실 수가 아니다. 초기 batched와 rowwise teacher의 correct flags 차이는0이었지만 wrong-class의 동일성이나 전체 logits parity를 증명한 것은 아니다.

실제 Adam 제안을1,.5,.25,.125로 한정 검토했다. alpha1은237회,.5는4회,.25는2회,.125는4회 채택했고137회는 모두 거절했다. 총815candidate reference 검사+초기/최종2=817Linear calls였다. 거절해도1920행/2731825UTF8B/94512critical노출과 finite cursor/scaler성장은 소비했다. Adam step은247, scaler growth tracker는384였다. 전체137거절은 weights+Adam을 복구했으며 자동 보충 학습을 하지 않았다. 100TRAIN보존은 unseen 의미 보증이 아니다. 각 과거 candidate의 전체 logits를 별도 보존하지 않아 그 전체를 CPU에서 다시 forward 검증했다는 주장도 하지 않는다.

동일 source의 Commit64개는 학습 전후 bitwise 동일했고 전체 모델·buffer를 직접 재구성해 마지막2개만 변경/나머지 및prefix5개 고정을 확인했다. 따라서 이번 출력 변화는 같은 입력 의미 표현 아래 lastLinear 갱신에서 발생했다. 입력이 추상 의미를 완벽하게 형성했다는 인증이나 모든 과거 오류의 decoder-only 원인 증명은 아니다. 데이터·grouping·guard를 함께 바꿨으므로 개선을 단일 요인에 귀속하지 않는다.

새64NLL .00668718→.00680720/byte99.8588→99.8396/literal44→43aux다. OWT decoded108→110, retainedEnglish3→3,old12854→53,oldexpression8→7이었다. V79보다 여러 지표가 회복됐으나 V75 대비 모든 항목이 개선된 것은 아니다. 기존 관측340+revised64+2long을 재사용했으며 새로운 blind benchmark나 E2 최종평가가 아니다.

Worker 333.079065초, encode3076+fullreadout3418+referenceLinear817+stream258=7569calls/max8500이었다. Longblock2322,reference patch work6116879/11515006cap, GPU peak3332758016B/teacher3264918528B다. 384attempt/8500calls/900wall780cut120reserve/31/32GiB owned budget은 **CLOSED**이고 미사용931calls/시간은 재사용하지 않는다.

Owner16780→28920(13:48:15.489/.515), worker23912→20032(13:48:24.147/.160)의 생성·parent·command·exe·생존을 ROOT13:48:44.664에 관측했다. Worker/ownerexit0 후13:54:34.260에 모든 owned/proxy 종료와freshPython0을 확인한 뒤 분석했다. 실제 학습 중 ROOT 상세 분석을 중단하고 OS 감독만 유지했다. 마지막 C116216197120B=108.23GiB로100GiB 보호 유지/현재 모델프로세스0다.

저장: 기존 source/raw는 바꾸지 않고 새 raw에서만 bit-exact FP32값을 BF16 carrier로 저장한 뒤 모든 reader가 원래FP32로 복원했다. Nonexact fullFP32 long/parity는 그대로다. ROOT1064pins/3parent-linked fullstates/2Adam·scaler·모든RNG·groupcursor·소비노출/teacher1024/분리TRAIN100bank/초기최종 argmax/470raw+parity1/실제UTF8FSM/NLL/모든sourceSHA를 독립 확인했다. 추가 postrun NNforward0이다. Initial reference argmax는 저장값에서 마스크를 재구성했으며 full initial reference logits를 저장·재계산했다는 주장은 하지 않는다.

최종29823411B SHA `588ceaad4a94dde38ae56e42239497534d69387814a4fa66a4fbcea974db3cd2` / algorithm `1c67b342a0df81aa993f97043f50b57d1d057858ed70c0750f3dfa1bb11b727c`. Plan `d6c527d0472925db57203811a5be23d7aa9eda62bd9ca5c69cf29e2b6c1262aa`. 실제 모델은 standalone이 아니며 mandatory standaloneV74 `734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7`+7stored delta(고정prefix5+trained2)와 full runtime을 함께 보존한다. 모델 가중치 압축 참조이며 Exact 문장 검색이 아니다.

준비 중 tinyfixture 경로정규화 및 상속 metadata의 no_feature_cache/384optimizer_steps를 수정했다. 실제 fullsizeCPU4encode4readout/optimizer0/CUDAfalse와 전체 tinyphase/scaler/STOP/거절 복구가 통과했다. GPU 실행중 sealed code는 수정하지 않았다. 상속 progress 명칭은 일부 남아 있었으며 loss packet/head source cache와 별도TRAIN-reference cache를 본문에 구분했다. 추가 severity분석은 V75의 서로 다른 옛 baseline파일을 잠시 선택한 오류를 source-ID 일치 검사로 바로잡고 이전 진단·코드를 보존했다. 최종 비교는 이 run의 V75 warm new_baseline이다.

자동 승인 검토가 이전 승인된 pip 캐시 삭제 명령을 정책상 차단했고 구체 사유는 없었다. 삭제는0이며 우회 삭제를 하지 않았다. 대신 새 출력 cap을31/32GiB로 낮춰 예상1023415424B/임시·STOP·bank 포함 조건으로 이번 실행을 완료했다. 사용자 원본/게임/가중치·AdamRNG/raw를 지우지 않았다.

K_P prior/K_C input long-recent fuzzy는 유지, K_E invalidNULL/content0/copy0, K_P certified writer는 미완료다. 현재 language→meaning→language 수리이며 미래 thought diffusion은 전체 입출력·메모리·binding을 고정하고 별도 사고 가중치만 학습한다. 다음 구체적 목적과 자원 계약은 FDT_V80_CONTINUATION_NEXT_CONTRACT_20261113_KR.md에 저장했다. 완료하지 못한 언어 복원과 자동 디퓨전 진입을 구분한다.
