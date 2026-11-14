# V72 영어 언어→퍼지 메모리→언어 분리 학습 결과

2026-11-12. ROOT 직접 수행. **실제 640attempt/640update/AMPskip0 학습 완료**이며 새 시간·호출 예산은 닫혔다. 영어 원문의 수량·시간·이름 보존을 개선했지만 새 이름 퇴행도 남았다. 아직 언어 사이클 전체의 완료나 추상 의미 표현의 무손실성을 선언하지 않는다. 의미 디노이징은 실행하지 않았다.

V71에서 출력기 변경만으로 생기는 오류와 Commit·결합 쪽 변경으로 생기는 오류를 나눠 관측했다. 이번에는 V70 전체 모델 123,497,080개 파라미터를 그대로 읽고, **추출기와 upstream role_projection을 모두 동결한 채 마지막 readout 7개 tensor, 2,908,192개 파라미터만** 학습했다. K_P 기본 퍼지 prior와 입력별 K_C 문맥 퍼지는 실제 입력마다 계산한다. K_E Exact는 휴면 NULL/내용0/복사0이다. K_P의 인증된 장기 writer·통합 기능은 아직 없고 prior read만 유지한다.

출력기는 실제 bound Commit만 받는다. 원문 H·바이트·정답 prefix·GT 길이·정답 역할·regex 결과를 추론 feature로 주지 않는다. 고정된 추출·결합에서 만든 Commit cache와 source id/version/hash/upstream digest를 묶었고, loss용 정답·중요 구간·초기 teacher p(gold)·정답 여부는 feature와 별도 저장했다. 인코더 학습을 한 것으로 세지 않는다. 다음에 상류 가중치가 바뀌면 이 cache를 학습용으로 재사용할 수 없다.

훈련은 기존 V70 TRAIN512행을 보존하고 새 수치 표현128행을 더한 640행의 한 유한 epoch다. 원문954,100 UTF-8바이트와 중요 구간43,992바이트를 노출했다. 추출640행, 초기 teacher640행, head640행은 같은 자료의 서로 다른 계산이며 1,920개 새 문장으로 세지 않는다. fresh7AdamW/LR3e-5/WD.01/clip1/BF16/enabled scaler1024를 사용했다. V70의 populated30Adam/scaler/모든 RNG와 모든 과거 모델은 보존했다.

손실은 V70 기본·hard·tail EOS·중요 구간 CE를 유지하면서 중요 구간을 시간·부호·소수·통화·범위·단위까지 넓히고, 초기 모델이 TRAIN에서 맞힌 비중요 body 위치에만 0.05 Bernoulli KL을 추가했다. 평가에는 정답 여부에 따른 필터가 없다. 자료·loss·학습 경계가 함께 달라졌으므로 개선을 특정 항 하나의 효과로 단정하지 않는다.

| 관측 패널 | V70/학습 전 | V72 | 범위 |
|---|---:|---:|---|
| 새 영어 수치 표현32행의 수치 구간 손상 | 14/104 | 4/104 | 원문의 구간별 실제 출력 비교 |
| 같은32행의 이름 구간 손상 | 5/64 | 2/64 | 미학습 이름8개, 표현 가족은 공유 |
| 같은32행의 문장 literal 일치, 보조 | 11/32 | 26/32 | 전체 일치를 의미 점수로 대체하지 않음 |
| 약26만 바이트 장문2개의 수량 구간 손상 | 20 | 12 | 기존 관측 Commit을 재사용한 출력기 비교 |
| 같은 장문2개의 이름 구간 손상 | 17 | 4 | 상관된2,718개 사건 문장, 독립 표본 아님 |
| 같은 장문의 실제 decoded byte 오류, 보조 | 157 | 67 | 단어 의미의 중요도는 별도 판정 |
| 기존 영어36행 실제 decoded 오류, 보조 | 16 | 9 | 미학습 관측 보존 패널 |
| 기존 자연 OWT16행 실제 decoded 오류, 보조 | 125 | 105 | 문서별 식별자·이름 손상은 여전히 있음 |

위 수량·이름 구간 점수는 해당 작성 문법에서 확인한 원문 범위에 출력 바이트를 대응해 구한 제한된 점수다. 일반 자연 문장의 의미 정확도나 독립적인 수천 사례 점수로 확대하지 않는다. 숫자가 잘못된 다른 숫자가 된 장문 구간도 7→6곳으로 줄었지만 **7→6, 17→16 같은 값 변경6곳이 남았다.** 글자가 조금 비슷하다는 이유로 올바른 수량으로 판정하지 않았다.

ROOT가 새32행의 원문·전후 출력, 장문의 남은 수량·이름 위험16개 사건 문맥, 기존 짧은 영어의 변화와 자연 OWT의 남은48개 오류 문맥을 읽었다. `6:45→6045`였던 기존 두 문장은 모두 `6:45`로 돌아왔고, `-%13.24→-$13.24`, `11541→11:41`, `Epric→Elric` 등이 교정됐다. 장문의 문서형 입력에서는 이름 구간 손상이9→0이었고, 하나의 긴 문장형 입력에서는8→4였다. 이전 위치의 정보가 모두 복구됐다는 뜻은 아니다.

새32행의 남은 비일치6행 중 `+18.59→ 18.59`는 명시적 plus가 빠졌지만 양의18.59°C라는 값은 보존한 것으로 읽었다. 나머지5행은 온도값 손상·통화 소수점 불확실성·이름 식별 손상으로 남겼다. `+16.48→616.4m`, `-22.89→-2%.89`, `Brynn→9rynn`은 의미 보존 성공으로 세지 않는다. 이 작은 비맹검 수동 판정은 일반 의미 정확도 인증이 아니다.

퇴행도 보존했다. 새32행의 `Yvette→Vvette`는 초기에는 없던 이름 오류다. 기존16KiB 영어에서 두 `Inez→IneV`가 새로 생겼고 기존 `Ine9`, `27→2&`도 남았다. 자연 OWT에서는 첫 `U-24→1-24`, `LGBTQ→LGBTY`, 1인칭 `I` 소실 같은 새 위험이 있다. 전체 byte 수의 개선만으로 모든 조건이 개선됐다고 보고하지 않는다. URL·이메일 같은 식별자의 손상도 Exact로 자동 회수·수정하지 않았다.

장문은 V69/V70에서 실제 원문으로 계산해 저장한 Commit을 사용했다. 상류의 전체 가중치·binding digest가 동일한 것을 검증해 이번에는 **출력기만 비교**했다. 따라서 이번 장문 점수를 새 원문 인코더의 추가 실행 또는 새로운 blind 장문 시험으로 부르지 않는다. 실제 원문→메모리→언어 경로를 별도4행에서 재실행했으며 Commit 상대 L2=0/모든 argmax 일치/최종 출력 일치였다. 원래 길이 확장은16,384바이트 TRAIN 한도를 넘는 최대262,042바이트의 선형 local-memory 실행이며, 고정 용량 압축·범용 장거리 추론 증명이 아니다.

실행은 새900초/780초 모델 cutoff/120초 종료 여유, 최대2,200calls/출력4GiB/GPU12GiB로 봉인했다. 실제 encode676+readout1,434=2,110calls/denoise0, worker191.898초였다. 미사용90calls·시간은 재사용하지 않는다. 초기0/cache완료0/head1/320/640의5개 fullstate와640개 feature/target cache 쌍, 모든 원시 결과를 보존했다. 최종 fullstate518,000,145바이트/SHA `74531e0baa91f8f3fd0bfb26aeedc73871c9bc4afbd5a896ad6c25d4158fe864`, recursive algorithm SHA `d2b59076b393dcfcafef768da2cc377cc95f47bba8ac874247409dabc4eb2e09`다. Plan SHA `bff7df9aa19fdf66235c081552f3053637711964e6a1c9adf7f2bfadbb214ac6`.

OwnedCPU preflight8.781초에서 실제 전체 모델 복원·추출1/readout2·기울기 소유권·부분 cache fullstate 복원을 확인했다. 작은 별도 CPU 계약에서 cache 중간/경계·populated7Adam/enabled scaler/모든 RNG·다음 update/AMPskip 행 소비의 정확한 복구를 확인했다. 실제 CUDA 재시작은 하지 않았다. ROOT는380개 봉인 pin과5fullstate SHA/safeload/재귀 digest,7Adam/scaler/모든 저장 RNG/cursor/order/exposure,모든 비학습 tensor 불변,7학습 tensor 변경,640cache 쌍의 source/mask/epoch/SHA를 독립 재구성했다. anchor909,222위치를 확인했으나 TRAIN teacher의 전체 logits는 저장하지 않았으므로 teacher argmax를 독립적으로 다시 계산했다고 주장하지 않는다.

실제 원시154개 결과의 target/argmax/NLL와 독립 UTF-8 FSM·EOS·출력 문자열을 CPU로 확인했다. 추가 모델 호출0/CUDA 초기화false다. 새DEV baseline의 일부 끝부분에 추가 문자가 있어 CPU 의미 집계 초안의 동일 길이 가정이 실패했다. 실패 기록을 보존하고, 실제 ASCII 선택 byte 위치를 검증한 뒤 원문 구간과 끝의 추가 문자를 분리해 집계했다. GPU 계산이나 평가 출력을 다시 실행·수정하지 않았다.

OSowner12884→26932/worker16080→6348의 생성 시각·parent·command·exe·생존을 ROOT가15:02:54.958KST에 관측했다. worker/owner exit0 뒤15:06:29.251에 모든4owned/proxy 종료와 fresh nativePython0을 먼저 확인한 후 분석했다. 학습 중 ROOT 상세 분석과 에이전트 사용은 없었다. 학습 GPU peak775,015,936바이트/추출 peak3,264,918,528바이트. 종료 당시 C여유약117.8GiB로100GiB 보호선을 유지했다.

다음 개선에 쓸 구체적 자료 편중도 확인했다. 중요 구간에서 `I`, `Y`, `z`가 각각0회였으나 전체 TRAIN에서는2,438/340/636회 존재했다. 따라서 해당 문자를 전혀 학습하지 않았다는 뜻은 아니다. 중요 구간의 plus는14회뿐이며16개 위치 중6·7·8번에만 있었다. 반면 숫자6/7/8/9는 중요 구간의 모든 위치에 최소54회 이상 있어 단순히 그 숫자를 본 적 없다는 설명은 맞지 않는다. 위치 기여는 actor 같은 추상 semantic role이 아닌 byte-position binding이다. 이 편중은 후속 범위를 정하는 근거이고 단일 원인 증명은 아니다.

증거: `runs/frozen_verbalizer_20261112_v72/{root_result.json,root_saved_review.json,root_meaning_review.json,root_manual_reading_cards.json,root_TRAIN_coverage_cpu.json,fresh_cache_parity.json,ROOT_terminal_native.json}`. 다음 계약은 `FDT_FUZZY_EXTRACTION_V73_NEXT_CONTRACT_20261112_KR.md`에 분리했다. 현재 유용한 V72 가중치는 보존하되 자동 승격이나 추가 무제한 epoch는 하지 않는다.
