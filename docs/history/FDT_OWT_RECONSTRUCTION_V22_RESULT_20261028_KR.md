# V22 — 동일 입력의 퍼지 추출·재복원, 4개 유한 epoch 결과

실제 **8,192 attempt / 8,188 update / AMP skip 4회** 완료했다. 누적 AE 39,636 attempt / **39,624 update**다. 목표는 X→원래 퍼지 앵커 Z(X)→동일 입력 X̂이며, 추론·출력 목적 의미 디노이징은 별도 동결했다. 원문 context 우회 없이 추출된 두 view와 BOS/자체 복원 prefix만 사용한다. 보조 teacher-prefix 측정과 상태만 받은 동일 입력 복원을 구분한다. 자유창작 능력의 평가가 아니다.

## 학습과 보조 예측

같은 4,096 train 문서/16,384 선택 문장·구간에 4 finite epoch(301–304)를 노출했다. 65,536행/target UTF8 5,837,512 bytes이며 두 decoder 손실 감독은 11,675,024 UTF8 occurrences/11,806,096 byte+EOS symbols다. Prefix mask 875,376/eligible5,837,512=14.9957%. Skip(at33,2106,2849,6853)도 행·순서·노출을 소비했다. 새 문서 다운로드·기존 held 편입·E2 final/registry 접근 0이다.

Online clean CE epoch 평균은 .901058→.806530→.679659→.583923, 손상 prefix CE는 1.436073→1.368841→1.287488→1.227073, 혼합 loss는 1.168566→1.087685→.983574→.905498이었다. 이는 각 시점 가중치의 온라인 값이며 고정 final train NLL이 아니다.

동일하게 관측해 온 dev 64문서/256구간/23,207 active byte+EOS symbols:

| 상태 조건 | V21 NLL | V22 NLL | V21 byte 정확도 | V22 byte 정확도 |
|---|---:|---:|---:|---:|
| 자기 입력의 추출 상태 | 1.269213 | 1.339957 | 63.0801% | 62.8173% |
| 다른 입력의 상태로 교환 | 4.300625 | 4.798403 | 29.3963% | 29.1205% |
| 상태 제거 | 3.377847 | 3.810289 | 37.7343% | 37.2991% |

정상 correct14,639→14,578, EOS251→253/256이다. 보조 정확도는 **-.26285pp**, NLL은 **+.070743**로 소폭 퇴행했다. 이를 개선으로 소급 보고하지 않는다. V21과 V22 전체 상태를 모두 보존하며 자동 승격하지 않는다. 정상-교환 NLL 차이3.458446/정상-제거2.470332는 입력별 상태 정보 이용의 관측 근거지만 완전한 의미 추출이나 decoder 단독 원인 증명은 아니다.

## 원문-상태-원문 복원 직접 판독

기존 dev16과 이미 학습한 고정 train16을 모두 직접 읽었다. 두 표본 모두 UTF8/EOS16/16, exact0/16, 충실한 전체 의미 재현0/16이었다. **훈련 표본에서 올바른 부분 명제 1개**가 새로 확인됐다. 이는 비맹검 정성 판독이지 독립적인 의미 정확도 수치가 아니다.

| 동일 입력 상태 조건 복원 | V21 dev16 | V22 dev16 | V21 train16 | V22 train16 |
|---|---:|---:|---:|---:|
| chrF | 21.3720 | 22.2914 | 27.7533 | 32.4742 |
| 단어 F1 | 13.9512% | 10.8918% | 23.6227% | 29.5695% |
| 내용어 F1 | .6579% | 1.0952% | 12.0755% | 15.7188% |
| 정답 시작 UTF8 byte 평균 | 2.9375 | 2.6250 | 8.6875 | 12.0000 |
| byte edit 유사도 | 24.7838% | 24.9734% | 35.9692% | 39.0765% |

Dev 자기 원문 unique top1은11→12/16이다. 정상 chrF22.2914는 상태 교환15.8598/제거18.1905보다 높지만 단어·도입부·관계 보존 지표가 모두 함께 좋아진 것은 아니다. Train16은 이미 본 서로 다른 문서들이며 dev16과 합쳐 일반화 점수로 세지 않는다. 반복 관측 dev/설계 선택/작은 표본/정성 판독/lexical-not-semantic 한계가 있다.

새 부분 명제의 실제 예:

> 원문: The tweets were flagged by a source and sent my way.
>
> 복원: The tweets were flagged by a sound the years waying.

**The tweets were flagged**는 주어-행위가 올바르게 복원됐다. `source`가 `sound`로 바뀌고 전달 관계가 빠졌으므로 전체 복원 성공으로 세지 않는다. 또한 `Craig Charles`와 `Eduardo`는 정확히 남았지만 다른 인물·행위 관계는 빠졌다. 주문 일정 문장에서는 `We’ll be taking`과 `March`가 남았으나 실제 2012년 날짜·주문·발송 관계가 없어졌다. 경찰 신분과 “움직이지 말라”는 금지는 여전히 복원되지 않았다. 원문 없는 부정이나 이름 비슷한 철자를 보존된 명제로 높여 세지 않았다.

Dev에서는 주어·행위·대상·부정·관계가 대부분 빠졌다. 실제 원문이 불완전한 train 구간은 존재하지 않는 완전한 정답 명제를 만들어 평가하지 않았다. 전체 corpus의 문장 단편 비율은 측정하지 않았다. 전체 직접 판독은 run의 `root_all32_state_reconstruction_reading.md`에 있다.

학습 표본의 내용어·정확한 도입부·한 부분 명제는 진전이지만, 전체 복원과 관측 dev 보조 점수는 아직 안정되지 않았다. 초기·작은 고유 corpus이며 추출기·복원기·학습량을 분리한 인과 실험이 아니다. 퍼지/FDT 전체의 실패나 완성으로 단정하지 않는다. 현재 공동 추출·언어화 학습을 이어가되 다음에는 고유 train 문서 다양성 확대를 구체화한다. 원래 reasoning/의미 디노이징은 계속 분리한다.

## 실행·상태 보존

7 focused CPU 검사13.77초 PASS/실제 worker CPU preflight/독립 OS validate PASS 뒤 ONE 실행했다. Owner23296→28608 생성21:53:19.487/.513KST, worker28032→24480 생성21:53:30.252/.269, ROOT는 생성·parent·command·생존을21:53:39에 관측했다. Worker-owner exit0/COMPLETE/allowned gone 후22:04:01KST Python0을 확인하고 분석했다. 임시 sleepguard는 finally에서 해제됐다.

실제618.227892초/encode8,232+decoder18,304+denoise0=**26,536 calls/max28,000**. 새900초/reserve90/output6GiB 예산은 CLOSED이며 미사용1,464calls·시간을 재사용하지 않는다. 누적111,852calls와 이번 allocation을 구분한다.

Plan SHA `2dcad61de3d51bdb6c4a06a3b8997d731c98c19c5c09d79e70d103f67e1a09f7`. Final `state_8192.pt`646,846,001B/SHA `aa95bc9b1871cb6a3af3cb59876f39df5d29ec5aab8fee475a98ae81a8e67320`, algorithm `fd9b3bf4aa8583966f539f17472905f203f7a9571a6277227ece3fe48356c812`다. Import0/mid4096/final8192/sourceV21 및 모든 이전 fullstate/Adam/scaler/RNG/history/raw/실패를 보존했다.

ROOT CPU 검산은 세 fullstate SHA/safeload/재귀 digest/source전체 component 동일성/동결 reasoning-prefix-basis/finite model-Adam/63Adamstep39624/scaler131072tracker1339를 확인했다. Source tracker1976부터 모든 성공·skip을 재생했고 네 finite epoch 순서·cursor·노출·mask·두 loss 합성·전체 call·저장23,207logits×3조건 argmax/NLL·raw byte/EOS를 재구성했다. 추가 model forward0이다. 근거는 run의 `root_result.json`, `root_state_reconstruction_similarity.json`, `root_all32_state_reconstruction_reading.md` 및 `scripts/review_owt_state_reconstruction_v22_cpu.py`다.

## 후속 학습을 위한 저장 여유 복구

종료 시 C free41,696,280,576B로 새 학습 시작 기준40GiB보다 낮았다. 기존 결과를 지우지 않고, 닫힌 자체 AE 체크포인트27개와 workspace 런타임 DLL3개에 한정 파일시스템 압축을 적용했다. Microsoft의 `compact` 문서에서 확인한 NTFS/읽기 위주 파일용 LZX 방식이며 `/U /EXE`로 해제할 수 있다. [Microsoft compact 문서](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact)

두 **별도 한정 CPU 저장 작업**은 각각 max32files/24GiB/480초와 max3files/3GiB/180초로 닫았다. 실제27files/108.656초,3files/20.328초였다. 각 파일 전후 논리 길이·SHA가 모두 동일하고 모든 소유 compact 프로세스 종료를 확인했다. 삭제·이동·체크포인트 재직렬화·패키지 버전 변경·모델 forward·optimizer update0이다. 물리 할당 절감은1,356,281,253+658,371,104=**2,014,652,357 bytes**다. 최신 V20–V22 체크포인트와 사용자 원본·typed core·raw 자료는 압축 대상에서 제외했다.

보수적40.5GiB 여유 목표에는 못 미쳤지만,22:16:40KST 실제 C free43,328,319,488B로 **필수40GiB 시작 기준은 복구**됐다. 다음 실행 전 다시 실측한다. Torch CPU import2.10.0+cu128/CUDAinitializedfalse/modelcalls0 확인, Python/compact0이다. 근거는 `research/artifacts/owt_checkpoint_lossless_space_20261028_v1`과 `owt_runtime_lossless_space_20261028_v2`, run의 `root_postreview_storage_native_observation.json`이다. 과거 체크포인트의 내용 해시와 사용 가능한 상태는 그대로 보존했다.
