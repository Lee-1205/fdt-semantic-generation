# V29 — 입력 언어 추출부 4개 block 적응 후 같은 문장 복원

ROOT가 실제 새 학습 4,096attempt/4,094update를 완료하고 모든 owned/proxy 종료를 확인한 뒤 저장 결과를 검토했다. 언어→원래 퍼지 입력 의미→동일 입력 재복원만 학습했으며, 출력 목적 의미로 바꾸는 추론·디노이징은 별도 동결 상태다. 총 111,485,711parameter와 퍼지 μ/ρ/anchors/value-prior-condition, 두 의미 view만 받는 decoder, p50 목적과 추론 경로는 유지했다. 기존 언어 block5·6을 추가로 풀어 마지막4개 block을 학습 대상으로 삼았으며 학습 가능 parameter는 38,309,762개다. 새 parameter는 없다.

## 실제 실행과 보존

- Run: `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261029_v29` ONE COMPLETE, worker/owner exit0. 신규 330.232초, encode4,136+decoder10,075+denoise0=14,211calls/max15,000. 900초/reserve90의 별도 예산은 CLOSED이며 남은789calls·시간을 재사용하지 않는다.
- 새 seed371의 한 유한 epoch, 기존8,192문서/32,768span을 한 번씩 선택했다. 신규 targetUTF8 2,926,478bytes, decoder byte/EOS 지도 기회5,918,492, 실제 past-byte mask1,463,686/2,926,478이다. 원래 dev64문서256행과 자료 분할·과거 자료는 보존했다. 새 corpus 다운로드·새 평가 패널·E2final/registry 접근은 없다.
- 기존63tensor의 populatedAdam group/moment/step88,005, 모델/scaler32768tracker975/모든 RNG를 정확히 보존했다. 기존 tensor18개만 별도 세 번째 group/baseLR3e-6으로 추가하고 첫 갱신 전 Adam은 비웠다. 기존 groupbaseLR1e-5/1e-4는 유지했다. 전체 optimizer dict의 구조는 의도적으로 확장됐으며 전체 불변이라고 주장하지 않는다. 옛 frozen hash와 새 phase frozen hash를 따로 검증했다.
- CPU focused7개 PASS/41.14초: 실제 새 두 층을 포함하는5block tiny모형, oldAdam 보존·new18빈state, gradient ownership, enabledscaler AMPskip/fullstate 정확 next/마지막7행, projected write reserve1GiB, 실제 schema의 postPopen 정상·예외·불확실 종료 경로. 첫 pytest의 시스템 임시폴더 PermissionError는 원문 XML로 보존했고 별도 새 workspace 임시폴더에서 통과했다. 모델 실패나 GPU 실행으로 세지 않는다. 실제 worker CPUpreflight를 독립OSvalidate로 통과했다.
- PlanSHA `b004c9dc1c510c3ed75cd03dab896faff4e615cbd79b23f4a23260bd87b523d3`, 시작1793236969.004129/끝1793237869.004129/cutoff1793237779.004129. Owner33004→4008 생성10:22:49.265/.293KST, worker31616→25924 생성10:23:10.317/.333KST. ROOT10:23:39의 native 생성정보/parent/command/생존과 owner의 identity_verified를 실제 관측했다. 마지막 ROOT training 관측10:25:12는1382attempt1381update였으며 이후 heartbeat/yield 간격이 있었다. 전 구간 ROOT 연속 관측이라고 주장하지 않는다. 독립 OSowner는 정상 완료와 전체 종료를 보존했고 ROOT12:45:52 새 조회에서 Python0/allownedgone부터 확인했다.

## 저장된 raw에서 재구성한 결과

보조 teacher 측정은 **정답 이전 byte를 제공한 조건**이다. 의미 보존 백분율이나 완전 복원률이 아니다.

| 같은 관측 dev256행/23,207symbols | V28 | V29 |
|---|---:|---:|
| 정상 teacher NLL |0.996612|0.991277|
| 정상 teacher byte정확도 |71.1768%|71.1682%|
| 정확 byte/EOS 수 |16,518|16,516|
| 정답 EOS 위치 |253/256|253/256|
| 다른 행 의미로 교환한 NLL |6.371324|6.407646|
| 의미를0으로 한 NLL |5.229656|5.284431|

NLL은0.005334 감소했지만 정확 byte는2개 줄었다. 정상보다 swap/zero가 크게 나쁜 것은 입력별 의미 상태의 사용을 지지한다. 완전한 의미 추출·decoder 무결성·독립적인 추론 능력·단일 병목 원인의 증명은 아니다. 추가 학습량/순서/LR와 층 해제 효과를 분리한 대조 실험이 아니므로 개선을 추가 두 층의 인과적 효과로 확정하지 않는다.

ROOT가 정답 prefix·정답 길이 없이 **그 입력에서 뽑은 의미와 BOS/자기 prefix만으로 복원한** dev16행과 고정 train16행을 모두 읽었다. 이것은 무조건 자유 창작의 점수가 아니다. 양쪽 UTF8/EOS16/16이지만 exact0/16, 전체 충실 복원0/16, 확실히 보존된 부분 명제0/16이었다.

| 사후 어휘 유사도 | dev V28→V29 | 고정 train V28→V29 |
|---|---:|---:|
| chrF |22.6713→23.0972|34.9578→32.3149|
| wordF1 |12.2564→10.4395%|30.1430→27.9395%|
| content-wordF1 |2.1946→1.6062%|21.7660→16.6967%|
| 처음부터 맞은 UTF8bytes 평균 |3.125→2.75|13.8125→12.5625|
| byte편집 유사도 |25.9390→25.5162%|42.2928→40.9750%|

`Craig Charles`는 유지됐지만 다른 참가자와 합류 관계는 사라졌다. 기존에 정확하던 `Eduardo`와 `NATO`, `Outside the court`는 이번 출력에서 흐트러졌다. `Donations and correspondence`는 더 잘 복원했으나 보내는 행위/가족/주소는 틀렸다. `25-year-old`는 여전히20으로 바뀌고 경찰 신분·움직이지 말라는 지시는 없었다. Dev의 부정·대상·사건·시간 관계도 복원됐다고 판단하지 않았다. 소스 자체가 선택된 문장 조각인 경우 존재하지 않는 완전한 정답 명제를 만들지 않았다. 모든 문장별 원문/출력/ROOT 판독은 run의 `root_all32_state_reconstruction_reading.md`에 남겼다.

전체 유한 epoch의 앞/뒤 절반 온라인 cleanCE .545391→.506469, corruptCE1.995880→1.970375, 혼합loss1.270636→1.238422로 내려갔다. 서로 다른 행·진행 중 가중치 평균이므로 고정 최종 trainNLL 비교가 아니다. 초기 추가 층 적응 한 epoch만으로 FDT/퍼지 전체 실패를 선언하지 않는다. 원래 typed 관계 예측의 강했던 관측 결과와 가중치는 별도로 보존했다.

## 독립 상태 검토

ROOT 새 CPU reviewer `scripts/review_owt_state_reconstruction_v29_cpu.py`는 추가 모델 construction/forward0으로 import0/mid2048/final4096의 SHA/safeload/재귀 digest와 세 optimizer group을 검사했다. Import에서 기존63개 state가 이름별/원래 group별 exact, 새18개 empty였고, 마지막에는 기존63 Adamstep92,099와 새18 step4,094였다. 새18개 tensor 모두 실제 변경됐다. 나머지 frozen모델/이유추론부/basis는 그대로다. 새 AMPskip은1379/3454였고 scaler32768tracker642까지 전체 history로 재구성했다. 모든 model/Adam 값은 finite, 모든 저장 RNG/순서/cursor/history/전체 ancestry와 외부 V23readonly1,936call 장부는 보존됐다. 실제 CUDA 복구 재실행을 했다는 뜻은 아니다.

- Import0 658,442,673B SHA `70c53b482e5185d44e9c3d7c97b25468acb49123e8bd0d2c04270167914a829e`.
- Mid2048 772,113,285B SHA `80ab18ef471b9c977e587eaf9eef526dd6bf2d98119c5ff5e64ca45bda1b6ec4`.
- Final4096 772,498,629B SHA `97165ab6e7fcf9e6b75174b5b8eff80801c7f95925e49c3c8923b70cbd0d22ae`, algorithm `7babea2452ea480fba1e917ee9286b9b27cc31a64cf1a2d04cea07187ffc59e0`.

누적92,132attempt/92,099update이며 과거32AMPskip+V18실패attempt1을 소급 고치지 않았다. source 학습 장부280,937calls는 별도 종료된 V23readonly1,936calls를 제외하므로 합산282,873이다. 모든23,207저장 logits×3의 argmax/NLL, rawbyte/EOS, 실제 batch 마지막EOS 기반 AR호출 수를 독립 재구성했다.

검토 당시 Cfree 약42.41GB는 strict시작40GiB=42,949,672,960B보다 작았다. 과거 같은 문제의 해결 기록을 먼저 읽고 [Microsoft compact 문서](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact)를 확인한 뒤, 별도 종료된 V20–V26 생성 checkpoint21개만 한정300초/max24files18GiB 범위에서 투명한 무손실 LZX 파일시스템 압축했다. 실제85.141초/할당byte1,174,096,197절약/모든21개 전후 논리SHA·길이 exact/모든 ownedcompact종료/삭제0·이동0·직렬화변경0·모델호출0이다. 최신V27–V29/source문서/core/typed/사용자원본은 대상에서 제외했다. 새 artifact `research/artifacts/owt_checkpoint_lossless_space_20261029_v3`에 원문·경로·각SHA·실측 할당량을 보존했다. 마지막C43,348,738,048B로 REQUIRED40GiB는 회복했으며 soft40.5GiB 목표는 미달로 그대로 기록했다. 다른 OS디스크 사용 때문에 할당 절감량과 전체 Cfree증가는 같지 않다. 보호선은 내리지 않았다.

첫 추가 층 적응은 한 epoch뿐이고 온라인 loss가 계속 내려간다. 다음 한정 목적은 같은4block/퍼지/추론/손실/자료를 유지하며 **이미 populated된 세 Adam group81개를 전부 이어서** 두 유한 epoch를 학습하는 것이다. 새로 moment를 비우거나 층을 더 풀지 않는다. V30새worker·복구·CPU·실제preflight·독립OSvalidate·시작자원을 검토한 뒤 별도 새900초/max28,000call 예산만 실행한다. V29남은예산은 재사용하지 않는다. 모든 원본/실패/fullAdamRNG/원래 typed core 보존, E2final/registry 접근 금지를 유지한다.
