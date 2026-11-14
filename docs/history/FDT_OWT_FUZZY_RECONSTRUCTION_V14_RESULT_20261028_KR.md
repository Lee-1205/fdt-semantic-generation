# OWT 퍼지 추출·복원 V14 — 같은 자료에서 학습량 확대

기존 퍼지 앵커 기반 **문장→추출 상태→복원**을 유지하며4,096회 추가 학습했다. AE 누적8,192업데이트, 이번8유한 epoch/32,768행/2,913,424targetUTF8bytes, AMP skip0이다. 모델·추론 함수·훈련1,024문서/4,096문장·개발64문서/256문장은 그대로다. 추론/출력 의미 디노이징은 동결했고 원문 context의 디코더 직접 접근도 없다.

## 관측 결과

| 추출 상태 조건 | V13 NLL | V14 NLL | V13 teacher byte 정확도 | V14 teacher byte 정확도 |
|---|---:|---:|---:|---:|
| 정상 |1.744130|**1.666882**|49.2567%|**52.9065%**|
| 다른 문장 상태 |2.525478|3.933796|35.0024%|27.5779%|
| 상태 제거 |2.179626|2.920832|39.2080%|35.3083%|

정상 상태의 정확도는**3.6498pp** 증가했다. 잘못된 상태로 교환한 NLL 비용은0.78135→2.26691nats, 제거 비용은0.43550→1.25395로 증가했다. 해당 문장의 추출 정보에 의존하는 정도가 커졌다는 관측이다. 잘못된 상태에서의 점수가 떨어진 사실을 전체 조건의 성능 개선으로 부르지 않는다. 교사 강제는 정답 prefix를 제공하므로 자유 복원이나 보편적 의미 정답 점수가 아니다.

같은16문서의 자유 복원은 정상·교환·제거 모두 UTF8/EOS16/16이다. 정상 EOS는15/16에서 회복됐지만, **완전 복원은0/16**이고 비문·잘못된 내용이 남았다. 정상/교환 출력은 각각16종, 제거 출력은1종이다. 사후 normalized byte 편집 오류는 정상0.78895→0.79454로 조금 높아졌다. Teacher 성능 개선과 자유 복원의 정체를 모두 보존한다. 반복4word 비율0을 복원 성공으로 세지 않는다.

저장된 온라인 훈련 batch CE의 epoch 평균은1.5932→1.4965→1.3948→1.2934→1.2003→1.1184→1.0566→1.0164였다. 이것은 갱신 중 batch 손실의 평균이고 고정 최종 모델의 train평가 NLL과 같지 않다. 관측 개발 NLL의 개선폭은0.07725로 작아졌다. 이것만으로 학습량이 충분하거나 전체 방식이 실패했다고 단정하지 않는다. 다음은 같은 모델을 유지하면서 훈련 문서 다양성을 확대하는 한정 학습이 적절하다.

## 퍼지 상태와 보존

원래 Router의 비배타적 top-k sigmoid `mu`, probabilistic-sum union, 정규화 retrieval `rho`, anchors/prior/value/condition을 그대로 사용했다. Held 활성 앵커는11→13/64, 뷰당8개, 소속도 합 평균2.2677→2.2805다. 앵커 수의 증가 자체가 의미 품질의 증거는 아니며, 상태 의존성도 각 앵커의 해석 가능성을 증명하지 않는다.

V13의 모든 모델/63Adam/scaler4096 tracker96/CPU·CUDA·Python·NumPy RNG를 정확히 가져왔다. Import component digest `1bce66cb8bd3eb38c2f13622ed4235ab9e8ffd0c71a04edceda28b0374bcc024`는 source와 동일하다. V13local2048와 그 ancestry V12local2048를 함께 유지해 source의 누적4,096을 정확히 처리했다. 새 local4,096을 더한 최종 AE누적은8,192다.

최종63Adam state는 모두 step8192, scaler16384/tracker192다. 동결된 reasoning/diffusion/prefix/basis는 import와 동일하다. Source의 전체 history/order/exposure/기존 ancestry를 유지했고 새 clock/order/call allocation만 별도로 만들었다. 원래 데이터·모델·Adam/RNG·과거 실패를 삭제하거나 수정하지 않았다.

## 실제 실행과 검증

- 6focusedCPU delta: 실제 V13schema의 전체 component import·source 불변·재귀 누적 counter, 새 fullstate exact next update/마지막7행/모든 RNG/누적-새 호출 분리,8유한 epoch/cap, 새 owner의 postPopen 자연·예외·UNCONFIRMED 처리. 실제 worker CPU preflight·독립 OS validate도 통과했다. 이전 광범위 검사나 초기 GPUbaseline은 반복하지 않았다.
- ONEowner24640→43928 생성16:21:56.241/.248KST, worker22492→7644 생성16:22:04.196/.212KST. ROOT가 생성/parent/command/생존을 직접 확인했다. Worker/owner exit0/COMPLETE/allowned/proxygone 후16:26:34KST Python0을 확인하고 분석했다. 임시 sleepguard는 finally에서 해제된다.
- 실제257.860초, 새 encode4132+decoder5552+denoise0=**9684calls/max10500**. 새900초/reserve90/output6GiB 예산은CLOSED이며 남은816call/시간은 재사용하지 않는다. 종료 후 Cfree약57.55GB다.
- Fullstate는 import0/mid2048/final4096의3개를 완전 보존해 기존5개 저장 방식보다 디스크 소비를 줄였다. 삭제·rotation은 없다.
- Plan SHA `999ee9bfdacd20b97766c0bc7fdde195b39cdc812bbeaa74b212833e553e55ee`.
- 최종640,579,825B SHA `a9e2c3819b818cfc3e1d968d6ad515c9f136732786bb1ee39919d852f0f151d6`, algorithm `f66a4b10639db7b6bfbd1e3df0e5e40bf14dbf8e7b3597b6c9c1b574d159b4cb`.
- Root가 source/import 전체 component equality, 최종 SHA/safeload/digest·동결 tensor·63Adam/scaler·재귀 ancestry·8epoch 순서/cursor/노출·호출 수를 CPU 재구성했다. 저장23,207기호×3조건 logits/원래 UTF8 target·EOS·PAD·fuzzy degrees·raw 생성도 확인했다. 추가 모델 forward0이며 raw argmax 집계는 worker와 일치했다.

근거는 `fdt_local_102m_20261026/runs/owt_fuzzy_reconstruction_20261028_v14/root_result.json`과 저장 raw/fullstates/native 관측이다. 같은 관측 개발 자료를 여러 번 사용한 한계와 OWT원문 권리·중복·과거 pretraining 불확실성은 그대로 유지한다. 이번 다운로드는0, E2final/registry 접근도0이다. 전체 모델 실패나 추론 완성으로 포장하지 않으며 다음도 추출·복원 학습으로 진행한다.
