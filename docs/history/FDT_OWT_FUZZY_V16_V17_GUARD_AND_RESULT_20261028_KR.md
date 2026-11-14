# V16·V17 — AMP 누적 중단 수정과 자유 복원 재확인

원래 퍼지 앵커 기반 문장→입력 의미 추출→재복원을 이어갔다. 추론/출력 의미 디노이징은 동결했고 decoder는 추출한 두 fuzzy view만 받는다. 모델·자료·추론 함수·원래 비배타적 mu/union/rho는 같다. ROOT가 직접 처리했으며 에이전트·새 다운로드·원문 context 우회·새 parameter·anchor reset·반복/균등화 벌점은 없다.

## 실제 중단 원인과 수정

V16은8,192attempt를 계획했으나6,867attempt/6,864update/3AMPskip에서 이전 `self.skips>=3` 조건으로 중단됐다. Supervisor의 실제 FAILED/workerexit2 기록은 그대로 보존했다. 이 결과를 COMPLETE로 바꾸지 않았다.

실제 AMPskip은2,613/4,719/6,867회에 있었고 간격은2,106/2,148회다. 각각 손실.90914/.73488/.63471이었고 scaler가131,072→65,536으로 낮아졌다. 중간의 성공적인 갱신은 계속됐으며 저장 모델·Adam의 모든 floating tensor가 유한하다. 즉 연속 수치 이상을 검출한 것이 아니라, 긴 run에서 멀리 떨어진 세 번의 scale backoff를 누적해 중단한 코드였다. [PyTorch 공식 AMP 설명](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)도 비유한 gradient에서 optimizer step을 건너뛰고 scaler를 갱신하는 동작을 설명한다. 이 사건으로 의미 형성이나 모델 전체 실패를 주장하지 않는다.

V16 emergency fullstate의6,864회 유용한 갱신과63Adamstep19,152/scaler65,536 tracker0/모든CPU·CUDA·Python·NumPy RNG/history/order/cursor54,936/전체 ancestry를 보존했다. Gradient는 비어 있고 skip된 세 행도 소비된 기록을 유지했다. 새 실행에서 해당 행이나 미사용1,325attempt·5,066call·시간을 재사용하지 않았다.

새 V17 엔진은 **3회 연속**, **16attempt 안에3회**, **새 allocation에서 누적8회**를 중단 조건으로 고정했다. Source 끝의 history도 연속·window 검사에 포함한다. Nonfinite loss, 성공 갱신의 nonfinite gradient,20배loss 급증, 동결 tensor·gradient 소유권, disk/call/deadline/process STOP은 유지했다. 원래 V16 봉인 코드는 수정하지 않았다. 모델·Adam·scaler의 역사나 정밀도 방식을 초기화하지 않았다.

9focusedCPU 계약은10.58초에 통과했다. 실제 V16schema import/복구/모든RNG/마지막7행/새·누적 budget/ownedOS cleanup, 실제 skip간격과 집중·연속·전체 cap, source 경계를 잇는 검사, enabledCPU GradScaler에 강제Inf를 주어 weight·moment 갱신을 건너뛰고 exact next restore가 되는 것을 확인했다. 실제 worker CPUpreflight·독립 OSvalidate도 통과한 뒤 새 V17을 한 번 실행했다.

V17은**별도2,048attempt/2,048update/AMPskip0/2유한 epoch251–252**를 완료했다. 이 run에서 세 번의 GPUskip을 재현했다는 주장은 하지 않는다. 수정 판정은 저장된 실제 V16history와 CPU 회귀로 확인했고, 수정된 엔진에서 실제 추가 학습과 종료를 확인한 것이다.

## 관측 품질: 부분 구별은 유지되지만 생성은 정체

아래는 V15의 마지막 저장 평가와 V17을 비교한다. V16에서는 최종 평가를 하지 못했으므로 차이는 **V16의6,864회+V17의2,048회** 학습을 합친8,912회 효과다. Guard 수정만의 효과나 V17만의 효과로 분리하지 않는다. 같은 기존64개발문서/256문장/23,207기호이며 새 blind benchmark가 아니다.

| 지표 | V15 | V17 |
|---|---:|---:|
| 정답 prefix 기반 byte 정확도 |58.4177%|58.4134%|
| 정상 NLL |1.431893|1.665974 **악화**|
| 정상 자유 chrF2 /100 |21.5155|21.4314|
| 정상 자유 wordF1 |13.4077%|9.6185%|
| 정상 자유 content-word exactF1 |0.5208%|1.1117%|
| 자기 원문 chrF 단독1위 /16 |9/16|11/16|
| 처음부터 맞는 byte 평균 |3.125|2.750|
| 자유 완전 복원 / ROOT 충실한 paraphrase 판단 |0/16·0/16|0/16·0/16|
| 자유 UTF8/EOS |16/16|16/16|

일부 내용어와 문서별 구별은 늘었지만 전체 자유 복원이 좋아졌다고 말할 결과는 아니다. ROOT가 새 정상16개도 모두 읽었다. Robby는 이번에 정확히 남았으나 인물·스카우팅·시점 관계는 없어졌다. 그 외에도 처음 철자 흔적과 짧은 주제 조각은 있으나 행위·개체·부정·관계가 빠지고 비단어가 이어진다. 이것은 비맹검 직접 판단이며 보편적 의미 정답률이 아니다. chrF/wordF1/편집거리도 사후 표면 유사도이며 의미 성공 확률로 사용하지 않았다.

정상 free chrF21.43은 상태교환15.55/상태제거16.08보다 높다. 정상·교환 출력16종/제거1종, swap 자기원문1위0/16이다. 추출 정보에 반응한다는 증거는 유지된다. 그러나 이것만으로 완전한 의미 상태나 충실한 언어화를 증명하지 않는다. Held 활성 anchor는13→10/64,뷰당8/소속도 합 평균2.15646이며 단독으로 의미 collapse나 균등화 필요를 단정하지 않는다.

## NLL 악화를 저장 logits로 분해

추가 forward 없이 같은 개발 위치를 짝지어 계산했다. V15/V17 모두 맞은 위치12,114개에서는 NLL.2591→.1404로 좋아졌다. 둘 다 틀린8,208위치에서는3.2299→4.0738로 악화됐고 전체 NLL 증가에+.29847nats를 기여했다. 새교정1,442위치와 새퇴행1,443위치가 거의 상쇄됐다.

틀린 위치의 평균 argmax확률은47.96→60.94%,전체 평균 최대확률64.94→75.47%로 증가했다. 같은 관측 dev에서 오답에 대한 확률 집중이 커졌다는 구체적 문제다.10bin 사후 ECE는.0652→.1706이며 bin선택에 의존하는 기술 통계다. 이를 독립적인 일반화 평가나 전체 방법의 실패로 확대하지 않는다. 더 큰 scale이나 잘못된 AMP update 때문에 생긴 NaN 문제도 아니다. Source V16과 final V17의 모델·189Adam floating tensors는 모두 유한했고 V17skip0이었다.

V17 온라인 batchCE epoch평균.7474→.5575는 갱신 중 값이며 고정 최종 모델의 trainNLL이 아니다. 전체 고유 훈련 문장은728,913UTF8bytes/2,048문서로 여전히 작다. 따라서 학습이 충분했다고도, 기존 관계 예측의 높은 성능이 사라졌다고도 단정하지 않는다. 다만 지금은 동일 objective의 반복만 더하기보다 **추출 상태에서 prefix 오류를 견디며 복원하는 학습**을 별도 후속으로 조사할 이유가 생겼다.

## 실행·예산·보존 근거

- V16:새8epoch 계획 중6,867attempt/6,864update,54,936행/4,889,157targetUTF8bytes,encode6,867+decoder6,867+denoise0=13,734/max18,800,실제413초경중단. 예산CLOSED/미사용5,066call·시간 재사용금지. Plan65180e264b9b4353fd7d0f8c0ab469a8c2ac97df7a7e71f0d83bead29909b7dc. Emergency642,351,153B SHA272ecc6b6ee6846e539bbdbbf0f6ff3947c7b84afea9e51a1b2e6dbdee5e53fb,algorithm0343c73f7f7103deceed2b1c03c6f461571c8f01194b50288bb278823bf9fd28.
- V16 owner26540→31772/worker29668→18960의 생성/parent/command/생존을 확인했고 전부 종료 후17:08:26Python0/Cfree53.50GB에서 CPU분석했다. 실제 FAILED/incident/raw/fullstates를 보존했다.
- V17:새2epoch16,384행/1,457,826targetUTF8bytes,기존1,024doc군728,356+다른1,024doc군729,470. 새900초/reserve90/max6,500calls/output6GiB 중 실제143.260초/encode2,084+decoder3,488+denoise0=5,572calls였다. CLOSED/미사용928call·시간 재사용금지. 두 run의 예산·시간·호출은 별개다.
- V17 owner16356→36168 생성17:14:15.037/.044KST,worker31364→41480 생성17:14:23.960/.975. ROOT17:14:32생존/parent/command 확인. Worker-ownerexit0/COMPLETE/allownedproxygone,17:16:52Python0/Cfree51,469,099,008B 확인 후 분석했다. 임시 sleepguard는 종료시 해제됐다.
- V17 plan9d4456da7f3b8a347b041ad1fdffdf171b5feacae0d48ff64c8ca87083f99819. Final642,817,649B SHAd3896e120ff3ed4d684437b541e3d93599b9524a81a005a51af9fb4a1ad56d8e,algorithm2c36b4dc2b05c88b4e0ced59077a8e547ccb6087b9314315a116880da767b485. Import0/mid1024/final2048 모두 완전 보존했다.
- Final63Adam 모두step21,200/scaler131,072 tracker48. Source-import component SHA2fc67cd4ffbefa7e12507a3ad8842d253273cfec870382febf56a01d95353ce5 동일. 모든 모델/Adam/scaler/RNG/전체nestedancestry/history/orders/exposure/calls를 보존했다. 전체 AE21,203attempt/21,200update/역사적skip3이다. V12–V17 AE누적 target노출15,082,395bytes이며 BPEtoken·새독립문서 수가 아니다.
- Root가 fullstateSHA/safeload/digest/동결 reasoning·prefix/63Adam/scaler·2epoch복구·노출·전체 저장 logits/NLL/argmax/raw생성·유사도를 CPU 재구성했다. 추가 model forward0,정확도 argmax worker와 일치. Source/final Adam finite 확인과 확률 분해도 저장 logits만 썼다.

근거:V16 `root_stop_review.json`,V17 `root_result.json`/`root_numeric_confidence_review.json`/`root_free_similarity.json`/`root_all16_free_reading.md`/native JSON. 현재 모델 프로세스는 없다. 원래core/typedpartial9/a877/모든source·weightsAdamRNG·실패·raw·immutable자료를 보존했다. E2final/registry 접근0,새 다운로드0,외부 게시0이다.
