# V36: 같은 언어 codec의 두 번째 넓은 학습 순회

사람의 최신 지시인 계속 학습을 수행한다. 언어→입력 의미→동일 원문 복원, 기본 퍼지K_P·문장별 퍼지K_C·ExactK_E, Commit만 받는 디코더를 유지한다. 디노이징·원문 전체 자동 복사·raw H/원문 바이트의 디코더 우회는0이다. V35 모델779648693B/SHAb8c32a8d3517152807ba9630509c903d6adb0f7703ebe3a4dfa188744612fe7f의 모든 가중치·91개 populated Adam·enabled scaler·저장된 모든 RNG를 정확히 가져온다. 새 파라미터/모멘트/optimizer step 초기화는 없다.

기존 불변 OpenWebText TRAIN8192문서/32768구간의 두 번째 넓은 유한 순회다. 새 seed35001, batch8,4096attempt/32768행/2,926,478target UTF8bytes. AMPskip도 행/attempt를 소비하며 되감지 않는다. 기존 개발은 학습에 넣지 않고, 새 corpus 다운로드/E2final/registry 접근은 없다. 동일한4개 그룹의 base LR에 새 순회의 유한 cosine 일정을 적용한다. **학습률 주기는 새로 시작하지만 Adam의 누적 모멘트와 step은 그대로 유지한다.**

## 새 실행의 시간 계산

이전 CURRENT의2048+2048 분할은 아직 코드·plan·실행이 없는 제안이었다. 이번에는 저장 중복/두 번의 시작 비용을 줄이고, 측정 시간에 맞춘 **별도 신규1200초** 실행으로 한 순회를 완료하도록 제안을 교체했다. 변경은 새 seal/모델 Popen 이전이며 과거900초/480초 실행의 시계나 한도를 연장하지 않는다.

- V34 실측 약5.46updates/초보다 보수적인5.0updates/초로 계산:4096/5=819.2초→학습820초.
- 시작60초+checkpoint60초+최종평가60초+종료reserve120초를 더해1120초.
- 새 고정 owner1200초, modelcutoff1080초, reserve120초. worker는 기존90초 최종평가 확보 규칙도 유지한다. 처리 속도가 예상보다 느리면 상태를 보존하고 고정 상한 안에서 종료하며 실행 중 상한을 바꾸지 않는다.
- 최대17000호출/코드최대15060,출력6GiB,C시작35GiB(20보호+6출력+1임시+8추가여유),GPU0free12GiB. import0/mid2048/final4096+필요시emergency fullstate를 보존하며 삭제/회전하지 않는다.

V35의 저장된 최종 평가를 같은 함수의 기준선으로 쓴다. 초기·중간 GPU 평가를 반복하지 않는다. 순회 끝에서 기존 학습64/개발256teacher 보조지표, 학습64/개발16의 입력 의미만 받는 BOS 재복원, 개발16메모리 제거 통제를 저장한다. ROOT가 원문과 출력80개를 읽어 이름·수치·부정·관계와 부분 명제를 확인한다. teacher 정확도를 의미 보존율로 부르지 않는다.

필요한 CPU7개검사PASS24.54초: 전체91개/model/scaler/RNG 이관, 새 유한 LR 주기와 Adam step 유지, enabled CPU scaler skip/마지막7행/정확한 next update를 포함한 full restore, 실제plan/timecost/유한순서, 네 가지postPopen종료 경로. 실제 전체 모델 worker CPU preflight도 독립OSowner--validate로 통과했다. 새 owner의 Windows CPU fixture에서도 native identity와 전체 종료를 확인했다. GPU 학습의 실행·성능 증거는 실제 native worker/진행/종료 후 별도로 기록한다.

K_P의 certified Writer, vNext 전체 층별 prefix read-before-write, Exact literal selector/transport는 여전히 미구현이다. 현재 부분을 전체 vNext·완성된 의미 추출로 포장하지 않는다. 미래의 명시적 Commit 전문 인용 계약을 보존한다. ROOT 직접 수행하며 agent는 사용하지 않는다. 모든 이전 원본/모델/Adam/RNG/실패를 보존한다.
