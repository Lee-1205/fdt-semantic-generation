# V77 준비: 실제 update 비중40/40/20과 V75 숫자 가중치 고정

ROOT 직접. V76을 자동 채택하지 않고 명시적으로 선택한 V75 NN에서 시작한다. K_P 기본 퍼지/K_C 문맥 퍼지→Commit→언어,K_E ExactNULL/복사0과 상류 고정·출력기7tensor만 학습하는 경계를 유지한다. 새로운 의미 추출/디노이징 학습이나 성능 향상은 아직 없다.

이번 변경은 V76의 부호 문장75% update 편중과 숫자 base weight 하락을 다룬다. 단순한 loss scalar만으로 기여를 맞추지 않는다. 전체 파라미터의 결합 norm을 제한하는 [PyTorch clip_grad_norm_](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)과1·2차 모멘트로 갱신하는 [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)의 동작을 다시 대조했다. clip이 작동하는 구간에서 양의 scalar를 곱한 같은 방향 gradient의 크기 차이가 줄 수 있다는 계산에 따라,이번에는 실제 finite order의 update 횟수를 맞춘다. 이것도 optimizer 후 파라미터 gradient 기여가 정확40/40/20이라는 뜻은 아니다.

새 TRAIN 원문을 만들지 않고 기존 V76 TRAIN의1,024개 고유 입력을 재사용한다. Teacher는 고유1,024행을 한 번만 실제 인코딩하고 loss 전용 label/확률/정답 여부만 저장한다. Head는1,920번 실제 원문을 다시 인코딩한다.5개 update를 unsigned/signed/OWT/unsigned/signed 순으로 묶은384개 cycle이다. 부호768행은 각1회,unsigned128행은 각6회,같은 OWT128행은 각3회다. 총 unsigned768/signed768/OWT384=40/40/20%이며1,920개 새 문서나1,024행의 한 epoch라고 하지 않는다.

Teacher 원문1,010,943 UTF-8바이트,head 반복 노출2,720,189바이트/중요구간86,108바이트/EOS1,920개다. 반복횟수·order/cursor를 미리 고정했고 AMPskip도 해당 노출을 소비한다. 추가 epoch나 실패 행 되감기는 없다. Base258개 symbol weight는 V75 TRAIN에서 계산한 배열을 부모 runtime과 비트 수준으로 일치 확인해 고정한다. 현재1,024행에서 다시 계산한 숫자 weight2.8473은 쓰지 않고 V75의3.6495를 쓴다. 나머지 V76 capped 중요 오답/보존 손실은 그대로다. 데이터 선택과 baseweight를 함께 바꾼 실험이라 각각의 단독 인과 효과를 분리한 것으로 부르지 않는다.

평가64행은 V76의 동일 사실·이름·quartet을 유지하면서 END32행의 소유격+관사 중복만 고쳤다.32행은 실제 prompt가 같고32행은 편집됐다. 이를64개의 새로운 블라인드 미학습 사례로 세지 않는다. 원래 V76 자료·정답·점수는 그대로 보존했다. 새 평가 feature는 id/prompt뿐이며 gold는 사후 score 전용이다. 예전 관측340행/262KiB장문2개 보존과 실제 source/native-stream parity를 함께 본다.

CPU 단계에서 finite1,920order의 각 source 노출횟수·384cycle·JSON nextcursor·V75 symbol array 동일·수정64행의 의미 label/UTF8span을 ROOT가 별도 검토했다. Tiny engine은 unique teacher3/head반복5에서 populated7Adam/enabledCPUscaler/모든RNG/parent-linked 다음 update 완전 일치,overflow skip 소비,weight/source mismatch 거절,in-flight STOP 복원과 자동 재실행 금지,상류 고정을 확인했다. 실제 대형 CUDA복구 증거는 아니다.

최종 실제 worker CPU preflight는18.219초/optimizer0/CUDAfalse다. V75 linked 전체 NN 정확 warm/standalone V74 storage base/고정 symbol profile/대형 encoder2+readout2·only7gradient/initial과partialteacher 전체복구/실제 legacy packet 참조/long wrapper strict copy를 통과했다. 모든 CPU owned/proxy 종료 후 ROOT가998pins·두 실제 상태를 별도 재구성했다.

초기 CPU 준비에서는 과거 평가 feature 경로를 새 TRAIN 자료에서 찾는 metadata 오류가 잡혔다. 잘못된 경로/실패 stdout·stderr/상태/코드 snapshot을 보존하고 구체적인 과거 V75 feature 경로로 고쳤다. 구형 OWT feature에 원본 구간 provenance 필드가 함께 있다는 점도 반영했다. 새 TRAIN feature의 id/prompt-only 규칙은 그대로다. ROOT 검토 script의 지역 변수 shadow는 실행 전에 고쳐 새 execution_review_v3에서 최종 코드 pins로 다시 preflight했다. 이 준비 문제를 GPU 학습·모델 계산 실패로 세지 않고 과거 상태를 소급 고치지 않는다.

새 계획은1,920attempt 상한/unique teacher1,024/encode3,076+readout3,418+stream258=6,752calls,cap7,000/longblock2,322/denoise0다. 새900초 wall/780modelcutoff/120reserve,output hard1.25GiB이며 예상1,104,832,864바이트에 모든raw/teacher/initial·teachercomplete·final·추가STOP/임시·32MiB slack을 포함한다. 시작 C100GiB보호+1.25GiBoutput+8GiBheadroom=109.25GiB/GPU12GiBfree를 요구하고 기존 guard를 낮추지 않는다.08:13 C120,259,584,000바이트/약112.0GiB,GPU15,283MiBfree/RAM20,090,441,728바이트/Python0을 확인했다. 기존 V76의 미사용240calls·시간은 재사용하지 않는다.

현재는 actual schema/CPU·ROOT 준비 완료이며 봉인/실행과 성능 개선은 별도다. 새 시계의 ONE hidden launch 후 실제 native 생성/parent/command/생존을 관측하고 ROOT는 상세 분석을 중단해 OS감독만 한다. terminal뒤 모든 owned/proxy 종료부터 확인하고,관측 장문의 새 숫자 손상·잘못된plus와 단위 개선의 공존을 실제 출력으로 검토한다. 에이전트·자동 재시작·실행중 코드 수정/삭제·과거 예산 확장·Exact 문장 회수는 없다.
