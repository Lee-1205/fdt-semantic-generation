# 원인에 대응한 V88 한정 학습 결과 — 2026-11-13

**실제 학습과 결과 검토를 수행했다.** 최대 192회 중 108회 시도, 16회 update, 92회 reject, AMP skip0이다. 사전 고정한 ‘32회 연속 수락 update 없음’ 조건으로 멈췄다. 192회 완료나 언어→의미→언어 완성으로 부르지 않는다. 원인을 좁히는 데는 유용했지만 의미 개선은 제한적이어서 기본 연구 참조 모델을 자동 승격하지 않는다.

## 변경한 것과 유지한 것

V81~V84 진단에서 확인한 두 문제에 대응했다. 출력기 계산을 고정 FP32로 바꾸고, 기존 TRAIN100 reference에 더해 **TRAIN 전체에서 처음 정답이면서 취약한 critical 위치**를 보호했다. 일반 byte는 top1이 유지되는 한 margin 감소를 허용했다.

학습 source는 기존 V80 fullstate다. 입력·기본/문장 fuzzy 메모리·역할 결합과 출력기 prefix5개 tensor는 고정했고, 마지막 Linear weight/bias 2개 tensor, 2,117,664개 parameter만 fresh Adam으로 학습했다. 기존 자료 1,024개를 유지했고 개발 문장·정답을 TRAIN이나 보호 bank에 넣지 않았다. 새 corpus 행이나 외부 다운로드도 없다. 기존 source의 Adam/scaler/모든 RNG는 원본 파일에 보존했다. 새 trajectory는 기존 optimizer의 남은 학습을 재개한 것이 아니다.

K_P/K_C 경로는 유지하며 K_E는 NULL/content0/copy0이다. 출력기의 정상 입력은 Commit이다. 원문, 정답 역할/span, 정답 prefix/길이, Exact 검색을 출력기에 넣지 않았다. 디노이징은 이번에도 0이다.

## 실제 실행 증거

- run: `runs/critical_repair_20261113_v88`
- Plan SHA: `a2cafb3de459e7759b47bbc03d829a48fe42455b0887439de6d3eb95af1ac347`
- source V80 SHA: `588ceaad4a94dde38ae56e42239497534d69387814a4fa66a4fbcea974db3cd2`
- native owner29688→31284, worker12096→6972의 생성·parent·command·exe·생존을 ROOT가 16:33:45 KST에 확인했다.
- worker 약609.053초. encode1,564 + readout1,888 + reference Linear393 = **3,845회**, max5,100 안에서 종료했다. 새 900초/모델 cutoff780초/종료 여유120초를 연장하지 않았다.
- supervisor는 작업 프로세스의 정상 종료를 뜻하는 `COMPLETE`, worker 학습 결과는 `EARLY_STOP_32_NO_ACCEPTED_UPDATES`다. 두 상태를 구분해 그대로 보존했다.
- ROOT가 16:44:29에 모든 owned/native/proxy 종료와 Python0을 확인한 뒤 분석했다. 이후 CPU 검토의 추가 모델 forward는 0이다.
- 미사용 1,255call, 최대 시도 중 미소비 84회, 남은 시간은 닫았다. 자동 재시작·같은 평가 반복·상한 확대를 하지 않았다.

최종 runtime 29,254,771B, SHA `ebe939a6c14b205bd92a6cad772b33cf2ee462473bb77daadcc6ba48926c84d7`, 재구성 fullstate SHA `bd3edf2d661231a571d6c0af8c45a2f725545793128ad114b263a9e2c06765dd`. standalone이 아니며 보존된 V74 standalone storage parent를 요구한다. 이 checkpoint 참조는 inference Exact/text 검색이 아니다.

실제 크기의 CPU worker에서 사전 4encode/4readout, 전 tensor warm 복원, 마지막2개 gradient 소유, initial/partial fullstate 복구를 검증했다. 이전 작은 CPU 엔진에서 populated Adam·enabled scaler·모든 RNG·reject·AMP skip·중단 복구를 검증한 결과와 구분한다. 실제 CUDA 중간 복구를 재실행한 증거는 아니다.

## 실제 학습량과 보호 범위

Teacher는 1,024행/1,002,559 UTF8 bytes를 한 번 보았다. Head는 108그룹×5행 = **540회 source 노출**, 475고유 행, 839,418 UTF8 bytes다. 행 하나의 최대 반복은 3회다. 540개의 새 문서나 192회 완료로 세지 않는다.

현재 V80 부모의 전체 FP32 teacher에서 정한 취약 critical 위치는 **152곳**이었다. 이 중 기존100 reference 안은15곳, 밖은137곳이다. 과거 V75/BF16 기록의 213/24/189와 부모·정밀도가 다르므로, 그 차이를 이번 학습의 개선량으로 해석하지 않는다.

실제 bank는 7,623patch, 초기 정답 보호119,769곳, 취약 critical margin 보호152곳이다. 초기 rowwise teacher와 batched reference의 정답 flag 불일치는0, margin 최대 차이는 약0.00008774로 사전 허용치0.001 안이었다. 최종 저장 결과에서도 **보호한 초기 정답 손실0, 취약 margin 하한 위반0**을 독립 재구성했다. 이는 유한 TRAIN bank의 보존이며 미학습 문서 전체의 보장이 아니다.

ROOT는 3개 fullstate SHA/safeload/재귀 digest, source 대비 마지막2개 외 모든 tensor의 bitwise 불변, optimizer step16/scaler tracker108, 저장된 모든 RNG, 유한 cursor/order/노출/history를 확인했다. Teacher 1,024개 packet의 SHA·원문 byte/EOS/PAD·critical span과 전체 취약 위치의 bank 편입도 재구성했다.

## 실제 문장 복원과 의미 검토

평가는 이미 관측한 64+64+64개 짧은 문장과 두 장문이다. 새 blind benchmark가 아니다. 입력·메모리·prefix가 고정돼 있어 같은 cached Commit으로 출력층 변화를 분리했다. 두 장문은 각각 262,042B/262,029B로, 이번에 입력 인코더를 새로 장문 실행한 결과는 아니다. 과거의 fresh-long/cached parity와 같은 고정 입력 tensor를 활용했다.

**짧은 192개 출력은 동일 정밀도 baseline과 전부 바이트 단위로 같았다.** 새64의 수치·단위 명확 보존은52/64로 그대로다. 명확한 값 변경4개와 비표준 접두사로 의미 미인증8개도 남았다. UTF8/EOS는 전체194/194다.

| 두 장문 source-bound 지표 | 학습 전 | 학습 후 |
|---|---:|---:|
| source bytes | 524,071 | 524,071 |
| decoded byte 오류 | 52 | 49 |
| 이동 수량 오류 /2,718 | 13 | 13 |
| 잔여 수량 오류 /2,718 | 2 | 2 |
| episode ID 오류 /2,718 | 8 | 7 |
| actor 이름 오류 /2,718 | 2 | 2 |
| item 표현 오류 /2,718 | 7 | 7 |
| `moved` 표현 오류 /2,718 | 5 | 5 |
| 위치 명사 shelf/drawer/box/gate 오류 /10,872 | 8 | 9 |

수량 두 종류를 합친15/5,436 오류는 그대로다. 이 위치들은 반복된 authored template에 속하므로 독립 사실 표본 5,436개라는 주장은 하지 않는다. 기존 제한된 색상·순서 score만으로 위치 단어까지 완전하다고 해석하면 안 된다.

ROOT가 달라진 **5곳의 원문/수정 전/수정 후를 모두 읽었다.**

- `did not removem in episode...`의 반복된 세 곳이 원문의 `did not remove; in episode...`로 복원됐다. 단어 끝과 경계가 회복된 것이며 독립 의미 사실 세 개를 새로 추론한 것은 아니다.
- `episode 1,59`가 원문의 `episode 1359`로 복구됐다.
- 반대로 원래 맞았던 `violet drawer` 한 곳이 `violet draber`가 됐다. 위치 단어를 훼손하는 새 오류로 보존하며, 무조건 의미가 같다고 면제하지 않는다.

두 장문의 시작·중간·끝 6개 구간도 원문과 대조했다. 대부분의 사건 구조·대상·순서는 유지됐다. 모든 524,071바이트를 수작업으로 읽은 검사는 아니며, 국소 핵심 오류가 남아 있으므로 장문이 완전히 보존됐다고 부르지 않는다. 장문 전체가 붕괴했다는 결론도 맞지 않는다.

NLL은 새64에서 약0.00681211→0.00676502, 장문에서 약0.00080919→0.00080397로 낮아졌다. 반면 나머지 두 짧은 패널 NLL은 조금 올라갔다. 정작 짧은 출력과 핵심 수량은 같으므로 NLL 감소를 의미 복원 성공으로 대신 세지 않는다.

## 왜 멈췄고 무엇을 해야 하는가

수락 update의 alpha는1.0이12회,0.5가1회,0.25가3회다. 마지막 수락은76번째 시도였다. 77~108의32회가 연속 거부돼 사전 조건대로 끝났다.

거부된92시도는 모두 가장 작은 alpha0.125에서도 **취약 critical margin 하한 위반**이 남았다. 그 작은 후보들에서는 실제 top1 flip은 없었다. 전체391개 후보 검사 중29개에서는 critical top1 flip도 있었고,375개에서는 취약 margin 위반이 있었다. 이를 ‘모든 거부가 곧 의미 오류였다’고 해석하지 않는다. 현재 목적 함수가 만든 실제 Adam 방향이 reference 제약과 충돌한 결과다.

최종 margin 여유가 가장 작은 위치는 TRAIN의 `Fayruz` 마지막 `z`였다. 초기 margin0.010353, 하한0.0051765, 최종0.0052719, 여유 약0.0000954다. 단위 `km`의 `m`도 여유가 작았다. 이들은 이름·단위라서 보호 자체를 불필요한 형식 검사라고 치부할 근거도 없다. 다만 **거부 후보의 위치 ID나 logits는 저장하지 않았으므로, 같은 `z`가92회 또는 마지막32회를 전부 막았다는 증거는 아니다.** 저장된 최종 상태에서 가까운 위치를 확인한 것까지다. 로그를 보강하려고 동일 학습을 다시 돌리지 않는다.

다음은 같은 방향의 SGD를 더 반복하거나 gate를 자동으로 낮추는 작업이 아니다. 실제 Adam delta를 원래 TRAIN 제약 안의 허용 방향으로 수정하는 방법, 또는 무엇이 보호되는지에 맞는 목적 함수를 먼저 CPU에서 검토한다. 마지막 두 Linear의 제약은 고정 prefix feature에 대한 선형 부등식이므로, 실제 delta에 대한 제한된 투영/보정과 rollback을 구체적으로 검증할 수 있다. 이 방법을 실제 연구 가중치에 적용해 성능을 얻었다는 뜻은 아직 아니다.

reference anchor는 보존해야 한다. V88에서 이미 줄어든 margin을 새 teacher 기준으로 다시 절반씩 낮추면 여러 실행에 걸쳐 보호 기준이 약해지는 문제가 생긴다. 다음 run이 V88을 warmstart하더라도 입력·prefix가 같은지 검증하고 원래 V80 FP32 reference anchor를 유지하는 계약이 필요하다. 새 시작을 핑계로 old anchor나 old budget을 초기화하지 않는다.

현재 선택은 V75의 balanced reference와 V80 고정 FP32 reference를 보존하고, V88을 보존 제약이 적용된 부분 연구 결과로 남기는 것이다. 16회의 유용한 가중치·Adam/RNG도 삭제하지 않는다. 새 위치 단어 오류와 남은 수치 오류를 고려해 자동 기본 승격은 하지 않는다.

## 디퓨전 판단

최신 사람 지시대로 필요하면 디퓨전도 진행할 수 있다. 이번에 입증된 원인은 출력층과 update 제약에 있으므로, 현재 수정 과제는 해당 경로에 둔다. 내부 의미가 완전히 형성됐다는 증거가 추가된 것도 아니다. 이후 의미 개선/사고를 학습할 때는 **입력·기본/문장/Exact 메모리·역할 결합·출력의 모든 가중치와 buffer를 고정**하고, 별도의 사고 부분만 학습한다. 지금 출력 오류를 사고 모델이 임의로 보정하게 만들거나 ordinary text 전체를 Exact로 복사하지 않는다.

## 보존과 증거 범위

시작 공간을 맞추기 위해 종료된 V80의 long raw 두 파일만 Windows `compact /EXE:LZX`로 투명 압축했다. 물리100,757,570B를 회수했으며 논리 크기·SHA·CPU safeload가 모두 같았다. 삭제한 파일은0개다. 관련 공식 명령 설명은 [Microsoft compact 문서](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact)다. 권한 차단된 삭제를 다른 경로로 반복하지 않았고 모든 가중치·raw·원본·실패·게임을 보존했다.

새 평가 기록은 공간을 줄이기 위해 모든 UTF8 상태의 neural 선택, raw argmax, gold log-probability와 선택한 patch의 전체258 logits를 저장한 **decision trace**다. 기존 full raw 파일은 유지했다. ROOT는 전체 trace의 UTF8 선택/IDs/출력을 재생하고, 3,360개 sampled byte 위치에서는 전체 클래스 score도 검증했다. 저장하지 않은 나머지 위치의 전체258 logits를 독립 재계산했다고 주장하지 않는다. gold는 score에만 사용하며 UTF8 선택에 영향을 주지 않는 CPU 검사도 통과했다.

주요 증거는 run의 `root_saved_review.json`, `root_meaning_and_rejection_review.json`, `root_decoded_panels.json`, `ROOT_terminal_native.json`과 최종 `root_result.json`이다. 실제 전후 output, source identity, teacher/bank SHA, fullstate와 불확실성을 함께 보존했다.
