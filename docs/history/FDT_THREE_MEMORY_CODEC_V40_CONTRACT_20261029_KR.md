# V40 두 순회 유한 원문 복원 학습 계약

같은 세 메모리 읽기와 언어→입력 의미→같은 원문 목표를 유지한다. V39전체 모델/91populated Adam/scaler/모든 저장 RNG를 정확 이관하고 새 NN parameter0이다. seed39001/39002,각TRAIN32768구간 한 번씩/합65536행선택5852956target UTF8bytes/8192attempt/batch8/AMPskip도 커서·노출 소비다. 같은8192문서 반복이며 새문서65536개가 아니다. 기존baseLR의 새8192stepcosine만 적용하고 Adamstep·moments는 초기화하지 않는다. denoise0/copy0/no raw H-byte bypass/Commit-only decoder 및K_PWriter·K_C전체층prefix·K_Eliteraltransport 미완성 범위를 그대로 둔다.

새 고정2100초/reserve180/cutoff1920/max30000호출/calculated27348=24576train+2772assessment다. 보수적5update/sec에서1639train+90startup+60checkpoint+60assessment+180종료=2029<2100. 모든 과거 시계와 미사용 예산은 CLOSED다. V39저장 결과를 기준선으로 써 초기·중간GPU평가는 반복하지 않고 마지막에 같은 teacher패널·80원문복원·3메모리 통제를 저장한다.

## 저장 정책과 대가

초기는 기존형식의 parent-required 참조로 저장한다. 보존된V39부모와 합쳐 전체model/Adam/scaler/ALLRNG/order/cursor/history를 정확히 복구한다. 이후에는 정상 최종 또는 중단 상태 한 벌의 self-contained fullstate를 보존하고 중간 fullstate를 생략한다. 삭제·회전·덮어쓰기는 없다. 따라서 최대8192attempt의 진행이 초기 참조 이후 내구 checkpoint로 확정되지 않을 수 있다.

최종 저장 뒤 상태보고 오류가 나면 model/Adam/scaler/RNG/cursor/history/호출을 포함한 전체digest가 동일한 경우에만 이미 저장된 최종을 복구점으로 사용한다. 허용하는 차이는 설명용worker_stage뿐이다. fullwrite 자체가 실패하면 후보 파일과 초기 참조·부모를 보존하고 현재 복구를UNCONFIRMED로 기록하며 두 번째 큰 파일을 쓰지 않는다. 이 정책을 모든 I/O실패에서 현재진행 완전복구 보장이라고 하지 않는다.

최대full800MiB+초기참조32MiB+평가128MiB+기타32MiB=992MiB<owned출력1GiB다. 보호20GiB+출력1+임시1+추가headroom8=새시작30GiB/GPUfree12GiB. 보호공간·headroom은 줄이지 않았고 과거32/35GiB계약은 바꾸지 않는다. checkpoint를 쓰기 전 실제디스크23GiB(기존22+임시1) 검사는 유지한다. owned출력의 다음 파일 추정800/32MiB를 free-disk 임시 여유와 별도로 계산해 작은 outputcap을 무효하게 하지 않는다.

ROOT12CPU PASS19.12초:실제plan/cost/storage정책/91Adam모델·RNG 이관/scaler skip·마지막7행포함전체복구/초기참조 exact next update/부모변조거부/최종후 보고오류재사용·상태변조거부·fullwrite실패후두번째쓰기금지/4종OSowner종료경로를 확인했다. 실제full-size V39 CPU safeload→새worker import→초기참조 safe serialization·full digest 일치도 독립OS --validate에서 PASS다. 실제full-size CPUforward/CUDA복구 재실행은0이다. 실제 Windows CPUownerfixture23596→25860의identity/자연종료/allgone PASS, owner Torchimport0.

실행하지 않은 원래plan_unsealed.json은 보존하고 plan_unsealed_v2.json을 현재 후보로 사용한다. 사용하지 않는7개 조상PTpin을 제외해 반복체크대상을8파일6.24GB→현재self-contained부모1파일784MB로 줄였다. 현재부모의 파일·algorithm SHA/원래core/실제runtime PY/data JSON검증은 유지하며 조상파일과 과거plan을 삭제하지 않는다. 모델·자료·학습함수 변경이 아니다.

CPU/OS준비 완료는 GPU실행·성능 증거가 아니다. 아직seal/실행clock/modelPopen/output없다. 현재GPU여유가12GiB시작기준 아래라 리소스 회복 후 기존 준비를 재사용해 새seal/ONEhiddenlaunch한다. 에이전트·하드웨어 재승인 질문·원본삭제·E2final/registry·자동재시작·실행중코드/상한변경은 없다.
