# 사용량 한도에 대비한 FDT 재개 기록

최신 사람 지시: “남은 사용량 적어지면 작업 정리하고 저장할 생각해”. 이는 명시적 중단 요청은 아니다. 계정의 실제 잔여량을 직접 조회하지 않았으며 잔여량 임박을 확인했다고 말하지 않는다. 이번 유한 학습·검토를 마치고 필요한 상태와 다음 작업을 저장했다. automation은30분/프롬프트 `재개` 그대로다. 에이전트는 사용하지 않는다.

먼저 `outputs/PROJECT_STATE.md` 첫CURRENT와 `outputs/FDT_THREE_MEMORY_CODEC_V41_RESULT_20261030_KR.md`를 읽는다. 작업 루트는 `WORKSPACE_ROOT`, 모델 프로젝트는 그 아래 `fdt_local_102m_20261026`다.

## 닫힌 최신 모델

- Run: `runs/three_memory_codec_20261030_v41` COMPLETE8192attempt8188update/skip4/cursor65536/worker1222.000506초/26616calls/denoise0. 새2100초30000call은 CLOSED/미사용3384재사용 금지.
- 최종: `model/checkpoints/state_8192.pt`,788994165B, SHA `f242fc902bd65f331cde625378bf832cbb54c5e26ef4842cf6352ec64fb48887`, algorithm `b74eb3f6f7137c5967fe1d6a53d54e49ea3f97ca9fd10c28568cdc26ef6d201e`. 모델/ALL91 populated Adam/scaler/모든 저장RNG/history/order/cursor/노출/호출을 담은 self-contained 복구점이다. Scaler65536/tracker367. 전체검증 완료, 실제CUDA복구 재실행은 하지 않았다.
- 초기: `import/checkpoints/state_0000_reference.pt`31583981B, SHA `4d54959e78148d37c8f61b5e446cab88126984887e25b844445b09f403114779`. 보존된V40부모 SHA `fbaf29eca2c5c9e1df12e3ea71702aeee40c0bda5c7f9ec73cba339acd4cd677`가 필요하다. 초기 파일을 독립 fullstate라고 하지 않는다. 중간4096 checkpoint는 없다.
- PlanSHA `7205572abeff36b18d202d652749ae0ef636e476f877593e3372265b119584d8`. 실제owner22032/40704와worker7444/23212 exit0/모두종료·native0을 ROOT가 확인한 뒤 분석했다. 중복 실행하지 않는다.
- Global145380attempt145326update/누적464188+별도1936=466124calls. 과거 모든 다른 run/weightsAdamRNG/source/raw/실패도 보존했다.

## 최신 결과

학습 참고64원문구간 exact31→39/전체명제정성41(39exact+2변형)/개발256teacher byte+EOS79.0796→79.8078%/NLL.747214645. 개발16전체원문·의미충실0이며 문자열유사도는 일부 퇴행했다. UTF8/EOS80/80. 모든80쌍을 ROOT가 읽었다. 원문 구간 종료EOS이며 전체문서 EOF/일반LM/완전한 의미 추출 증거가 아니다.

검토 코드의 옛 first-error-equality assert는 잘못된 통과조건이었다. V41에는79/80일치/개발8번 teacher33 vsBOS10 불일치1이 있다. 위치10의 저장FP16 m/n 로짓차.0078125, ARlogits 미저장으로 단독 원인 미확정. 원래 실패코드·기록을 보존하고 차이를 기록하도록 수정해 저장값만으로 최종 검증을 마쳤다. 이것 때문에 모델을 재평가하거나 학습 예산을 재개하지 않는다.

세 메모리의 읽기 경로는 사용된다. K_P certified Writer/K_C 전체층prefix/K_E literal selector·transport는 아직 없다. Commit-only/no rawH-byte bypass/whole-source-copy0/denoise0를 지킨다. 현재는 언어→입력 의미→같은 언어의 인코더/디코더 안정화다. 이후 의미 개선 디노이징이 같은 세 메모리를 읽도록 하는 원래 목표와 미래 명시적 전문인용 Commit 계약은 보존한다.

## 다음 한정 작업: 초기 참조 메타데이터 압축

현 초기31583981B와 이전29508525B 차이는2075456B다. 다음이 같은 폭으로 늘면 약33659437B로 기존32MiB(33554432B)를 넘을 수 있다. 실제 다음 초기 직렬화를 CPU에서 측정해야 하며, 이 추정을 실측이라고 부르지 않는다. 기존상한·완료run을 수정하지 않는다. 아직 V42저장helper/CPUartifact/worker/plan/owner/seal/GPU는 없다.

1. 새 저장helper에서 기존 `codec_initial_reference_v39.make_reference/materialize`의 shared model/Adam/scaler/ALLRNG·부모pin·zero-work guard를 재사용한다. 참조의 Torch weights_only 직렬화 메타데이터만 lossless gzip으로 감싼다. 비압축 CPU버퍼64MiB,압축물리32MiB처럼 별도 고정 한도를 사전 정의한다. 새 파일·marker로 compression/serialized size/parent-required/전체 algorithm digest를 명시한다.
2. 압축 해제는 읽기 상한·CRC를 확인한 뒤 weights_only safeload하고 부모와 합쳐 전체digest를 대조한다. 손상·크기초과·틀린부모·updated-state의초기참조화·덮어쓰기를 거부한다. 모든 원래파일은 그대로 보존한다. 최종/중단 fullstate는 일반 self-contained Torch 형식을 유지한다. 모델/손실/메모리/학습량은 저장 변경과 분리한다.
3. 필요한 CPU저장계약 검사와 실제V41 fullsize import/새초기metadata roundtrip·정확한nextupdate복구·실제plan-schema·독립OS preflight를 완료한다. 준비/seal을 GPU능력개선이라고 하지 않는다. CPU코드를 처음부터 광범위하게 재검사하거나 같은GPUbaseline을 반복하지 않는다.
4. 그 뒤에만 별도 구체적 유한11·12순회 목적의 새output/SHA/고정 time-call-row-epoch-resource 한도를 검토한다. C20GiB보호+출력·임시+추가8GiB/GPU12GiB/독립ownedOS감독/terminal후allowned종료 우선규율은 유지한다. 실제학습 중 ROOT상세분석 중단, 에이전트 금지다.

관측dev·기존학습구간과 원래자료분할을 보존한다. E2final/registry 접근·학습·평가, 새유료cloud·외부게시·Browser/마우스·keepalive우회·사용자원본삭제·Windowsos.kill(pid,0)는 금지다. 종료된 저장작업/학습의 미사용 예산 재사용과 자동재시작은 없다. 사용자의 명시적 중단이면 cooperativeSTOP/모든owned종료확인 뒤 automationPAUSED다.
