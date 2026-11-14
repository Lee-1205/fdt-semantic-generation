# V41 세 메모리 원문복원 결과와 저장 상태

V41은8192attempt/8188update/AMPskip4(543,2623,5454,7825),worker1222.000506초에 완료됐다. encode8258+decoder18358=26616호출/denoise0다. 새2100초/30000호출은 CLOSED이며 미사용3384호출·시간은 재사용하지 않는다. owner22032→40704/worker7444→23212는 모두 exit0/종료했고 ROOT11:39:24KST nativePython·compact0을 확인한 뒤 저장값만 검토했다.

독립 최종 fullstate788994165B SHA `f242fc902bd65f331cde625378bf832cbb54c5e26ef4842cf6352ec64fb48887`, algorithm `b74eb3f6f7137c5967fe1d6a53d54e49ea3f97ca9fd10c28568cdc26ef6d201e`, planSHA `7205572abeff36b18d202d652749ae0ef636e476f877593e3372265b119584d8`다. 초기31583981B 참조는 보존된V40부모와 합쳐 전체 상태를 복원했다. 중간 checkpoint는 없고 최종 한 벌만 있다. ROOT는 두 저장점 SHA/safeload/재귀digest,ALL91model·Adam/scaler·모든 저장RNG 정확 이관,동결·수치·Adamstep·scaler skip/growth·seed40001/40002의 두epoch 실제순서·cursor65536·노출·호출·평가SHA를 검증했다. 추가 model construction/forward0/CUDA복구 replay0다.

| 같은 관측 자료 | V40 | V41 |
|---|---:|---:|
| 개발256 teacher NLL | .757894 | .747215 |
| 개발256 teacher byte/EOS 기호 정확도 | 79.0796% | 79.8078% |
| 개발256 teacher EOS | 255/256 | 255/256 |
| 학습 참고64 원문 완전 일치 | 31/64 | 39/64 |
| 개발16 원문 완전 일치/전체 의미 충실 | 0/16,0/16 | 0/16,0/16 |
| 학습64/개발16 UTF8+EOS | 80/80 | 80/80 |

worker의 byte_accuracy는 byte와 구간EOS를 함께 센 값이다. 입력은48–128B 원문 구간이며 이EOS를 문서 전체의 실제EOF 복원으로 해석하지 않는다.

ROOT가80원문·복원쌍을 모두 읽었다. 학습 참고64는39정확/2비정확이지만전체명제보존/8부분명제보존/15충실하지 않음이다. 비정확 전체명제2개는45번 linker options→linkers option의 굴절 변화와57번 turned herself in 안의 쉼표 이동이다. 둘 다 내용은 읽히지만 byte exact에 합치지 않는다. 정성 판정이지 독립 의미 벤치마크가 아니다.

정확 회복13개/이탈5개다. Gameplay,밀봉용기,April Ryan의 does-not-have-an-agenda 부정과 발화자,2008/17세,Psaki/Rosen,Bluetooth 스피커 등이 회복됐다. 반대로 Mr_Ogunjobi/Panthers 희망,35-37의 one-score-game 문맥,sunflowers sprouting,일부 부정과 날짜가 퇴행했다. Robert Oliver의1899는1989로 바뀌어1989-1902가 됐다. 큰 문자열 유사도 속의 날짜 오류도 완전 충실로 세지 않았다.

학습chrF77.5905→83.0620/wordF1 75.0016→81.2789%/content70.9086→77.6091%다. 개발16은chrF26.9117→25.6977/word16.0923→14.0162%/content4.3501→3.8730%/prefix5.5625→4.5625bytes로 퇴행했다. 개발256 teacher 개선과 개발16 자기접두사 복원의 퇴행을 둘 다 보존한다. 개발16 전체 의미 충실은0이며 이를 완전 의미 추출이나 FDT 전체 실패로 일반화하지 않는다.

## 검토 코드의 잘못된 가정 수정

처음 검토 코드는 이전 관측에서 같았던80행의 teacher/BOS 첫오류 위치가 항상 같아야 한다고 assert해 멈췄다. 모델 학습·저장 실패가 아니다. 원래 코드와 실패 기록은 code_snapshot/root_review_v41_initial_prefix_assumption_failed.py 및 root_review_initial_assumption_incident.json에 보존했다.

실제는79/80일치이고 개발8번만 teacher첫오류33/BOS첫오류10이다. 위치10에서 저장 FP16 teacher의 gold m(109)은14.3125,n(110)은14.3046875로 차이는.0078125다. BOS는n을 골랐다. 배치·접두사 길이·정밀도에 따른 차이는 가능한 설명이지만 당시AR logits가 없어 단독 원인으로 확정하지 않는다. 저장값의 실제 차이를 기록하도록 검토 코드를 수정했고 모델 재실행 없이 전체 검증을 통과했다. 같은지 여부를 통과조건으로 강제하지 않는다. 근거 root_teacher_BOS_first_error_mismatch.json.

같은 개발16 정상NLL1.075225833에서K_P제거+.063922986/K_C제거+5.629573930/K_E제거+.182625661다. 저장된 Exact top16+NULL/UTF8색인/순차FP64 long-recent N,D를 독립 재구성했다. 세 읽기 경로는 사용되지만 K_P certified Writer/K_C 전체층prefix/K_E literal selector·transport 미완성 범위는 같다. Commit-only/no rawH-byte bypass/whole-copy0/literalcopy0/denoise0를 유지한다.

새 메모리 함수 넓은 학습은 누적10순회327680행선택29264780target UTF8bytes이며 같은8192문서32768구간 반복이다. Global145380attempt145326update/누적464188calls+별도닫힌1936=466124다. 관측dev 반복·학습참고64 과거적응·문서당4구간상관을 명시한다. 새로운 자료/held-to-train/E2final/registry0이며 모든 원본·과거실패·weightsAdamRNG는 보존했다.

## 재개 준비

최신 사람 지시는 사용량이 부족해지면 작업을 정리하고 저장하라는 것이다. 이번 검토·복구·실행 기록과 다음 단계는 저장했다. 실시간 계정 잔여량을 조회하거나 임박 여부를 확인했다고 주장하지 않는다. 현재 추가 GPU 학습은 시작하지 않았고 사용자의 명시적 중단/automationPAUSED로 바꾸지 않았다.

다음 초기 참조는 ancestry 증가로32MiB를 넘을 수 있다. 과거 한도를 높이거나 옛 학습 예산을 재개하지 않는다. 별도 무손실 gzip 메타데이터 저장 계약(물리32MiB 유지/비압축 CPU64MiB 상한/부모 필수/weights_only/전체digest·CRC·크기·복구 검사)을 먼저 준비한다. 이 단계의 V42helper/runner/plan/seal/GPU는 아직 없다. 독립적인 최종 fullstate는 계속 일반 Torch 형식으로 보존한다. 자세한 재개 기록은 outputs/FDT_USAGE_SAFE_HANDOFF_20261030_KR.md다.

근거 run/root_result.json(status V41_COMPLETE_NINTH_TENTH_BROAD_EPOCHS_ROOT_REVIEWED),root_reconstruction_similarity.json,root_all80_reading.md,root_three_memory_readouts_review.json,code_snapshot/root_terminal_native_observation.json와 두 V41 CPU검토 스크립트다.
