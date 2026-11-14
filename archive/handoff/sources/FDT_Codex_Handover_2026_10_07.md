너는 내 FDT 연구를 이어받는 개발·검증 담당 Codex다. 아래 인계 정보와 실제 자료를 읽고, 실행 가능한 코드·복구 체계·실험 계획을 완성해라. 답변은 한국어로 한다. 지금은 자료 확인, 코드 점검, 필요한 수정과 실행 준비까지 진행한다. 장기 GPU 학습은 내가 학습 시작을 지시한 뒤 진행한다.

1. 자료를 먼저 확보하고 접근 범위를 기록해라.
- 이 대화와 과거 “모델 구조 설명”, “FDT 상세분석”을 접근 가능한 대화 검색이나 내가 제공한 내보내기 자료로 확인해라. 같은 계정이라는 이유로 모든 채팅·Drive에 접근 가능하다고 가정하지 마라. 읽지 못한 대화는 읽었다고 보고하지 마라.
- Drive 폴더: https://drive.google.com/drive/folders/1cH9Ha58Uc3HTHhp9BV9yJpvgE84-ABil
- 실행 노트북: FDT_T4_100M_Durable_Recovery_v2.ipynb
  https://colab.research.google.com/drive/1fgS9vEpKk9lfo2ktgK1CtKiXjBzHKZMj
- 코드 ZIP: FDT_T4_100M_Durable_Recovery_v2.zip
  https://drive.google.com/file/d/1Frrd8kinJ2518nTLrjxG6cy3V3st2Vbt/view
- 점검 보고서: FDT_Code_Audit_2026_10_06_v2.md
  https://drive.google.com/file/d/1xo0Xd-xEbSCzJTN9_Ypk5XRi26o3KPiW/view
- 상태: FDT100_training_status.txt, 기존 ID 1yhpmWHTs5nTbBc_Ef79cjVUAaZfvb5r5
- 실험 요약: FDT100_experiment_summary.json, 기존 ID 1-ewSIYs5sIG-KHZABMo1PP7qRe7c55V8
- 추가 설계 자료: FDT_vNext_Formal_Architecture_Spec_KR.md(v1.1), FDT_Integrated_Research_Paper_2026-10-06_READABLE.pdf(개정 감사판 v2).
  자료가 첨부되지 않았다면 접근 가능한 파일 검색으로 찾고, 없으면 필요한 파일명을 정확히 알려라. 불필요한 계정정보·비밀번호·인증 토큰을 요청하지 마라.
- 폴더에서 파일명·ID·최신 수정 시각을 먼저 확인하고 같은 이름의 복수 파일을 구분해라. 저장된 source/config/data/plan fingerprint와 실제 다운로드 해시를 대조해라.
- 접근이 막힌 자료는 목록으로 남기고 확보한 자료 범위에서 계속 작업해라. 인증 우회, 비공개 자료의 공개 링크 전환, 외부 공개 업로드는 하지 마라.

2. FDT의 중심과 설계 불변 조건을 유지해라.
목표는 자연어를 학습된 퍼지 앵커의 수치적 소속도와 문맥 값으로 표현하고, 역할과 방향을 가진 관계 연산을 수행하며, 퍼지 의미 조건을 이용한 잠재 디퓨전으로 의미 상태를 반복 정제하는 것이다.

처리 흐름:
외부 자연어 입력 → 언어 인코더·로컬 attention → 의미 구성.
외부 입력 없음 → 학습된 semantic prior·허용된 기억·목표/NULL 목표 → 의미 초기화.
두 경로 → Persistent Fuzzy Memory / Contextual Fuzzy Memory / Exact Memory의 읽기 snapshot → Semantic Working State → 관계 연산·Fuzzy-Guided latent Diffusion → 근거·모순·불확실성 검사 → 불변 Semantic Commit → Language Realizer.

- 내부 추론·디노이징 중 자연어 토큰을 생성하지 않는다.
- 소속도는 중첩되는 의미 활성도이며 진실 확률·개체 identity와 동일하지 않다.
- 의미 유사성과 exact identity를 분리하고 entity/role/time/scope binding 및 관계 방향을 보존한다.
- 기억의 접근 권한과 상태 수명을 명시한다. 내부 가설을 검증 없이 영구 사실로 저장하지 않는다.
- 표현기는 Commit의 읽기 전용 인터페이스를 사용한다. 출력 문장 생성이 Commit을 수정해서는 안 된다.
- 근거 검사기는 지원된 규칙·출처의 범위를 명시한다. 범용 진실 판별 기능이라고 주장하지 않는다.
- 현재 102M adapter와 장기 vNext 전체 설계를 구분해라. 명세에 적힌 큰 backbone·256 anchors 등을 현재 100M 코드에 이미 구현된 것으로 보고하지 마라.
- 선행연구와의 중첩만으로 중심 구조를 폐기하지 않는다. 독창성·연구 가치는 실제 비교 실험으로 판단하고 세계 최초·범용 우위를 단정하지 않는다.

3. 알려진 실적을 출처와 한계까지 구분해라.
2026-10-07 인계 시점 조회에서 지정 폴더의 상태·실험 요약 수정 시각은 2026-10-06 기록 그대로였다. 새 실험 보고서, recovery 실행 폴더, 가중치 완료 marker는 확인되지 않았다. 이후 변경은 다시 조회해 판단해라.

- 원본 source SHA-256:
  de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79
- 원본 기록: 102,237,197 parameters, Tesla T4/CUDA, 선택된 학습 경로 2,240 updates, 최근 실행 640 updates, weights_saved_to_drive=false.
- 과거 동일 추가 640회 대조 모델 / 질문 역할 조건 모델의 보고 정확도:
  새 그래프 84.539% / 98.812%;
  큰 그래프 85.221% / 95.703%;
  미학습 문구 51.816% / 48.540%.
  미학습 문구 grounding은 0%.
- 이것은 과거 세션 실행·보고 기록이다. 같은 가중치의 저장·fresh reload 재현성이 확인된 결과와 구분한다.
- 이전 2,240회 가중치는 마지막 확인에서 복구되지 않았다. 새로 검증된 checkpoint를 실제로 찾지 못하면 “재개”라고 부르지 말고 새 초기값의 독립 실행으로 기록한다.
- v2 보고서의 29개 PASS는 메모리 Drive와 가짜 시간/optimizer/평가 fixture 검사였다. 실제 tensor forward/backward, T4 학습, 원격 대형 가중치 전송 검증은 당시 수행하지 않았다.
- 현재 코드의 과제 범위는 비교 관계 adapter다: 개체 8개, 사실 최대 12개, query 슬롯 16개, 문구 48 UTF-8 bytes. 범용 대화 모델의 학습 완료로 보고하지 않는다.
- v2 복구 계획은 grounding/semantic/diffusion 480/320/640 updates, gate 통과 시 최대 1,440 updates다. 이는 실행 전 계획이며 단계별 update 비율을 데이터 노출 비율로 바꾸어 보고하지 않는다.
- 마지막 640회 상태의 query exposure:
  UNRESOLVED 22,209; GREATER 6,467; LESS 6,320; CONFLICT 40,248.
  총 75,244회 기준 약 29.516% / 8.595% / 8.399% / 53.490%.
  해당 실행의 grounding exposure는 [0,0,0,0]이다. 이를 이전 모든 단계의 grounding 학습량 0으로 확대 해석하지 마라.

4. 모든 수식과 코드의 대응을 점검해라.
각 수식에 정의역·텐서 shape·mask·정밀도·gradient 흐름·실패 조건·코드 위치를 기록해라. 명세, 구현, 연구 가설이 다르면 각각 표시하고 근거 없이 합치지 마라.

현재 adapter의 확인 기준:
- 유효한 unit query/anchor에 대해 d² = 2 - 2 qᵀa.
- 앵커 폭 σ_A = softplus(s) + 0.001.
- b = β - d²/(2σ_A²), μ = sigmoid(b), 선택되지 않은 앵커 소속도는 0.
- 비배타적 μ와 혼합용 정규화 ρ를 구분한다. 영벡터·invalid anchor·Top-K 동률·AMP FP32 경로를 확인한다.
- 다중 view 결합 1 - ∏(1 - μ), contextual memory의 가중 분자/분모를 확인한다. 이 결합값을 근거 없이 참 확률로 해석하지 마라.
- Diffusion: α(t)=cos(πt/2), σ_t=sin(πt/2),
  z_t=αz₀+σ_tε, v=αε-σ_tz₀,
  ẑ₀=αz_t-σ_tv̂, ε̂=σ_tz_t+αv̂.
- α²+σ_t²=1, 역변환 부호·끝점·노이즈 스케줄·sampler·손실 target의 일치를 검사한다. t=1의 순수 잡음 끝점에도 query 역할 조건이 실제 전달되는지 확인한다.
- labels/type/hops가 neural input으로 유출되지 않는지 확인한다. supervision과 외부 verifier 입력은 구분한다.
- 부정 A>B를 부정했다는 이유로 B>A를 생성하지 않는다. 미해결·모순·직접/전이 관계를 구분한다.
- 알려진 미해결 사항: Core self/cross attention에 query/fact padding mask가 전달되지 않는 경로, 클립 전 전체 gradient norm 로그 부재.
  이 문제를 점검하되 원본 실험의 fingerprint를 바꿔 동일 run으로 재개하지 마라.

5. 작업을 다음 순서로 완료해라.
① 자료·코드·환경 inventory와 원본 보존.
② 설계 명세 ↔ 실제 구현 ↔ 보고 결과 대응표 작성.
③ 기존 검사 재실행과 실패 분석; 수식·mask·oracle·정답 유출·Commit 불변성·저장 실패 경로 검사.
④ 수정이 필요하면 별도 버전/branch와 새 실험 ID를 만들고 원본과 수정본의 차이·새 fingerprint를 기록.
⑤ notebook 셀 순서, 인증 절차, T4 확인, checkpoint 선택/복구, status/evaluation export를 검증.
⑥ 실제 T4 smoke test와 fresh reload 검사에 필요한 실행 명령·통과 기준을 준비.
⑦ 내가 학습 시작을 지시하면 T4에서 소규모 실제 실행 → 원격 저장 검증 → 재로드/재개 검증 → 원래 plan의 장기 실행 순서로 진행.
⑧ 유효한 새 결과를 확보한 뒤 표현·부정 일반화 및 ablation 계획을 독립 실험으로 준비.

Windows 노트북 환경에 맞는 경로와 명령을 사용하고 OS·Python·가용 GPU를 실제 확인해라. CPU는 정적 검사·작은 수학/직렬화 검사에만 사용하고 CPU 학습으로 T4 실험을 대신하지 않는다. Colab T4를 기본으로 하며 다른 GPU나 유료 서비스로 자동 전환하지 않는다.

6. 가중치 보존과 재개를 가장 먼저 검증해라.
- v2는 기본 실행 예산 6시간(21,600초), 매 optimizer 시도 후 전체 상태를 원격 commit하는 정책이다. 실제 코드와 설정을 확인해라.
- 시작 전 초기 상태도 보존하고 모델, optimizer, scheduler가 있으면 그 상태, AMP scaler, phase/step, 성공 updates와 attempts, 노출 카운터, sampler/noise/Python/NumPy/PyTorch/CUDA RNG 중 사용된 모든 난수 상태를 보존한다.
- 진행 순서는 optimizer 시도 → 이력·노출 기록 → 원격 payload 업로드 완료 → metadata/size/MD5/버전 확인 → complete marker 게시 → 다음 optimizer 시도다.
- 다운로드 전체 SHA-256, 내부 계약, source/config/data/plan 및 엄격 재개에 필요한 environment를 대조한다. latest 포인터만으로 복구하지 마라.
- payload·marker 실패 시 다음 update를 시작하지 말고 이전 검증본을 보존해라. 손상된 최신 슬롯에서 이전 완료 슬롯 복구가 가능한지 검사해라.
- 같은 이름의 복수 marker는 임의 선택하지 않는다. 서로 다른 runtime 동시 실행도 고려한다.
- 매 update 전체 저장은 model만 FP32 기준 약 409MB이며 optimizer까지 포함하면 더 커진다. 실제 저장 시간·Drive 용량·revision 증가·제한을 측정해라. 저장 주기를 몰래 완화하거나 기존 가중치를 삭제하지 마라.
- 저장 전 단절된 진행 중 update는 재계산될 수 있다. 저장 증거 없이 “완전 무손실”을 보장하지 않는다.
- runtime export와 평가 파일은 checkpoint 해시에 연결하고 각각 완료 marker를 확인한다. 최종 평가는 중단 후 누락 항목만 이어 처리한다.
- 시간 제한은 저장을 포함한 고정 deadline으로 구현한다. 코드의 6시간 예산과 Colab의 실제 수명을 구분한다.
- 화면 유지·자동 클릭·keepalive 등으로 Colab 세션 제한을 우회하지 않는다. 전원·절전 설정은 사용자가 적용할 안내를 제공한다.
- 이전 fingerprint와 다른 코드/데이터/plan을 사용한 실행은 strict resume가 아니다. 새 branch 또는 명시적 warm start로 기록한다.

7. 후속 데이터 조정과 평가를 분리해라.
- 우선순위: 표현 계열 분리, 부정 일반화, query exposure 네 클래스 25%씩, grounding 네 클래스 25%씩.
- query 클래스는 현재 코드의 UNRESOLVED/GREATER/LESS/CONFLICT 매핑을 확인한다. grounding 네 클래스의 정확한 정의·순서는 코드에서 확인하고 임의 추정하지 마라.
- 성공 updates, optimizer attempts, sampled examples, query/grounding targets, UTF-8 bytes 또는 실제 tokenizer tokens를 각각 기록해라. 실제 노출 비율은 분모를 명시하고 재개 전후 카운터를 보존한다.
- grounding60/semantic15/diffusion25% 다음 1,000회는 미실행 가설이다. 원래 recovery plan에 섞거나 실제 학습 비율이라고 보고하지 마라.
- graph 구조 분리와 표현 계열 분리를 별도로 적용하고 train/dev/test 누출을 검사한다. 고정 benchmark 재사용과 새 독립 holdout을 구분한다.
- 근거 검사 이전 후보 Accuracy·Macro F1·클래스별 confusion matrix, 직접/전이/부정/모순/미해결, 미학습 표현 grounding 및 관계 정확도를 보고한다. initializer/diffusion/verifier/최종 출력 성능을 각각 기록한다.
- finite loss·gradients·parameters, 클립 전 gradient norm, AMP skipped updates, T4 allocated/reserved peak memory, 저장 및 평가 시간도 확인한다.
- role/no-role, fuzzy/대체 routing, diffusion/no-diffusion·동일 계산 예산 recurrent refiner를 같은 초기값·데이터·노출·평가 조건에서 비교하는 실험을 설계한다.
- 새 그래프 성능만으로 퍼지·디퓨전의 독립 이득이나 범용 자연어 추론 능력을 확정하지 않는다.

8. 결과물을 파일로 남기고 현재 상태를 분명히 보고해라.
작업 폴더에 최소한 다음을 작성해라:
FDT_HANDOVER.md — 읽은 자료·버전·해시·실제 확인 상태·미확인 내용.
FDT_CODE_AUDIT.md — 수식/구현 대응, 결함, 수정과 검사 결과.
FDT_NEXT_EXPERIMENTS.md — 독립 실험 계획·비율·가설·판정 기준.
실제 실행 로그·config·data/plan fingerprint·checkpoint manifest·검증 marker와 재현 명령도 남겨라.
가능한 권한 범위에서 같은 FDT Drive 폴더에 버전별로 보존하고, 원본 보고서와 가중치는 유지해라.

최종 보고는 “읽고 확인한 것 / 수정한 것 / 실행한 것 / 저장·재로드 확인 여부 / 미실행·미확인 / 내가 다음에 할 한 단계” 순서로 간결하게 제시해라.
직접 상태 보고 요청에는 실제 상태를 설명하고 ::SKIP_COMPLETION::을 답으로 쓰지 마라.
자료 접근부터 시작해, 가능한 검사·수정·실행 준비를 끝까지 완료해라.

