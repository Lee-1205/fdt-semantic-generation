# QueryCount 학습 감독 오류와 안전 중단 — 2026-10-27

새 질문 수 정렬 학습은 실제로9시도/9성공 업데이트를 수행한 뒤 중단했다. 원인은 모델 발산이 아니라 감독 코드에서 owner_start로 바꾼 변수의 오래된 start 참조가 남아 있었던 NameError다. Root의 최종 검토와 CPU 준비 검사에서도 이 post-Popen 실행 경로를 놓쳤다. 독립 감독 프로세스가 먼저 종료했고, root의 fresh native 검사로 작업자만 살아 있는 것을 확인해 즉시 cooperative STOP을 기록했다.

Worker는18:31:50.522KST STOPPED/COOPERATIVE_STOP으로 끝났고 모든4개프로세스43568→24428→14552→14416 종료를 root와 agent가 확인했다. 사용자 중단 요청은 아니므로 자동화는 ACTIVE로 유지한다. 자동 재시작은 하지 않았다. 원래 source/plan/log/STOP/terminal과 모든 이전 가중치를 그대로 보존한다.

Root는 payload_000010.pt 전체SHA와636586945B 크기,CPU safe-load, model·populatedAdamW·scaler·모든RNG·history·sampler72cursor·동결 tensor·실제 노출을 검증했다. 실제72행/72고유sourcegraph/522활성query, Q1/4/8/16 각각18행, AMPskip0,816forwardattempts다. 모델 개선이나 전체128회 완료를 주장하지 않는다.

PayloadSHA5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675
Algorithm stateSHA d346ee6410ca7627fd63e4dbb82d04057c715cd7765a0eb33e28c9917910034a
정확한 필드 목록·모델hash·다음10번째batch의sourcegraph/Q구성은root_failure_review.json에보존했다.

수정은봉인된원본을고치지않는별도operational recovery_v2 CPU준비로진행한다. 감독의실제post-Popen전체경로를거치는mock검사와예외시ownedtreeSTOP/종료확인을추가한다. 재개한다면현재9회가중치만이아니라populatedAdamW/scaler/모든RNG/sampler/history/counters/원래baseline을정확히이어가고,import전후algorithmhash와다음10번째batch일치·첫복구저장을검증해야한다. 아직새GPU/재개는승인하거나실행하지않았다.

원래절대끝1793094360.9948046=18:46:00.995KST와trainingcutoff18:44:00.995,global128attempt/8192callcap을유지한다. 이미9attempt/816calls를써서남은119attempt/7376calls뿐이다. 새프로세스의900초를더하지않는다. 준비·root검토가원래cutoff이전에끝나지않으면재개GPU를시작하지않고미완료checkpoint를보존한다. 시간연장이나새epoch로메우지않는다.

기존128회계획이시간만료되면그실험은미완료로닫는다. 이후판단에필요하면9회부분가중치의효과를별도작은읽기전용평가계약으로검토할수있지만이는아직계획봉인·실행되지않았으며추가optimizer나옛deadline연장을허용하는것이아니다. E2최종holdout접근0/원래core·source·weights·AdamRNG보존/C20GiB보호를계속지킨다.

근거: runs/querycount_alignment_20261027_v1/root_failure_review.json, recovery.json, final_status.json, supervisor.stderr.log, model/STOP 및 outputs/QUERYCOUNT_ALIGNMENT_RUNTIME_INCIDENT_20261027.json.

18:40KST root 최종 결정으로 이번 할당의 recovery_v2 optimizer 실행을 취소했다. CPU 검사가 cutoff 전에 끝나더라도 실행하지 않는 결정이며 위의 이전 조건부 재개 설명보다 우선한다. 18:46:00.995KST 원래 절대 종료도 지났고 실험은 STOPPED9/9/미완료로 닫혔다. 새 native Python 관측은0이며 모든 원래 owned 프로세스 종료를 다시 확인했다. 남은119시도나7376call은 재사용하지 않는다.

별도 recovery_v2는 아직 CPU 미완료 초안이다. 새 감독의 변수 참조는 고쳤지만 정확한 전체 알고리즘 복구 및 Popen 이후 전체 감독·예외 정리 회귀가 끝나기 전 준비 완료로 부르지 않는다. 원래 봉인된 파일을 수정하거나 복구 후보를 봉인·실행하지 않았다. CPU 수리가 끝난 뒤 필요한 다음 작업은9회 부분 가중치와 저장된 원래 baseline을 비교하는 별도 한정 읽기 전용 평가 준비다. 추가 SGD를 뜻하지 않으며 아직 새 평가 실행은 없다. 시간 종료 관측 근거는 outputs/QUERYCOUNT_ALLOCATION_CLOSED_20261027.json에 보존했다.

후속 CPU 수리 검토: Root는 실제 partial9 상태의 재귀 SHA 재계산, inherited Engine.restore를 거치는 CPU 복원 하네스, import 저장 hook과 정확한 다음10번째 batch를 확인했다. 자연 종료·Popen 이후 주입 예외·progress 없는 예외·살아남은 자식의 UNCONFIRMED 처리까지8개 focused CPU 검사를 직접 재실행해 통과했고 실제 worker CPU preflight 및 독립 supervisor validate도 통과했다. 하네스의 CPU model/optimizer/scaler/CUDA RNG bridge를 이용한 직렬화·제어흐름 증거이며 실제 CUDA 복구를 실행한 것은 아니다.

후보 worker e1225345348f1e63935b4671f4af8d8646104e5cbe0fb1f1a8f1b9d6e809779c, 감독fcbe4603244475a06175e37d4c6cab1d45c2aa7477dc843f3bb18d39a7f8e558, 초안plan7dd4072ef2f17d5a2fca5eb8a98f5213bfa24a8c0ef221b30e245d5f389a4ceb는 authorized:false/미봉인/미실행으로 보존한다. 원래 봉인된3개 SHA는 그대로다. 만료된 원래 학습을 다시 실행하지 않으며 새 읽기 전용 평가의 CPU 준비로 넘어간다. 자세한 검토는 outputs/QUERYCOUNT_RECOVERY_CPU_REVIEW_20261027.json, 평가 범위는 FDT_QUERYCOUNT_PARTIAL9_ASSESSMENT_PLAN_20261027_KR.md에 있다.
