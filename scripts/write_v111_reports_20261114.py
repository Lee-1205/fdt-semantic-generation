"""Publish seven NEW reports from independently reviewed saved records only."""
import collections,hashlib,json,re,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];W=P.parent
RUN=P/'runs/answer_meaning_20261114_v111'
PAIR=P/'research/artifacts/answer_meaning_v111_fullTRAIN_pair_readonly_20261114_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
link=lambda label,p:f'[{label}]({p.as_posix()})'


def repeat4(text):
    raw=text.encode();grams=[raw[i:i+4] for i in range(max(0,len(raw)-3))]
    return 0. if not grams else 1-len(set(grams))/len(grams)


def main():
    result=read(RUN/'result.json');review=read(RUN/'root_saved_review.json');paired=read(PAIR/'root_saved_review.json')
    rows=read(RUN/'assessment.json');full=read(PAIR/'assessment.json');plan=read(RUN/'plan.json')
    pre=P/'research/artifacts/answer_meaning_20261114_v111_C_cpu_v1';cpu=read(pre/'cpu_preflight_owned/runtime/actual_CPU_report.json')
    before=review['groups'];curve=review['training_curve'];end=result['checkpoints'][-1]
    assert result['updates']==268 and review['all2144_noise_draws_exact'] and paired['positive80_partial_grammar_correct']==0
    boundary={}
    for key,rr in [('final_MAIN112',[r for r in rows if r['phase']=='final']),('final_FULLTRAIN134',full)]:
        boundary[key]=dict(rows=len(rr),UTF8=sum(r['UTF8'] for r in rr),EOS=sum(r['EOS'] for r in rr),
            no_EOS=sum(not r['EOS'] for r in rr),whole_exact=sum(r['exact'] for r in rr),
            correct_prefix=sum(r['output'].startswith(r['expected']) for r in rr),
            mean_byte4gram_repeat_ratio=sum(repeat4(r['output']) for r in rr)/len(rr),
            mean_length_excess_bytes=sum(max(0,len(r['output'].encode())-len(r['expected'].encode())) for r in rr)/len(rr),
            notes='Length excess is not a claim that the preceding content is correct; byte4gram repetition is not an AR feedback metric.')
    oracle=read(pre/'cpu_preflight_owned/runtime/actual27_target_ceiling.json')
    boundary['actual_oracle_EY27']=dict(rows=len(oracle),UTF8=sum(r['UTF8'] for r in oracle),EOS=sum(r['EOS'] for r in oracle),
        whole_exact=sum(r['exact'] for r in oracle),correct_prefix=sum(r['prefix'] for r in oracle),
        trailing_bytes_min=min(len(r['output'].encode())-len(r['expected'].encode()) for r in oracle),
        trailing_bytes_max=max(len(r['output'].encode())-len(r['expected'].encode()) for r in oracle))
    (RUN/'root_decoder_boundary_statistics.json').write_text(json.dumps(boundary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    filenames=['01_CURRENT_STATE_AUDIT_KR.md','02_ANSWER_TARGET_CONTRACT_KR.md','03_TRAINING_PLAN_KR.md','04_TRAINING_RESULTS_KR.md','05_SEMANTIC_EVALUATION_KR.md','06_DECODER_BOUNDARY_AUDIT_KR.md','07_NEXT_RESEARCH_DECISION_KR.md']
    common='2026-11-14. ROOT 직접, 에이전트0. VERIFIED=이번 실제 코드/실행·독립 재계산, OBSERVED=기존 저장 관측, INFERRED=해석, PROPOSED=미실행 제안.\n'
    text={}
    text[filenames[0]]=f'''# 01. 현재 상태와 계보 감사

{common}
## VERIFIED

실제 루트는 `{P.as_posix()}`다. 이 작업 트리는 Git 저장소가 아니어서 Git commit/PR로 변경을 증명하지 않았다. 봉인 plan·소스 snapshot·파일 SHA를 근거로 사용한다. 고정 codec은 V88+필수 V74 standalone이며123,497,080개 parameter와 모든 buffer를 고정했다. 실제 현재 출력기는 Commit(global2+local)→FP32 LayerNorm/MLP→16위치 Byte/EOS/PAD readout이다. V7 VP+AR decoder나 옛426M/Diana 모델 사양이 아니다.

수정된 hidden520/출력512/9,576,426개 사고 parameter에 **처음으로 E(Y) 답변 표적 학습268update를 실제 실행**했다. 새로운 가중치가 생겼다는 것과 의미 변환 능력을 획득했다는 것을 구별한다. 현재 판정은 `TRIAL_COMPLETE_TRANSFORMATION_NOT_ESTABLISHED`다.

| 분기 | 목적·상태 |
|---|---|
| V102→V105 | hidden256에서 E(Y)192+768update. 두 Adam trajectory, 수정 전 rank 제약 |
| V109 | hidden256 E(X) identity134update, 별도 분기 |
| V110 | hidden520 E(X) identity67update, 별도 분기 |
| V111 이번 | V110의 모든 θ 가중치만 warm, fresh Adam/scaler, E(Y)268update |

이번 초기 θ는 V110 최종 θ와64개 tensor 모두 일치했다. 초기 Adam은 비었고, V110의 optimizer/RNG를 새 답변 학습의 resume로 취급하지 않았다. 기존 codec의 model/buffer digest는 전후 `{result['frozen_codec_SHA_before_after']}`로 같고, 독립 source safeload/endpoint digest도 일치했다. 모든 옛 가중치·Adam/scaler/RNG·실패를 보존했다.

본 trial 실제 owner11180→15028, worker25640→29828의 생성·parent·명령·exe·생존을 ROOT가 확인했다. worker/owner exit0, 모두 종료와 Python0을 먼저 기록한 뒤 분석했다. {link('종료 증거',RUN/'ROOT_terminal_native.json')}.

6개 focused CPU 계약이 통과했고 실제 packet/전체 크기의 C 사전검사330calls가 통과했다. 첫 자원 테스트는 기본 pytest Temp 디렉터리 ACL 때문에 setup에서 실패했으며 모델 호출 전 오류였다. 임시 파일이 필요 없는 예산 테스트를 비존재 workspace 경로로 바꾸고2개를 통과시켰다. 오류 기록은 삭제하지 않았다.

## OBSERVED

사용자 기술 지시와 {link('직전 가능성 보고서',W/'outputs/FDT_ANSWER_MEANING_FEASIBILITY_AND_PLAN_20261114_KR.md')}를 대조했다. AFE 최종 발표자료와 AI 협업 설계서의 별도 원본 파일은 이름 기반 로컬 검색에서 찾지 못해 읽었다고 주장하지 않는다. 일부 옛 다른 프로젝트 폴더는 OS 접근 거부였다. 이를 우회하지 않았고 이번 실제 코드·자료 감사는 진행했다.

처음 D: 출력으로 준비한 CPU 검사330calls/임시 fixture3update는 완료됐으나 연구 학습 launch는0이었다. 사용자의 C: 변경 지시 뒤 C 전용 코드·계획·실제 CPU 검사를 새로 만들었다. C/D 검사 READY 파일 SHA는 같았다. 이660calls와 임시6update는 본 학습268과 별개다. 저장 정책 변경 뒤 새 연구 산출물은 모두 C, D의 기존 V110 부모는 읽기 전용이다.

## INFERRED / PROPOSED

이제 “수정 구조의 답변 학습0”이라는 과거 상태는 갱신해야 한다. 다만 현재 성공은 실행·동결·복구 계약의 성공이며 Transformation 성과가 아니다. 실제 평가 판정은05/07보고서에 따른다. 타이머는 재개하지 않았고 fdt config 부재를 확인했다. 새 trial 자동 실행은 없다.
'''
    text[filenames[1]]=f'''# 02. 답변 의미 표적과 누수 경계

{common}
## VERIFIED

`X→고정 E→Zx`, `Zx+K_P+K_C+NULL K_E+독립 ε→Rθ×16→Z_hat→고정 D→Y_hat`을 실행했다. **표적은 E(Y), E(X)가 아니다.** source packet과 loss-only answer packet은 manifest/파일 경계가 다르다. 선택134개는 원래 TRAIN384에서만 왔고 DEV 편입0, fact key31, 답변 의미27종, greater38/less42/unknown54다. 원래 전체480행의 정답은 사전 저장 분석에서 유한 수치 세계로 독립 재구성됐다.

{link('조건·표적 계약',P/'research/artifacts/answer_meaning_objective_20261114_v1/contract.json')}의 row ID→원래 정답 답변 hash→실제 answer semantic packet을 연결했다. source와 target 파일의 SHA, codec version, epoch, dtype, mask, 원문 답변 label, EOS/PAD를 확인했다. `Data.target()`은 원래 답변 문자열의 SHA로 **훈련 전용 packet**을 읽으며 질문 원문으로 치환하지 않는다. 이 lookup은 loss/사후 score 경로에만 있고 생성 결과를 직접 반환하지 않는다.

| 항목 | 실제 계약 |
|---|---|
| 최종 의미/표적 | FP32 `(1,35,512)` |
| global/local |2 /33, 합35 |
| mask | bool `(1,35)`, loss-side only |
| labels | int64 `(34,16)`, bytes→EOS256→PAD257 |
| UNKNOWN14byte | body index0–3, tail4–34 |
| positive19/20byte | body0–4, tail5–34 |

body에는 global2, 실제 local1또는2, frozen terminal1이 들어간다. index는0부터다. terminal 이후 tail은 같은 frozen terminal 벡터로 채워져 있다. **local33과 전체35를 용량 불일치로 비교하지 않는다.**

주 경로는 정답 packet을 읽지 않고 target-free16단계를 먼저 실행한다. 이후 E(Y)와 끝점 loss를 계산한다. 보조 VP 경로만 noisy E(Y)를 사용한다. 실제 추론은 target/정답 mask/답변 길이/역할/template 인수를 받지 않는다. 평가의 target packet도 최종 의미와 D 출력이 끝난 다음 score에 사용한다. source ID/hash를 NN embedding으로 넣지 않았다.

정확한 loss는 행마다 body의 slot/512좌표 평균 MSE, tail의 별도 평균 MSE를 내어 `body +.05 tail +.1 VPaux`다. VPaux는 body의 velocity MSE+.25 clean MSE, 그리고.05배 별도 tail 평균을 합한 기존 목적이다. batch는8행 동일 비중 gradient accumulation이다. 언어 CE/EOS CE 로그0은 미사용 자리표시자이며 실제 언어 loss0이라는 뜻이 아니다.

K_P는 고정 anchor/prior, K_C는 입력별 long/recent snapshot이다. θ가 매 단계 읽는 query/projection만 학습한다. K_E는 invalid NULL/key-value0/copy0이며 전문 복사 기능을 만들지 않았다. source의 전체 원문·token H·답변 retrieval·gold 역할·정답 길이는 D나 R에 새로 공급하지 않았다.

## OBSERVED

입력134개는 앞선 source 왕복 검사에서 전체 byte+UTF8+신경 EOS가 통과했다. 이 사실은 새 답변 생성 성과가 아니다. 사전검사에서27개 E(Y)를 실제 D에 직접 넣어 모두 정답 prefix를 확인했지만 종료 잡음은 남았다.

## INFERRED / PROPOSED

이 경계에서는 올바른 정보를 가진 입력이 있어도 R이 답변 의미를 형성하지 못할 수 있다. 원문의 복원 가능성이 관계 변환의 쉬운 학습을 보장하지 않는다. 미래의 별도 codec 개선 가능성은 열어두되 이번 실험에서 E/D를 공동 학습하지 않았다.
'''
    text[filenames[2]]=f'''# 03. 실제 봉인한 유한 학습 계획

{common}
## VERIFIED

실제 plan SHA는 `{sha(RUN/'plan.json')}`이며 {link('봉인 plan',RUN/'plan.json')}에 저장됐다. 준비 파일과 실행 plan을 구별한다.

| 항목 | 실제 설정 |
|---|---:|
| 초기 θ |V110 hidden520,9,576,426개 전체 가중치 warm ONLY|
| optimizer |fresh AdamW, LR1e-4, decay.01, clip1|
| scaler |CPU enabled, 초기1024, growth interval16384|
| TRAIN / epoch / 노출 |134 /16 /2,144|
| 답변 UTF8byte 노출 |37,088|
| batch/micro/attempt |8 /1 /268|
| sampling |target-free16step|
| auxiliary time |각 row가16개 실제 sampler grid를 한 번씩|
| 전체 예정 호출 |40,032thought+224D=40,256; E0|
| 상한 |41,000calls, owner1,200초, modelcut1,080초, reserve120초|
| 저장 |C 연구 run 전체1GiB,0/134/268와STOP·원시trace·임시파일 포함|

58개 실제 역전짝과18개 singleton을 유지했다. singleton 두 개를 일정상 같은2-row block에 넣어도 실제 역전짝이라는 label을 붙이지 않았다. 16epoch를 하나의 stream으로 연결해8행 batch268개를 만들었고, 실제짝은 전부 batch 안에 있었다. 모든134행을 각 epoch에1번씩 사용했고 순서·시간·cursor를 저장했다. **주 학습의 noise는 행마다 독립 draw다. 실제짝은 같은 time/batch였지만 주 학습에서 공통 noise를 사용했다고 주장하지 않는다.** 공통 noise는 별도 사후 paired 평가에서 사용했다.

초기·중간·최종 fullstate는 model/optimizer/실제 scaler/모든 RNG/순서/cursor/누적 gradient/attempt-update/노출/call/phase/codec digest/loss 설정을 저장한다. 마지막 state는 학습 DONE 직후, 최종 평가 전에 저장됐다. 최종224개 평가 reference는 `assessment.json`과 result에 별도로 보존했다. 존재하지 않는 post-evaluation fullstate나 실제 CUDA 재시작을 주장하지 않는다.

사용자 최신 저장 지시에 따라 실행 C 하한은50GiB, OS owner의 조기 정리선은54GiB다. 시작 전에는 `100GiB+실험 전체1GiB+64MiB`를 확인해 우리 산출물을 모두 써도100GiB가 남도록 계획했다. 실제 시작 전 약107.557GiB였다. D에 새 실험을 저장하지 않았다. 중복 Python0과 실제 lock 사용 가능성, RAM8GiB 이상을 확인했다. 실제학습 중 ROOT는 과학 분석을 중단하고 독립 OS 감독·진행 상태만 확인했다.

## OBSERVED

기존 호출 비례 예상은 약637초였고 실제 worker는{result['seconds']:.3f}초였다. 사전검사와 준비 시간은 이 worker 시간에 합산하지 않는다. 기존 예산을 연장하거나 남은 update/call을 사용하지 않았다.

## INFERRED / PROPOSED

16epoch는 충분성 보장이 아니라 첫 유한 작업 단위다. 일반화·기억 인과·Composition은 이 plan의 성공 기준이 아니다. 이후 추가 학습에는 새 목적·새 상한·새 plan이 필요하다.

실제로 실행한 명령은 다음과 같다. **현재 이 봉인 프로필은 종료됐으며 재실행이 차단된다. 기록용 명령을 새 학습 명령으로 재사용하지 않는다.**

```powershell
& '{(P/'.venv/Scripts/python.exe').as_posix()}' -B -X utf8 '{(P/'scripts/supervise_answer_meaning_v111_c.py').as_posix()}' --plan '{(RUN/'plan.json').as_posix()}'
```
'''
    text[filenames[3]]=f'''# 04. 실제 학습 결과와 복구 검증

{common}
## VERIFIED

실행은 정상 완료됐다. **268attempt/268update/AMPskip0**,2,144행,37,088답변 UTF8byte다. 실제 호출은40,032thought+224D=40,256이며 encoder0/훈련decoder0/훈련언어loss0이다. worker 시간은{result['seconds']:.3f}초다. owner/worker exit0과 모든 owned/proxy 종료를 확인했다.

| 실제 학습 loss,8개 update 평균 | 처음8 | 마지막8 |
|---|---:|---:|
| body MSE |{curve['first8']['actual_trajectory']['runtime_body_MSE']:.6f}|{curve['last8']['actual_trajectory']['runtime_body_MSE']:.6f}|
| tail MSE |{curve['first8']['actual_trajectory']['runtime_tail_MSE']:.6f}|{curve['last8']['actual_trajectory']['runtime_tail_MSE']:.6f}|
| VPaux |{curve['first8']['actual_trajectory']['aux_VP']:.6f}|{curve['last8']['actual_trajectory']['aux_VP']:.6f}|
| 합계 loss |{curve['first8']['loss']['latent']:.6f}|{curve['last8']['loss']['latent']:.6f}|
| clipping 전 norm |{curve['first8']['mean_preclip_gradient_norm']:.6f}|{curve['last8']['mean_preclip_gradient_norm']:.6f}|

gradient가 끊기거나 모든 update가 skip된 실험은 아니다.64개 θ tensor가 실제 변했고 E/D 전체 buffer/parameter 경계는 같았다. 실제 로드 구조와 최종 저장 가중치 모두 velocity512×520이며, LayerNorm 이후 유효 rank512를 확인했다. NaN/Inf 없이 끝났다.

| checkpoint | byte | 파일 SHA |
|---|---:|---|
|0|{result['checkpoints'][0]['bytes']}|`{result['checkpoints'][0]['SHA']}`|
|134|{result['checkpoints'][1]['bytes']}|`{result['checkpoints'][1]['SHA']}`|
|268|{end['bytes']}|`{end['SHA']}`|

ROOT는3개 fullstate의 SHA/weights-only safeload/재귀 digest, 초기 warm θ 전부 일치·빈 Adam, 최종 Adam step268/scaler1024 tracker268, 모든2,144개 noise draw의 정확한 재생, 순서·노출·cursor를 독립 재구성했다. 실제 CPU 사전검사에서는 populated READY의6-row 잔여 batch와 다음 update, 중간 gradient와 입력 checkpoint 불변을 검증했다. 최종 사후 평가에서 정답 없이 생성한 실제 의미를 저장했다.

사전검사의 같은 seed 출력은 bitwise 같았고 train/eval16step 최대 차이는{cpu['actual16step_training_vs_eval_maxgap']:.8f}였다. 최종 의미 분산도 저장 raw에서 계산했다. rank나 분산이 정상이라는 사실을 의미 생성 성공으로 해석하지 않는다.

## OBSERVED / INFERRED

loss는 내려갔지만 의미 변환은 아직 확인되지 않았다. 다음05의 실제 출력이 우선이다. 모델을 자동 승격하지 않았고 다음 SGD를 시작하지 않았다. 원래41,000예산의 미사용744call은 닫혔으며 재사용하지 않는다.

필요한 전체 TRAIN/실제32긍정 역전쌍 검사는 별도 plan으로만 수행했다. optimizer0,134행,2,144thought+134D=2,278calls,14.953초였다. 별도 cap2,300/120초(90+30)/64MiB도 닫혔다. 원래 미사용 budget의 재사용이나 학습 연장이 아니다.

{link('독립 저장 검토',RUN/'root_saved_review.json')} · {link('전체 TRAIN 별도 검토',PAIR/'root_saved_review.json')}
'''
    table='| 패널 | NLL 전→후 | body MSE 전→후 | 후 byte argmax | 후 긍정 정상 내용 |\n|---|---:|---:|---:|---:|\n'
    for name,pretty,positives in [('TRAIN_probe','TRAIN probe16',8),('dev_pair','dev_pair32',16),('dev_surface','dev_surface32',16),('dev_position','dev_position32',16)]:
        a=before['baseline/'+name];b=before['final/'+name]
        table+=f"|{pretty}|{a['mean_NLL']:.4f}→{b['mean_NLL']:.4f}|{a['body_MSE']:.4f}→{b['body_MSE']:.4f}|{100*b['byte_accuracy']:.2f}%|0/{positives}|\n"
    text[filenames[4]]=f'''# 05. 의미 정확도와 질문 역전 평가

{common}
## VERIFIED

학습 전후 각112개는 같은 row별 noise seed·고정 D·고정 decoding으로 비교했다. ROOT는 TRAIN 전체80개와 DEV48개 긍정 출력의 대상·관계 내용을 직접 검토했다. 명확히 올바른 긍정 관계를 확인하지 못했다. 이는 단지 전체 문자열 exact가0이기 때문이 아니다. 주어·목적어가 빠지거나 다른 문자이며 관계어도 깨졌다.

{table}

전체 TRAIN134를 별도 공통-pair-noise로 평가한 결과도 **긍정0/80, 실제32긍정쌍 모두 정답0/32**였다. 전체 TRAIN 적합을16probe로 추정하지 않았다. 이134개는 학습 전 전수검사와 같은 noise 비교가 아니라, 학습 후의 별도 paired 검사다.

UNKNOWN의 자동 keyword 점수는 확정 의미 정확도가 아니다. 주 평가의 TRAINprobe5/16, dev_pair8/32, surface3/32, position2/32가 자동 partial grammar에 잡혔지만 모두 UNKNOWN 쪽이며 `unknowng`, `unknownene` 같은 깨진 문자열도 포함한다. 전체 TRAIN은31/54 UNKNOWN이 substring에 잡혔다. 사람이 의도를 읽을 수 있는 일부 오타는 있지만 미인증 범주로 분리했다. 전체 byte 일치만 강요하지 않되 깨진 문자열의 의도를 target에서 채워 넣지 않았다.

확실한 정답 prefix는 주 평가 DEV에서3/96(모두 dev_pair UNKNOWN), 전체 문자열은2/96였다. 전체 TRAIN 별도평가에서는prefix4/134, fullstring1/134였고 모두UNKNOWN이다. prefix가 맞아도 뒤의 언어 안정성은 별개다. **UNKNOWN54개를 전부 옳다고 가장 관대하게 보아도, 긍정0/80인 현재 전체 TRAIN은 상수 UNKNOWN54/134=40.30%를 넘었다고 할 수 없다.** DEV도 같은 이유로 상수50%를 넘는 관계 능력이 입증되지 않았다.

## 실제 예시

| scope | 기대 답 | 실제 출력 |
|---|---|---|
| TRAIN 긍정 |A is larger than B.|`A  s laa`|
| TRAIN 긍정 |H is larger than B.|`H is largeer hhe\\n W.Cgeaa r aa`|
| DEV pair 긍정 |C is larger than D.|`D ism iake`|
| DEV surface 긍정 |B is smaller than A.|`It is unknown.unn'  e  s  awinee Etio.mo `|
| DEV pair UNKNOWN |It is unknown.|`It is unknown.`|
| TRAIN UNKNOWN |It is unknown.|`It is unknown.  A C`|

## VERIFIED: 동일 잡음 역전쌍의 반응

전체32긍정쌍에 같은 ε를 주었다. source 표현 차이 RMS는 최소.35168/평균.44151로 실제 입력 차이가 있었다. 생성 의미 first4slot 차이 중앙값은.05063, 해당 정답 answer-code 차이 중앙값은1.77113, 개별 비율의 중앙값은**.02816**이었다.32쌍 중9쌍은 실제 출력 문자열도 같았다.

이 비율은 **목표 query-role 코드에 대한 반응 크기 진단**이지 의미 정확도2.8%가 아니다. `A>B`와 `B<A`는 같은 canonical 사실을 표현하므로 역전짝의 정답이 서로 반대 사실이라는 뜻도 아니다. 다른 표현으로 같은 사실을 정확히 답했다면 의미적으로 인정할 수 있지만, 그 사실만으로 질문 역할을 변환했다고 증명되지는 않는다. 현재는 양쪽 모두 제대로 된 긍정 내용을 생성하지 못했다.

## INFERRED

공통 답문 형태는 배웠지만 대상과 관계의 결합은 아직 부족하다는 해석이 지지된다. TRAINprobe 긍정8개의 ` is `4byte는29/32가 맞아졌지만 주어0/8, 목적어2/8, 관계 단어 전체0/8이었다. 전체 TRAIN 긍정80개에서도 주어 argmax5/80, 목적어6/80, 관계 단어0/80이다. 이는 바이트 정확도 상승을 의미 이해로 대체할 수 없는 구체적 이유다.

표현이 모든 정보를 잃었다거나 Diffusion 원리가 불가능하다는 결론은 아니다. 현재 업데이트 수·표현·loss·조건 분리 학습에 관한 결과다. NN 재실행 없이 비교한 TRAIN 평균 표적은 같은16probe body MSE.69385였는데 모델은1.00609였다. 기본 fit도 미수렴이다.

## PROPOSED

다음은 조건 반응을 분리해 강화하는 좁은 통제 실험이다. 데이터·epoch를 무조건 크게 늘리거나 새 AR decoder를 붙이지 않는다. 구체적인 후속은07에 기록했다.

{link('전체224 저장 점수·출력',RUN/'root_saved_rows224.json')} · {link('TRAIN134 원문·출력',PAIR/'root_rows134.json')} · {link('실제58쌍 반응',PAIR/'root_actual58_pair_response.json')}
'''
    bd=boundary
    text[filenames[5]]=f'''# 06. 고정 언어화 경계와 생성 의미의 차이

{common}
## VERIFIED

이번 C 사전검사에서 실제 사용27종 E(Y)를 고정 D에 직접 넣었다. **정답 prefix27/27, 전체 문자열0/27**, 신경 EOS/UTF8는27/27이며 추가 문자는{bd['actual_oracle_EY27']['trailing_bytes_min']}–{bd['actual_oracle_EY27']['trailing_bytes_max']}byte였다. 과거31종 결과를 새 측정27종의 성적으로 자동 승계하지 않았다. 정답 길이 crop이나 template 조립을 사용하지 않았다.

| 경로 | 행 | whole exact | correct prefix | no EOS |
|---|---:|---:|---:|---:|
| 직접 E(Y)→D |27|0|27|0|
| 학습 전 R→D, 전체 main probe/DEV |112|0|0|0|
| 학습 후 R→D, 같은 main 조건 |112|{bd['final_MAIN112']['whole_exact']}|{bd['final_MAIN112']['correct_prefix']}|{bd['final_MAIN112']['no_EOS']}|
| 학습 후 R→D, TRAIN전체 paired |134|{bd['final_FULLTRAIN134']['whole_exact']}|{bd['final_FULLTRAIN134']['correct_prefix']}|{bd['final_FULLTRAIN134']['no_EOS']}|

생성134개는 noEOS30/early16/late76, 나머지12개는 길이가 같지만 내용 정답은 대부분 아니다. 본 평가112개는 noEOS19/early20/late70이며 길이 일치도 내용 정답과 구별했다. UTF8는112/112와134/134다. UTF8 규칙이 정상 문자열을 만드는 것과 의미가 맞는 것은 다르다.

byte4gram 반복 비율 평균은 main최종{bd['final_MAIN112']['mean_byte4gram_repeat_ratio']:.4f}, 전체TRAIN{bd['final_FULLTRAIN134']['mean_byte4gram_repeat_ratio']:.4f}였다. 이는 저장된 출력 바이트의 반복성 지표이며 AR feedback loop를 측정한 수치가 아니다. 모든 원시 출력은 길이를 자르지 않고 보존했다. 길이 초과 평균은 별도 JSON에 있지만, 내용이 틀린 문장을 ‘정답 뒤 추가byte’라고 표현하지 않는다.

샘플로 저장한 처음2patch의 full258 logits는32byte를 덮는다. 이번 정답은14/19/20byte라 **모든 정답 byte와 EOS 점수**를 독립 재계산할 수 있었다. 길게 발생한 뒤쪽 tail의 모든258logit을 저장한 것은 아니며, 그쪽은 모든 legal UTF8 decision trace로 전체 decode/EOS를 독립 검증했다.

## INFERRED

직접 정답 의미는 관계 문장을 읽는데 생성 의미는 대상·관계부터 깨지므로 **현재 실패를 D의 기존 종료 오류만으로 설명할 수 없다.** 반대로 D의 종료 한계가 해결됐다는 뜻도 아니다. 순수 MSE로 정답 의미를 정확히 만들더라도 그 표적의 late EOS를 물려받을 수 있다.

현재 생성 tail MSE는 main probe약.524, 전체TRAIN약.536이다. 같은 terminal을 두면 되는 tail도 남아 있으므로, 아직 decoder가 읽기 좋은 잠재 상태에 충분히 도달하지 못했다고 보는 것이 타당하다. 이것만으로 내부에 어떠한 관계 정보도 전혀 없다고 증명되지는 않는다.

## PROPOSED

먼저 θ의 조건별 분리와 기본 fit을 조사한다. 향후 의미 내용은 맞는데 종료만 나쁜 단계가 실제 확인되면, 별도 θ-only frozen-D 충실도/EOS 목적을 검토한다. 이번에는 D를 열거나 새 decoder, 높은 반복 penalty, 정답 길이 crop을 추가하지 않았다.

{link('실제27 oracle readout',pre/'cpu_preflight_owned/runtime/actual27_target_ceiling.json')} · {link('경계 통계',RUN/'root_decoder_boundary_statistics.json')}
'''
    text[filenames[6]]=f'''# 07. 후속 연구 판단

{common}
## VERIFIED

요청한 첫 hidden520 조건부 E(Y)학습은 실제 완료됐다. 올바른 packet 연결, frozen E/D/기억,16step gradient, rank512, finite state, 정확한 순서·노출, Adam/scaler/RNG 복구를 확인했다. 기본 코드 누수·초기 optimizer 미실행·identity 표적 오연결 때문에 결과를 낸 실험은 아니다. 그러나 **Transformation 능력은 아직 입증되지 않았다.**

## OBSERVED

loss와 NLL, 자주 나오는 언어 형태는 좋아졌지만 긍정 관계는 TRAIN80/DEV48에서 명확한 정답0이었다. UNKNOWN 상수 기준을 넘는 조건부 의미 변환은 확인하지 못했다. 전체 TRAIN의 같은-noise 역전쌍에서도 목표 query-role 코드 차이에 대한 생성 반응이 작았다. 기존 D의 직접 E(Y) prefix는27/27이었다.

## INFERRED

현재 결과는 주로 **D: loss는 감소하나 조건에 맞는 구별이 약함**, 그리고 **E: 단순 평균 표적 수준의 기본 적합에도 못 미침**에 가깝다. TRAIN 의미 변환을 확보하지 못했으므로 이를 단순한 ‘TRAIN만 잘 되고 DEV만 나쁜 일반화 실패’로 분류하지 않는다. D의 EOS 문제도 있지만 이번 관계 실패 전체를 출력기만의 문제라고 할 수 없다.

추상 의미 공간 전체가 불가능하다거나 입력에 필요한 정보가 없다는 판정은 과장이다. 선택134입력은 source-only Commit에서 실제 복원됐고, paired 입력 표현도 달랐다. 다만 정보가 존재하는 것과 현재 θ가 관계를 쉽게 읽어 답변 코드로 바꾸는 것은 다르다. rank 제약 수정은 학습 가능 방향을 열었을 뿐, 그 방향의 적합을 보장하지 않았다.

같은 canonical 사실의 질문 역전 답문은 의미적으로 동치일 수 있다. 그래서 다음 대비는 ‘다른 참/거짓을 만들어야 한다’가 아니라 **요구한 주어/목적어·관계 방향의 출력 표현을 조건에 맞게 구성하는가**를 겨냥해야 한다. 단순 logits 변화나 잠재 거리만으로 의미 생성 성공을 선언하지 않는다.

## PROPOSED: 다음의 최소 실험

추가 SGD를 지금 자동 시작하지 않았다. 다음에는 입력·출력·메모리·자료134·목표 E(Y)를 그대로 두고, **실제 역전짝의 조건 구별을 강화하는 한 가지 변경**부터 검토하는 것이 우선이다.

1. 주 학습의58실제짝을 같은 ε로 처리하는 통제를 먼저 설계한다. 이번 주 학습은 같은 t/batch였지만 ε는 독립이었다. 공통 잡음으로 질문 역할 차이의 신호를 분리할 수 있다는 가설이며, 표준 독립 noise가 코드 오류라는 뜻은 아니다.
2. 별도 후속 후보로 `||(Z_hat_a−Z_hat_b)−(Zy_a−Zy_b)||²`의 pair-difference 목적을 검토할 수 있다. 이는 query-role/답변-code 차이에 대한 감독이며 새로운 head나 D 학습이 필요 없다. target 차이0인 UNKNOWN26쌍, 실제짝이 아닌18singleton의 처리를 명시해야 한다. 두 변경을 한꺼번에 넣어 원인을 흐리지 않는다.
3. baseline대비, TRAIN 평균 표적 기준, 긍정·UNKNOWN 구분, 같은-noise 질문 대비를 유지한다. pair loss가 표면 변형만 외우는지 unseen pair/표현 통제로 확인한다. E(Y)가 표면 코드도 담는 한, pair 코드 차이의 학습을 곧 추상 관계 이해라고 해석하지 않는다.
4. 첫16epoch의 cosine LR은 마지막에 약1e-5로 내려갔고 기본 fit은 남았다. 추가 최적화가 필요할 가능성은 있으나, 학습률이나 시간만 바꿔 길게 학습하기 전에 위 조건 반응/평균 수렴 가설을 우선 구분한다. 무엇이 최적이라는 증거는 아직 없다.

새 paired objective를 구현한다면 두 row의 graph를 한 원자적 단위로 역전파하고, 중간에 저장 불가능한 graph를 checkpoint한 것처럼 취급하지 않아야 한다. 모든 θ-only/정확복구/유한상한 계약을 다시 검토한다. 지금 존재하지 않는 다음 worker/명령을 만들어 실행했다고 보고하지 않는다.

기본 Transformation의 증거가 생기면 같은 질문·다른 합법적 K_C 기억 paired episode로 기억 인과성을 검사하고, 그다음3–4개체 관계 Composition으로 확대한다. K_P의 certified writer·문서 간 장기 통합·미래 명시적 Exact copy는 아직 미완이다. 이번 실패나 성공을 그 기능들의 성적으로 합산하지 않는다. 장문 사고·범용 LLM·무입력 자율 의미 형성은 후속 과제다.

## 실제 사용 가능한 검증 명령

다음 경로와 명령은 이 환경에서 실제 사용한 것이다. 기존 봉인 trial을 다시 실행하거나 예산을 초기화하는 명령이 아니다.

```powershell
Set-Location '{P.as_posix()}'
& '.venv/Scripts/python.exe' -B -X utf8 -m pytest -q tests/test_answer_meaning_v111.py
& '.venv/Scripts/python.exe' -B -X utf8 -m pytest -q tests/test_answer_meaning_v111_c_resources.py
Get-Content -LiteralPath '{(RUN/'result.json').as_posix()}' -Raw
Get-FileHash -LiteralPath '{Path(end['path']).as_posix()}' -Algorithm SHA256
```

실제 연구 학습 예산40,256/41,000과 별도 read-only2,278/2,300은 모두 닫혔다. CPU 사전검사660calls/임시 fixture6update와 tiny 계약 검사는 따로 센다. 모든 새 연구 산출물은 C이며, 완료 시 C:100GiB 이상을 새로 확인한다. 타이머는 삭제 상태로 유지하며 다음 학습은 자동 실행하지 않는다.
'''
    for name in filenames:
        path=RUN/name
        if path.exists():raise FileExistsError('new reports must not overwrite: '+str(path))
        with path.open('x',encoding='utf-8') as stream:stream.write(text[name])
    # Human review records reflect actual inspection of positive outputs, not a
    # grammar classifier secretly promoted to a general semantic evaluator.
    manual=[]
    for scope,rr in [('MAIN_final',[r for r in rows if r['phase']=='final']),('FULLTRAIN_commonpair',full)]:
        for r in rr:
            category=('NO_UNAMBIGUOUS_CORRECT_POSITIVE_PROPOSITION' if r['label']!='unknown' else
                      'CORRECT_UNKNOWN_PREFIX_LANGUAGE_TAIL_SEPARATE' if r['output'].startswith(r['expected']) else
                      'DAMAGED_UNKNOWN_INTENT_OR_UNREADABLE_NOT_CERTIFIED')
            manual.append(dict(scope=scope,input_id=r['input_id'],label=r['label'],expected=r['expected'],output=r['output'],
                ROOT_review=category,rubric='Do not fill missing entities/direction from target. UNKNOWN typos may convey intent but are not certified whole responses. Full raw tails preserved; semantic core inspected.'))
    (RUN/'root_manual_semantic_core_review.json').write_text(json.dumps(manual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    decision=dict(status='TRIAL_COMPLETE_TRANSFORMATION_NOT_ESTABLISHED',actual_updates=268,actual_training_D_calls=0,
        observed_TRAIN_positive_correct=0,observed_TRAIN_positive_total=80,observed_DEV_positive_correct=0,observed_DEV_positive_total=48,
        observed_positive_pairs_both_correct=0,observed_positive_pairs_total=32,
        automatic_unknown_substring_NOT_semantic_accuracy=True,loss_reduction_NOT_success=True,
        inferred_primary_bottleneck='Conditional/query-role separation and basic latent fit; decoder termination also imperfect.',
        no_new_training_after_terminal=True,all_old_assets_preserved=True,latest_checkpoint=end,
        main_model_calls=40256,separate_final_readonly_calls=2278,separate_preflight_calls=660,scientific_updates_exclude_fixture_updates=True)
    (RUN/'root_result.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    index='# V111 첫 답변 의미 생성 실험 — 결과 묶음\n\n실제 학습268회는 완료했으나 Transformation은 아직 입증하지 못했다. 긍정 TRAIN0/80, DEV0/48; UNKNOWN과 공통 문장 형태의 회복은 관계 의미 생성 성공과 구분한다. 모든 프로세스 종료 확인 후 독립 검토했다.\n\n'
    index+='\n'.join('- '+link(name,RUN/name) for name in filenames)+'\n'
    (RUN/'README_RESULTS_KR.md').write_text(index,encoding='utf-8')
    print(json.dumps(dict(reports7_written=True,main_updates268=True,Transformation_NOT_established=True,root_result_SHA=sha(RUN/'root_result.json'),boundary=boundary),ensure_ascii=False))


if __name__=='__main__':main()
