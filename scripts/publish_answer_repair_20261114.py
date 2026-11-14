"""Saved evidence -> reports; no neural execution, no old run mutation."""
import json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];W=P.parent
sys.path.insert(0,str(P/'scripts'))
import learning_io_v2 as io


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:f.write(value)
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a=P/'runs/answer_pair_contrast_20261114_v112';b=P/'runs/answer_codec_fidelity_20261114_v113'
    q=read(b/'root_saved_review.json');d=read(b/'root_dataset_generalization_audit.json');m=read(b/'root_manual_semantic_review.json')
    r=read(b/'result.json');p=read(b/'plan.json');c=read(b/'root_TRAIN_constant_reference.json')
    g=q['groups'];pair=q['pairs'];curve=q['training_curve'];header='2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.\n\n'
    reports={}
    reports['01_CURRENT_STATE_AUDIT_KR.md']='''# 01. 계보와 현재 상태

**VERIFIED** V111 뒤 실제 문제를 분석하고 서로 다른 목적의 두 trial을 수행했다. V112는 실제짝 차이의 학습, V113은 고정 D content 피드백을 추가했다. 각각268attempt/268update/skip0, 이번 연구추가536update다. 각각부모 θ만 warm/freshAdam-scaler였으므로 같은Adamtrajectory나하나의연속loss곡선으로합산하지않는다.

|분기|목적|worker초|model calls|
|---|---|---:|---:|
|V111 기존|E(Y) endpoint/VP|536.539|40256|
|V112 이번|공통ε+실제짝 차이|642.193|40358|
|V113 이번|위 목적+.5 고정D contentCE|657.888|42502|

전부CPU4threads이며새GPU학습이아니다. hidden520/출력512/θ9576426, 기존codec123497080parameter와ALLbuffers/E/D/기존K_P,K_C형성·binding을고정했다. K_E는invalidNULL/content0copy0. 기존V88+필수V74고정codec이며V7 AR/Diana/426M사양과혼동하지않는다.

실제V112 owner4584→36084/worker30056→2760, V113 owner30172→37836/worker40400→32908 생성·parent·command·생존을ROOT가관측했고worker-ownerexit0/allgone/freshPython0 FIRST후분석했다. 타이머삭제 유지, 새C출력만사용, 게임·원본·과거weights/AdamRNG/실패 삭제이동압축0. D기존부모는읽기만했다.

현재 V113은 TRAIN 부분적 조건변환을 보인 연구후보이며 자동승격·기존분기폐기0이다. 일반화는미해결이다. 이전보고의엄격관계문법0을의미0으로해석하지않는다. V112에서도 H is largeer hhn B.r라는손상된 core1개이상은H>B로읽을수있었다. 당시0은자동정상문법성적이다.
'''
    reports['02_ANSWER_TARGET_CONTRACT_KR.md']='''# 02. 수정 수식과 누수 경계

**VERIFIED** source E(X)+고정K_P/K_C+K_E NULL+독립ε → 실제16step Rθ → 확정 Zhat → 고정D이다. 정답문장·E(Y)·정답길이는추론조건이아니다. target은여전히해당행E(Y), FP32(1,35,512)=global2+local33이며body/tail mask는loss에만쓴다. labels는(34,16)의byte/EOS256/PAD257이다.

V112에서실제58역전짝에만공통독립ε를주고 .25*||(Zhat_a−Zhat_b)−(Zy_a−Zy_b)||²의body평균을추가했다. 같은body mask에서행별endpoint평균은 ||e_mean||²+.25||e_a−e_b||²이므로차이계수가.25→.5가된다. UNKNOWN26짝의target차이는0,18singleton은독립ε·차이loss0이다. 서로역전한답문이같은canonical사실을표현할수있음을보존한다.

V113 L=bodyMSE+.05tailMSE+.1VPaux+.25실제짝차이+.5contentbyteCE다. CE는실제target-freeZ를고정D에넣은뒤계산하며정답prefix로decoder를조건화하지않는다. EOSaux0/새AR0/template0/NN검색0/Exactcopy0. E/D가중치는동결되고고정D입력에대한미분만θ에전달된다. 따라서여러모델을공동학습한것이아니다. 생성중의언어피드백은없고훈련손실의피드백만있다.

추가CE는MSE에충분히반영되지않은개체·관계의D결정경계를감독하려는후보이다. 계수.25/.5가최적이라는증거는없다. V112의공통잡음+차이손실, V113의추가학습량/freshAdam 영향은각각의인과실험으로분리되지않았다. VP는16논리단계가아니라16노이즈복원step이다.
'''
    reports['03_TRAINING_PLAN_KR.md']='''# 03. 실행한 유한 계획과 사전검사

**VERIFIED** 두run각각134TRAIN×16finiteepoch=2144행/37088정답UTF8byte, batch8/원자단위2행/268attempt, LR1e-4/weightdecay.01/clip1/같은cosine를사용했다. 58실제짝+18singleton을보존했고각행매epoch정확1회,1216독립noise draw다. 노출행을새독립자료로세지않는다. DEV편입0/E2final·registry접근0.

V112예산41000call/expected40358, V11343000/42502. 각각owner1200초/cut1080/reserve120/C전체1GiB. C실행하한50GiB/owner54GiB조기정리/시작101GiB+64MiB로종료100GiB보호. output3fullstates와STOP임시여유까지포함했다. 이전미사용예산재사용0/자동재시작0/상한확대0/실행중봉인코드변경0/체크포인트회전삭제0.

V112 focusedCPU7PASS,실제fullsizepreflight388calls/fixture·재생4optimizer실행, V113 focusedCPU3PASS/fullsize424calls/fixture·재생4실행이다. 각각populated Adam/scaler/ALL RNG/중간scaledgradient/MID와6행READY다음update정확,입력checkpoint불변/codec고정/이전TRAIN·DEV2개출력bitwise검증했다. 본연구536update와fixture8회는별도다. 실제V113고정D CE→θ gradient norm6.230322>0을확인했다. 이는한행진단이며최적계수의증거가아니다. CUDA복구검사라고하지않는다.

기존230baseline출력을재사용해반복baselineforward0이었다. 최종230행은동일row/ε/decoder조건으로실행했고V113훈련D2144call이추가됐다. D는학습가중치가아닌고정미분경로였다. 독립Windowsowner는Torch를불러오지않고모든postPopen예외·자연종료의verifiedowned cleanup을보장하는기존코드를유지했다.
'''
    ck='\n'.join(f"|{i}|{ref['bytes']}|`{ref['SHA']}`|" for i,ref in zip((0,134,268),r['checkpoints']))
    reports['04_TRAINING_RESULTS_KR.md']=f'''# 04. V113 학습과 저장 검증

**VERIFIED** 268attempt268update/AMPskip0/2144행/37088UTF8byte, 학습thought36448+D2144/평가thought3680+D230, 전체42502calls/worker{r['seconds']:.3f}초다. 이전V112는40358calls/642.193초였다. V113목적에서고정D forward를썼다는점과D가중치를학습했다는것을혼동하지않는다.

|V113 checkpoint update|bytes|SHA256|
|---|---:|---|
{ck}

3fullstates의파일SHA/크기/weights-onlysafeload/재귀digest, 초기θ전부부모일치와빈Adam,최종64θtensor변경/Adamstep268/scaler1024tracker268/gradient정리/모든134행16회/순서·노출/RNG1216draws정확재생을검증했다. 고정codec전체modelSHA는 `42a2032db3137ffb9f876585cf26979854d2555e19639fa5c2f1e8d61201b96a`, 전후같다. frozen endpoint digest도3상태에서같다.

|학습8update평균|처음|마지막|
|---|---:|---:|
|bodyMSE|{curve['first8']['body']:.6f}|{curve['last8']['body']:.6f}|
|contentCE|{curve['first8']['byte_CE']:.6f}|{curve['last8']['byte_CE']:.6f}|
|combinedloss|{curve['first8']['combined_loss']:.6f}|{curve['last8']['combined_loss']:.6f}|
|짝차이MSE|{curve['first8']['contrast']:.6f}|{curve['last8']['contrast']:.6f}|

추가NN/optimizer0인독립사후검토에서460전후raw의파일SHA/UTF8·EOSdecode/NLL/body-tailMSE와모든정답byte/EOS위치full258logit을검증했다. 긴생성tail전체258는미저장이다. 저장goldLP의gather−logsumexp는460행전부bitwise재현됐고FP64 log_softmax기준NLL최대오차2.5814e-6였다. 초기사후검사의고정2e-6허용오차실패는FP32연산형태차이여서실제식bitwise+FP64오차경계검사로교정했다. 데이터·점수·모델을바꾼것이아니다.

최종checkpoint는학습DONE후/최종평가전상태이며230최종평가참조는별도assessment/result에보존했다. 모든새trial은CLOSED다.
'''
    table='\n'.join(f"|{name}|{g['baseline/'+name]['NLL']:.6f}→{g['final/'+name]['NLL']:.6f}|{100*g['baseline/'+name]['byte_accuracy']:.3f}→{100*g['final/'+name]['byte_accuracy']:.3f}%|{g['final/'+name]['positive_grammar']}/{g['final/'+name]['positive_rows']}|" for name in ('TRAIN_all','dev_pair','dev_surface','dev_position'))
    reports['05_SEMANTIC_EVALUATION_KR.md']=f'''# 05. 실제 의미·일반화 평가

**OBSERVED** V113은제한된TRAIN에서실제정상관계core를생성하기시작했다. ROOT가전체80TRAIN긍정출력을모두읽었다(max149byte). 정상문법으로맞는관계core34/80와 `C is smallertthan F.`라는명확한철자손상1개를별도로확인했다. 최소35/80core를읽을수있었다. 이는전체80개의일반의미정확도를인증한숫자가아니며, 정답뒤잡음이있어서긍정전체문자열exact는0/80이다. 전체TRAIN exact17/134는모두UNKNOWN이다. UNKNOWN은54/54가정답prefix지만뒤noise를완전성공으로숨기지않는다.

|동일조건패널|NLL 전→후|byte accuracy 전→후|자동정상관계core(긍정)|
|---|---:|---:|---:|
{table}

직접읽은예시:
- `H is larger than B.rly` → H>B core맞음, 뒤잡음.
- `F is smaller than C.CA    .faC a &` → F<C core맞음, 뒤잡음.
- `C is smallertthan F.` → C<F 의미를읽을수있으나철자손상(자동문법은miss).
- 정답F>C에 `C is larger than F..Ceia f` → 관계가반대라실패.
- 정답A>G에 `A is larger than B.Cltnp...` → 객체오류.
- 새DEV정답C>D에 `H is larger than D.Smer iemana` → 관계word는맞지만주체오류.

DEV긍정48개앞150자를직접읽었고확인된정상정답core0이었다. 긴DEV잡음tail전부를수동판독한것은아니다. UNKNOWN상수기준은TRAIN54/134=40.30%, DEV각16/32=50%다. 전체정답률이나byte향상만으로이기준을넘는일반의미이해를주장하지않는다.

32긍정역전쌍정상관계bothcorrect0→8/32, 동일출력2→0, target방향cosine.668333→.845888, target차이에대한투영비율.310958→.524632, 차이MSE1.479526→.861468. 이잠재거리비율들은의미정확도가아니다. 8짝의TRAIN질문역할반응은제한된부분학습의증거이며메모리개별인과·독립추상사고·Composition증거는아니다.

TRAIN만으로맞춘최적상수bodyMSE{c['TRAIN_only_fitted_constant_body_MSE']:.6f}보다현재{c['current_allTRAIN_body_MSE']:.6f}가낮다. 따라서현재를전부상수출력이라고부를수없다. 반면미관측사실로일반화했다고도할수없다.

**VERIFIED 자료점검** TRAIN134/31factkeys/27answers/family0–3(23/31/50/30),layout0/1=56/78. dev_pair는fact/groupseen0/32이고긍정정답seen0/16이다. 최종DEV긍정16중12개는TRAIN의정형답문core를대신출력했다. 이는새개체조합일반화부족과일치하지만메모리검색답안반환코드가있다는뜻은아니다. 실제추론에답안검색/GT조건은없었다.

dev_surface/position은fact와답문이모두TRAIN에있으나실패했다. 단순미관측답문문제만으로설명할수없고질문표현·위치과의존도남는다. NLL은모든DEV에서악화했으므로새모델을전체개선으로승격하지않는다. 이미관측한개발자료의다중사용·선별bias·짧은A-H자료한계를유지한다.
'''
    reports['06_DECODER_BOUNDARY_AUDIT_KR.md']='''# 06. 고정 출력기 경계

**OBSERVED** 같은고정D는실제27종E(Y)직접입력에서정답prefix27/27, whole0/27,뒤1–12byte였다. V111실제검사와hash가같아재호출하지않았다. 따라서정답내용읽기능력과종료문제를구분한다.

V112에서조건구별은좋아졌지만 lmaleer 등혼합철자가남았다. V113에서고정D의contentCE를θ로역전파하자TRAIN 관계word정확3→74/80, 주체9→46/목적어29→63/80, 정상관계core34개가생겼다. 출력기자체는변하지않았다. 따라서생성잠재가고정D의결정경계에맞도록조정되는것이내용충실도에중요하다는관측이다. CE만의단독인과효과는별도통제하지않았다.

TRAIN noEOS5→2, lateEOS114/134, positivewholeexact0/80다. UTF8는모든raw가유효하지만고정된neuralUTF8결정trace의성질이며정상문장·의미정확성증거가아니다. 정답길이crop으로성적을올리지않았고EOS보조loss는이번에추가하지않았다.

새DEV에서문법이정상인데주체가틀리는사례는남은실패전체가EOS만의문제가아님을보여준다. 또한TRAINcore성공을고정E의추상의미완성/긴입력사고/범용LM으로승격하지않는다. 현재D는비AR Commit→MLP Byte/EOS, V7의context우회·ARprior반복진단을여기에자동적용하지않는다.

후속종료충실도는별도θ-onlyEOS실험후보이지만현재최우선은새사실·표현에서올바른개체·관계를생성하는조건일반화다. 메모리필수counterfactual episode와장문Composition은아직미실행이다.
'''
    reports['07_NEXT_RESEARCH_DECISION_KR.md']='''# 07. 다음 연구 판단

**VERIFIED / OBSERVED** 단순반복이아닌두수정·유한학습·전체평가를수행했다. 짝차이수식으로조건반응을키웠고고정Dcontent피드백으로TRAIN언어내용이개선됐다. TRAIN정상관계core34/80+명확한typo1, 역전짝bothcorrect8/32는이전보다진전이다. 하지만DEV긍정은아직해결되지않았고NLL도악화해전체품질개선이나일반화완료를선언하지않는다.

**INFERRED** 현재다음병목은134행/27답문에대한조건부적합에서새개체·관계조합과표현으로넘어가는일반화다. dev_pair의정답positive16개전부학습에없고12개에서학습답문을대신출력했다. 표현·위치패널은학습에있는fact/answer도틀리므로새답문조합뿐아니라배치·질문읽기도다뤄야한다. 이근거로같은134행을더오래반복하는것을다음주목적으로삼지않는다. 의미기억의내용이부재하거나디퓨전이불가능하다는판정도하지않는다.

**PROPOSED 다음별도목적**

1. 원래TRAIN분할의허용된fact/group으로새source-only문장을작성하고, 관계방향/주체/목적어/질문순서/문맥위치를균형화한다. 관측DEV의실제행·heldpair/group·family4/5를TRAIN으로옮기지않는다. 미관측답문구성은별도사전등록분할로둔다. 기존384중탈락250행을무단복구하지않는다.
2. 사람지시대로새TRAIN후보도고정E→D왕복검사를먼저수행해통과한행과실패행을별도manifest로보존한다. 이선별은codec검증이며semanticgeneralization증거가아니고선별bias를보고한다. 정답E(Y)는loss-onlyfile, 조건packet은source-only로계속분리한다.
3. 셋업전에성분·literal/family/pairgroup중복·label균형·실제노출을계산한다. 새실행은별도코드/정확복구/고정time·call·output상한/C100GiB종료보호를검증한ONEtrial만쓴다. 지금아직새data/runner/학습은없으며본CLOSED예산을재사용하지않는다.
4. 같은질문·서로다른합법적K_C의기억필수episode는기본Transformation일반화의증거가생긴뒤별도실험으로한다. K_Pcertifiedwriter/장기통합·미래Exactcopy/Composition·장문사고는성적에섞지않는다. 입력·출력가중치는여전히고정한다.

기존V111/V112/V113fullstates·Adam/scaler/RNG·실패·raw는모두보존했다. V113의훈련성과만으로전역기본모델이나외부발표성적을자동변경하지않는다. 타이머재개0이다.

실제가능한안전검증명령(프로젝트루트):
```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m pytest -q -p no:cacheprovider tests/test_answer_codec_fidelity_v113.py
Get-Content -LiteralPath 'runs/answer_codec_fidelity_20261114_v113/root_result.json' -Raw
Get-FileHash -LiteralPath 'runs/answer_codec_fidelity_20261114_v113/checkpoints/state_0268.pt' -Algorithm SHA256
```
닫힌owner/worker명령을다시실행하지않는다. 다음자료분기용미구현명령을만들어제출하지않는다.
'''
    for name,text in reports.items():write(b/name,header+text)
    summary=dict(status='PARTIAL_TRAIN_TRANSFORMATION_LEARNED_DEV_UNRESOLVED',actual_updates=268,new_research_updates_this_turn536=True,
        checkpoint=r['checkpoints'][-1],training_D_calls=2144,D_weights_frozen=True,E_and_memory_extractors_frozen=True,
        TRAIN_positive_normal_relation_core34_of80=True,manual_readable_positive_core_lower_bound=35,
        positive_whole_exact=0,DEV_positive_correct_observed=0,DEV_positive_total=48,
        TRAIN_positive_pair_bothcorrect=8,TRAIN_positive_pairs=32,automatic_promotion=False,
        all_old_assets_preserved=True,closed_budget=True,next_goal='New TRAIN-only role/composition/surface coverage; no blind repeats')
    io.atomic_json(b/'root_result.json',summary)
    corrections=dict(no_training_failure=True,review_initial_absolute_tolerance_failure='log_softmax vs gather-logsumexp FP32 roundoff up to2.86e-6; actual stored460 answerLP reproduced bitwise,FP64max2.58e-6',
        dataset_initial_saved_audit_KeyError='source_fact.key is actual schema, not flat fact_key; corrected before deriving counts',
        automatic_grammar0_NOT_semantic0=True,no_model_data_or_scores_changed=True)
    io.atomic_json(b/'root_analysis_corrections.json',corrections)
    index='# V112→V113 문제 분석·수정·학습 결과\n\n이번 추가536update를 완료했다. V113 TRAIN긍정core 최소35/80, DEV긍정0/48로일반화는미해결이다. 전체입출력·메모리는동결했고모든프로세스종료를먼저확인했다.\n\n'
    index+='\n'.join(f'- [{name}]({(b/name).as_posix()})' for name in reports)
    index+=f'\n\n[V112 실제 결과]({(a/"RESULT_AND_NEXT_KR.md").as_posix()})\n'
    write(b/'README_RESULTS_KR.md',index)
    write(W/'outputs/FDT_ANSWER_REPAIR_V112_V113_RESULT_20261114_KR.md',index)
    print(json.dumps(dict(status=summary['status'],reports7=True,checkpoint=summary['checkpoint'],core35=True)))


if __name__=='__main__':main()
