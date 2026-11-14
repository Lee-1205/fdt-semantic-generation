"""Close the actual scope experiment; preserve all older models and evidence."""
import datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    run=ROOT/'runs/last_linear_20261113_v79';work=ROOT.parent
    p=io.load_json(run/'plan.json');a=io.load_json(run/'result.json');s=io.load_json(run/'root_saved_review.json');m=io.load_json(run/'root_meaning_review.json')
    t=io.load_json(run/'root_training_and_boundary_review.json');d=io.load_json(run/'root_scope_delta_and_next_math.json')
    terminal=io.load_json(run/'ROOT_terminal_native.json');native=io.load_json(run/'ROOT_postreview_native.json');owner=io.load_json(run/'final_status.json')
    assert owner['status']=='COMPLETE' and owner['exit_code']==0 and owner['exit']['all_owned_processes_gone']
    assert terminal['all4owned_and_proxy_gone'] and terminal['before_result_analysis'] and not terminal['PythonProcesses'] and not native['PythonProcesses']
    assert s['status']=='ROOT_V79_SAVED_REVIEW_PASS' and m['status']=='ROOT_V79_SOURCE_BOUND_MEANING_READY'
    assert a['attempts']==a['updates']==384 and a['AMP_skips']==0 and sum(a['calls'].values())==6752 and native['C_free_bytes']>100*1024**3
    cards=io.load_json(run/'root_manual_reading_cards.json');observed=io.load_json(run/'observed_V75_DEV_reading_cards.json')
    assert (len(cards['new_after']),len(observed['after']),len(cards['long_after']),len(cards['long_regions']),len(m['long_after']['unique_changed_field_patterns']))==(22,20,72,6,42)
    assert m['new64_two_values_after']['all_two_value_fields_clear_rows']==56 and m['long_paired']['quantities']['after']==29
    assert sum(v['changed'] for v in d['model_delta_vs_V75'].values())==2
    now=datetime.datetime.now().astimezone().isoformat();stamp=now[:16].replace('T',' ')
    selection=dict(time=now,decision='KEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V79',selected_version='V75',selected_runtime=p['cycle_source'],selected_SHA=p['cycle_sha'],
        mandatory_standalone_base=p['storage_parent'],mandatory_base_SHA=p['storage_parent_sha'],rejected_as_default='V79',V79_final_SHA=a['latest']['sha256'],all_V76_V77_V78_V79_states_preserved=True,
        reason='Same revised64 two quantities54->56/64, but long quantities16->29,episode10->22,decoded58->77; observed75 stock14/32 unchanged vs V75 and below V78 20/32. Scope restriction alone does not solve retention.',
        deployed_registry_changed=False,weight_files_modified=False,automatic_extra_training=False,all_unused_budgets_closed=True)
    config=ROOT/'configs/research_base_selection_v79_review_20261113.json';assert not config.exists();io.atomic_json(config,selection);io.atomic_json(run/'root_research_base_selection.json',selection)
    reading=dict(status='ROOT_V79_ACTUAL_READING_COMPLETE',time=now,new64_nonliteral_full_clauses_read=22,observed75_nonliteral_full_clauses_read=20,
        all72_long_changed_event_contexts_read=True,long_unique_changed_field_patterns=42,start_middle_end_contexts_read=6,
        all524071bytes_individually_manually_scored=False,all_old128_OWT_semantics_individually_manually_certified=False,
        semantic_judgment='450.25 vs +50.25 and 22->+2 change the quantity. 17->16 and actor Cyril->Ckril are real fidelity risks. Reviet and drawer->draber are output word damage; meaning cannot be fully certified by quantity slots alone.',
        retained_structure='Restricted before/after ordering, color-location labels and negation fields unchanged in both long rows. Full UTF8/EOS length retained, with localized errors; not global collapse.',
        boundary_caveat='New64 review7109;->71090 and Review7110->Review7 10 outside the two quantity slots; long145+/169+ outside fixed episode spans. Fixed span counts preserved, adjacent risk cards separate.',
        inference='Frozen input/memory/binding and frozen five readout feature tensors plus only two changed finalLinear tensors locate this variant output changes downstream of the same Commit. They do not certify abstract meaning quality or establish cause of all older errors.',
        extra_model_calls=0)
    io.atomic_json(run/'root_cause_and_reading_review.json',reading)
    nextfile=work/'outputs/FDT_READOUT_UPDATE_RETENTION_NEXT_CONTRACT_20261113_KR.md'
    nextfile.write_text('''# V79 이후: 출력 갱신의 보존 조건과 TRAIN coverage

다음 초기화는 V75이며 V79 자동 연장/승격은 없다. V75~V79의 코드·자료·실제 결과를 먼저 참조한다. 현재 목표는 영어 language→meaning→language의 안정화이며, 디퓨전은 아직 학습하지 않는다. 미래 디퓨전은 전체 입력·메모리·바인딩·출력 가중치와 buffer를 고정하고 별도 사고 모듈만 학습한다.

V79는 V78과 동일한 loss/weak margin/계수/그룹/1,920행 노출/384회/LR/seed에서 출력 prefix까지 고정했지만, 긴 수량 오류29/episode22로 V75의16/10을 보존하지 못했다. 따라서 다음 실험을 같은 2개 텐서에 같은 384회 더 적용하거나 λ만 올리는 작업으로 정하지 않는다. 입력 의미가 모두 완벽하다는 뜻도, Diffusion이 실패했다는 뜻도 아니다.

구체적인 다음 CPU 작업은 TRAIN 내 보존 대상과 수리 대상의 **실제 갱신 방향** 계약이다. 현재 correct weak margin<=3는 critical 중1,943/45,063위치만 선택했다. 반복 노출과 그룹 loss 평균은 unseen margin 보존 보장이 아니다. 기존 unsigned128/OWT128/signed768의 표현·자리·숫자 경계 coverage를 먼저 좁게 점검하고, 평가의 실제 문장/정답/장문 event를 TRAIN에 가져오지 않는다. 새로운 prospective TRAIN을 만들 경우 이름·숫자·어순/문장길이·정규화/중복/상관 view와 노출을 따로 기록한다.

학습 수리 gradient와 올바른 TRAIN 사실의 retention gradient를 구분하는 후보는 검토할 가치가 있다. [GEM 원 논문](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)과 [PCGrad 원 논문](https://arxiv.org/abs/2001.06782)은 gradient 제약/충돌 완화의 근거다. 이 논문이 현재 모델의 충돌이나 개선을 입증하는 것은 아니다. 논문의 episodic replay는 FDT K_E Exact memory 활성화와 전혀 다른 학습 표본 재사용 기법이며, K_E는 계속 invalid NULL/content0/copy0다.

**그대로 PCGrad를 Adam 앞에 붙여 안전하다고 부르지 않는다.** ROOT의 CPU 수치 반례에서 g_ret=[1,1], g=[2,-1]이면 dot=1로 raw-gradient 검사는 통과한다. 그러나 양의 diagonal preconditioner [.1,10]에 따른 Δθ=[-.2,10]은 g_ret·Δθ=9.8로 retention 손실을 1차적으로 증가시킨다. 모멘텀/weight decay도 실제 Δθ와 함께 검토해야 한다. 이것은 analytic toy 계산이며 실제 FDT gradient 계측이나 optimizer 구현이 아니다.

후보가 채택되려면 최종2개 optimizer 소유권/AMP unscale/부분 그룹/skip/거절된 step의 Adam·scaler·모든 RNG·cursor·노출 보존을 실제 driver CPU 계약으로 정한다. raw gradient projection만으로 보호를 주장하지 않고, 실제 Adam 제안 update 및 TRAIN 참조 손실/경계의 수치 변화를 기록한다. 같은 이미 관측한 개발 패널로 세부 threshold를 역산하지 않는다. 한정 실험마다 사전 고정한 목적/caps와 독립 검토를 갖고 실행하며, broad audit나 반복 baseline을 무한 선행조건으로 만들지 않는다.

현재 C: 약109.08GiB는 다음 동일 owner 시작조건109.25GiB보다 작다(100GiB 보호는 유지). 다음 GPU 예산을 무작정 다시 사용하지 않는다. 새 raw 저장이 float32라면 lossless BF16 roundtrip 여부를 실제 저장값에서 CPU 확인하고, 정확 복원되는 경우에만 새 저장 형식/실제 estimate/hard cap을 낮추는 경로를 검토한다. 원시값이 달라지면 그 형식은 거절한다. 기존 raw/체크포인트/AdamRNG/source/게임은 지우지 않는다. 준비는 새 GPU 실행이나 능력 향상 증거가 아니다.

아직 V80 worker/plan/seal/GPU는 없다. V79의 미사용248 calls/시간은 닫혔다. K_P prior와 K_C 장단기 fuzzy를 유지하고 raw input/GT prefix/GT length/정답 역할/regex를 출력 feature로 우회하지 않는다. K_P certified writer는 미완료다.
''',encoding='utf8')
    report=work/'outputs/FDT_LAST_LINEAR_V79_RESULT_20261113_KR.md'
    report.write_text(f'''# V79 실제 결과 — 마지막 출력층만 학습해도 보존 문제 잔여 ({stamp} KST)

V79는 한 번의 한정 실행을 완료했다. V75의 전체 입력·퍼지메모리·Commit 바인딩 및 출력 terminal/LayerNorm/첫 Linear 5개를 고정하고 **마지막 Linear weight/bias 2개(2,117,664 parameters)**만 학습했다. V78의 loss, weak teacher margin<=3, 계수, 384개5행 그룹, 1,920행 노출, LR1e-5와 seed를 유지해 trainability 범위만 비교했다. Clipping/Adam 부분공간도 달라지므로 개별 prefix 부품의 인과증명은 아니다.

결정은 **V75를 다음 연구 초기화로 유지**다. 짧은 두 수량 패널의 일부 회복은 있었지만 장문 숫자 보존 개선으로 이어지지 않았다. V79와 이전 모든 실제 가중치/Adam/scaler/RNG/raw/실패 자료를 보존했으며 기존 모델/registry를 덮어쓰지 않았다.

| 같은 관측 지표 | V75 | V78(7텐서) | V79(2텐서) |
|---|---:|---:|---:|
| 두 수량 모두 읽히는 행 /64 |54|56|56|
| 첫 수량 /64 |64|64|64|
| B kg /16 |10|12|12|
| B 온도 /16 |14|16|16|
| B km /16 |14|14|14|
| B items /16 |16|14|14|
| 다른 사실을 바꿔도 유지된 값 /64 |59|60|60|
| 기존 V75 패널 stock 엄격 보존 /32 |14|20|14|
| 기존 양의 kg 엄격 보존 /72 |59|60|59|
| 장문 수량 오류 /5436 필드 |16|28|29|
| 장문 episode fixed-span 오류 |10|25|22|
| 장문 인명 오류 |2|2|1|
| 장문 item 오류 |6|6|8|
| 장문 decoded byte 차이 /524071 bytes |58|76|77|

장문은 실제262042/262029UTF8 bytes의 두 입력으로 최대 TRAIN16384보다 약16배 길다. 2718개의 상관 event이며 5436개의 수량 필드다. V75→V79 수량은 교정0/신규13/기존16; V78→V79는 교정2/신규3/잔여26이다. Episode V75→V79는 교정1/신규13/잔여9다. 제한된 before/after 관계·색상 위치·부정 필드는 계속0변화였고 UTF8/EOS 전체 길이는 유지했다. 드문 오류가 실제로 있으나 문서 전체 의미가 붕괴했다고 과장하지 않는다. 반대로 이를 자연어 의미가 완벽하다는 인증으로 쓰지도 않는다.

직접 읽은 실제 오류: +50.25kg→450.25kg는 값 변화, 75 washers→7; washers는 수량 훼손, 22 ribbons→+2 ribbons와17→16은 사실 변화다. `Reviet`, `drawer→draber`, Cyril→Ckril, silver→s lver도 남았다. 긴 행1007에서는 원문 `28 silver`가 `2Üsilver`가 되어 정렬된 수량 substring만의 판별보다 위험하다. 숫자 뒤 경계도 확인해 review7109;→71090 및145+/169+를 별도 위험으로 기록했다. V77/V78의1089→10893,53→533 병합은 V79에서 관측되지 않았다. 기존 fixed-span 점수를 소급 변경하지 않았다.

ROOT는 새 패널 비문자일치22개, 기존V75 패널 비문자일치20개, **장문 변경 event72개 전체 문맥**과 시작·중간·끝6개를 원문과 대조했다. 변경 field pattern42개다. 전체524071byte를 사람이 하나씩 의미 채점하거나 모든 OWT/old128 문장을 수동 인증한 것은 아니다. 관측 개발 자료를 여러 번 재사용했으며 새로운 blind benchmark가 아니다.

새64 패널 NLL은 .00668718→.00636117로 낮아졌지만 byte/EOS accuracy는99.8588%→99.8460%였고 auxiliary literal은44→42/64였다. NLL·byte accuracy·수량 보존은 서로 다르다. OWT decoded 오류108→117, 기존128은54→56, 표현패널8→11, retained English3→3이었다. 기존V75 primary64개는 유지했지만 stock의 unit 누락2개/값과unit동시훼손2개도 남아 숫자 core만으로 성공 처리하지 않았다.

전체 NN과 buffer를 독립 재구성해 upstream과 출력 prefix5개가 bitwise 동일하고 **정말 마지막2개만 달라졌음**을 확인했다. Δweight L2=.3988028/max=.00244912, Δbias L2=.0170553/max=.00107134였다. 같은 새64 Commit도 bitwise 같았다. 따라서 이번 출력 변화는 같은 Commit 아래 마지막 출력층 갱신에서 발생했다. 이것은 모든 입력 의미가 완벽하다는 증거나 모든 역사적 오류가 출력층만의 문제라는 증거가 아니다.

TRAIN teacher에서 correct critical45063 중 weak1943, correct noncritical964665 중 weak2995만 margin 대상으로 선택됐다. 실제 margin 항은 적용됐고 critical84/noncritical416 반복 위치 위반, 293/384그룹 clipping, 평균 margin항 .00339214/.00181836이었다. 미관측 사실 보존 보장은 아니다. wrong-plus는8→33(V78 36), digit→plus8→30(V78 34)였다. prefix 고정만으로 sign/숫자 간 출력 간섭을 해결하지 못했다. 실제 task gradient 충돌을 계측했다고 주장하지 않는다.

실행:384attempt/384update/AMPskip0, {a['seconds']:.6f}worker초, encode3076+readout3418+chunks258=6752calls, longblock2322, denoise0. Teacher1024행1010943UTF8B, head1920행2720189UTF8B/86108critical positions. 새7000calls/900wall780cut120reserve/1.25GiB budget은 **CLOSED**, 미사용248calls/시간 재사용·자동 재학습은 없다. GPU peak {a['training_GPU_peak_bytes']}B/teacher{a['teacher_GPU_peak_bytes']}B.

Owner508→19884(10:01:14.859/.884), worker17124→2136(10:01:23.073/.087)의 생성·parent·command·exe·생존을 ROOT10:01:43.881에 관측했다. Worker/owner exit0 후 ROOT10:07:12.864에 모든 owned/proxy 종료와 freshPython0을 확인한 뒤 분석했다. 실제 학습 중 ROOT는 상세 분석을 중단하고 OS 감시만 했다. 마지막 C{native['C_free_bytes']}B={native['C_free_bytes']/1024**3:.2f}GiB로100GiB 보호 유지/현재 모델 프로세스0.

ROOT는 {s['all_pins_rehashed']}pins/3fullstates, populated2Adam step384/scaler tracker384/모든 저장RNG/finiteorder·cursor·384그룹1920micro history·노출·teacher1024packets와470raw+parity1을 독립 재구성했다. CPU3focused/tiny실제엔진 복구 및 actualfullsizeCPU2encode2readout/optimizer0/CUDAfalse도 구분해 통과했다. Teacher full logits는 미저장이므로 teacher전체argmax/margin을 raw에서 재계산했다는 주장은 하지 않는다. 추가 postrun NNforward0/실제CUDA 복구 재실행0.

저장 수정도 검증했다. V75의 prefix5개는 standaloneV74와 다르므로 파일에는7개 delta를 저장하되 optimizer는2개만 소유한다. 2개만 저장하면 V75 prefix가 사라지는 사례를 CPU에서 거절했다. Direct standaloneV74 base를 참조하며 recursive model link/Exact 문장 검색이 아니다. 최종{a['latest']['bytes']}B SHA `{a['latest']['sha256']}`, algorithm `{a['latest']['algorithm_sha256']}`; plan `{support.sha(run/'plan.json')}`. Mandatory base `{p['storage_parent_sha']}`를 보존한다.

K_P learned prior/K_C input long/recent fuzzy는 유지, K_E는 invalidNULL/content0/copy0다. K_P certified writer는 아직 없다. 현재 언어 복원 수리 단계이고 디퓨전 호출0이다. 미래 사고학습은 전체 input-memory-binding-output codec을 고정하고 별도 사고 모듈만 학습한다. Whole language→meaning→language 완료/독립 abstract 의미 인증/자동 승격은 하지 않는다.

다음 구체적 계약은 [출력 갱신 보존·TRAIN coverage](FDT_READOUT_UPDATE_RETENTION_NEXT_CONTRACT_20261113_KR.md)에 저장했다. 동일2개를 같은 목적·같은384회로 자동 반복하지 않는다. GEM/PCGrad 원 논문은 그 문서의 출처 링크로 확인했으며, raw gradient 제약이 Adam 실제 update에 그대로 성립하지 않는 수치 반례도 계산했다. 새 GPU/생성 능력 개선 증거는 아니다.
''',encoding='utf8')
    files=[run/n for n in ('plan.json','result.json','final_status.json','ROOT_native_start_observation.json','ROOT_terminal_native.json','ROOT_postreview_native.json','root_saved_review.json','root_meaning_review.json','root_manual_reading_cards.json','observed_V75_DEV_reading_cards.json','root_training_and_boundary_review.json','root_scope_delta_and_next_math.json','root_cause_and_reading_review.json','root_research_base_selection.json')]
    files.extend([config,report,nextfile,ROOT/'scripts/review_last_linear_v79_saved_cpu.py',ROOT/'scripts/review_last_linear_v79_meaning_cpu.py',ROOT/'scripts/review_v79_training_and_token_boundaries_cpu.py',ROOT/'configs/reasoning_only_training_policy_20261113_v1.json',Path(__file__)])
    result=dict(status='ROOT_V79_ACTUAL_COMPLETE_INDEPENDENTLY_REVIEWED_NOT_PROMOTED',time=now,actual_attempts384_updates384_skip0=True,actual_seconds=a['seconds'],actual_calls=a['calls'],source_exposure=a['exposure'],
        all_owned_proxy_gone_before_analysis=True,latest_native=native,active_parameters=2117664,active_tensors=2,serialized_deltas=7,frozen_prefix_deltas=5,
        final_SHA=a['latest']['sha256'],final_algorithm_SHA=a['latest']['algorithm_sha256'],final_bytes=a['latest']['bytes'],mandatory_standalone_V74_SHA=p['storage_parent_sha'],parent_required_NOT_standalone=True,
        saved_review=s['status'],meaning_review=m['status'],selection=selection,old_weights_Adam_scaler_RNG_raw_failures_sources_and_games_preserved=True,
        closed_budget=dict(attempts=384,max_calls=7000,actual_calls=6752,unused248_calls_time_NOT_reusable=True,wall900_cut780_reserve120=True,automatic_retry=False),
        extra_postrun_model_calls=0,Exact_content_entries=0,copy_bytes=0,denoise_calls=0,whole_language_meaning_language_complete=False,
        latest_HUMAN_future_diffusion_freezes_ALL_codec_and_trains_ONLY_thought_unchanged=True,actual_next_worker_plan_seal_GPU_exists=False,
        owned_output_bytes=sum(f.stat().st_size for f in run.rglob('*') if f.is_file()),evidence_SHA={str(f):support.sha(f) for f in files})
    assert result['owned_output_bytes']<p['max_output_bytes'];io.atomic_json(run/'root_result.json',result)
    current=f'''## CURRENT — {stamp} KST V79 COMPLETE reviewed / lastLinear-only does not solve retention / keep V75

- LatestHUMAN future diffusion MUST freeze ENTIRE input-memory-binding-output codec and train separate thought weights ONLY. CurrentEnglish language→meaning→language repair/meaningPRIMARY/denoise0/ROOTONLY/noagents/K_Pprior+K_Cdynamicfuzzy/K_EExactNULLcontent0copy0/Commit-onlyD/noHraw-GTprefix-length/K_Pcertifiedwriterabsent/timer30min재개/C100GiBfloor/alloldweightsAdamRNGsourcesrawfailuresgamespreserved.
- V79 ONEactualCOMPLETE384attempt384update/AMPskip0/{a['seconds']:.6f}worker sec/1920live source micros. V75warmexact; ONLYlastLinear2tensors2117664paramsfreshAdam1e-5. Entireinput-memory-binding+Dterminal/norm/firstLinear5tensorsfrozen. SAMEV78loss/weakteacher<=3/.25-.10/groups384x5(U/S/O/U/S)/.2each/1920headrows/1024teacher/seed77113/sameobservedpanels; trainability/clippingAdam-subspaceonlychanged/notindividualprefixcausality.
- PlanSHA{support.sha(run/'plan.json')}. Owner508→19884born10:01:14.859/.884/worker17124→2136born10:01:23.073/.087;ROOTall4creation-parent-command-exe-live10:01:43.881. Worker-ownerexit0/ROOTallownedproxygone+freshPython0verified10:07:12.864 BEFOREanalysis. ActualtrainingdetailedanalysisOFF/OSonly/noagents. LatestC{native['C_free_bytes']}B~{native['C_free_bytes']/1024**3:.2f}GiB>100floor/currentmodelprocess0.
- Actualencode3076+readout3418+chunks258=6752/longblock2322/denoise0/GPUtrain{a['training_GPU_peak_bytes']}B/teacher{a['teacher_GPU_peak_bytes']}B. New384attempt/7000call/900wall780cut120reserve/owned1.25GiB CLOSED/unused248calls-timeNOTreusable/noautorestart-clockreset-capgrowth-sealededit-rotation/alloldbudgetsclosed. Final{a['latest']['bytes']}BSHA{a['latest']['sha256']}/algorithm{a['latest']['algorithm_sha256']}/mandatorySTANDALONEV74base734a...NOTstandalone/NOTExactretrieval.
- ROOT{s['all_pins_rehashed']}pins/3linkedfullstatesindependentlyreconstructed/only2changed/frozen5prefix+allupstreambitwise/2Adamstep384/scalertracker384/ALLstoredRNG/ordercursor384groups1920micros/exposure/1024teacherpackets/lossmarginverified. Storagecorrected: serialize7V75readoutdeltas inclfrozen5vsV74, optimizerONLY2; rejectmissingfrozenprefix/recursiveparent. ActualfullsizeCPU2encode2readoutopt0CUDAfalse+3focused+tinyenginefullrestore/AMPskip/oneMicroSTOPNOreplayPASS;NOTactualCUDArestart. Raw470+parity1verified/same64Commitbitwise/native-stream4096+two262KiBparityPASS/postrunNN0. FullteacherlogitsNOTsaved/noindependentfullteacherargmax-marginclaim.
- SAMErevised64 V75→V78→V79 twoquantity54→56→56/64,A64all/Bkg10→12→12,temp14→16→16,km14→14→14,items16→14→14/otherfact59→60→60. V79NLL.00668718→.00636117 butbyte99.8588→99.8460/literal44→42aux. Sameobserved64facts/16quartets/multipleDEVuse/NOTnewblind. Observed75primary64maintained/stockstrict14→20→14of32;oldkg72strict59→60→59/clearwrong2→2→3. OWTdecoded108→114→117,old12854→55→56,expression8→10→11,retainedEnglish3same. KeepgainsANDregressions.
- Same262042/262029B~16xTRAINlong2 quantities16→28→29of5436(V75zeroFixed13new16residual),episodesfixedspan10→25→22(V75oneFixed13new9residual),names2→2→1,items6→6→8,decoded58→76→77/524071B. V78→V79quantity2fixed3new,episode5fixed2new. UTF8EOSfull-lengthretained/restrictedorder-locationcolor-negation0changes;drawer→draber/itemworddamageexists/notglobalcollapse/notabstractNLUcertificate. ROOTREADall72changedlongcontexts/42patterns/6regions/new22nonliteral+observed75all20;NOTall524071Bwordbymanual/notallOWTold128manualcertified.
- Actualriskexamples450.25vs+50.25,22→+2,17→16,Cyril→Ckril,28silver→2Üsilver. Adjacentepisode10893/533(V77V78)notseenV79,but145+/169+remain;review7109;→71090 outsidequantityslots. Keepoldfixedspancounts+separateadjacentrisks/nooutputrepair. Wrongplus8→36→33/digitplus8→34→30. TRAINcritical45063weak1943/noncritical964665weak2995/marginviolation84/416repeatedpositions/293of384clipped/meanmargin.00339214/.00181836;notunseenmarginbound/actualtaskgradientconflictproof.
- SelectionKEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V79/configs/research_base_selection_v79_review_20261113.json;V76V77V78V79allstatespreserved. Evidence runs/last_linear_20261113_v79/root_result.json/root_saved_review.json/root_meaning_review.json/root_training_and_boundary_review.json/root_scope_delta_and_next_math.json/root_cause_and_reading_review.json. Reportoutputs/FDT_LAST_LINEAR_V79_RESULT_20261113_KR.md.
- NEXToutputs/FDT_READOUT_UPDATE_RETENTION_NEXT_CONTRACT_20261113_KR.md:doNOTblindlyrepeatlast2same384orλincrease. TRAINcoverage+actualupdate-retentiondirectionCPUcontract;GEM/PCGradprimaryresearchreviewed,analyticCPUcounterexamplegret·g=1butgret·(-Dg)=9.8showsrawgradientprojectionNOTAdamsafety. Noactualmodelgradientprobe/nextworker-plan-seal-GPUyet. CurrentC109.08<sameowner109.25start; beforeNEWGPU derive smallerlosslessrawestimate/cap or routineauthorizedcacheoptions,neverdeletevaluableweights/raw. Don'trepeatcompletedboundaryaudit/broadbaselinesasgate. Futurethought-onlypolicyunchanged/noautomaticdiffusionphaseadvance/E2registry0/wholelanguagecycleunfinished.

'''
    state=work/'outputs/PROJECT_STATE.md';old=state.read_text(encoding='utf-8-sig');first,rest=old.split('\n',1);state.write_text(first+'\n'+current+rest,encoding='utf8')
    print(json.dumps(dict(status=result['status'],selection=selection['decision'],root_result_SHA=support.sha(run/'root_result.json'),report_SHA=support.sha(report),owned_bytes=result['owned_output_bytes'],CURRENT_saved=True)))


if __name__=='__main__':main()
