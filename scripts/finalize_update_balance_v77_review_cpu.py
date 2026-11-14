"""Publish saved-evidence decision and handoff; no neural computation."""
import collections, datetime, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    run=ROOT/'runs/update_balance_20261113_v77';workspace=ROOT.parent
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    p=io.load_json(run/'plan.json');s=io.load_json(run/'root_saved_review.json');m=io.load_json(run/'root_meaning_review.json')
    drift=io.load_json(run/'root_saved_margin_drift.json');guard=io.load_json(run/'root_plan_metadata_and_future_guard.json')
    assert s['status']=='ROOT_V77_SAVED_REVIEW_PASS' and m['status']=='ROOT_V77_SOURCE_BOUND_MEANING_READY'
    assert drift['status']=='ROOT_V77_SAVED_MARGIN_DRIFT_PASS' and guard['tests_passed']==8
    terminal=io.load_json(run/'ROOT_terminal_native.json');native=io.load_json(run/'ROOT_postreview_native.json')
    assert terminal['all4owned_and_proxy_gone'] and terminal['before_result_analysis'] and not terminal['PythonProcesses']
    assert not native['PythonProcesses'] and native['C_free_bytes']>100*1024**3
    assert support.sha(run/'plan.json')==guard['actual_V77_plan_unmodified_SHA']
    features=io.load_json(Path(p['data'])/'train_features.json');targets=io.load_json(Path(p['data'])/'train_targets.json')
    orders=io.load_json(Path(p['balance_data'])/'orders.json')
    groups=collections.Counter('OWT' if targets[x[0]]['source_kind']!='prospective_joint_quantity' else targets[x[0]]['mode'] for x in orders)
    assert groups==dict(unsigned=768,signed=768,OWT=384)
    cards=io.load_json(run/'root_manual_reading_cards.json');observed=io.load_json(run/'observed_V75_DEV_reading_cards.json')
    assert (len(cards['new_after']),len(observed['after']),len(cards['long_after']),len(cards['long_regions']))==(22,12,60,6)
    assert len(m['long_after']['unique_changed_field_patterns'])==38
    summaries=drift['summaries']
    plus={v:sum(summaries[f'{v}_row{i}']['wrong_plus'] for i in range(2)) for v in ('V75','V76','V77')}
    digitplus={v:sum(summaries[f'{v}_row{i}']['digit_to_plus'] for i in range(2)) for v in ('V75','V76','V77')}
    role=collections.Counter()
    for i in range(2):role.update(summaries[f'V77_row{i}']['wrong_plus_patch_roles'])
    review=dict(time=now,status='ROOT_V77_CAUSE_AND_ACTUAL_READING_COMPLETE',
        actual_head_update_groups=dict(groups),actual_head_row_fraction={k:v/1920 for k,v in groups.items()},
        frozen_V75_symbol_profile_verified=True,new64_Commit_identical=True,all_frozen_upstream_identical=True,
        V75_to_V77_long=m['long_paired'],V76_to_V77_long=m['V76_to_V77_long_paired'],
        wrong_plus=plus,digit_to_plus=digitplus,V77_wrong_plus_patch_roles=dict(role),
        final_bias_decomposition=drift['final_bias_decomposition'],plan_metadata_discrepancy=drift['plan_metadata_discrepancy'],
        actual_ROOT_reading=dict(revised64_nonliteral_full_clauses=22,observed75_nonliteral_full_clauses=12,
            long_changed_event_local_contexts=60,long_unique_changed_field_patterns=38,long_start_middle_end_contexts=6,
            all524071bytes_individually_manually_scored=False),
        severity='Localized quantity/episode/name damage despite intact long output and restricted order/location/negation. No whole-model-collapse claim. V77 partly recovers V76 but does not retain V75 long facts.',
        causal_limit='Exposure, class weights and update count changed together; final-bias arithmetic not gradient/causal mediation proof. Observed authored panels are correlated, not blind natural benchmark.',
        new_GPU_or_model_calls=0,held_facts_TRAIN=0,old_weights_or_sealed_plan_modified=False)
    io.atomic_json(run/'root_cause_and_reading_review.json',review)
    selection=dict(time=now,decision='KEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V77',
        selected_version='V75',selected_runtime=p['cycle_source'],selected_SHA=p['cycle_sha'],
        mandatory_standalone_base=p['storage_parent'],mandatory_base_SHA=p['storage_parent_sha'],
        rejected_as_default='V77',V77_final_SHA=s['all3_parentlinked_states_independently_reconstructed'][-1]['SHA'],
        all_V76_V77_states_preserved=True,
        reason='V77 long quantities28 vs V75 16 and episode17 vs10; OWT110 vs108; revised64 both-facts54/64 unchanged. V76-to-V77 eight quantity repairs and observed75 stock22/32 gains preserved separately.',
        deployed_model_registry_changed=False,weight_files_modified=False,automatic_extra_training=False,all_unused_budgets_closed=True)
    selectionfile=ROOT/'configs/research_base_selection_v77_review_20261113.json'
    assert not selectionfile.exists()
    io.atomic_json(selectionfile,selection);io.atomic_json(run/'root_research_base_selection.json',selection)
    report=workspace/'outputs/FDT_UPDATE_BALANCE_V77_RESULT_20261113_KR.md'
    nextcontract=workspace/'outputs/FDT_READOUT_MARGIN_NEXT_CONTRACT_20261113_KR.md'
    files=[run/n for n in ('plan.json','result.json','final_status.json','ROOT_terminal_native.json','ROOT_postreview_native.json',
        'root_saved_review.json','root_meaning_review.json','root_manual_reading_cards.json','observed_V75_DEV_reading_cards.json',
        'root_saved_margin_drift.json','root_plan_metadata_and_future_guard.json','root_cause_and_reading_review.json','root_research_base_selection.json')]
    files += [selectionfile,report,nextcontract,ROOT/'scripts/review_v77_margin_drift_cpu.py',ROOT/'scripts/plan_scope_contract_v78_cpu.py',ROOT/'scripts/check_v77_plan_metadata_and_v78_guard_cpu.py',Path(__file__)]
    result=dict(status='ROOT_V77_ACTUAL_COMPLETE_INDEPENDENTLY_REVIEWED_NOT_PROMOTED',time=now,actual_training=io.load_json(run/'result.json'),
        worker_owner_exit0_all_owned_gone_before_analysis=True,latest_native=native,saved_review=s['status'],meaning_review=m['status'],
        actual_attempts=1920,actual_updates=1920,AMP_skips=0,active_parameters=2908192,active_tensors=7,
        final_SHA=s['all3_parentlinked_states_independently_reconstructed'][-1]['SHA'],final_algorithm_SHA=s['all3_parentlinked_states_independently_reconstructed'][-1]['algorithm_SHA'],
        final_runtime_bytes=35964614,parent_required_NOT_standalone=True,mandatory_standalone_V74_SHA=p['storage_parent_sha'],
        closed_budget=dict(max_calls=7000,actual_calls=6752,unused_calls248_NOT_reusable=True,wall900_cut780_reserve120=True,automatic_retry=False),
        selection=selection,root_additional_neural_calls=0,denoise_calls=0,Exact_entries=0,copy_bytes=0,
        K_P_certified_writer_complete=False,whole_language_meaning_language_complete=False,
        plan_descriptive_metadata_bug_disclosed=True,future_scope_guard_CPU_tests=8,actual_V78_worker_plan_seal_GPU=False,
        all_prior_models_Adam_scaler_RNG_raw_failures_and_sources_preserved=True,
        evidence_SHA={str(f):support.sha(f) for f in files},owned_output_bytes=sum(f.stat().st_size for f in run.rglob('*') if f.is_file()))
    assert result['owned_output_bytes']<p['max_output_bytes']
    io.atomic_json(run/'root_result.json',result)
    stamp=now[:16].replace('T',' ')
    current=f'''## CURRENT — {stamp} KST V77 COMPLETE independently reviewed / partial V76 recovery / keep V75 research base

- ROOTONLY/noagents/English language→meaning→language BEFOREdiffusion/meaningPRIMARY/untrained+verylong/C100GiBfloor/timerACTIVE30min재개. K_Pprior+K_Cdynamicfuzzyactive/K_EExactNULLcontent0copy0/Commit-onlylanguage/noHraw-sourcebytes-GTprefix-GTlength;K_Pcertifiedwriterabsent. AllweightsAdamRNGrawsourcesfailures/useroriginals/gamespreserved.
- V77 ONEactualCOMPLETE1920attempt1920update/AMPskip0/359.951450workersec. ONLY7readout2908192parametersfreshAdam1e-5/allupstreamfrozen/V75warmexact. UniqueTRAIN1024 teacheronce/head1920=signed768×1,unsigned128×6,OWT128×3/actualrow40-40-20/head2720189UTF8B/86108critical/V75symbolprofilefrozen. NOT1920newdocs/NOT1uniqueepoch/NOTpostclipAdamgradientmassguarantee. encode3076+readout3418+chunks258=6752/longblock2322/denoise0. New900wall780cut120reserve/7000call/1.25GiBCLOSED/unused248timecallsNOTreusable/noautorestart-capgrowth-oldclockreset-sealededit.
- Owner35008→30756born08:15:14.120/.147;worker19540→5584born08:15:21.406/.420;ROOTcreation-parent-command-exe-live08:15:37.856. Worker-ownerexit0/ROOTall4ownedproxygone+Python0confirmed08:21:57.544BEFOREanalysis. ActualtrainingROOTdetailedanalysisOFF/ownedOSonly/noagents. EndC119248814080B>{100*1024**3}Bfloor;latestC{native['C_free_bytes']}B/NOmodelprocess. PlanSHAd136ac4a5996fa6d3cfd027a67179557514ca346564bb8bdc8a85a8b5ba959bc/final35964614BSHA964aeeaaebd0029c370025a1f8c3b5e9cb9d60b5ac21d84d198650608c5e3fcc/fullalgorithmeaff304b6c9a46e9706fd429bf8e841f6ea5d7f8acc8786096fa829a3674eaf4/mandatorySTANDALONEV74base734a2eed...NOTstandalone/NOTExactretrieval.
- ROOT1000pins/3linkedfullstatesindependentlyreconstructed/only7changed/allupstreamfrozen/7Adamstep1920/enabledscalertracker1920/ALLstoredRNG/ordercursorhistory/1024teacherp(gold)packets/finite1920repeatexposureverified. Raw470+parity1SHA-target-argmax-NLL-independentUTF8FSM-outputverified/new64Commitbitwiseidentical/native-stream4096+two262KiBfreshsourceparityPASS. Teacherfull logitsNOTsaved/noindependentteacherargmaxclaim/postrunNN0/CUDAfalse.
- SameoldpanelsV75→V76→V77:observed75secondaryBstock14→20→22of32/primary64allpreserved;oldpositivekg72strict59→61→58;OWTdecoded108→122→110;old12854→52→55;oldexpression8→10→9;retainedEnglish3→4→4. V77stock32=22clear,6differentvalues,2valueunitdamage,2extra-symbolNOTcertified;missingkg0/wrongnegative0.
- Sameactual262042/262029B(~16xTRAIN16384)long2:quantities16→36→28;V76→V77eightfixed/0new/28residual BUTV75→V770fixed/12new(11cleardifferentvalues+1malformed)/16residual. Names2→2→2(V75onefixedonenewAveline→Eveline);episodes10→10→17(V75onefixed8new;V76twofixed9new);items6→8→7;decoded58→66→63. AllUTF8EOSfulllengthretained/restrictedorder-location-negative0changes/2718correlatedevents5436quantityfields(notabstractNLU100%/naturalblindbenchmark/fixedmemorylossless/1MiB). ROOTactualreadnew22nonliteral+observed75all12nonliteral+ALL60changedlongeventcontexts/38patterns/6startmiddleendregions;NOTall524071Bwordbymanualscore.
- Evalgrammarrevision64samefacts/names/quartets=32unchanged+END32possessivearticlefixed;NOT64newblindcases/NOTdirectV76scorecomparison/oldV76rawscorespreserved/noheldTRAIN. SameREVISED64V75→V77A64→64/Bbothclear54→54/kgB10→12of16/temp14→16/km14→12/items16→14;pairunchanged-correct59→59of64/literal44→42aux/NLL.00668718→.00619608. Mainvaluerecoveryoffsetotherdamage;meanNLLNOTmeaninggain.
- SavedmarginCPUcode/mathNOforward:wrongplus8→22→21/digitplus8→20→21;V77wrongplus19of21atpatchroles12/13. V75correct→V77plus12positionsmediancompetinggapdelta1.920780vsfinalbiasdelta.001123/all12needotherpathchange;NOTjustfinalplusbias/NOTgradientcausalproof. Supportsreadoutconditional-positioninterference/notdiffusionfailure/notupstreammeaningperfection.
- CODEissue:inheriteddict(old)leftplanactual_trainable_parameters17677856andobsoleteoutputestimates5656937760/3324668416;actualCPUgradient+states7/2908192andwiredauthoritativeestimate1104832864/hardcap1342177280verified. Disclosed/preservedsealedplanSHA/noactualbudgetgrowth. NEWplan_scope_contract_v78_cpu+8CPUchecksrejectstaleparams/multipleestimates/badcap-headroom-cost;FUTUREguardonly/NOTV78workerplansealGPU. Mustwireactualoptimizer-ownedtensorsandbuildexplicitcanonicalscopebeforefutureseal,doNOTinherituncheckeddescriptions.
- SelectionKEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V77/configs/research_base_selection_v77_review_20261113.json;V75f01e8512...+mandatoryV74basepreserved;allV76V77weightsAdamscalerRNGrawkept/noexistingweightsregistryoverwritten. LatestACTUALtrainedV77butNEXTdefaultV75. PartialrecoveryNOTwholecyclecompletion/globalfailure/automaticextraSGD.
- Evidence runs/update_balance_20261113_v77/root_result.json/root_saved_review.json/root_meaning_review.json/root_manual_reading_cards.json/root_cause_and_reading_review.json/root_saved_margin_drift.json/root_plan_metadata_and_future_guard.json/ROOT_terminal_native.json. Reportoutputs/FDT_UPDATE_BALANCE_V77_RESULT_20261113_KR.md;NEXToutputs/FDT_READOUT_MARGIN_NEXT_CONTRACT_20261113_KR.md. NEXTspecificTRAIN-onlycorrect-teachermargin/strongestcompetitor+same-updategroupretentionCPUcontract thennewboundedtrialafteractualpreflight-schema-resource-ownedOSreview. NoextraGPUautomaticprerequisite/broadbaseline/repeatedtrace/devfactsTRAIN/Exactcopy/oldbudgetreuse. Denoise0/wholecycleNOTcomplete.

'''
    state=workspace/'outputs/PROJECT_STATE.md';previous=state.read_text(encoding='utf-8-sig');first,rest=previous.split('\n',1)
    state.write_text(first+'\n'+current+rest,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],root_result_SHA=support.sha(run/'root_result.json'),selection_SHA=support.sha(selectionfile),
        report_SHA=support.sha(report),owned_output_bytes=result['owned_output_bytes'],current_C_free=native['C_free_bytes'],first_CURRENT_saved=True)))


if __name__=='__main__':main()
