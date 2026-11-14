"""Close V78 evidence/selection; preserve old models, budgets, and human policy."""
import collections,datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    run=ROOT/'runs/competing_margin_20261113_v78';work=ROOT.parent
    p=io.load_json(run/'plan.json');s=io.load_json(run/'root_saved_review.json');m=io.load_json(run/'root_meaning_review.json');training=io.load_json(run/'root_training_and_boundary_review.json');actual=io.load_json(run/'result.json')
    terminal=io.load_json(run/'ROOT_terminal_native.json');native=io.load_json(run/'ROOT_postreview_native.json')
    assert s['status']=='ROOT_V78_SAVED_REVIEW_PASS' and m['status']=='ROOT_V78_SOURCE_BOUND_MEANING_READY'
    assert terminal['all4owned_and_proxy_gone'] and terminal['before_result_analysis'] and not terminal['PythonProcesses'] and not native['PythonProcesses']
    assert actual['attempts']==actual['updates']==384 and actual['AMP_skips']==0 and native['C_free_bytes']>100*1024**3
    cards=io.load_json(run/'root_manual_reading_cards.json');observed=io.load_json(run/'observed_V75_DEV_reading_cards.json')
    assert (len(cards['new_after']),len(observed['after']),len(cards['long_after']),len(cards['long_regions']),len(m['long_after']['unique_changed_field_patterns']))==(22,14,73,6,46)
    env=torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True)
    proposed_names=('readout.readout.net.2.weight','readout.readout.net.2.bias')
    proposed_count=sum(env['active_weights'][n].numel() for n in proposed_names);assert proposed_count==2117664
    now=datetime.datetime.now().astimezone().isoformat()
    selection=dict(time=now,decision='KEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V78',selected_version='V75',selected_runtime=p['cycle_source'],selected_SHA=p['cycle_sha'],
        mandatory_standalone_base=p['storage_parent'],mandatory_base_SHA=p['storage_parent_sha'],rejected_as_default='V78',V78_final_SHA=actual['latest']['sha256'],all_V76_V77_V78_states_preserved=True,
        reason='Revised64 both quantities54->56/64 but long quantities16->28,episode10->25,decoded58->76 vs V75; V77 episode17->25. Preserve useful gains, no overall promotion.',
        deployed_registry_changed=False,weight_files_modified=False,automatic_extra_training=False,all_unused_budgets_closed=True)
    config=ROOT/'configs/research_base_selection_v78_review_20261113.json';assert not config.exists();io.atomic_json(config,selection);io.atomic_json(run/'root_research_base_selection.json',selection)
    rows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2];plus={}
    for version,folder in [('V75',Path(p['baseline_run'])),('V77',ROOT/'runs/update_balance_20261113_v77'),('V78',run)]:
        records=io.load_json(folder/'long_reconstructions.json');counts=collections.Counter()
        for row,record in zip(rows,records):
            for a,b in zip(row['prompt'].encode(),record['output'].encode()):
                if a!=b:counts['different_bytes']+=1;counts['wrong_plus']+=int(b==43);counts['digit_to_plus']+=int(48<=a<=57 and b==43)
        plus[version]=dict(counts)
    cause=dict(status='ROOT_V78_CAUSE_AND_ACTUAL_READING_COMPLETE',time=now,all_upstream_and_new64_Commit_frozen=True,
        actual_margin_and_grouping_applied=training['training'],long_symbol_interference=plus,adjacent_episode_token_scan=training['episode_adjacent_risks_not_in_fixed_span_counts'],
        root_reading=dict(new64_nonliteral_full_clauses=22,observed75_nonliteral_full_clauses=14,all73_long_changed_event_contexts=True,long_unique_field_patterns46=True,start_middle_end6=True,all524071bytes_individually_manually_scored=False),
        inference='New readout variant improves some quantities but TRAIN-only weak-margin retention does not bound unseen decision margins; not semantic diffusion failure or all-upstream-perfect certificate.',
        causal_limit='Margin,grouped clipping/Adam and number of optimizer updates changed jointly; no single cause proof or parameter-gradient attribution.',
        proposed_next_scope=dict(names=list(proposed_names),parameters=proposed_count,freeze_norm_and_first_MLP_and_terminal=True,actual_next_worker_or_plan_or_GPU=False),extra_model_calls=0)
    io.atomic_json(run/'root_cause_and_reading_review.json',cause)
    report=work/'outputs/FDT_COMPETING_MARGIN_V78_RESULT_20261113_KR.md';nextcontract=work/'outputs/FDT_READOUT_SCOPE_NEXT_CONTRACT_20261113_KR.md'
    files=[run/n for n in ('plan.json','result.json','final_status.json','ROOT_native_start_observation.json','ROOT_terminal_native.json','ROOT_postreview_native.json','root_saved_review.json','root_meaning_review.json','root_manual_reading_cards.json','observed_V75_DEV_reading_cards.json','root_training_and_boundary_review.json','root_cause_and_reading_review.json','root_research_base_selection.json')]
    files.extend([config,report,nextcontract,ROOT/'configs/reasoning_only_training_policy_20261113_v1.json',ROOT/'scripts/review_competing_margin_v78_saved_cpu.py',ROOT/'scripts/review_competing_margin_v78_meaning_cpu.py',ROOT/'scripts/review_v78_training_and_token_boundaries_cpu.py',Path(__file__)])
    result=dict(status='ROOT_V78_ACTUAL_COMPLETE_INDEPENDENTLY_REVIEWED_NOT_PROMOTED',time=now,actual_attempts384_updates384_skip0=True,
        actual_seconds=actual['seconds'],actual_calls=actual['calls'],group_source_exposure=actual['exposure'],all_owned_proxy_gone_before_analysis=True,latest_native=native,
        active_parameters=2908192,active_tensors=7,final_SHA=actual['latest']['sha256'],final_algorithm_SHA=actual['latest']['algorithm_sha256'],final_bytes=actual['latest']['bytes'],
        mandatory_standalone_V74_SHA=p['storage_parent_sha'],parent_required_NOT_standalone=True,
        saved_review=s['status'],meaning_review=m['status'],selection=selection,old_weights_Adam_scaler_RNG_raw_failures_and_sources_preserved=True,
        closed_budget=dict(attempts=384,max_calls=7000,actual_calls=6752,unused248_calls_time_NOT_reusable=True,wall900_cut780_reserve120=True,automatic_retry=False),
        extra_postrun_model_calls=0,Exact_content_entries=0,copy_bytes=0,denoise_calls=0,whole_language_meaning_language_complete=False,
        latest_HUMAN_future_diffusion_freezes_ALL_codec_and_trains_ONLY_thought_unchanged=True,actual_next_trial_or_diffusion_plan_exists=False,
        owned_output_bytes=sum(f.stat().st_size for f in run.rglob('*') if f.is_file()),evidence_SHA={str(f):support.sha(f) for f in files})
    assert result['owned_output_bytes']<p['max_output_bytes'];io.atomic_json(run/'root_result.json',result)
    stamp=now[:16].replace('T',' ')
    current=f'''## CURRENT — {stamp} KST V78 COMPLETE reviewed / limited two-quantity gain, long episode regression / keep V75

- LatestHUMAN future diffusion MUST freeze ENTIRE input-memory-binding-output codec and train separate thought weights ONLY; detailed boundary audit alreadyCOMPLETE/11CPU+fullsize fixture,don'trebuild. Current phaseEnglish language→meaning→language repair/meaningPRIMARY/denoise0. ROOTONLY/noagents/K_Pprior+K_Cdynamicfuzzy/K_EExactNULLcontent0copy0/Commit-onlyD/noHraw-GTprefix-GTlength/K_Pcertifiedwriterabsent. Timer30min재개/C100GiBfloor/ALLoldweightsAdamRNGrawsourcesfailuresgamespreserved.
- V78 ONEactualCOMPLETE384attempt384update/skip0/312.383624worker sec/1920live source micros. Sourceunchanged1024teacheronce/headsameV77schedule1920=U/S/O/U/S×384 at SAMEparams/.2eachloss/oneunscale-clip1-Adam pergroup;NOT1920updates/newdocs/uniqueepoch. Teacher1010943UTF8B/head2720189B/86108critical/frozenV75symbolprofile. ONLY7D2908192paramsfresh1e-5/allupstreamfrozen. TRAIN-onlycorrectweakmargin<=3 targetmin(m,1),critical/noncritical.25/.10+existingcap8CE;notinferencefeatures. Margin+grouping+stepcountjointlychanged/not single-causeproof.
- PlanSHA3bac6f51460e4a518d26b1f83de71158a021bd7f4d487117aca0c6b79bf1425a. Owner22684→34400born09:26:30.685/.710;worker6164→31288born09:26:38.821/.835/ROOTall4creation-parent-command-exe-live09:26:56.861. Worker-ownerexit0/ROOTall4ownedproxygone+freshPython0verified09:32:16.183 BEFOREanalysis. DuringactualtrainingROOTdetailedanalysisOFF/OSonly/noagents. EndC118165704704B/finalC{native['C_free_bytes']}B~{native['C_free_bytes']/1024**3:.2f}GiB>100floor/currentmodelprocess0.
- Actualencode3076+readout3418+chunks258=6752/longblock2322/denoise0/GPUtrain3334297088B/teacher3264918528B. New384attempt/7000calls/900wall780cut120reserve/owned1.25GiB CLOSED/unused248calls-timeNOTreusable/noautoretry-oldclockreset-capgrowth-sealededit-rotation. AlloldbudgetsCLOSED. Final36007558BSHAe2e2cd004b6b96f6347d71e89f06dbc7bdb4b7b0ef3fe1483ee3cfff90cffca5/fullalgorithma0d934f83c085e72eed4ee5f0b97b0e24292a0cd943b7327f59374e22224d313/mandatorySTANDALONEV74base734a...NOTstandalone/NOTExactretrieval.
- ROOT1019pins/3parentlinkedfullstatesindependentlyreconstructed/V75warmallNNexact/upstreamfrozen/all7changed/7Adamstep384/enabledscalertracker384/ALLstoredRNG/order-groupcursorhistory/exposure/1024teacherpackets/1920micro-loss-cap-margin-statseachverified. Raw470+parity1SHA-target-argmax-NLL-independentUTF8FSM-outputverified/new64Commitbitwiseidentical/native-stream4096+2fresh262KiBsourceparityPASS. Teacherp(gold)/correct/margin/competitorsavedNOTfullteacherlogits/noindependentfullteacherargmax-marginclaim/postrunNN0.
- SAMErevised64 sourceV75→V78 A64→64/Bbothclear54→56of64/kgB10→12/temp14→16/km14→14/items16→14;independent-otherfactcorrect59→60/64;literal44→42aux/NLL.00668718→.00608507. Allsameobserved64facts/names+16quartets/notblindnewdata/noheldTRAIN. Actualread22nonliteralfullclauses incl75→7;,450.25vs+50.25,Reviewtypo/IDmerge,notallmeaning56/64certificate. Observed75Bstock14→20/32(primary64allmaintained),V77was22/32;oldpositivekg72strict59→60(V77was58);observed74primary61→62/Bstock34→36. OWTdecoded108→114(V77 110),old12854→55,oldexpression8→10,retainedEnglish3→3. Preserveallgainsandregressions.
- Same262042/262029B~16xTRAINlong2 quantities16→28(V75onefixed13new15residual);V76→V78eightfixed0new;V77→V78threefixedthreenew28sameTOTAL. Names2→2(onefixedonenewvsV75);episodesfixedspan10→25(V77 17→25;twofixed10newvsV77);items6→6;decoded58→76(V77 63). FullUTF8EOSlengthretained/2718correlatedevents5436qtyslots/restrictedorder-locationcolor-negative0changes;drawer→draberworddamageexists/notglobalcollapse/notnaturalblindorabstractNLU/fixedmemorylossless. ROOTreadALL73changedeventcontexts/46patterns/6startmiddleend,observed75all14nonliteral;NOTall524071Bwordbymanualscore.
- ADDEDscore-onlytokenboundaryreview:fixedspanepisodecheckmisses1089,→10893 and53,→533(V77V78both/V75none);V78also145+,169+malformedtoken. Old25/17fixedspanscoresPRESERVED/extraadjacentrisksseparate/noinferenceregex/outputrepair/TRAINchanges. Longwrongplus8→36(V77 21),digitplus8→34(V77 21). TRAINanchorACTUALLYactive:correctcritical45063/weak1943(~4.31%);correctnoncritical964665/weak2995;headcriticalviolations101/noncritical478 occurrences;249of384groupsclipped. Groupmeanmargin.00402044/.00298059. TRAINscalarcountsNOTunseenmarginbound/gradientmass;notdiffusionfailure/notupstreamcompletecertificate.
- PreparationbugsfixedBEFOREGPU:clamp_min at0 pushed teacher-equal margins→ReLUzero-subgradient(9CPUchecks inclreportidempotence);firstownedCPUmetadata diagnostic doublewrap mismatchSTOPPED beforeencode/readout/optimizer0/allgone/originalcode-planpreserved. execution_review_v2fullsizeCPU2encode2readoutoptimizer0CUDAfalse+ROOT1017pins/initialpartialrestore/canonical7scope+singleestimate1158797952B verified;ownerpostPopenbodyidenticalV77. Tinygroupengine populated7Adam-enabledCPUscaler-ALLRNG-AMPskip5rows-oneMicroSTOPexactNOreplayPASS;NOTactualCUDArestart.
- SelectionKEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V78/configs/research_base_selection_v78_review_20261113.json. V75f01e8512...+mandatoryV74base retained;V76V77V78ALLstatesAdamRNGrawkept/noexistingweights-registryoverwrite. LatestACTUALtrainedV78, NEXTdefaultV75. Evidence runs/competing_margin_20261113_v78/root_result.json/root_saved_review.json/root_meaning_review.json/root_manual_reading_cards.json/root_training_and_boundary_review.json/root_cause_and_reading_review.json/ROOT_terminal_native.json. Reportoutputs/FDT_COMPETING_MARGIN_V78_RESULT_20261113_KR.md.
- NEXToutputs/FDT_READOUT_SCOPE_NEXT_CONTRACT_20261113_KR.md:doNOTblindlyrepeat7D384/1920orλincrease. CPUcandidatefreezeDnorm/firstLinear/terminal + ALLupstream;ONLYlastLinearweight+bias2tensors2117664params,TRAINcorrect-margineligibilityscope/finitecost-gradient-restore review before newboundedtrial. No nextcandidateworker/plan/seal/GPUexistsyet. Don'trepeatcompletedfullboundaryaudit/broadbaseline/repeatedtrace. Futurethought-onlypolicyunchanged/designonly/noautomaticdiffusionphaseadvance;answermeaningnotrawtext/GTrole-lengthcondition. WholelanguagecycleNOTcomplete.

'''
    state=work/'outputs/PROJECT_STATE.md';old=state.read_text(encoding='utf-8-sig');first,rest=old.split('\n',1);state.write_text(first+'\n'+current+rest,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],root_result_SHA=support.sha(run/'root_result.json'),selection_SHA=support.sha(config),report_SHA=support.sha(report),C_free=native['C_free_bytes'],owned_bytes=result['owned_output_bytes'],CURRENT_saved=True)))


if __name__=='__main__':main()
