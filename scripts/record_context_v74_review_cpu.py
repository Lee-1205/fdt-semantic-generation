"""Record completed V74 evidence and source-output reading; CPU metadata only."""
import collections,datetime,difflib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    run=ROOT/'runs/context_verbalizer_20261112_v74';out=ROOT.parent/'outputs'
    report=out/'FDT_CONTEXT_VERBALIZER_V74_RESULT_20261112_KR.md';nextfile=out/'FDT_CRITICAL_RETENTION_NEXT_CONTRACT_20261112_KR.md'
    result=io.load_json(run/'result.json');saved=io.load_json(run/'root_saved_review.json');meaning=io.load_json(run/'root_meaning_review.json');cards=io.load_json(run/'root_manual_reading_cards.json')
    retention=io.load_json(run/'root_retention_code_review.json');confusion=io.load_json(run/'root_value_and_confusion_cpu.json');term=io.load_json(run/'ROOT_terminal_native.json')
    assert saved['status']=='ROOT_V74_SAVED_REVIEW_PASS' and result['attempts']==result['updates']==640 and term['all4owned_and_proxy_gone'] and term['before_result_analysis']
    assert len(cards['new64_after_changed_clauses'])==41 and len(cards['old128_after_nonliteral'])==43 and len(cards['long_changed_after'])==92 and len(cards['long_start_middle_end'])==6
    patterns=collections.Counter()
    for card in cards['long_changed_after']:
        a,b=card['source'],card['output']
        for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            if tag!='equal':patterns[(a[max(0,i-15):min(len(a),j+18)],b[max(0,k-15):min(len(b),l+18)])]+=1
    assert len(patterns)==83 and sum(patterns.values())==95
    manual=dict(status='ROOT_V74_SOURCE_OUTPUT_READING_COMPLETE',
        new64_changed_clauses_read=41,old128_changed_rows_all_local_edits_read=43,long_changed_events=92,long_unique_local_patterns_read=83,long_regions_read=6,
        new64_primary_numeric_same_value_before=42,new64_primary_numeric_same_value_after=63,new64_primary_total=64,
        findings=[
            'Fixed encoder/binding, readout-only learning improved old positive-kg values21/72 to64/72. Some needed source information was usable in the same Commit; not a whole abstract-semantic certificate.',
            'New held primary quantities improved42/64 to63/64; separate stored mass only10/64 to26/64. +2.25kg to22.25kg and +0.50degreesC to20.50degreesC are actual value changes.',
            'Ilyaz/Ilyai, Ezrien/Eqrien, Zephyr/Sephyr, Boaz/Boak are identity risks, not automatically repaired spelling variants.',
            'Existing two maximum longs retain source length/UTF8/EOS and correct start-middle-end examples, but number/reserve changes11 to29 and decoded74 to110 regress. episode129/1+9 and29/26 are content changes.',
            'Wrong plus outputs7 to29 including17 originally digit bytes support interference in this controlled readout comparison; no single causal-mechanism conclusion.',
            'OWT2013/2012 and U-24/1-24 affect content; representative local contexts read. No claim every natural-document word was manually independently scored.',
            'Space16.50kg and +16.50kg can preserve the same positive value; malformed currency/unit or changed name/amount are not counted as meaning-preserved via similarity.',
            'No global semantic accuracy fabricated; limited source-aligned fields, Decimal value checks, raw reconstruction and manual reading remain separate.'
        ],not_all_long_words_individually_manually_scored=True,extra_model_calls=0)
    io.atomic_json(run/'root_manual_reading_review.json',manual)
    free=shutil.disk_usage(ROOT).free;owned=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    assert free>100*1024**3 and owned<3758096384
    panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in saved['panels'].items()}
    review=dict(status='ROOT_V74_COMPLETE_EVIDENCE_VERIFIED_MIXED_CONTENT_RESULTS',all_owned_and_proxy_gone_before_analysis=True,
        actual=dict(attempts=640,updates=640,AMP_skips=0,seconds=result['seconds'],calls=result['calls'],budget_closed=True,unused108_calls_not_reusable=True),
        final_state=result['latest'],plan_SHA=support.sha(run/'plan.json'),active_tensors=7,active_parameters=2908192,upstream_all_frozen=True,
        standalone_fullstates=saved['two_standalone_fullstates'],initial_runtime_requires_pinned_V73_parent=True,
        panels=panels,source_bound_meaning_summary=dict(new_primary_value42_to63_of64=True,old_positive_kg21_to64_of72=True,long_numeric11_to29_regression=True,OWT_decoded99_to120_regression=True),
        protected_C_free_bytes=free,owned_output_bytes=owned,Exact_entries=0,copy_bytes=0,denoise_calls=0,postrun_model_calls=0,
        automatically_promoted=False,language_meaning_language_complete=False,next_GPU_prepared_or_launched=False,
        report=str(report),next_contract=str(nextfile),
        evidence_SHA={str(f):support.sha(f) for f in [run/'root_saved_review.json',run/'root_meaning_review.json',run/'root_manual_reading_review.json',run/'root_value_and_confusion_cpu.json',run/'root_retention_code_review.json',run/'ROOT_terminal_native.json',report,nextfile]})
    io.atomic_json(run/'root_result.json',review)
    now=datetime.datetime.now().astimezone();heading=f'## CURRENT — {now:%Y-%m-%d %H:%M} KST V74 COMPLETE / readout correction gains AND long-natural retention regressions verified'
    current=heading+'''\n\n- Latest HUMAN:ROOTONLY/noagents;continue English language→meaning→language with MANYuntrained/long inputs BEFOREdiffusion. Meaning/content PRIMARY/literalaux/Koreandeferred;C100GiBfloor/timerACTIVE30min재개. K_Pprior+K_Cdynamicfuzzy/K_EExactNULLcontent0/copy0. Commit-onlydecoder/noHraw/sourcebytes/GTprefix/GTlength. K_Pcertifiedwriter absent. AlloldsourceweightsAdamRNGrawfailures/gamespreserved. CurrentmodelprocessNONE;allV74owned/proxyterminated BEFOREanalysis.
- NEW V74 ONE COMPLETE640attempt640update/AMPskip0/227.721446worker sec;entireV73encoder+bindingfrozen,ONLYfinalreadout7tensors2908192paramslearned/fresh7Adam;wholeNN123497080. TRAIN640=512authored256correlatedsignpairs+128EXACToldOWTTRAIN/1finiteepoch1039965UTF8B/34194critical/max16384. Primary+kg/-kg/+degreesC/-degreesC each128;heldZephyr/BoazNOTTRAIN/newDEV64newwrappernamesnotTRAIN. Actual640sourcecache/teacher640/head640separateexposuresNOT1920newdocs. Rev1EzraalreadyOWTTRAINcaught/emptyartifact+builderfailurepreserved;v2Ezrienused/noexclusionweakened.
- Actualowner30620→17700born18:50:37.491/.520,worker16372→29200born18:50:42.086/.102;ROOTALL4creation-parent-command-exe-liveobserved18:50:47.488. Worker-ownerexit0;ROOTALL4ownedproxygone/freshPython0verified18:54:57.586BEFOREanalysis. TrainingROOTdetailedanalysisOFF/OSonly/noagents. New900wall780cut120reserve/max2700/output3.5GiB CLOSED. Actualencode708+readout1626+chunks258=2592/block2322/denoise0. UNUSED108call/timeNOTreusable/noautoretry/deadlinereset/capgrowth/sealededit/automaticpromotion.
- PlanSHA30ed4ba49e9afc237635de3cd2bd52057a7a903781eeb815a3f1e977dce4a41c. Final518012241B SHA734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7/algorithm b768fcd932a33a868ba6399275a64c6c8751c1483d4b146211dd104cbdd9f911. ActualfullsizeCPUowned11.546sec/fullparentwarm+fresh7Adam/encoder1readout2gradient+linkedinitialexactreconstruction/optimizer0/CUDAfalse/allCPUownedgonePASS. EngineCPUphase/populated7Adam/enabledscalerALLRNG/AMPskip/inflightSTOPexactrestoreNOautoreplayPASS;NOTactualCUDArestartproof.
- ROOTall536pins/linkedinitialruntimefullreconstructedrecursiveDigest/twoSTANDALONEcachecomplete0+final640/fullSHA+safeload/ALLnonheadNNfrozen+7changed/7Adamsteps640/scaler1024tracker640/allstoredRNG/ordercursorhistoryexposure/calls/cache640sourceepochfeatures-targetseparation PASS. InitialruntimeNNweightsreferencesONLYmandatorypinnedstandaloneV73parent;NOTstandalone/NOTinputtextorExactretrieval. Parent23AdamRNGpreserved. TeacherfullrawlogitsNOTsaved/noindependentteacherargmaxrecomputeclaim. All344actualevalraw+parity1SHA/targets/argmax/NLL/independentUTF8FSM/actualoutputverified;extraNN0/CUDAfalse.
- NewDEV64sameunchangedupstream baseline→final:primarynumericvalue42/64→63/64(+kg5/16→16/16,+temp5/16→15/16,bothnegative16/16maintained). Secondaryphysicalmass10/64→26/64 ONLY;remaining18validDIFFERENTvalue+20malformed/missingunit. Numberexpressionchanged79/128→39/128/name11/128→12/128/decoded101→61/literal6→23AUX. +2.25kg→22.25kg,+0.50degreesC→20.50degreesC areREALcontenterrors;Ilyaz→Ilyai/Ezrien→Eqrien identityrisks. 2oldtail4B omissionspersist. UTF8EOS64/64NOTmeaningcompletion.
- OldV73untrained128 NOWobservedretention:decoded95→58/literal60→85AUX/numericchanged62/653→22/653/name25/794→26/794. Oldpositivekg sameVALUE21/72→64/72;validdifferent24→2/malformed27→6/explicitminusflips3→0. SameboundCommitimprovedthroughreadoutsupportsusableSOURCEinformationforspecificcases,NOTcompleteabstractsemanticspace/allupstreamcorrect. Actualevalprompts/namesNEVERTRAIN.
- TWOmaxlongs262042/262029B(~16xTRAINmax) actualfreshsourcestream+cachedreadout BOTHtested:CommitrelL2/allargmax/outputparityPASS/native-stream4096rawparityPASS. BUT numeric11→29/name7→5/episode7→15/decoded74→110REGRESSION;wrongplus7→29(including17digitbytes). Bothfulloutputlength/UTF8EOS/first-middle-end6examplescorrectNOTpurelengthcausal/generalreasoning/fixedmemoryproof. LocalCommitgrowswithinputlength. OldOWT16decoded99→120/literal8→6REGRESSION;retainedEnglish36decoded7→5/literal33→32;oldexpression32decoded6→5/literal26→28. DoNOTclaimallconditionsimproved.
- ROOTreadALLnew64nonliteral41changedclauses/old128nonliteral43localedits/long92changedevents83uniquelocalpatterns+6regions/OWTdate2013→2012,U-24→1-24representativeregressions. AlllongwordsNOTindividuallymanualscored/sourceboundfield+Decimal+manualscopeNOTglobalmeaningpercentage. PostrunCODEreview:KLprotects1005022initialcorrectNONcriticalTRAINbytes;33509correctcriticalexcluded/685wrongcritical. Criticalstillordinary+.25CEsupervised;KLchoiceNOTsinglecauseproof/automaticbugfix. NoheldlabelsforTRAIN.
- Evidence runs/context_verbalizer_20261112_v74/root_result.json/root_saved_review.json/root_meaning_review.json/root_manual_reading_review.json/root_retention_code_review.json/root_value_and_confusion_cpu.json/ROOT_terminal_native.json. Report outputs/FDT_CONTEXT_VERBALIZER_V74_RESULT_20261112_KR.md;NEXT outputs/FDT_CRITICAL_RETENTION_NEXT_CONTRACT_20261112_KR.md. PreserveV73parent81f4... andALLpriorstates. Cafterterminal~109.12GiB;GPUhead775015936B/cache3264918528B<12GiB.
- NEXT targetcorrection-retentioninterference withTRAIN-onlyseparatecorrectcriticalprotection+wrongcriticalcorrection andprospectivenewunsigned/signed/physicalmass/eventcontexts. NoactualV75data/runner/plan/seal/GPUyet. Storagecurrent~109GiBcannotstartanother3.5GiB+8GiBheadroomtrial;DONOTdeleteoldmodelsorlower100floor. Firstimplementpinnedactive7delta+runtimeFULLreconstruction/readonlysameepochcache-referenceandactualbyte/temp/STOPbudget;NOTyetimplemented/NOTstandalone. ActualCPUrecovery/preflight/independentvalidate/newfixedcaps/resourcesBEFOREnewONElaunch. Keepimprovinglanguagecycle;nodiffusion/wholecyclecompletionclaim.\n'''
    state=out/'PROJECT_STATE.md';old=state.read_text(encoding='utf-8');idx=old.index('## CURRENT')
    if heading not in old[:idx+len(heading)+2]:state.write_text(old[:idx]+current+old[idx:],encoding='utf-8')
    print(json.dumps(dict(status=review['status'],C_free_bytes=free,owned_output_bytes=owned,report=str(report),root_result_SHA=support.sha(run/'root_result.json'),PROJECT_STATE_first_CURRENT=heading)))
if __name__=='__main__':main()
