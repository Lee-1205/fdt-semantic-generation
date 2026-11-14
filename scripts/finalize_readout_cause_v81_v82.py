"""ROOT cause summary/data audit/state handoff; no model forward or training."""
import collections,hashlib,json,re,shutil,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    r81=ROOT/'runs/readout_cause_20261113_v81';r82=ROOT/'runs/precision_fix_check_20261113_v82';a=io.load_json(r81/'root_saved_review.json');b=io.load_json(r82/'root_saved_review.json')
    p=io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json');train=io.load_json(Path(p['data'])/'train_targets.json');features=io.load_json(Path(p['data'])/'train_features.json');dev=io.load_json(Path(p['eval_data'])/'features.json');labels=io.load_json(Path(p['eval_data'])/'targets.json');selected=io.load_json(Path(p['scope_data'])/'anchor_selection.json')['indices']
    def width(t,f):return len(re.match(r'[+-]?(\d+)',t[f+'_span']['text'])[1])
    def describe(rows):return dict(rows=len(rows),A_B_digit_width_pairs=dict(collections.Counter(str((width(t,'A'),width(t,'B'))) for t in rows)),B_patch_roles=dict(collections.Counter(str(t['B_span']['start']%16) for t in rows)),layouts=dict(collections.Counter(str(t['layout']) for t in rows)))
    trainhash={hashlib.sha256(x['prompt'].encode()).hexdigest() for x in features};leak=sum(hashlib.sha256(x['prompt'].encode()).hexdigest() in trainhash for x in dev)
    assert leak==0 and len(train)==1024 and len(selected)==100 and len(features)==1024
    data=dict(TRAIN_numeric=describe(train[:896]),TRAIN_reference_numeric=describe([train[i] for i in selected if i<896]),observed_new_dev=describe(labels),reference_sources=100,TRAIN_total_sources=1024,
        source_literal_DEV_in_TRAIN=leak,all84_authored_sign_width_cells_not_all_joint_contexts=True,
        detected_limitation='A/B sign varies independently, but integer digit widths are tied: (1,1)/(2,2)/(3,3) only, no mixed-width numeric TRAIN pairs. This does NOT explain current64, whose widths are all(2,2); it is a distinct prospective generalization gap, not the demonstrated cause of this regression.',
        B_patch_roles_covered_all16=True,not_simple_missing_byte_role=True,all_old_splits_unchanged=True,eval_rows_used_for_training=0)
    io.atomic_json(r82/'root_targeted_data_audit.json',data)
    regress=[v for v in b['affine_final_layer_conflicts'] if v['old_FP32_correct'] and not v['new_FP32_correct']]
    assert len(regress)==5 and a['counts']['diagonal_focus_reproduced']==12 and b['counts']['raw_fixed']==1 and b['counts']['raw_new_wrong']==0
    reading=dict(ROOT_read_all_changed_source_and_outputs=True,changed_row=49,source_fact='remaining stock 75 copper washers; separate delivery 34 copper washers',
        before_fact='remaining stock 7: copper washers; separate delivery 34 copper washers',after_fact='remaining stock 75 copper washers; separate delivery 34 copper washers',
        judgment='A real exact numeric-content repair, not merely a cosmetic punctuation normalization. Source/whole before/whole after were manually read. Other63 outputs byte-identical; no full abstract meaning certificate.')
    io.atomic_json(r82/'ROOT_manual_reading.json',reading)
    t=datetime.now().astimezone().isoformat();free=shutil.disk_usage(ROOT).free
    result=dict(status='ROOT_CAUSE_ANALYSIS_AND_NUMERIC_READOUT_FIX_COMPLETE_LANGUAGE_CYCLE_STILL_UNFINISHED',time=t,
        no_new_optimizer_training=True,actual_CUDA_readout_calls=197,actual_CPU_preflight_readout_calls=4,encoder_calls=0,denoise_calls=0,new_checkpoints=0,
        V81=dict(plan_SHA=support.sha(r81/'plan.json'),result=io.load_json(r81/'result.json'),diagonals_reproduced=12,diagnostic_selected_not_random=True),
        V82=dict(plan_SHA=support.sha(r82/'plan.json'),result=io.load_json(r82/'result.json'),counts=b['counts'],NLL=b['NLL'],numeric_severity={k:v['counts'] for k,v in b['B_numeric_severity'].items()}),
        confirmed_causes=[
            'BF16 readout arithmetic changes the discrete winner in at least row49 75->7:; fixed FP32 restores75 with same Commit/weights. It is a partial numerical cause, not the main remaining error explanation.',
            'With bitwise identical upstream Commit and five readout prefix tensors, the V75->V80 lastLinear2 update alone is sufficient for five selected full-FP32 correct->wrong transitions. The semantic input did not change between those endpoints.',
            'V80 TRAIN100-reference safeguard protected categorical top1 only, not margins or all1024/unseen contexts. It correctly reported no reference flips; interpreting that as global preservation was invalid.'],
        unsupported_or_refuted_in_selected_scope=[
            'Not all errors are BF16 artifacts: strong wrong short cases and nativeFP32 long failures persist.',
            'No target B winner flip for nine matched A-change local/global memory swaps; the tested other-fact interference hypothesis is not supported here. Zeroing globals is an off-manifold destructive control, not a repair.',
            'No current DIANA AR feedback-loop cause: current output is prefix-free CommitPatchReadout MLP; inherited AR parameters are unused in this path.',
            'All16 B byte roles are present in TRAIN/anchors, so a wholly missing role does not explain the current errors.',
            'No complete abstract semantic-memory certificate: bound Commit mixes E+binding; V71 contains upstream as well as output causes.'],
        code_fixes=dict(fixed_precision='research/readout_precision_policy_v81.py; fixed fullFP32, no labels/text/Exact, no changed weights',
            retention_candidate='research/train_margin_retention_v81.py; prospective TRAIN-only FP32 margin floor in addition to top1. CPU tested, not trained or integrated into an actual optimizer driver.',CPU_tests_passed=7),
        independent_data_audit=data,manual_reading=reading,last_layer_regression_cases=regress,global_alpha_conflict=b['global_alpha_conflict_example'],
        next='Keep V75 balanced reference and preserve V80 gains; no blind repeat/automatic rollback or SGD. Next targeted work: same-precision TRAIN feature+margin preservation covering actual TRAIN rather than only100, independent A/B width/predicate context coverage without DEV ingestion, then one separately bounded repair only after concrete driver review. No diffusion; future entire codec frozen.',
        Cfree=free,C100GiB_protected=True,all_owned_and_proxy_gone=True,all_sourceweights_AdamRNG_raw_failures_games_preserved=True)
    io.atomic_json(r82/'root_result.json',result)
    evidence=[r81/'plan.json',r81/'result.json',r81/'focused_patch_logits.pt',r81/'root_saved_review.json',r81/'ROOT_terminal_native.json',r82/'plan.json',r82/'result.json',r82/'root_saved_review.json',r82/'root_targeted_data_audit.json',r82/'ROOT_manual_reading.json',r82/'ROOT_terminal_native.json',r82/'root_result.json',ROOT/'research/readout_precision_policy_v81.py',ROOT/'research/train_margin_retention_v81.py',ROOT/'tests/test_precision_readout_v81.py',ROOT/'tests/test_train_margin_retention_v81.py']
    io.atomic_json(r82/'root_evidence_SHA.json',{str(f):support.sha(f) for f in evidence})
    io.atomic_json(ROOT/'configs/readout_cause_fix_v81_v82_review_20261113.json',dict(time=t,learned_model='V80 unchanged parameters, V75 reference preserved',precision_fix='fixed full FP32 Commit readout verified on SAME64',
        automatic_global_promotion=False,whole_language_cycle_complete=False,next_SGD_authorized_by_this_record=False,reason='Cause-first HUMAN instruction; retain useful historical states. FP32 fixes one reproduced byte; remaining output-boundary regressions require revised data and margin preservation, not blind repeated training.',result=str(r82/'root_result.json')))
    current='''## CURRENT — '''+t+''' ROOT cause-first V81/V82 READONLY COMPLETE / numerical fix verified / no new SGD

- LatestHUMAN cause analysis + concrete repair FIRST / no blind training. ROOTONLY/noagents. English language→meaning→language unfinished; denoise0; future thought training freezes ENTIRE codec. K_P/K_C active, K_E invalidNULLcontent0copy0; no raw/H/GTprefix/GTlength as decoder features. No E2final/registry/cloud/browser/deletions. Timer30min재개 unchanged. All old budgets CLOSED.
- V81 ONEactual133/160 readout calls,3.146054s,12purpose-selected observed byte cases/11cachedCommits/4precisionmodes/9paired local-global interventions. Noencoder/optimizer/denoise/checkpoint. All12nativefocusreproduced; sameCommit+fiveprefixfixed, V75→V80 lastLinear2 alone gives5selected FP32correct→wrong. Nine other-A/same-B local/global swaps did NOT flip B; zero-global destructive control NOTfix. Notallerrorsprecision/notARcurrentpath/notabstractmeaningcomplete. Native5652→32404→18936→23372 parent/command/exe/liveROOT14:30:28 observed;worker-ownerexit0/all4gone/Python0 confirmed14:30:40 BEFOREanalysis. Plan0c6ed4915af7cdd6e50101a3add7522f0adebfd2d4630311f204a6895f386530;oldclosedunused27NOTreuse.
- Fixed fullFP32 output arithmetic implemented research/readout_precision_policy_v81.py without weight/feature changes. V82 ONEactual64/70cachedCommit readout-only verification5.730154s on SAME64correlatedobserved rows, noencoder/optimizer/denoise/checkpoint. Source/output manuallyread: remainingstock75→7: repairedto75; other63outputs identical. Rawbyteerrors25→24, fixed1/new0; Bclear51→52/64,4clearwrong+8nonstandardprefixUNCERTIFIED remain; literal43→44aux; UTF8EOS64/64. NLL .0068071958→.0068121099 slightlyWORSE; no general performance-promotion claim. Native5036→3968→12740→15612 liveROOT14:37:12;worker-ownerexit0/all4gone/Python0 confirmed14:37:31 BEFOREanalysis. Plan dbeea4cf4c9043a5615d034c6e1a2f874eae63f762fcb6e7ef4236fe1cf302e4;unused6NOTreuse. Both new120wall90cut30reserve/32MiB budgets CLOSED.
- Confirmed retention limitation: TRAIN100top1 preservation is NOTmargin/wholeTRAIN/unseen preservation. New TRAIN-only FP32margin-floor candidate research/train_margin_retention_v81.py/7focusedCPUtotalPASS; noactualoptimizerdriverintegration/training yet. Same-precision TRAINfeatures must be rebuilt in any next trial; don't silentlyuseBF16prefixbankwithfullFP32runtime. Exact affine endpoint calculation: improving new8 needsalpha>=.571627 while retainingnew28 needsalpha<=.349073; simpleglobalold/newblend or smallerstepalongSAMEdirection cannot satisfyboth. These DEVdiagnostics are NOTtraining selectors.
- Targeteddataaudit: numericTRAIN896+OWT128,reference84numeric+16OWT; all16Brolescovered/literalDEVtrain0. A/B signsindependent butdigitwidths tied(1,1)320/(2,2)288/(3,3)288; mixedwidthgap existsbutcurrentDEV64all(2,2), soNOTprovenreasonforthisregression. Preservealloriginaldata/splits/weightsAdamRNGrawfailuresgames; noAUTOrollbackorSGD. BoundCommit doesnotseparatelyidentifyEvsbinding; V71upstreamcases preserved/no blanketallsemanticinformationperfectclaim. Codecfuturefreezeunchanged/denoise0.
- Evidence runs/precision_fix_check_20261113_v82/root_result.json/root_saved_review.json/root_targeted_data_audit.json/ROOT_manual_reading.json/root_evidence_SHA.json and runs/readout_cause_20261113_v81/root_saved_review.json. Report outputs/FDT_READOUT_CAUSE_AND_FIX_20261113_KR.md. CurrentCfree'''+str(free)+'''B>100GiB. Noactive modelprocess atlastterminal; noSGDthisturn. NEXT concrete train-margin/coverage remedy before ONEseparatelyboundedrepair; don'trepeat broadbaselinegates or assumecompletion. V75balancedreference/V80usefulresearchweightsbothpreserved.

'''
    state=WORK/'outputs/PROJECT_STATE.md';oldstate=state.read_text(encoding='utf-8-sig');first=oldstate.find('## CURRENT');state.write_text(oldstate[:first]+current+oldstate[first:],encoding='utf-8')
    print(json.dumps(dict(status=result['status'],counts=b['counts'],Cfree=free,resultSHA=support.sha(r82/'root_result.json'),evidence=len(evidence))))

if __name__=='__main__':main()
