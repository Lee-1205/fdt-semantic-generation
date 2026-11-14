"""Combine121+73 independently reviewed traces from the SAME saved6-update model."""
import collections,gc,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from research.readout_decision_trace_v88 import decode
from review_critical_repair_v88_saved_cpu import review_trace
from review_long_extrapolation_v69_saved_cpu import PATTERN
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    torch.set_num_threads(4);train=ROOT/'runs/feasible_repair_20261113_v91';run=ROOT/'runs/missing_assessment_20261113_v93'
    p=io.load_json(train/'plan.json');rp=io.load_json(run/'plan.json');result=io.load_json(run/'result.json')
    term=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert term['AllOwnedGone'] and term['BeforeAnalysis'] and not term['PythonProcesses']
    assert owner['status']=='COMPLETE' and owner['exit_code']==0 and owner['exit']['all_owned_processes_gone']
    assert result['calls']==dict(readout=73) and result['optimizer_steps']==result['encoder_calls']==result['denoise_calls']==0
    assert support.sha(rp['source'])==result['source_SHA']==io.load_json(train/'incident.json')['emergency']['SHA']
    for f,sha in rp['files'].items():assert support.sha(f)==sha,f
    items=io.load_json(p['eval_inputs']);records=io.load_json(train/'root_partial_records.json')+io.load_json(run/'assessment/final/reconstructions.json')
    baseline=io.load_json(p['baseline_records']);assert len(records)==len(items)==len(baseline)==194
    rows=[];changes=[];panels=collections.defaultdict(lambda:dict(rows=0,symbols=0,before_correct=0,after_correct=0,before_NLL_sum=0.,after_NLL_sum=0.,UTF8_EOS=0,changed=0,before_decoded_errors=0,after_decoded_errors=0))
    samples=1936
    for i,(info,record,old) in enumerate(zip(items,records,baseline)):
        assert record['id']==old['id']==info['row']['id'] and record['panel']==info['panel']
        if i>=121:samples+=review_trace(info,record,p)['sample_bytes_verified']
        else:assert support.sha(record['trace'])==record['SHA']
        before=decode(torch.load(old['trace'],map_location='cpu',weights_only=True))['text']
        after=decode(torch.load(record['trace'],map_location='cpu',weights_only=True))['text'];source=info['row']['prompt']
        b,a,s=before.encode(),after.encode(),source.encode()
        err=lambda x:sum(c!=v for c,v in zip(s,x))+abs(len(s)-len(x))
        row=dict(global_index=i,id=record['id'],panel=info['panel'],source=source,before=before,after=after,
            before_decoded_errors=err(b),after_decoded_errors=err(a));rows.append(row)
        m=panels[info['panel']];m['rows']+=1;m['symbols']+=record['symbols'];m['before_correct']+=old['correct'];m['after_correct']+=record['correct']
        m['before_NLL_sum']+=old['NLL_sum'];m['after_NLL_sum']+=record['NLL_sum'];m['UTF8_EOS']+=record['UTF8'] and record['EOS']
        m['changed']+=before!=after;m['before_decoded_errors']+=err(b);m['after_decoded_errors']+=err(a)
        if b!=a:
            if len(a)!=len(b):changes.append(dict(global_index=i,length_before=len(b),length_after=len(a),source=source[:180],before=before[:180],after=after[:180]))
            else:
                for offset,(u,v) in enumerate(zip(b,a)):
                    if u!=v:changes.append(dict(global_index=i,byte_offset=offset,source=s[max(0,offset-65):offset+80].decode(errors='replace'),before=b[max(0,offset-65):offset+80].decode(errors='replace'),after=a[max(0,offset-65):offset+80].decode(errors='replace'),fixed_to_source=v==s[offset],new_error=u==s[offset] and v!=s[offset]))
        gc.collect()
    for m in panels.values():
        m['before_NLL']=m['before_NLL_sum']/m['symbols'];m['after_NLL']=m['after_NLL_sum']/m['symbols']
        m['before_byte_EOS_accuracy']=m['before_correct']/m['symbols'];m['after_byte_EOS_accuracy']=m['after_correct']/m['symbols']
    oldplan=io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json')
    targets=io.load_json(Path(oldplan['old_eval_data'])/'long_targets.json')[:2];longs=[r for r in rows if r['panel']=='long'];metrics={};snippets=[]
    for phase in ('before','after'):
        counts=collections.Counter()
        for r,t in zip(longs,targets):
            source=r['source'].encode();actual=r[phase].encode();assert len(source)==len(actual)
            for event in t['events']:
                a,b=event['byte_start'],event['byte_end'];text=source[a:b].decode();out=actual[a:b].decode();match=PATTERN.fullmatch(text);assert match
                for group in ('quantity','reserve','actor','checker','item','order'):
                    a0,b0=match.span(group);counts[group+'_positions']+=1;counts[group+'_errors']+=text[a0:b0]!=out[a0:b0]
                pos=text.index(' moved ')+1;counts['predicate_positions']+=1;counts['predicate_errors']+=out[pos:pos+5]!='moved'
            for match in re.finditer(rb'episode (\d+)',source):counts['episode_positions']+=1;counts['episode_errors']+=source[match.start(1):match.end(1)]!=actual[match.start(1):match.end(1)]
            for match in re.finditer(rb'\b(?:shelf|drawer|box|gate)\b',source):counts['location_noun_positions']+=1;counts['location_noun_errors']+=source[match.start():match.end()]!=actual[match.start():match.end()]
            if phase=='after':
                for offset in (0,len(source)//2,len(source)-260):snippets.append(dict(id=r['id'],offset=offset,source=source[offset:offset+260].decode(),output=actual[offset:offset+260].decode()))
        metrics[phase]=dict(counts)
    e=torch.load(train/'emergency/runtime.pt',map_location='cpu',weights_only=True);residual=[];same=collections.Counter()
    for h in e['runtime']['history']:
        checks=h['update_receipt']['reference_checks']
        if len(checks)!=2 or checks[1]['accept']:continue
        first={(d['source_id'],d['byte_offset'],d['gold'],d['rival']) for d in checks[0]['diagnostics']}
        for d in checks[1]['diagnostics']:
            same['same_active_pair' if (d['source_id'],d['byte_offset'],d['gold'],d['rival']) in first else 'new_pair_or_context']+=1
            residual.append(d['deficit'])
    proof=dict(status='ROOT_V91_PLUS_V93_COMPLETE194_SAME6UPDATE_ASSESSMENT_REVIEWED_NO_AUTO_PROMOTION',
        training_owner_original_status='FAILED/exit2 preserved',training_attempts70=True,accepted_corrected_updates=6,rejected64=True,
        V92_readonly_API_failure_actual_model_calls=0,V93_readonly_owner='COMPLETE',V93_readout_calls=73,V93_worker_seconds=result['seconds'],
        combined_outputs194=True,old121_not_GPU_repeated=True,all_native_owned_gone_before_analysis=True,
        panels=dict(panels),changed_contexts=changes,long_source_bound_fields=metrics,start_middle_end_6=snippets,
        total_sampled_full258_scores_verified=samples,full258_not_saved=True,
        corrected_rejection_same_pair_counts=dict(same),corrected_rejected_gap_deficit_min=min(residual),
        corrected_rejected_gap_deficit_max=max(residual),correction_padding=1e-6,
        original_reference_119769top1_152floors_preserved=True,source_ALLAdam_RNG_unchanged=True,
        extra_model_calls_in_ROOT_review=0,CUDA_initialized=False,
        interpretation='6 actually corrected proposals preserved finite TRAIN references, but decoded output/meaning gains must follow observed panel results. Same-active-pair residuals after affine correction support insufficient FP32 numerical headroom, not newly-discovered unseen semantic failure. Saved rejected weights absent; numerical mechanism not uniquely proven.',
        next='Preserve6updates and failures. Existing V52 scanner fix now used in NEW readonly code; shared make_engine.device verified by actual CPU assessment path. No more missing73 retry. Next training change should address affine-vs-FP32 guard numerical headroom without lowering ORIGINAL floors; no blindSGD or broadbaseline.')
    io.atomic_json(train/'root_combined_decoded_panels.json',rows);io.atomic_json(train/'root_combined_assessment_review.json',proof)
    files=[Path(__file__).resolve(),train/'root_saved_review.json',train/'root_combined_assessment_review.json',train/'root_combined_decoded_panels.json',train/'incident.json',train/'ROOT_terminal_native.json',run/'plan.json',run/'result.json',run/'ROOT_terminal_native.json',run/'final_status.json',Path(rp['source'])]
    io.atomic_json(train/'root_combined_evidence_SHA.json',{str(f):support.sha(f) for f in files})
    print(json.dumps({k:v for k,v in proof.items() if k!='start_middle_end_6'},ensure_ascii=False))
    print('ROOT_LONG_READING_CARDS '+json.dumps(snippets,ensure_ascii=False))


if __name__=='__main__':main()
