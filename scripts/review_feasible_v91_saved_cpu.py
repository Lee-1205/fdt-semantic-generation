"""Review the FAILED V91 run's preserved weights and121 completed traces.

No extra forward. Does not retroactively mark owner/run COMPLETE or assume the
missing73 outputs. Original floors/Adam/RNG/failed worker remain preserved.
"""
import collections,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from research.readout_decision_trace_v88 import decode
from research.critical_bank_storage_v87 import load as load_bank
from review_critical_repair_v88_saved_cpu import review_trace
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    torch.set_num_threads(4);r=ROOT/'runs/feasible_repair_20261113_v91';p=io.load_json(r/'plan.json')
    term=io.load_json(r/'ROOT_terminal_native.json');incident=io.load_json(r/'incident.json');owner=io.load_json(r/'final_status.json')
    assert term['AllOwnedGone'] and term['BeforeAnalysis'] and not term['PythonProcesses']
    assert owner['status']=='FAILED' and owner['exit_code']==2 and owner['exit']['all_owned_processes_gone']
    for f,sha in p['files'].items():assert support.sha(f)==sha,f
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True)
    before=torch.load(p['source'],map_location='cpu',weights_only=True);warm=dict(storage['model']);warm.update(before['active_weights'])
    e=torch.load(incident['emergency']['path'],map_location='cpu',weights_only=True);model=dict(storage['model']);model.update(e['active_weights']);s=dict(e['runtime'],model=model)
    assert support.sha(incident['emergency']['path'])==incident['emergency']['SHA']
    assert support.state_sha(s)==incident['emergency']['reconstructed_fullstate_SHA']==e['reconstructed_fullstate_SHA']
    assert all(torch.equal(v,warm[k]) for k,v in model.items() if k not in p['trainable_names'])
    assert s['attempts']==70 and s['updates']==6 and s['rejections']==64 and s['skips']==0 and s['inflight'] is None
    assert len(s['optimizer']['state'])==2 and all(int(v['step'])==6 for v in s['optimizer']['state'].values())
    assert s['scaler']['_growth_tracker']==70 and s['cache_refs']==before['runtime']['cache_refs'] and s['bank_ref']==before['runtime']['bank_ref']
    assert s['teacher_sha']==before['runtime']['teacher_sha'] and not s['exposure']['teacher_rows']
    assert s['exposure']['head_rows']==s['exposure']['head_encoder_rows']==350
    assert len(s['history'])==70 and [h['rows'] for h in s['history']]==s['orders'][:70]
    assert all(h['update_receipt']['acceptance']=='bounded_corrected_actual_proposal' for h in s['history'] if h['update_receipt']['accepted'])
    assert all(not h['update_receipt']['accepted'] for h in s['history'][-32:]) and s['history'][-33]['update_receipt']['accepted']
    assert sum(len(h['update_receipt']['reference_checks']) for h in s['history'])+2==s['calls']['reference_linear']==135
    bank=load_bank(s['bank_ref'],ROOT,support.sha);c=bank['calibration'];f=torch.load(s['reference_final']['path'],map_location='cpu',weights_only=True)
    assert support.sha(s['reference_final']['path'])==s['reference_final']['SHA']
    assert not ((f['argmax'].long()!=c['labels'])&c['top1_mask']).any()
    assert not ((f['margin']<c['required_margin'])&c['weak_floor_mask']).any()
    items=io.load_json(p['eval_inputs']);baseline=io.load_json(p['baseline_records']);records=[];cards=[];changes=[];sampled=0
    for i in range(121):
        file=r/f'assessment/final/trace_{i:03d}.pt';t=torch.load(file,map_location='cpu',weights_only=True);d=decode(t)
        info=items[i];raw=info['row']['prompt'].encode();n=len(raw)+1;y=torch.full_like(t['argmax'],257,dtype=torch.long)
        y[:len(raw)]=torch.tensor(list(raw));y[len(raw)]=256;valid=y!=257
        record=dict(id=info['row']['id'],panel=info['panel'],index=info['index'],trace=str(file),SHA=support.sha(file),
            EOS=d['EOS'],UTF8=d['UTF8'],exact_auxiliary=d['text']==info['row']['prompt'],symbols=n,
            correct=int(((t['argmax'].long()==y)&valid).sum()),NLL_sum=float(-t['gold_log_probability'][valid].double().sum()))
        reviewed=review_trace(info,record,p);sampled+=reviewed['sample_bytes_verified']
        old=decode(torch.load(baseline[i]['trace'],map_location='cpu',weights_only=True))['text']
        if old!=d['text']:changes.append(dict(global_index=i,id=record['id'],panel=record['panel'],source=info['row']['prompt'],before=old,after=d['text']))
        records.append(record);cards.append(dict(global_index=i,id=record['id'],panel=record['panel'],source=info['row']['prompt'],output=d['text']))
        del t;gc.collect()
    assert len(list((r/'assessment/final').glob('trace_*.pt')))==121
    io.atomic_json(r/'root_partial_records.json',records);io.atomic_json(r/'root_partial_decoded.json',cards)
    io.atomic_json(r/'root_partial_changes.json',changes);io.atomic_json(r/'root_missing_inputs.json',items[121:])
    summary=dict(status='ROOT_V91_FAILED_ASSESSMENT_PRESERVED_MODEL_AND_PARTIAL121_INDEPENDENTLY_REVIEWED',
        worker_owner_status_preserved='FAILED/exit2',attempts=70,accepted=6,rejected=64,AMPskip=0,last_accepted=38,
        all_accepted_are_actual_corrected_proposals=True,fullstate_SHA=incident['emergency']['SHA'],algorithm_SHA=support.state_sha(s),
        all_frozen_except_last2_bitwise=True,populatedAdam6_scaler70_ALLstoredRNG_preserved=True,
        original_reference=dict(top1=119769,weak_floor152=True,flips=0,floor_violations=0),
        calls=s['calls'],total_calls=sum(s['calls'].values()),unused_calls_closed=1400-sum(s['calls'].values()),unconsumed_max_attempts_closed=26,
        static_teacher_rows=1024,new_teacher_rows=0,head_rows=350,head_UTF8bytes=s['exposure']['head_UTF8_bytes'],
        partial_outputs_reviewed=121,partial_changed_outputs=len(changes),sampled_full258_byte_scores_verified=sampled,
        missing_outputs=73,missing_long_outputs=2,whole_model_performance_NOT_assessed=True,
        cause='Worker reused old unsafe size() while owner already used V52 atomic-rename-safe scanner. A state.json.*.tmp disappeared between is_file and stat. Assessment I/O supervision race, not failed tensor/SGD computation.',
        fix='Use existing owned_output_size_v52.owned_output_bytes in NEW code. Preserve sealedV91; no training restart; complete only73missing assessments under NEW bounded readonly purpose if reviewed.',
        extra_model_calls=0,CUDA_initialized=False)
    io.atomic_json(r/'root_saved_review.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
