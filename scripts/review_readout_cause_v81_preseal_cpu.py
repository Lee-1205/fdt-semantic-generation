"""ROOT independent plan/data/ownership review; zero model calls/CUDA."""
import ast,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import unpack
from saved_commit_reference_v76_v2 import resolve_saved_commit

def main():
    folder=ROOT/'research/artifacts/readout_cause_20261113_v81_cpu_v1';p=io.load_json(folder/'plan_unsealed.json');pins=p['files']
    assert not p['authorized'] and p['attempts']==p['epochs']==p['optimizer_steps']==p['encoder_calls']==p['denoise_calls']==p['checkpoints']==0
    assert p['max_calls']==160 and p['expected_calls']==133 and p['wall_seconds']==120 and p['max_output_bytes']==32*1024**2
    for file,digest in pins.items():assert support.sha(file)==digest,file
    assert support.core_sha(ROOT)==support.CORE_SHA
    worker=(ROOT/'scripts/run_readout_cause_20261113_v81.py').read_text(encoding='utf-8');ast.parse(worker)
    assert 'torch.optim' not in worker and '.backward(' not in worker and '.encode_input(' not in worker
    old=(ROOT/'scripts/supervise_update_retention_v80.py').read_text(encoding='utf-8');new=(ROOT/'scripts/supervise_readout_cause_v81.py').read_text(encoding='utf-8')
    assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
    pre=io.load_json(folder/'cpu_preflight_owned/final_status.json');assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['report']['actual_CPU_readout_calls']==2
    inputs=io.load_json(p['inputs']);cases=io.load_json(p['cases']);assert len(inputs)==11 and len(cases)==12
    embed='3ea6672a445e984751fc0489b1e2067eaf6858c704291bc8848f0f81f5c4805c';paired=0
    for c in cases:
        info=inputs[c['input']];raw=unpack(torch.load(info['raw'],map_location='cpu',weights_only=True));row=info['row'];z,_=resolve_saved_commit(raw,row,embed,pins,ROOT,512)
        assert c['gold']==row['prompt'].encode()[c['offset']]==int(raw['targets'].flatten()[c['offset']])
        if 'partner' in c:
            partner=c['partner'];other=unpack(torch.load(partner['raw'],map_location='cpu',weights_only=True));oz,_=resolve_saved_commit(other,partner['row'],embed,pins,ROOT,512);s=c['fact_span'];a,b=s['start'],s['end']
            assert z.shape==oz.shape and row['prompt'].encode()[a:b]==partner['row']['prompt'].encode()[a:b]==s['text'].encode()
            assert row['prompt']!=partner['row']['prompt'];paired+=1
    assert paired==9 and len(inputs)*2*4+paired*5==133
    free=shutil.disk_usage(ROOT).free;assert free>=108*1024**3+32*1024**2 and not torch.cuda.is_initialized()
    report=dict(status='ROOT_INDEPENDENT_READONLY_V81_READY',pins=len(pins),actual_readout_restore_proof=pre['report'],cases=12,inputs=11,matched_B_context_interventions=9,exact_cost=133,
        max_calls=160,wall120_cut90_reserve30=True,owned32MiB=True,Cfree=free,start_required=108*1024**3+32*1024**2,unchanged_reviewed_postPopen_cleanup=True,
        no_new_optimizer_or_encoder=True,CUDA_initialized=False,actual_new_GPU_calls=0,prepared_not_performance=True,
        limitation='diagnostic offsets choose interventions/scores only; bound local includes original E+binding. No all-semantic or unseen certificate.')
    io.atomic_json(folder/'root_independent_validate.json',report);print(json.dumps({k:v for k,v in report.items() if k!='actual_readout_restore_proof'}))

if __name__=='__main__':main()
