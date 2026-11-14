import ast,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import unpack

def main():
    a=ROOT/'research/artifacts/precision_fix_check_20261113_v82_cpu_v1';p=io.load_json(a/'plan_unsealed.json');items=io.load_json(p['input_list'])
    assert len(items)==64 and len({r['row']['id'] for r in items})==64 and not p['authorized']
    assert p['attempts']==p['epochs']==p['optimizer_steps']==p['encoder_calls']==p['denoise_calls']==0 and p['max_calls']==70 and p['expected_calls']==64 and p['wall_seconds']==120 and p['max_output_bytes']==32*1024**2
    for f,s in p['files'].items():assert support.sha(f)==s,f
    expected=0
    for item in items:
        packet=unpack(torch.load(item['raw'],map_location='cpu',weights_only=True));b=item['row']['prompt'].encode();z=packet['fuzzy_commit'];labels=packet['targets'];y=torch.full_like(labels.flatten(),257);y[:len(b)]=torch.tensor(list(b));y[len(b)]=256
        assert packet['source_id']==item['row']['id'] and packet['Exact_entries']==0 and torch.equal(labels.flatten(),y) and z.shape==(1,(len(b)+15)//16+2,512)
        expected+=labels.numel()*258*4
    assert expected==17502720 and expected<20*1024**2 # ample 32MiB incl output metadata
    text=(ROOT/'scripts/run_precision_fix_check_20261113_v82.py').read_text(encoding='utf-8');ast.parse(text);assert '.backward(' not in text and 'torch.optim' not in text
    old=(ROOT/'scripts/supervise_readout_cause_v81.py').read_text(encoding='utf-8');new=(ROOT/'scripts/supervise_precision_fix_check_v82.py').read_text(encoding='utf-8');assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
    pre=io.load_json(a/'cpu_preflight_owned/final_status.json');assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and not pre['report']['CUDA_initialized']
    free=shutil.disk_usage(ROOT).free;assert free>=108*1024**3+32*1024**2 and not torch.cuda.is_initialized()
    r=dict(status='ROOT_V82_FIXED_PRECISION_CHECK_READY',pins=len(p['files']),same_observed_rows=64,exact_calls=64,max_calls=70,logits_bytes=expected,hard_output_bytes=p['max_output_bytes'],
        wall120_cut90_reserve30=True,all_old_CPU_owned_gone=True,Cfree=free,no_training_or_encoder=True,unchanged_reviewed_postPopen_cleanup=True,CUDA_initialized=False,extra_model_calls=0)
    io.atomic_json(a/'root_independent_validate.json',r);print(json.dumps(r))

if __name__=='__main__':main()
