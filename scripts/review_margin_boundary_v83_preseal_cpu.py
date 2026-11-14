import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    a=ROOT/'research/artifacts/margin_boundary_diagnostic_20261113_v83_cpu_v1';p=io.load_json(a/'plan_unsealed.json')
    assert not p['authorized'] and p['expected_calls']==4 and p['max_calls']==6 and p['max_output_bytes']==32*1024**2 and p['attempts']==p['epochs']==0
    for f,s in p['files'].items():assert support.sha(f)==s,f
    pre=io.load_json(a/'cpu_preflight_owned/final_status.json');assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['report']['actual_CPU_projection_calls']==2 and not pre['report']['CUDA_initialized']
    f=torch.load(p['bank_features'],map_location='cpu',weights_only=True);t=torch.load(p['bank_targets'],map_location='cpu',weights_only=True)
    assert f['source_indices']==t['source_indices'] and len(set(f['source_indices']))==100 and len(f['hidden'])==7487 and f['hidden'].dtype==torch.bfloat16
    assert int(t['initial_correct'].sum())==117600 and torch.equal(t['initial_correct'],(t['initial_argmax'].long()==t['labels'])&(t['labels']!=257))
    for version in p['heads'].values():assert p['files'][version['path']]==support.sha(version['path'])
    old=(ROOT/'scripts/supervise_precision_fix_check_v82.py').read_text(encoding='utf-8');new=(ROOT/'scripts/supervise_margin_boundary_diagnostic_v83.py').read_text(encoding='utf-8');assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
    free=shutil.disk_usage(ROOT).free;assert free>=108*1024**3+32*1024**2 and not torch.cuda.is_initialized()
    io.atomic_json(a/'root_independent_validate.json',dict(status='ROOT_V83_TRAIN_ONLY_READONLY_READY',actual_projection_calls=4,max_calls=6,preflight_actual_CPU_projection_calls=2,
        all_CPU_owned_exit_confirmed=True,initial_correct=117600,pins=len(p['files']),Cfree=free,protected100GiB=True,start108plus32MiB=True,
        feature_target_separate=True,unmodified_postPopen_cleanup=True,CUDA_initialized=False,no_SGD_encoder_denoise=True,
        limitation='BF16 cached prefix held fixed; final-projection precision diagnostic, not chosen fullFP32 runtime parity or whole TRAIN guarantee.'))
    print('ROOT_V83_INDEPENDENT_READY')

if __name__=='__main__':main()
