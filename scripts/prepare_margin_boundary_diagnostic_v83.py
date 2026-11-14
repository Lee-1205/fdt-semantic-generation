import ast,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    source=io.load_json(ROOT/'runs/precision_fix_check_20261113_v82/plan.json');env=torch.load(ROOT/'runs/update_retention_20261113_v80/model/runtime.pt',map_location='cpu',weights_only=True)['runtime']
    a=ROOT/'research/artifacts/margin_boundary_diagnostic_20261113_v83_cpu_v1';a.mkdir(exist_ok=False)
    old=(ROOT/'scripts/supervise_precision_fix_check_v82.py').read_text(encoding='utf-8');new=old.replace('fdt_precision_fix_readonly_v82','fdt_TRAIN_margin_boundary_readonly_v83').replace("p['max_calls']!=70","p['max_calls']!=6").replace('run_precision_fix_check_20261113_v82.py','run_margin_boundary_diagnostic_v83.py')
    owner=ROOT/'scripts/supervise_margin_boundary_diagnostic_v83.py'
    with owner.open('x',encoding='utf-8',newline='\n') as f:f.write(new)
    assert new.split("    state=dict(status='PREPARING'",1)[1]==old.split("    state=dict(status='PREPARING'",1)[1]
    extras=[owner,ROOT/'scripts/run_margin_boundary_diagnostic_v83.py',ROOT/'scripts/prepare_margin_boundary_diagnostic_v83.py',Path(env['bank_ref']['features']),Path(env['bank_ref']['targets']),Path(env['reference_final_ref']['path'])]
    files=dict(source['files'])
    for f in extras:files[str(f.resolve())]=support.sha(f)
    assert files[env['bank_ref']['features']]==env['bank_ref']['features_SHA'] and files[env['bank_ref']['targets']]==env['bank_ref']['targets_SHA'] and files[env['reference_final_ref']['path']]==env['reference_final_ref']['SHA']
    for f in extras[:3]:ast.parse(f.read_text(encoding='utf-8'))
    f=torch.load(env['bank_ref']['features'],map_location='cpu',weights_only=True);t=torch.load(env['bank_ref']['targets'],map_location='cpu',weights_only=True)
    assert f['hidden'].shape==(7487,512) and t['labels'].shape==(7487,16) and len(f['source_indices'])==100 and int(t['initial_correct'].sum())==117600
    free=shutil.disk_usage(ROOT).free;assert free>=108*1024**3+32*1024**2 and not torch.cuda.is_initialized()
    p={k:source[k] for k in ('root','storage_parent','heads')};p.update(schema='fdt_TRAIN_margin_boundary_readonly_v83',authorized=False,output=str(ROOT/'runs/margin_boundary_diagnostic_20261113_v83'),
        wall_seconds=120,max_calls=6,expected_calls=4,max_output_bytes=32*1024**2,attempts=0,epochs=0,files=files,
        bank_features=env['bank_ref']['features'],bank_targets=env['bank_ref']['targets'],bank_final_predictions=env['reference_final_ref']['path'],
        optimizer_steps=0,encoder_calls=0,denoise_calls=0,checkpoints=0,e2_final_accessed=False,purpose='Measure real TRAIN100 margin erosion under accepted V80 endpoint, not another training/evaluation baseline. All old budgets CLOSED.')
    io.atomic_json(a/'plan_unsealed.json',p);io.atomic_json(a/'root_CPU_data_review.json',dict(status='ROOT_TRAIN_ONLY_FROZEN_FEATURE_INPUT_REVIEW_PASS',pins=len(files),hidden_rows=7487,TRAINsources100=True,
        feature_target_files_separate=True,initial_correct=117600,actual_GPU_or_CPU_projection_calls=0,CUDA_initialized=False,postPopen_body_unchanged=True,Cfree=free,
        max_saved_output_estimate_bytes=26*1024**2,max_output_bytes=32*1024**2,projection_calls=4,full_native_argmax_expected_for_two_BF16_endpoints=True,
        labels_used_for_loss_and_diagnostic_score_only=True,not_full_FP32_prefix_runtime=True))
    print(json.dumps(dict(plan=str(a/'plan_unsealed.json'),pins=len(files),expected_calls=4,CUDA_initialized=False)))

if __name__=='__main__':main()
