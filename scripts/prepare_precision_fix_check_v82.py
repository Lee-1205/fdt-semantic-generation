import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    src=io.load_json(ROOT/'runs/readout_cause_20261113_v81/plan.json');old=io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json')
    a=ROOT/'research/artifacts/precision_fix_check_20261113_v82_cpu_v1';a.mkdir(exist_ok=False);rows=io.load_json(Path(old['eval_data'])/'features.json')
    items=[dict(raw=str(ROOT/f'runs/update_retention_20261113_v80/raw_new_English_{i:02d}.pt'),row=r) for i,r in enumerate(rows)];io.atomic_json(a/'inputs.json',items)
    owner=(ROOT/'scripts/supervise_readout_cause_v81.py').read_text(encoding='utf-8').replace('fdt_readonly_readout_cause_v81','fdt_precision_fix_readonly_v82').replace("p['max_calls']!=160","p['max_calls']!=70").replace('run_readout_cause_20261113_v81.py','run_precision_fix_check_20261113_v82.py')
    op=ROOT/'scripts/supervise_precision_fix_check_v82.py'
    with op.open('x',encoding='utf-8',newline='\n') as f:f.write(owner)
    base=(ROOT/'scripts/supervise_readout_cause_v81.py').read_text(encoding='utf-8');assert base.split("    state=dict(status='PREPARING'",1)[1]==owner.split("    state=dict(status='PREPARING'",1)[1]
    files=dict(src['files'])
    extra=[a/'inputs.json',op,ROOT/'scripts/run_precision_fix_check_20261113_v82.py',ROOT/'scripts/prepare_precision_fix_check_v82.py',ROOT/'research/readout_precision_policy_v81.py',ROOT/'research/train_margin_retention_v81.py',ROOT/'tests/test_train_margin_retention_v81.py']+[Path(i['raw']) for i in items]
    for f in extra:files[str(f.resolve())]=support.sha(f)
    p={k:src[k] for k in ('root','storage_parent','heads')};p.update(schema='fdt_precision_fix_readonly_v82',authorized=False,output=str(ROOT/'runs/precision_fix_check_20261113_v82'),
        wall_seconds=120,max_calls=70,expected_calls=64,max_output_bytes=32*1024**2,attempts=0,epochs=0,input_list=str(a/'inputs.json'),targets=str(Path(old['eval_data'])/'targets.json'),
        files=files,optimizer_steps=0,encoder_calls=0,denoise_calls=0,checkpoints=0,e2_final_accessed=False,purpose='Verify fixed full-FP32 output arithmetic once on same cached 64 Commit, no encoder or SGD. All old budgets CLOSED.')
    io.atomic_json(a/'plan_unsealed.json',p);print(json.dumps(dict(plan=str(a/'plan_unsealed.json'),expected_calls=64,pins=len(files),FP32_logit_bytes=sum(((len(r['prompt'].encode())+15)//16+1)*16*258*4 for r in rows))))

if __name__=='__main__':main()
