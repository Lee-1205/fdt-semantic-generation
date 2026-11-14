"""Actual cost/schema/source contract for a NEW frozen-Commit decoder probe."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_commit_readout_20261111_v54 import validate,SOURCE_SHA

def main():
    artifact=ROOT/'research/artifacts/commit_readout_20261111_v54_cpu'
    if (artifact/'plan_unsealed.json').exists():raise FileExistsError('new plan already prepared')
    manifest=json.loads((artifact/'manifest.json').read_text());old=json.loads((ROOT/'runs/prefix_independent_codec_20261111_v53/plan.json').read_text())
    source=ROOT/'runs/prefix_independent_codec_20261111_v53/model/checkpoints/state_8192.pt'
    output=ROOT/'runs/commit_readout_20261111_v54'
    files={name:digest for name,digest in old['files'].items() if Path(name).suffix=='.py'}
    for path in [ROOT/'research/commit_patch_readout_v54.py',ROOT/'scripts/commit_readout_engine_v54.py',
        ROOT/'scripts/run_commit_readout_20261111_v54.py',ROOT/'scripts/supervise_commit_readout_v54.py',
        ROOT/'scripts/prepare_commit_readout_data_v54.py',ROOT/'scripts/test_commit_readout_v54_cpu.py',Path(__file__),
        artifact/'train.json',artifact/'dev.json',artifact/'manifest.json',artifact/'root_CPU_contract.json',
        ROOT/'runs/prefix_independent_codec_20261111_v53/root_saved_review.json',source]:files[str(path)]=support.sha(path)
    for name,digest in files.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_frozen_commit_readout_v54',authorized=False,root=str(ROOT),output=str(output),data=str(artifact),
        source=str(source),source_sha=SOURCE_SHA,files=files,capacity=16384,attempts=3624,epochs=6,batch=256,
        amp_dtype='bfloat16',wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=5000,max_output_bytes=2*1024**3,
        raw_indices=[0,4,8,12],source_encoder_optimizer_updates=0,cost=manifest['cost'],
        exposure=dict(TRAIN_documents=512,DEV_documents=16,encoder_train_UTF8_bytes=manifest['train_UTF8_bytes'],
            readout_train_patch_selections=6*manifest['train_active_patches'],
            readout_target_UTF8_bytes=6*manifest['train_UTF8_bytes'],epochs=6),
        optimizer='fresh AdamW only on new shared readout,LR1e-3,WD.01,clip1,enabledGradScaler1024; old96Adam/scaler/RNG retained in immutable parent',
        objective='all-active-symbol CE + .1 mean EOS CE; no answer-prefix feature',
        feature_boundary='input -> frozen V53 three memories -> Commit only; head never receives raw source or target bytes',
        terminal_role='one learned structural role after final committed slot; no gold byte length and no forced EOS',
        recovery='head model/Adam/scaler/RNG/phase/extraction journal/cache SHA/order/cursor/history; nonstandalone parent-required references',
        maximum_head_checkpoint_writes=4,source_copy_bytes=0,semantic_denoise_calls=0,automatic_model_promotion=False,
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    path=artifact/'plan_unsealed.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    validate(path,output)
    print(json.dumps(dict(CPU_schema_PASS=True,cost=manifest['cost'],estimated_cache_bytes=2*manifest['estimated_feature_target_bytes'])))
if __name__=='__main__':main()
