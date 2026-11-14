"""New role-binding hypothesis, immutable same-data references and actual caps."""
import ast,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_ordered_fuzzy_20261111_v55 import validate,SOURCE_SHA,HEAD_SHA,EOS_LOG_BIAS

def main():
    artifact=ROOT/'research/artifacts/ordered_fuzzy_20261111_v55_cpu'
    if (artifact/'plan_unsealed.json').exists():raise FileExistsError('new immutable plan')
    data=ROOT/'research/artifacts/commit_readout_20261111_v54_cpu'
    manifest=json.loads((data/'manifest.json').read_text())
    previous=json.loads((ROOT/'runs/commit_readout_20261111_v54/plan.json').read_text())
    source=ROOT/'runs/prefix_independent_codec_20261111_v53/model/checkpoints/state_8192.pt'
    head=ROOT/'runs/commit_readout_20261111_v54/model/checkpoints/state_3624.pt'
    output=ROOT/'runs/ordered_fuzzy_20261111_v55'
    files={k:v for k,v in previous['files'].items() if Path(k).suffix=='.py'}
    paths=[ROOT/'research/ordered_fuzzy_readout_v55.py',ROOT/'scripts/ordered_fuzzy_engine_v55.py',
        ROOT/'scripts/run_ordered_fuzzy_20261111_v55.py',ROOT/'scripts/supervise_ordered_fuzzy_v55.py',
        ROOT/'scripts/test_ordered_fuzzy_v55_cpu.py',Path(__file__),artifact/'root_CPU_contract.json',
        data/'train.json',data/'dev.json',data/'manifest.json',source,head,
        ROOT/'runs/commit_readout_20261111_v54/root_saved_review.json',ROOT/'runs/commit_readout_20261111_v54/root_EOS_recalculation.json']
    for path in paths:files[str(path)]=support.sha(path)
    for name,digest in files.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    cost=dict(encode=540,train_readout=3624,normal_and_role_reversal_readout=32,bank_readout=12,total=4208)
    p=dict(schema='fdt_ordered_fuzzy_v55',authorized=False,root=str(ROOT),output=str(output),data=str(data),
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=HEAD_SHA,EOS_log_bias=EOS_LOG_BIAS,
        files=files,capacity=16384,attempts=3624,epochs=6,batch=256,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=4500,max_output_bytes=8*1024**3,
        raw_indices=[0,4,8,12],source_encoder_optimizer_updates=0,cost=cost,
        exposure=dict(TRAIN_documents=512,DEV_documents=16,unique_source_UTF8_bytes=manifest['train_UTF8_bytes'],
            patch_selections=6*manifest['train_active_patches'],target_UTF8_bytes=6*manifest['train_UTF8_bytes'],epochs=6),
        optimizer='fresh AdamW projection1e-4/readout3e-4 WD.01 clip1 enabledGradScaler1024 BF16; old source96Adam and head7Adam retained in required parents',
        objective='uniform active-symbol prefix-free CE',
        feature_boundary='same fuzzy N/D split by input role -> shared256to32 ->16 disjoint role subspaces -> Commit -> language; no raw H/bytes/target prefix',
        terminal_role='one learned structural tail; no gold output length or forced EOS',
        recovery='all8 active tensors/Adam/scaler/allRNG/phase/shards/cache refs/orders/cursor/history; both immutable parent references required',
        maximum_head_checkpoint_writes=4,source_copy_bytes=0,semantic_denoise_calls=0,automatic_model_promotion=False,
        estimated_role_cache_bytes=manifest['train_cached_patches']*16*256*4,
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    path=artifact/'plan_unsealed.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    validate(path,output);print(json.dumps(dict(actual_schema_PASS=True,cost=cost,estimated_role_cache_bytes=p['estimated_role_cache_bytes'])))
if __name__=='__main__':main()
