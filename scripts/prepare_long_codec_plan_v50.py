"""Create an unsealed long codec plan. Sealing requires actual CPU owner evidence."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'research/artifacts/long_codec_20261111_v50'
OUT=ROOT/'runs/long_three_memory_codec_20261111_v50'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    path=ART/'plan_unsealed.json'
    if path.exists():raise FileExistsError('new plan already prepared')
    old=json.loads((ROOT/'runs/three_memory_codec_20261111_v47/plan.json').read_text())
    files={str(Path(n)):d for n,d in old['files'].items() if Path(n).suffix=='.py'}
    source=ROOT/'runs/three_memory_codec_20261111_v47/model/checkpoints/state_2048.pt'
    extra=[ROOT/'research/long_memory_codec_v49.py',ROOT/'research/trainable_long_codec_v50.py',
        ROOT/'scripts/long_codec_engine_v50.py',ROOT/'scripts/run_long_codec_20261111_v50.py',
        ROOT/'scripts/supervise_long_codec_v50.py',ROOT/'scripts/owned_cpu_preflight_v50.py',
        ROOT/'scripts/prepare_long_codec_data_v50.py',ROOT/'scripts/test_long_codec_v50_cpu.py',Path(__file__),
        ART/'train.json',ART/'dev.json',ART/'orders.json',ART/'manifest.json',ART/'root_cpu_contract.json',
        ROOT/'runs/long_codec_capacity_profile_20261111_v49_r3/result.json',source]
    for x in extra:files[str(x)]=sha(x)
    for name,digest in files.items():
        if sha(name)!=digest:raise ValueError('code pin changed '+name)
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_long_codec_training_v50',authorized=False,root=str(ROOT),output=str(OUT),data=str(ART),
        source=str(source),source_sha='2c79076246164ec80c01b68b8d711f5fe0759cc1342159fb2423380523d320cf',
        files=files,capacity=16384,chunk_bytes=16,attempts=1728,epochs=4,amp_dtype='bfloat16',
        wall_seconds=2100,start=None,cutoff=None,deadline=None,max_calls=140000,max_output_bytes=3*1024**3,
        raw_indices=[0,4,8,12],raw_max_symbols=16385,
        memory_contract='K_P/K_C/K_E -> Commit only; no raw source or copy; semantic denoise0',
        objective='uniform half clean CE + half p50 corrupted-prefix CE; exact same input bytes/EOS',
        optimizer='old92 Adam tensors+moments+steps and four LRs unchanged;3 new tensors group LR1e-4',
        AMP='bfloat16 with preserved source enabled GradScaler; FP16 profile overflow retained as separate evidence',
        cost=dict(training=5184,teacher=64,raw_upper=131080,total_upper=136328),
        data_exposure=dict(unique_train_documents=432,epochs=4,row_selections=1728,target_UTF8_bytes=4161520),
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3,
        checkpoints='initial+one final or emergency; full standalone model/Adam/scaler/RNG/finite cursor/order/history')
    path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(str(path))
if __name__=='__main__':main()
