"""Root preparation of a new one-shot measured length allocation."""
import ast
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def main():
    output=ROOT/'runs/long_codec_capacity_profile_20261111_v49_r2'
    if output.exists():raise FileExistsError('do not reopen a prior profile')
    old=json.loads((ROOT/'runs/three_memory_codec_20261111_v47/plan.json').read_text(encoding='utf-8'))
    source=ROOT/'runs/three_memory_codec_20261111_v47/model/checkpoints/state_2048.pt'
    source_sha='2c79076246164ec80c01b68b8d711f5fe0759cc1342159fb2423380523d320cf'
    if sha(source)!=source_sha:raise ValueError('source')
    corpus=Path(old['artifact'])
    files={str(Path(name)):digest for name,digest in old['files'].items() if Path(name).suffix=='.py'}
    for path in [ROOT/'research/long_memory_codec_v49.py',ROOT/'scripts/profile_long_memory_v49_r2.py',
                 ROOT/'scripts/supervise_long_profile_v49_r2.py',Path(__file__),
                 ROOT/'research/artifacts/long_memory_codec_20261111_v49_cpu/result.json',
                 corpus/'documents.json',corpus/'manifest.json',source]:
        files[str(path)]=sha(path)
    for name,digest in files.items():
        if sha(name)!=digest:raise ValueError('inherited source pin changed '+name)
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'),filename=name)
    start=time.time()
    plan=dict(schema='fdt_long_codec_capacity_profile_v49',output=str(output),source=str(source),
        source_sha=source_sha,corpus=str(corpus),data_sha=sha(corpus/'manifest.json'),files=files,
        start=start,deadline=start+300,cutoff=start+255,wall_seconds=300,max_calls=30,
        max_allocated_bytes=12*1024**3,min_remaining_GPU_bytes=1024**3,max_output_bytes=16*1024**2,
        lengths=[256,512,1024,2048,4096,8192,12288,16384,24576,32768],
        purpose='measure practical same-input codec training length; three memories; Commit only; denoise0/copy0',
        optimizer_steps=0,checkpoint_writes=0)
    output.mkdir()
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    review=dict(plan_sha=sha(output/'plan.json'),syntax_pins_verified=len(files),CPU_review_verified=True,
                parent_preserved=True,CUDA_run_started=False)
    (output/'root_prelaunch_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    snapshot=output/'code_snapshot';snapshot.mkdir()
    for name in ['profile_long_memory_v49_r2.py','supervise_long_profile_v49_r2.py','prepare_long_profile_v49_r2.py']:
        (snapshot/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    (snapshot/'long_memory_codec_v49.py').write_bytes((ROOT/'research/long_memory_codec_v49.py').read_bytes())
    print(json.dumps(review))

if __name__=='__main__':main()
