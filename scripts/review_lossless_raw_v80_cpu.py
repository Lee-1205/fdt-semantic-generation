"""ALL V79 raw packets in-memory bitwise roundtrip; never rewrite old raw."""
import io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io_tools
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import pack,unpack


def independent_same(a,b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
        return 1
    assert type(a) is type(b)
    if isinstance(a,dict):
        assert set(a)==set(b);return sum(independent_same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);return sum(independent_same(x,y) for x,y in zip(a,b))
    assert a==b;return 0


def main():
    torch.set_num_threads(4);began=time.monotonic();root=ROOT/'runs/last_linear_20261113_v79';out=ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1'
    records=[];total_original=total_packed=tensors=0
    for f in sorted(root.glob('raw*.pt')):
        if time.monotonic()-began>100:raise TimeoutError('fixed CPU storage review100sec')
        original_sha=support.sha(f);original=torch.load(f,map_location='cpu',weights_only=True);buf=io.BytesIO();torch.save(pack(original),buf);size=buf.tell();buf.seek(0)
        restored=unpack(torch.load(buf,map_location='cpu',weights_only=True));tensors+=independent_same(original,restored)
        assert support.sha(f)==original_sha
        total_original+=f.stat().st_size;total_packed+=size
        records.append(dict(path=str(f),SHA=original_sha,original_bytes=f.stat().st_size,packed_in_memory_bytes=size))
    assert len(records)==471 and not torch.cuda.is_initialized()
    proof=dict(status='ROOT_ALL471_V79_RAW_LOSSLESS_CARRIER_CPU_PASS',files=471,tensors_bitwise_checked=tensors,original_raw_bytes=total_original,
        packed_in_memory_bytes=total_packed,potential_new_output_saving_bytes=total_original-total_packed,old_files_written_or_deleted=0,
        full_FP32_long_and_parity_preserved=True,all_original_dtypes_and_bits_restored=True,model_forwards=0,FDT_optimizer_updates=0,new_GPU=False,
        CUDA_initialized=False,actual_free_space_reclaimed_bytes=0,seconds=time.monotonic()-began,
        limitations='Storage-only future new files. Future loader/reviewer must unpack before NLL and FSM. Full worker/budget including temporary and STOP not implemented or approved by this calculation.',
        source_SHA={str(ROOT/n):support.sha(ROOT/n) for n in ('research/lossless_raw_tensor_v80.py','tests/test_lossless_raw_tensor_v80.py','scripts/review_lossless_raw_v80_cpu.py')},records=records)
    io_tools.atomic_json(out/'root_lossless_raw_review_v3.json',proof);print(json.dumps({k:v for k,v in proof.items() if k!='records'}))


if __name__=='__main__':main()
