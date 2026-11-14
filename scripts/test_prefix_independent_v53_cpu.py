"""No answer-prefix leakage; exact restored next update; full corpus provenance."""
import copy,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import numpy_state
from research.prefix_independent_codec_v53 import PrefixIndependentCodec
from research.text_diffusion_v1 import pack_answers
from prefix_independent_engine_v53 import PrefixEngine
import text_generation_support_v1 as support
from test_long_codec_v50_cpu import Budget

def main():
    torch.set_num_threads(2);torch.manual_seed(53);random.seed(53);np.random.seed(53)
    c=Config.smoke()
    def model():
        m=PrefixIndependentCodec(c,input_capacity=256);m.enable_position_adapter();m.enable_long_capacity();m.enable_local_banks()
        return m
    old=model();named=dict(old.named_parameters());names=[n for n,p in named.items() if p.requires_grad]
    groups=[names[i::6] for i in range(6)]
    opt=torch.optim.AdamW([dict(params=[named[n] for n in g],lr=1e-5) for g in groups])
    for n in names:named[n].grad=torch.full_like(named[n],.001)
    opt.step();opt.zero_grad(set_to_none=True)
    source=dict(model=copy.deepcopy(old.state_dict()),optimizer=opt.state_dict(),optimizer_parameter_names=groups,
        scaler=torch.amp.GradScaler('cpu',init_scale=1024.).state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=[],
        python_rng=random.getstate(),numpy_rng=numpy_state(),plan_sha256='parent',attempts=1,updates=1,cursor=1,
        history=[],actual_calls={},evaluations={},ancestry={})
    rows=[dict(id=str(i),prompt=('A 2026 한글. '*20)[:100]) for i in range(9)];orders=[[0],[1],list(range(2,9))]
    def engine():
        m=model();m.load_state_dict(source['model'],strict=True)
        e=PrefixEngine(m,groups,rows,orders,Budget(),'cpu','cpu');e.import_parent(source);return e
    e=engine();a=pack_answers(['abcd'],5,'cpu');b=pack_answers(['WXYZ'],5,'cpu')
    pa,_,_=e.model.corrupt_teacher_prefix(a);pb,_,_=e.model.corrupt_teacher_prefix(b)
    assert torch.equal(pa,pb) and pa.tolist()==[[258,257,257,257,257]]
    e.step();saved=copy.deepcopy(e.export());e.step(force_overflow=True);e.step();final=e.export()
    other=engine();other.restore(saved);other.step(force_overflow=True);other.step()
    assert support.state_sha(other.export())==support.state_sha(final)
    assert (e.attempts,e.updates,e.skips,e.cursor)==(3,2,1,9)
    artifact=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu'
    documents=json.loads((ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final/documents.json').read_text())
    data=json.loads((artifact/'train.json').read_text());by_id={d['id']:d for d in documents['train']}
    for row in data:assert row['prompt'].encode()==by_id[row['document_id']]['text'].encode()[:row['source_byte_end']]
    assert len({d['normalized_sha256'] for d in documents['train']})==8192
    report=dict(prefix_identical_under_different_answers=True,no_past_answer_bytes=True,
        enabled_scaler_full_restore_next_update_exact=True,AMP_skip_consumes_cursor=True,last7=True,
        independently_reconstructed_train_documents=8192,CUDA_initialized=torch.cuda.is_initialized())
    (artifact/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
