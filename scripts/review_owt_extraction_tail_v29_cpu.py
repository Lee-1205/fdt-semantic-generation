"""One actual-source CPU trainability inventory; no model forward or optimizer."""
import json,sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE,TRAINABILITY,new_trainable_names
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(2);run=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v28'
    out=ROOT/'research/artifacts/owt_extraction_tail_20261029_v29_cpu';out.mkdir(exist_ok=False)
    source=run/'model/checkpoints/state_8192.pt';before=support.sha(source)
    assert before=='bc0fe9d59cf39d59290b64377d40e53e6f214c083eaaa3206b2bcf84961c4246'
    s=torch.load(source,weights_only=True,map_location='cpu');rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model=ExtractionTailFuzzyAE(Config(**s['core']));model.load_state_dict(s['model'],strict=True)
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert support.state_sha(model.state_dict())==support.state_sha(s['model'])
    groups=copy.deepcopy(s['optimizer_parameter_names']);old=set(n for g in groups for n in g)
    new=new_trainable_names(model,groups);named=dict(model.named_parameters())
    assert len(old)==63 and len(new)==18
    assert all(n.startswith(('core.language.blocks.5.','core.language.blocks.6.')) for n in new)
    assert not model.core.language.embedding.weight.requires_grad
    assert all(not p.requires_grad for b in model.core.language.blocks[:5] for p in b.parameters())
    assert all(p.requires_grad for b in model.core.language.blocks[5:] for p in b.parameters())
    previous=sum(named[n].numel() for n in old);extra=sum(named[n].numel() for n in new)
    result={'status':'ACTUAL_SOURCE_CPU_TRAINABILITY_CANDIDATE_ONLY','source_sha256':before,
        'source_model_sha256':support.state_sha(s['model']),'strict_model_tensor_copy_all_identical':True,
        'total_parameters':sum(p.numel() for p in model.parameters()),'old_trainable_parameters':previous,
        'additional_existing_parameters_thawed':extra,'candidate_trainable_parameters':previous+extra,
        'old_optimizer_tensor_names':len(old),'new_existing_tensor_names':new,'new_parameter_count':0,
        'source_Adam_step_values':sorted({int(v['step']) for v in s['optimizer']['state'].values()}),
        'additional_FP32_Adam_moment_bytes_after_first_update_estimate':2*4*extra,
        'source_file_bytes':source.stat().st_size,'payload_bytes_before_new_history_estimate':source.stat().st_size+2*4*extra,
        'CPU_RNG_scope_restored':True,'CUDA_initialized':torch.cuda.is_initialized(),'actual_model_forwards':0,
        'optimizer_instantiated_or_updated':False,'full_driver_scaler_RNG_ownership_migration_verified':False,
        'GPU_training_performed':False,'training_policy':TRAINABILITY,
        'graph_cost_note':'Frozen embedding and first5blocks avoid input-gradient backprop through all9blocks; new backward path covers last4 rather than last2. Same3logicalNNcalls does not mean same compute cost.',
        'next':'Implement explicit preserved63Adam ownership plus fresh appended18-state group, new frozen digest/source digest distinction, full enabledscaler/allRNG exact restore and actual plan before any bounded GPU run. No whole extractor/decoder cause claim.',
        'E2_final_accessed':False}
    assert before==support.sha(source)
    with (out/'root_cpu_inventory.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,ensure_ascii=True))
if __name__=='__main__':main()
