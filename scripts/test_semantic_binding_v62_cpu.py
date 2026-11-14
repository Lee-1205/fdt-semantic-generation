"""Actual fuzzy-role binding and source/Commit boundaries, not ability scores."""
import copy,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from research.semantic_binding_v59 import MemoryTokens,best_spans
from research.semantic_binding_v61 import SemanticBinderV61
from research.semantic_binding_v60 import BoundMeaningState
from dataclasses import replace
from research.commit_patch_readout_v54 import CommitPatchReadout
from semantic_binding_engine_v62 import SemanticEngine,select_packet
import text_generation_support_v1 as support

class Budget:
    def __init__(self):self.calls={'encode':0,'bind':0,'denoise':0};self.parent_reference={'path':'fixture','sha256':'fixture'}
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

def main():
    torch.set_num_threads(2);torch.manual_seed(59);random.seed(59);np.random.seed(59)
    count,n=9,16;valid=torch.ones(count,n,dtype=torch.bool);valid[-1,7:]=False
    f=dict(mu=torch.rand(count,n,64),contextual_values=torch.randn(count,n,256),exact_keys=torch.randn(count,n,128),exact_values=torch.randn(count,n,256),
        valid=valid,starts=valid.clone(),ends=valid.clone(),source_revisions=[(str(i),'v','hash') for i in range(count)],source_epoch='fixture')
    f['starts'][:,3]=False;f['ends'][:,2]=False  # a two-byte UTF8 codepoint
    targets=dict(spans=torch.tensor([[[0,2],[-1,-1],[4,6]]]*count),relation=torch.tensor([0,1,2]*3),
        polarity=torch.tensor([1,0,1]*3),quantity=torch.tensor([0,0,12]*3),meaning_ids=torch.tensor([0,0,1,1,2,2,3,3,3]))
    prior=torch.randn(64,256);orders=[[0],[1],list(range(2,9))]
    def create():return SemanticEngine(SemanticBinderV61(prior,'fixture',hidden=128),f,targets,orders,Budget(),'test','cpu')
    engine=create();packet=select_packet(f,torch.tensor([0,1]),'cpu');commit,raw=engine.model(packet,'test')
    # Independently accumulate all role N/D on a small real network output.
    num=torch.zeros_like(commit.role_N);den=torch.zeros_like(commit.role_D)
    for t in range(n):
        w=packet.mu[:,t].double()[:,None]*raw['role_degree'][:,:,t].double()[...,None]
        num+=w[...,None]*raw['projected_values'][:,t].double()[:,None,None];den+=w[...,None]
    assert torch.allclose(num,commit.role_N,atol=1e-12) and torch.allclose(den,commit.role_D,atol=1e-12)
    starts,ends=raw['start_logits'].detach(),raw['end_logits'].detach();pred=best_spans(starts,ends)
    for b in range(2):
        for r in range(3):
            candidates=[(float(starts[b,r,-1]+ends[b,r,-1]),(-1,-1))]
            candidates += [(float(starts[b,r,i]+ends[b,r,j]),(i,j+1)) for i in range(n) for j in range(i,n)]
            assert tuple(pred[b,r].tolist())==max(candidates,key=lambda item:item[0])[1]
    before=commit.semantic_vector().detach().clone();targets['polarity']=1-targets['polarity']
    assert torch.equal(before,engine.model(packet,'test')[0].semantic_vector())
    targets['polarity']=1-targets['polarity']
    try:CommitPatchReadout().from_commit(commit)
    except TypeError:pass
    else:raise AssertionError('new meaning commit accepted by old surface decoder')
    bad=copy.copy(packet);object.__setattr__(bad,'source_epoch','stale')
    try:engine.model(bad,'test')
    except ValueError:pass
    else:raise AssertionError('stale exact/memory epoch accepted')
    try:engine.model(packet,'test',targets=targets)
    except TypeError:pass
    else:raise AssertionError('inference accepted gold labels')
    state,raw_state=engine.model.extract_state(packet,'test')
    reference,factors=engine.model.read_state(state)
    try:engine.model.read_state(packet)
    except TypeError:pass
    else:raise AssertionError('token packet bypass accepted')
    frozen_factors={k:v.detach().clone() for k,v in factors.items()}
    packet.contextual_values.add_(10)
    assert all(torch.equal(v,engine.model.read_state(state)[1][k]) for k,v in frozen_factors.items())
    empty=replace(state,role_values=torch.zeros_like(state.role_values))
    assert (engine.model.read_state(empty)[1]['relation_logits']-factors['relation_logits']).abs().max()>1e-6
    donor=replace(state,donor_revisions=((3,('donor','v','hash')),))
    try:engine.model.read_state(donor)
    except ValueError:pass
    else:raise AssertionError('diagnostic donor allowed in normal path')
    assert engine.model.read_state(donor,diagnostic=True)[0].role_values.shape[1]==4
    engine.phase='semantic_training';engine.step();saved=copy.deepcopy(engine.export());engine.step(True);engine.step();final=engine.export()
    other=create();other.restore(saved,False);other.step(True);other.step()
    assert support.state_sha(other.export())==support.state_sha(final)
    assert (other.attempts,other.updates,other.skips,other.cursor)==(3,2,1,9)
    assert torch.equal(engine.model.persistent_prior,prior) and engine.scaler.is_enabled() and not torch.cuda.is_initialized()
    from research.semantic_binding_objective_v62 import memory_contrast
    identity=torch.tensor([0,0])
    value=memory_contrast(reference.role_values,identity)
    assert torch.autograd.grad(value,factors['relation_logits'],allow_unused=True,retain_graph=True)[0] is None
    labels=reference.relation_probabilities.detach().clone().flip(-1)
    altered=replace(reference,relation_probabilities=labels)
    assert torch.equal(value,memory_contrast(altered.role_values,identity))
    probe=torch.randn(4,4,128,requires_grad=True)
    memory_contrast(probe,torch.tensor([0,0,1,1])).backward()
    assert probe.grad.abs().sum()>0
    report=dict(contrast_excludes_factor_heads=True,contrast_reaches_actual_memory=True,factors_read_bound_memory_only=True,source_mutation_cannot_change_already_bound_readout=True,event_role_four=True,diagnostic_interchange_isolated=True,role_ND_independent_recurrence=True,hard_spans_equal_bruteforce=True,no_gold_forward_inputs=True,
        labels_do_not_change_inference=True,source_epoch_rejects_stale=True,meaning_commit_not_surface_commit=True,
        enabled_scaler_allRNG_exact_next_update=True,AMPskip_and_last7_rows_exact=True,prior_frozen=True,
        actual_new_module_parameters=sum(p.numel() for p in engine.model.parameters()),CPU_fixture_not_model_ability=True,CUDA_initialized=False)
    out=ROOT/'research/artifacts/semantic_binding_20261112_v62_cpu'
    (out/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
