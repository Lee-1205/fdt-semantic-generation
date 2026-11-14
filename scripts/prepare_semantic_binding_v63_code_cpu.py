"""Create new objective-specific engine/owner/fixture; old files immutable."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def emit(name,text):
    path=ROOT/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text);path.write_text(text,encoding='utf8')


def main():
    engine=(ROOT/'scripts/semantic_binding_engine_v62.py').read_text(encoding='utf8').replace('objective_v62','objective_v63').replace('binding_v62_fullstate','binding_v63_fullstate')
    emit('scripts/semantic_binding_engine_v63.py',engine)
    owner=(ROOT/'scripts/supervise_semantic_binding_v62.py').read_text(encoding='utf8').replace('v62','v63').replace('2200','2500').replace('1*1024**3','5*1024**3')
    emit('scripts/supervise_semantic_binding_v63.py',owner)
    test=(ROOT/'scripts/test_semantic_binding_v62_cpu.py').read_text(encoding='utf8').replace('engine_v62','engine_v63').replace('objective_v62','objective_v63').replace('20261112_v62_cpu','20261112_v63_cpu')
    # The contrast function remains implemented in the V62 base loss.
    test=test.replace('from research.semantic_binding_objective_v63 import memory_contrast','from research.semantic_binding_objective_v62 import memory_contrast')
    test=test.replace('    report=dict(',"    from research.semantic_binding_objective_v63 import role_overlap_loss\n    overlapping=torch.tensor([[[.9,.8,0.,0.],[.7,.8,0.,0.],[0.,0.,0.,0.]]],requires_grad=True)\n    disjoint=torch.tensor([[[0,2],[2,4],[-1,-1]]])\n    penalty=role_overlap_loss(overlapping,disjoint)\n    assert penalty>0\n    penalty.backward();assert overlapping.grad.abs().sum()>0\n    separated=torch.tensor([[[1.,1.,0.,0.],[0.,0.,1.,1.],[0.,0.,0.,0.]]])\n    assert role_overlap_loss(separated,disjoint)==0\n    shared=torch.tensor([[[0,2],[0,2],[-1,-1]]])\n    assert role_overlap_loss(overlapping,shared)==0\n    report=dict(overlap_loss_gradient=True,nested_gold_not_penalized=True,no_inference_hard_constraint=True,")
    emit('scripts/test_semantic_binding_v63_cpu.py',test)

if __name__=='__main__':main()
