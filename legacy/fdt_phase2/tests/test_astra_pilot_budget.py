import sys
from pathlib import Path
import copy
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_supervised_limit_excludes_prompt_pad_and_shift():
    from astra_bounded_pilot import restrict_row
    row={'input_ids':torch.tensor([[1,4,5,6,2,0]]),
         'labels':torch.tensor([[-100,-100,5,6,2,0]]),
         'attention_mask':torch.tensor([[1,1,1,1,1,0]])}
    result,stats=restrict_row(row,2,20)
    assert stats=={'active':5,'supervised':2}
    assert result['labels'].tolist()==[[-100,-100,5,6,-100,-100]]
    assert row['labels'][0,4]==2
    assert restrict_row(row,10,4)[0] is None
    assert restrict_row(row,0,20)[0] is None


def test_optimizer_restore_is_equivalent_at_committed_boundary(tmp_path):
    from astra_bounded_pilot import write_recovery,read_recovery
    torch.manual_seed(47)
    model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    x=torch.randn(4,3)
    def step(m,o):
        o.zero_grad(); loss=m(x).square().mean();loss.backward();o.step()
    step(model,opt)
    path=tmp_path/'recovery.pt'
    write_recovery(path,model,opt,{'step':1,'supervised':7,'active':12},'same-identity')
    restored=torch.nn.Linear(3,2);ropt=torch.optim.AdamW(restored.parameters(),lr=1e-3)
    state=read_recovery(path,restored,ropt,'same-identity')
    assert state['supervised']==7
    step(model,opt);step(restored,ropt)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for i,s in opt.state_dict()['state'].items():
        for k,v in s.items():torch.testing.assert_close(v,ropt.state_dict()['state'][i][k],atol=0,rtol=0)
    with pytest.raises(RuntimeError,match='identity'):
        read_recovery(path,restored,ropt,'different-code')


def test_hard_budget_survives_recovery_and_includes_elapsed():
    from astra_bounded_pilot import budget_stop
    state={'supervised':249999,'active':1000000,'elapsed_seconds':0}
    assert budget_stop(state,0) is None
    assert budget_stop({**state,'supervised':250000},0)=='SUPERVISED_BUDGET'
    assert budget_stop({**state,'active':1200000},0)=='ACTIVE_BUDGET'
    assert budget_stop(state,14400)=='TIME_BUDGET'
