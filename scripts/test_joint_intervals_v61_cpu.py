"""Exhaustive distribution/gradient oracle and linear-length numerical check."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research.semantic_binding_v61 import joint_intervals


def main():
    torch.set_num_threads(2);torch.manual_seed(611)
    s=torch.randn(2,3,7,requires_grad=True);e=torch.randn(2,3,7,requires_grad=True);u=torch.randn(2,3,6,requires_grad=True)
    spans,degree,presence,z,_,_=joint_intervals(s,e,u)
    zz=[];dd=[];pp=[]
    for b in range(2):
        zr=[];dr=[];pr=[]
        for r in range(3):
            pairs=[None]+[(a,c) for a in range(6) for c in range(a,6)]
            values=torch.stack([s[b,r,-1]+e[b,r,-1]]+[s[b,r,a]+e[b,r,c]+u[b,r,a:c+1].sum() for a,c in pairs[1:]])
            probs=values.softmax(0);zr.append(values.logsumexp(0));pr.append(1-probs[0])
            dr.append(torch.stack([sum(probs[i] for i,pair in enumerate(pairs) if pair and pair[0]<=t<=pair[1]) for t in range(6)]))
            a,c=spans[b,r].tolist();score=s[b,r,-1]+e[b,r,-1] if a<0 else s[b,r,a]+e[b,r,c-1]+u[b,r,a:c].sum()
            assert torch.allclose(score,values.max(),atol=2e-6)
        zz.append(torch.stack(zr));dd.append(torch.stack(dr));pp.append(torch.stack(pr))
    zz,dd,pp=torch.stack(zz),torch.stack(dd),torch.stack(pp)
    assert torch.allclose(z,zz,atol=2e-6) and torch.allclose(degree,dd,atol=2e-6) and torch.allclose(presence,pp,atol=2e-6)
    actual=torch.autograd.grad(z.sum()+degree.square().sum(),(s,e,u),retain_graph=True)
    oracle=torch.autograd.grad(zz.sum()+dd.square().sum(),(s,e,u))
    assert all(torch.allclose(a,b,atol=3e-6,rtol=3e-5) for a,b in zip(actual,oracle))
    n=16384;a=torch.zeros(1,1,n+1);inside=torch.zeros(1,1,n)
    _,long_degree,long_presence,long_z,_,_=joint_intervals(a,a,inside)
    total=n*(n+1)/2+1;index=torch.arange(n).float()
    expected=(index+1)*(n-index)/total
    assert torch.allclose(long_degree[0,0],expected,atol=2e-6)
    assert abs(float(long_z)-__import__('math').log(total))<2e-6 and torch.isfinite(long_presence).all()
    report=dict(exhaustive_joint_distribution_and_occupancy=True,exhaustive_gradient_oracle=True,
        long16384_linear_partition_arithmetic=True,long_check_is_not_model_ability=True,CUDA_initialized=torch.cuda.is_initialized())
    assert not report['CUDA_initialized']
    (ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu/root_joint_math_contract.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
