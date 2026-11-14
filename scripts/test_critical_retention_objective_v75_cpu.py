"""Numerical and gradient contracts for a new loss, not a model trial."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from research.critical_retention_objective_v75 import bernoulli_anchor,decoder_losses,COEFFICIENTS
from research.fuzzy_cycle_objective_v67 import cycle_losses as base_losses

class LossContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(75);self.y=torch.full((2,16),257,dtype=torch.long);self.y.flatten()[:8]=torch.tensor([ord('2'),ord('+'),ord('9'),ord(' '),ord('A'),ord('z'),ord('.'),256])
        self.z=torch.randn(2,16,258);self.z.scatter_add_(-1,self.y[...,None],torch.full((2,16,1),4.))
        self.mask=torch.zeros_like(self.y,dtype=torch.bool);self.mask.flatten()[:3]=True
        self.correct=self.z.argmax(-1)==self.y;self.correct.flatten()[2]=False
        self.p0=self.z.softmax(-1).gather(-1,self.y[...,None]).squeeze(-1);self.weights=torch.ones(258);self.weights[257]=0
    def test_full_loss_decomposition_and_padding_gradient(self):
        z=self.z.clone().requires_grad_();loss,p=decoder_losses(z,self.y,self.weights,self.mask,self.p0,self.correct);base,_=base_losses(z,self.y,self.weights)
        self.assertTrue(torch.allclose(loss,base+sum(COEFFICIENTS[k]*p[k] for k in COEFFICIENTS)))
        self.assertEqual(p['correct_critical_positions'],2);self.assertEqual(p['wrong_critical_positions'],1)
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue((z.grad[self.y==257]==0).all())
    def test_same_teacher_distribution_anchor_zero(self):
        z=self.z.clone().requires_grad_();anchor=bernoulli_anchor(z,self.y,self.p0,self.mask&self.correct)
        self.assertLess(abs(float(anchor)),1e-6);anchor.backward();self.assertLess(float(z.grad.abs().max()),1e-6)
    def test_correct_critical_drift_restores_gold_probability(self):
        z=self.z.clone();z.flatten(0,1)[0,ord('2')]-=2;z.requires_grad_()
        loss=bernoulli_anchor(z,self.y,self.p0,self.mask&self.correct);loss.backward()
        self.assertGreater(float(loss),0);self.assertLess(float(z.grad.flatten(0,1)[0,ord('2')]),0)
    def test_wrong_teacher_is_not_anchored(self):
        z=self.z.clone().requires_grad_();_,p=decoder_losses(z,self.y,self.weights,self.mask,self.p0,self.correct)
        p['correct_critical_KL'].backward();self.assertTrue((z.grad.flatten(0,1)[2]==0).all())
        z=self.z.clone().requires_grad_();_,p=decoder_losses(z,self.y,self.weights,self.mask,self.p0,self.correct);p['wrong_critical_CE'].backward()
        self.assertLess(float(z.grad.flatten(0,1)[2,ord('9')]),0)
    def test_more_noncritical_bytes_do_not_dilute_critical_anchor(self):
        z=self.z.clone();z.flatten(0,1)[0,ord('2')]-=2
        a=bernoulli_anchor(z,self.y,self.p0,self.mask&self.correct)
        yz=torch.cat([self.y,self.y],0);zz=torch.cat([z,self.z],0);pp=torch.cat([self.p0,self.p0],0);mask=torch.cat([self.mask&self.correct,torch.zeros_like(self.mask)],0)
        self.assertTrue(torch.allclose(a,bernoulli_anchor(zz,yz,pp,mask)))
    def test_extreme_p0_empty_anchor_and_bad_mask(self):
        p0=self.p0.clone();p0.flatten()[:2]=torch.tensor([0.,1.]);z=self.z.clone().requires_grad_()
        a=bernoulli_anchor(z,self.y,p0,self.mask);a.backward();self.assertTrue(torch.isfinite(a));self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(bernoulli_anchor(z,self.y,p0,torch.zeros_like(self.mask))),0)
        bad=self.mask.clone();bad.flatten()[7]=True
        with self.assertRaises(ValueError):decoder_losses(z,self.y,self.weights,bad,p0,self.correct)

if __name__=='__main__':
    torch.set_num_threads(2);suite=unittest.defaultTestLoader.loadTestsFromTestCase(LossContracts);r=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized()
    if not r.wasSuccessful():sys.exit(1)
