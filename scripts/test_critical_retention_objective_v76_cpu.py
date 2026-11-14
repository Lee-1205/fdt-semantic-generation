"""Focused gradient contracts, not full worker/CUDA recovery or model ability."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from research.critical_retention_objective_v76 import bounded_wrong_mean,decoder_losses,COEFFICIENTS
from research.critical_retention_objective_v75 import decoder_losses as previous_losses
from research.fuzzy_cycle_objective_v67 import cycle_losses as base_losses

class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(76);self.y=torch.full((10,16),257,dtype=torch.long)
        self.y.flatten()[:129]=ord('3');self.y.flatten()[129]=256
        self.z=torch.randn(10,16,258);self.z.scatter_add_(-1,self.y[...,None],torch.full((10,16,1),3.))
        self.mask=torch.zeros_like(self.y,dtype=torch.bool);self.mask.flatten()[:128]=True
        self.correct=torch.ones_like(self.mask);self.correct.flatten()[0]=False
        self.p=self.z.softmax(-1).gather(-1,self.y[...,None]).squeeze(-1)
        self.w=torch.ones(258);self.w[257]=0.
    def test_sparse_gradient_has_eightfold_position_bound(self):
        ce=torch.linspace(0.,127.,128,requires_grad=True);m=torch.ones(128,dtype=torch.bool);wrong=torch.zeros_like(m);wrong[5]=True
        value,s=bounded_wrong_mean(ce,m,wrong);value.backward()
        self.assertEqual(s['wrong_critical_denominator'],16.);self.assertEqual(s['wrong_critical_position_amplification'],8.)
        self.assertAlmostEqual(float(ce.grad[5]),1/16);self.assertEqual(float(ce.grad[~wrong].abs().sum()),0.)
    def test_dense_wrong_mask_is_exact_original_mean(self):
        ce=torch.arange(32.).requires_grad_();mask=torch.ones(32,dtype=torch.bool);wrong=torch.zeros_like(mask);wrong[:8]=True
        value,s=bounded_wrong_mean(ce,mask,wrong);self.assertTrue(torch.equal(value,ce[wrong].mean()))
        self.assertEqual(s['wrong_critical_position_amplification'],4.)
    def test_empty_set_is_finite_zero_and_zero_gradient(self):
        ce=torch.rand(20,requires_grad=True);mask=torch.ones(20,dtype=torch.bool);wrong=torch.zeros_like(mask)
        a,s=bounded_wrong_mean(ce,mask,wrong);a.backward();self.assertEqual(float(a),0.);self.assertTrue((ce.grad==0).all())
        self.assertEqual(s['wrong_critical_position_amplification'],0.)
    def test_full_loss_only_wrong_term_changes(self):
        z=self.z.clone().requires_grad_();p0=self.p.clone().requires_grad_()
        loss,parts=decoder_losses(z,self.y,self.w,self.mask,p0,self.correct)
        old,oldparts=previous_losses(z,self.y,self.w,self.mask,p0,self.correct)
        self.assertTrue(torch.allclose(parts['wrong_critical_CE'],oldparts['wrong_critical_CE']/16))
        self.assertTrue(torch.allclose(loss,old-.25*(oldparts['wrong_critical_CE']-parts['wrong_critical_CE'])))
        for name in ('correct_critical_CE','correct_noncritical_KL','correct_critical_KL'):
            self.assertTrue(torch.equal(parts[name],oldparts[name]))
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue((z.grad[self.y==257]==0).all());self.assertIsNone(p0.grad)
    def test_wrong_teacher_not_anchored_and_gold_corrected(self):
        z=self.z.clone().requires_grad_();_,parts=decoder_losses(z,self.y,self.w,self.mask,self.p,self.correct)
        parts['correct_critical_KL'].backward();self.assertTrue((z.grad.flatten(0,1)[0]==0).all())
        z=self.z.clone().requires_grad_();_,parts=decoder_losses(z,self.y,self.w,self.mask,self.p,self.correct)
        parts['wrong_critical_CE'].backward();self.assertLess(float(z.grad.flatten(0,1)[0,ord('3')]),0.)
    def test_subset_and_EOS_as_critical_rejected(self):
        with self.assertRaises(ValueError):bounded_wrong_mean(torch.ones(2),torch.tensor([True,False]),torch.tensor([False,True]))
        bad=self.mask.clone();bad.flatten()[129]=True
        with self.assertRaises(ValueError):decoder_losses(self.z,self.y,self.w,bad,self.p,self.correct)
    def test_cap_holds_for_all_C_W_and_padding_ignores(self):
        for c in (1,7,8,9,16,128,445):
            for w in sorted({1,c,max(1,c//8)}):
                ce=torch.ones(c);m=torch.ones(c,dtype=torch.bool);wrong=torch.arange(c)<w
                value,s=bounded_wrong_mean(ce,m,wrong)
                self.assertLessEqual(s['wrong_critical_position_amplification'],8.+1e-12)
                self.assertAlmostEqual(float(value),w/max(w,c/8),places=6)

if __name__=='__main__':
    torch.set_num_threads(2);suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized()
    if not result.wasSuccessful():sys.exit(1)
