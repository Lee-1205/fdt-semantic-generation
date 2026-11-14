"""Thought-only optimizer enforcement with cheap per-step and full saved audits.

Use the existing full endpoint digest at initialization/checkpoint/handoff.
Per-step tensor version checks avoid transferring/hash-reading the frozen123M
endpoint on every update. They supplement, never replace, the full audit:
untracked .data writes can evade version counters and fail the final digest.
"""
import torch


class ThoughtOnlyUpdate:
    def __init__(self,boundary,optimizer):
        self.boundary=boundary;self.optimizer=optimizer
        boundary.assert_optimizer(optimizer);boundary.assert_intact()
        self.frozen_versions=self._versions()

    def _versions(self):
        codec=self.boundary.codec
        entries=[('parameter',n,id(v),v._version) for n,v in codec.named_parameters()]
        entries += [('buffer',n,id(v),v._version) for n,v in codec.named_buffers()]
        return tuple(entries)

    def assert_fast(self):
        b=self.boundary;b.assert_optimizer(self.optimizer)
        if (self._versions()!=self.frozen_versions or any(m.training for m in b.codec.modules())
                or any(p.requires_grad or p.grad is not None for p in b.codec.parameters())):
            raise ValueError('frozen endpoint mode/identity/version/gradient changed')

    def step(self,max_grad_norm=1.):
        """CPU/unscaled transaction interface; future AMP driver unscales first.

        This is NOT a worker/scaler/RNG recovery implementation or launch plan.
        """
        self.assert_fast()
        parameters=list(self.boundary.reasoner.parameters())
        gradients=[p.grad for p in parameters if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('finite thought gradients required before any step')
        norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm)
        self.optimizer.step();self.assert_fast()
        return float(norm)

    def audit_endpoint(self):
        self.assert_fast();self.boundary.assert_intact()
        return self.boundary.endpoint_digest
