"""Future reasoning-only ownership contract. No diffusion model is supplied.

Language extraction, fuzzy binding, memory parameters, legacy modules, language
readout, and all codec buffers are one frozen endpoint. A *separate* reasoner
owns its exact optimizer allowlist. A frozen realizer can remain differentiable
with respect to its input; no_grad around that path would break this contract.
"""
import hashlib
import torch


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _storages(module):
    tensors = list(module.parameters()) + list(module.buffers())
    return {(str(t.device), t.untyped_storage().data_ptr()) for t in tensors if t.numel()}


class FrozenLanguageBoundary:
    """Contract checker; not an execution owner, trainer, or phase transition."""
    def __init__(self, codec, reasoner, allowed_reasoner_names):
        if codec is reasoner or _storages(codec) & _storages(reasoner):
            raise ValueError('reasoner must not share endpoint parameter/buffer storage')
        names = tuple(allowed_reasoner_names)
        named = dict(reasoner.named_parameters())
        if not names or len(set(names)) != len(names) or set(names) != set(named):
            raise ValueError('explicit exact reasoning parameter ownership required')
        self.codec, self.reasoner, self.names = codec, reasoner, names
        codec.requires_grad_(False)
        codec.eval()
        for parameter in codec.parameters(): parameter.grad = None
        reasoner.requires_grad_(True)
        for parameter in reasoner.parameters(): parameter.grad = None
        self.endpoint_digest = state_digest(codec)
        self.endpoint_ids = {id(p) for p in codec.parameters()}
        self.reasoner_ids = {id(named[n]) for n in names}
        self.assert_intact()

    def assert_optimizer(self, optimizer):
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        if len(ids) != len(set(ids)) or set(ids) != self.reasoner_ids or set(ids) & self.endpoint_ids:
            raise ValueError('optimizer must own reasoning weights only, exactly once')
        if set(dict(self.reasoner.named_parameters())) != set(self.names):
            raise ValueError('reasoner structure changed after scope was fixed')
        if {id(p) for p in self.reasoner.parameters()} != self.reasoner_ids:
            raise ValueError('reasoner parameter identity changed after optimizer scope was fixed')
        if any(not p.requires_grad for p in self.reasoner.parameters()):
            raise ValueError('reasoner ownership became inconsistent')
        if _storages(self.codec) & _storages(self.reasoner):
            raise ValueError('endpoint/reasoner storage alias appeared')
        return True

    def assert_intact(self):
        if {id(p) for p in self.codec.parameters()} != self.endpoint_ids:
            raise ValueError('frozen language endpoint parameter identity changed')
        if any(p.requires_grad or p.grad is not None for p in self.codec.parameters()):
            raise ValueError('language or memory endpoint acquired trainability/stale gradient')
        if any(module.training for module in self.codec.modules()):
            raise ValueError('frozen language endpoint must remain in eval mode')
        if state_digest(self.codec) != self.endpoint_digest:
            raise ValueError('frozen endpoint parameter or buffer changed')
        return True
