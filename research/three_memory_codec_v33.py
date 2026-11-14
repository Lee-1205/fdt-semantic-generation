"""Neural three-memory *language codec* candidate; no semantic denoising.

Warmstarts the preserved V31 language/fuzzy/decoder tensors, but changes the
extraction function. This is not an inference-equivalent resume or whole vNext.
The input-complete contextual snapshot is not a per-layer prefix memory writer.
Exact contains an input-only learned index; literal selection/copy intent is
still untrained. No raw bytes, token H, live store, or copy reach the realizer.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE
from research.text_context_decoder_v2 import BOS
from research.text_diffusion_v1 import PAD
from research.three_memory_contract_v32 import (
    PHASE, ExactSource, PersistentFuzzySnapshot, assert_codec_training_only,
)


@dataclass(frozen=True)
class ContextSnapshot:
    long_N: torch.Tensor
    long_D: torch.Tensor
    recent_N: torch.Tensor
    recent_D: torch.Tensor
    decay: torch.Tensor
    source_revisions: tuple
    model_epoch: str

    def means(self):
        def mean(n, d):
            return torch.where(d > 1e-8, n / d.clamp_min(1e-8), torch.zeros_like(n)).float()
        return mean(self.long_N, self.long_D), mean(self.recent_N, self.recent_D)


def contextual_snapshot(mu, values, valid, decay, revisions, epoch):
    """Final observed-input N/D, equivalent to valid-token sequential updates.

    Each input byte has its own contextual H/value. Invalid padding neither
    contributes nor ages recent memory. No documents are merged across B.
    """
    if mu.ndim != 3 or values.shape[:2] != mu.shape[:2] or valid.shape != mu.shape[:2]:
        raise ValueError('context dimensions')
    if valid.dtype != torch.bool or decay.shape != (mu.shape[-1],):
        raise ValueError('context mask/decay dimensions')
    if not ((decay > 0) & (decay < 1)).all() or len(revisions) != len(mu):
        raise ValueError('context decay/source revisions')
    weight = mu.double() * valid[..., None].double()
    later_valid = valid.sum(1, keepdim=True) - valid.long().cumsum(1)
    recency = decay.double()[None, None, :] ** later_valid[..., None]
    recent = weight * recency
    return ContextSnapshot(
        torch.einsum('bnk,bnv->bkv', weight, values.double()), weight.sum(1)[..., None],
        torch.einsum('bnk,bnv->bkv', recent, values.double()), recent.sum(1)[..., None],
        decay.clone(), tuple(revisions), epoch,
    )


@dataclass(frozen=True)
class ExactIndexSnapshot:
    sources: tuple
    spans: tuple
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    model_epoch: str

    def assert_epoch(self, epoch):
        if epoch != self.model_epoch:
            raise ValueError('stale Exact embedding index')


@dataclass(frozen=True)
class NeuralMemoryBundle:
    persistent: PersistentFuzzySnapshot
    contextual: ContextSnapshot
    exact: ExactIndexSnapshot
    model_epoch: str

    def __post_init__(self):
        if self.model_epoch != self.persistent.model_epoch or self.model_epoch != self.contextual.model_epoch:
            raise ValueError('memory epoch mismatch')
        self.exact.assert_epoch(self.model_epoch)
        if self.contextual.source_revisions != tuple((s.source_id, s.version, s.sha256) for s in self.exact.sources):
            raise ValueError('memory source revisions mismatch')


@dataclass(frozen=True)
class DifferentiableCodecCommit:
    """Training boundary: own clone keeps encoder gradients, readers clone again.

    A Python type/access boundary, not a security sandbox or immutable storage.
    Only semantic views + immutable revision metadata are retained. No permits
    are issued by this candidate; future explicit copy uses the V32 contract.
    """
    _semantic: torch.Tensor
    memory_revisions: tuple
    model_epoch: str
    phase: str = PHASE

    def __post_init__(self):
        if self.phase != PHASE or self._semantic.ndim != 3 or not self.model_epoch:
            raise ValueError('codec Commit shape/phase/epoch')
        object.__setattr__(self, '_semantic', self._semantic.clone())
        object.__setattr__(self, 'memory_revisions', tuple(self.memory_revisions))

    @classmethod
    def from_semantic(cls, z, revisions, epoch):
        return cls(z, tuple(revisions), epoch)

    def reader_tensor(self):
        return self._semantic.clone()

    def seal(self):
        z = self._semantic.detach().to('cpu', torch.float32).contiguous()
        return SealedCodecCommit(tuple(z.shape), z.numpy().tobytes(), self.memory_revisions, self.model_epoch)


@dataclass(frozen=True)
class SealedCodecCommit:
    shape: tuple
    semantic_bytes: bytes
    memory_revisions: tuple
    model_epoch: str
    phase: str = PHASE

    def __post_init__(self):
        if self.phase != PHASE or len(self.shape) != 3 or not self.model_epoch or len(self.semantic_bytes) != 4*math.prod(self.shape):
            raise ValueError('sealed codec Commit shape/phase/epoch')

    def reader_tensor(self):
        return torch.frombuffer(bytearray(self.semantic_bytes), dtype=torch.float32).clone().reshape(self.shape)


@dataclass(frozen=True)
class EncodedEpisode:
    memories: NeuralMemoryBundle
    commit: DifferentiableCodecCommit
    diagnostics: dict


class MemoryCodecHeads(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.recent_decay_logits = nn.Parameter(torch.full((c.anchors,), math.log(.9 / .1)))
        self.long_gate = nn.Sequential(nn.Linear(c.lang_dim + 2*c.value_dim + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        self.exact_key = nn.Linear(c.lang_dim, c.key_dim, bias=False)
        self.exact_query = nn.Linear(c.z_dim, c.key_dim, bias=False)
        self.exact_null = nn.Parameter(torch.zeros(()))
        self.exact_gate = nn.Linear(c.z_dim, 1)


class ThreeMemoryCodec(ExtractionTailFuzzyAE):
    def __init__(self, c):
        super().__init__(c)
        self.memory_codec = MemoryCodecHeads(c)
        assert_codec_training_only(self.named_parameters())

    def extract(self, tokens):
        raise TypeError('new branch requires encode_input with source revisions/model epoch; not the legacy driver')

    def token_features(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= 128 or tokens.dtype != torch.long:
            raise ValueError('input byte capacity/dtype')
        if ((tokens < 0) | (tokens > 256)).any():
            raise ValueError('input byte range')
        valid = tokens != 256
        if not valid.any(1).all() or ((~valid).long().cumsum(1).bool() & valid).any():
            raise ValueError('nonempty contiguous valid input prefix')
        language = self.core.language
        x = language.embedding(tokens)
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        band = (pos[:, None] >= pos[None]) & (pos[:, None] - pos[None] < self.c.window)
        eye = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool)[None, None]
        mask = band[None, None] & (valid[:, None, None, :] | eye)
        for block in language.blocks:
            if self.c.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, mask, valid, use_reentrant=False)
            else:
                x = block(x, mask, valid)
        weights = (x.float() @ language.pool.float().T).masked_fill(~valid[..., None], -1e4).softmax(1)
        pooled = torch.einsum('blv,bld->bvd', weights, x.float())
        return x.float(), pooled, valid

    def _exact_index(self, tokens, h, valid, source_ids, source_versions, epoch):
        if len(source_ids) != len(h) or len(source_versions) != len(h):
            raise ValueError('one source identity/version per input')
        sources, spans, span_values = [], [], []
        # UTF8 codepoint boundaries are mechanical input metadata. No gold
        # literal span, entity/role annotation, desired output or regex feature.
        for row, (source_id, version) in enumerate(zip(source_ids, source_versions)):
            if not isinstance(source_id, str) or not source_id or not isinstance(version, str) or not version:
                raise ValueError('explicit source identity/version')
            raw = bytes(tokens[row, valid[row]].detach().cpu().tolist())
            source = ExactSource.from_input(source_id, version, raw)
            boundaries, features, start = [], [], 0
            for char in raw.decode('utf-8'):
                end = start + len(char.encode('utf-8'))
                boundaries.append((start, end))
                features.append(h[row, start:end].mean(0))
                start = end
            sources.append(source); spans.append(tuple(boundaries)); span_values.append(torch.stack(features))
        revisions = {}
        for source in sources:
            identity = (source.source_id, source.version)
            if identity in revisions and revisions[identity] != source.sha256:
                raise ValueError('conflicting bytes under same source revision')
            revisions[identity] = source.sha256
        # Pad after extraction; padding is excluded from candidates/NULL softmax.
        features = nn.utils.rnn.pad_sequence(span_values, batch_first=True)
        mask = torch.arange(features.shape[1], device=h.device)[None] < torch.tensor([len(x) for x in spans], device=h.device)[:, None]
        keys = self.memory_codec.exact_key(features)
        roles = features.new_zeros(*features.shape[:2], 16)
        values = self.core.value(torch.cat((features, roles), -1))
        return ExactIndexSnapshot(tuple(sources), tuple(spans), keys, values, mask, epoch)

    def read_exact(self, index, z, epoch):
        index.assert_epoch(epoch)
        q = F.normalize(self.memory_codec.exact_query(z), dim=-1)
        k = F.normalize(index.keys, dim=-1)
        scores = (q @ k.transpose(-1, -2)).masked_fill(~index.valid[:, None], -torch.inf)
        count = min(16, scores.shape[-1])
        ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
        chosen = scores.gather(-1, ids)
        weights = torch.cat((chosen, self.memory_codec.exact_null.expand(*chosen.shape[:-1], 1)), -1).softmax(-1)
        values = index.values[:, None].expand(-1, z.shape[1], -1, -1)
        retrieved = values.gather(2, ids[..., None].expand(-1, -1, -1, values.shape[-1]))
        # Retain NULL probability rather than renormalizing selected source mass.
        result = (weights[..., :-1, None] * retrieved).sum(-2)
        return result, {'candidate_ids': ids, 'candidate_weights': weights[..., :-1], 'null_weight': weights[..., -1]}

    def encode_input(self, tokens, *, source_ids, source_versions, model_epoch, disabled_memories=()):
        if not isinstance(model_epoch, str) or not model_epoch or not set(disabled_memories) <= {'persistent', 'contextual', 'exact'}:
            raise ValueError('explicit model epoch/memory control')
        assert_codec_training_only(self.named_parameters())
        h, pooled, valid = self.token_features(tokens)
        with torch.autocast(tokens.device.type, enabled=False):
            persistent = PersistentFuzzySnapshot.prior_only(self.core.anchors, self.core.prior, model_epoch)
            # Each call builds a fresh Exact index under this embedding epoch.
            # No H/index is cached across updates. Never retained on the module.
            exact = self._exact_index(tokens, h, valid, source_ids, source_versions, model_epoch)
            revisions = tuple((s.source_id, s.version, s.sha256) for s in exact.sources)
            token_mu, _, _ = self.core.phrase_router(h, persistent.anchors)
            token_roles = h.new_zeros(*h.shape[:2], 16)
            values = self.core.value(torch.cat((h, token_roles), -1))
            decay = self.memory_codec.recent_decay_logits.sigmoid()
            contextual = contextual_snapshot(token_mu, values, valid, decay, revisions, model_epoch)
            mu, rho, gamma = self.core.phrase_router(pooled, persistent.anchors)
            kp = persistent.read(mu)
            long_values, recent_values = contextual.means()
            long_read = gamma[..., None] * (rho @ long_values)
            recent_read = gamma[..., None] * (rho @ recent_values)
            masses = torch.cat((rho @ contextual.long_D.float(), rho @ contextual.recent_D.float()), -1).log1p()
            gate = self.memory_codec.long_gate(torch.cat((pooled, long_read, recent_read, masses), -1)).sigmoid()
            kc = gate * long_read + (1-gate) * recent_read
            if 'persistent' in disabled_memories: kp = torch.zeros_like(kp)
            if 'contextual' in disabled_memories: kc = torch.zeros_like(kc)
            roles = pooled.new_zeros(len(tokens), self.c.meaning_views, 16)
            base = self.core.condition(torch.cat((kp + kc, roles), -1))
            ke, retrieval = self.read_exact(exact, base, model_epoch)
            exact_gate = self.memory_codec.exact_gate(base).sigmoid()
            if 'exact' in disabled_memories: ke = torch.zeros_like(ke)
            z = self.core.condition(torch.cat((kp + kc + exact_gate*ke, roles), -1))
            bundle = NeuralMemoryBundle(persistent, contextual, exact, model_epoch)
            commit = DifferentiableCodecCommit.from_semantic(z, revisions, model_epoch)
            return EncodedEpisode(bundle, commit, {
                'memberships': mu, 'token_memberships': token_mu, 'token_values': values,
                'retrieval_weights': rho, 'confidence': gamma, 'long_gate': gate,
                'exact_gate': exact_gate, **retrieval, 'denoise_calls': 0, 'copy_bytes': 0,
            })

    def context(self, commit, mode='normal'):
        if type(commit) not in (DifferentiableCodecCommit, SealedCodecCommit) or commit.phase != PHASE:
            raise TypeError('realizer accepts Codec Commit only; never episode/live memory/H')
        views = commit.reader_tensor().to(self.decoder.embedding.weight.device)
        if mode == 'swap': views = views.roll(max(1, len(views)//2), 0)
        elif mode == 'zero': views = torch.zeros_like(views)
        elif mode != 'normal': raise ValueError('explicit semantic-state control')
        valid = torch.ones(views.shape[:2], device=views.device, dtype=torch.bool)
        return self.decoder.context(views, valid, views.new_empty(len(views), 0, views.shape[-1]))

    def teacher_logits(self, commit, target_ids, mode='normal'):
        prefix = torch.cat((torch.full_like(target_ids[:, :1], BOS), target_ids[:, :-1]), 1)
        return self.decoder(prefix, self.context(commit, mode)).float()

    def reconstruction_loss(self, tokens, target_ids, *, source_ids, source_versions, model_epoch, charge=lambda kind: None):
        if not self.training: raise RuntimeError('targets are training-only')
        if tokens.ndim != 2 or target_ids.shape != (len(tokens), 129) or target_ids.dtype != torch.long:
            raise ValueError('same-input reconstruction target shape')
        expected = torch.full_like(target_ids, PAD)
        expected[:, :tokens.shape[1]] = tokens.masked_fill(tokens == 256, PAD)
        expected.scatter_(1, (tokens != 256).sum(1, keepdim=True), 256)
        if not torch.equal(expected, target_ids):
            raise ValueError('codec target must be the SAME input bytes plus EOS')
        charge('encode')
        episode = self.encode_input(tokens, source_ids=source_ids, source_versions=source_versions, model_epoch=model_epoch)
        charge('decoder'); clean_logits = self.teacher_logits(episode.commit, target_ids)
        prefix, selected, eligible = self.corrupt_teacher_prefix(target_ids)
        charge('decoder'); corrupt_logits = self.decoder(prefix, self.context(episode.commit)).float()
        active = target_ids != PAD; labels = target_ids.masked_fill(~active, -100); count = active.sum()
        clean = F.cross_entropy(clean_logits.transpose(1, 2), labels, ignore_index=-100, reduction='none').sum()/count
        corrupt = F.cross_entropy(corrupt_logits.transpose(1, 2), labels, ignore_index=-100, reduction='none').sum()/count
        self.last_objective_stats = {
            'clean_CE': float(clean.detach()), 'corrupted_prefix_CE': float(corrupt.detach()),
            'masked_prefix_bytes': int(selected.sum()), 'eligible_prefix_bytes': int(eligible.sum()),
            'active_target_symbols_per_pass': int(count), 'decoder_supervised_symbol_occurrences': 2*int(count),
            'denoise_calls': 0, 'copy_bytes': 0,
        }
        return .5*clean + .5*corrupt, episode

    @torch.no_grad()
    def reconstruct(self, commit, charge=lambda kind: None, mode='normal'):
        return self.decoder.generate(self.context(commit, mode), charge)

    def import_legacy_weights(self, state):
        """Strict old tensors, only new memory heads; not optimizer restoration."""
        own = self.state_dict()
        old_keys = {k for k in own if not k.startswith('memory_codec.')}
        if set(state) != old_keys: raise ValueError('legacy model key inventory mismatch')
        if any(state[k].shape != own[k].shape or state[k].dtype != own[k].dtype for k in old_keys):
            raise ValueError('legacy tensor shape/dtype mismatch')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != set(own)-old_keys:
            raise ValueError('only new memory heads may remain initialized')
        return tuple(result.missing_keys)
