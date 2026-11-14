"""CPU candidate: live language-tail learning + VP + causal byte continuation.

Separate research branch. It does not alter the typed core, old model instances,
or completed experiments. Natural text goes directly to the neural byte encoder;
there are no entity slots, gold semantic factors, regex roles or answer templates.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import Config
from research.text_diffusion_v1 import TextDiffusion, TextConfig, PAD
from research.text_context_decoder_v2 import ContextDecoder, BOS


@dataclass(frozen=True)
class ContinuationConfig:
    prompt_bytes: int = 256
    target_bytes_with_eos: int = 129
    diffusion_steps: int = 16
    train_language_tail_blocks: int = 2
    training_eos_weight: float = 4.0


class LiveLanguageVP(TextDiffusion):
    def __init__(self, core, text, train_tail):
        super().__init__(core, text)
        if not 0 < train_tail <= len(self.language.blocks):
            raise ValueError('explicit nonempty trainable language tail required')
        self.train_tail = train_tail
        for block in self.language.blocks[-train_tail:]:
            block.requires_grad_(True)

    def encode(self, tokens):
        valid = tokens != 256; n = tokens.shape[1]
        if n > self.text_config.prompt_bytes:
            raise ValueError('prompt capacity exceeded')
        pos = torch.arange(n, device=tokens.device)
        band = (pos[:, None] >= pos[None]) & (pos[:, None]-pos[None] < self.core_config.window)
        mask = band[None, None] & (valid[:, None, None, :] |
            torch.eye(n, device=tokens.device, dtype=torch.bool)[None, None])
        # The frozen prefix is deliberately detached; the last language blocks
        # and prompt projection remain in the autograd graph. No final detach.
        with torch.no_grad():
            x = self.language.embedding(tokens)
            for block in self.language.blocks[:-self.train_tail]:
                x = block(x, mask, valid)
        for block in self.language.blocks[-self.train_tail:]:
            x = block(x, mask, valid)
        context = self.prompt_project(x) * valid[..., None]
        return (torch.cat((context, context.new_zeros(len(tokens), 1, context.shape[-1])), 1),
                torch.cat((valid, torch.ones(len(tokens), 1, device=tokens.device, dtype=torch.bool)), 1))


class LanguageContinuation(nn.Module):
    def __init__(self, core=Config(), policy=ContinuationConfig()):
        super().__init__(); self.core_config = core; self.policy = policy
        self.vp = LiveLanguageVP(core, TextConfig(policy.prompt_bytes,
            policy.target_bytes_with_eos, policy.diffusion_steps), policy.train_language_tail_blocks)
        self.decoder = ContextDecoder(core, length=policy.target_bytes_with_eos)

    @staticmethod
    def resize_positions(value, shape):
        if value.ndim != 2 or value.shape[1] != shape[1]:
            raise ValueError('only learned sequence positions may resize')
        return F.interpolate(value.float().T[None], size=shape[0], mode='linear', align_corners=True)[0].T.to(value.dtype)

    def warmstart(self, generator_state, span_generator_state):
        """Copy V4 VP+decoder weights; never restore V4 Adam trajectory."""
        mapping = {}
        for key, value in self.vp.state_dict().items():
            if key not in generator_state:
                raise ValueError('missing VP source key '+key)
            old = generator_state[key]
            if old.shape != value.shape:
                if key != 'positions.weight':
                    raise ValueError('unexpected VP shape change '+key)
                old = self.resize_positions(old, value.shape)
            mapping[key] = old
        if set(mapping) != set(generator_state):
            raise ValueError('unexpected VP source keys')
        self.vp.load_state_dict(mapping, strict=True)
        decoder_source = {k[len('decoder.'):]: v for k, v in span_generator_state.items() if k.startswith('decoder.')}
        decoder = {}
        for key, value in self.decoder.state_dict().items():
            if key not in decoder_source:
                raise ValueError('missing decoder source '+key)
            old = decoder_source[key]
            if old.shape != value.shape:
                if key != 'positions.weight':
                    raise ValueError('unexpected decoder shape change '+key)
                old = self.resize_positions(old, value.shape)
            decoder[key] = old
        if set(decoder) != set(decoder_source):
            raise ValueError('unexpected decoder source keys')
        self.decoder.load_state_dict(decoder, strict=True)
        return {'VP_keys': len(mapping), 'decoder_keys': len(decoder),
            'resized_only': ['vp.positions.weight', 'decoder.positions.weight'],
            'position_resize': 'linear align_corners=True',
            'old_semantic_reader_imported': False, 'old_optimizer_restored': False}

    @torch.no_grad()
    def free_proposal(self, encoded, noise, charge):
        # The signature intentionally has no target bytes or gold semantics.
        context = (encoded[0].detach(), encoded[1]); z = noise.float()
        if z.shape[1] != self.policy.target_bytes_with_eos:
            raise ValueError('proposal noise capacity')
        schedule = torch.linspace(1, 0, self.policy.diffusion_steps+1).tolist()
        for t, nxt in zip(schedule, schedule[1:]):
            charge('denoise')
            velocity = self.vp.denoise(z, z.new_full((len(z),), t), context).float()
            a, s = (0., 1.) if t == 1 else (math.cos(t*math.pi/2), math.sin(t*math.pi/2))
            clean, eps = a*z-s*velocity, s*z+a*velocity
            z = math.cos(nxt*math.pi/2)*clean + math.sin(nxt*math.pi/2)*eps
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite free VP proposal')
        return z

    def training_loss(self, tokens, ids, t, objective_noise, proposal_noise, charge=lambda kind: None):
        if not self.training:
            raise RuntimeError('gold targets permitted only in explicit training loss')
        if ids.shape != tokens.new_empty((len(tokens), self.policy.target_bytes_with_eos)).shape:
            raise ValueError('target capacity')
        charge('encode'); encoded = self.vp.encode(tokens)
        # Free proposal is detached and generated WITHOUT reading target IDs.
        # No-grad proposal casts must not poison the supervised pass's AMP cache.
        device_type = proposal_noise.device.type
        with torch.autocast(device_type, enabled=torch.is_autocast_enabled(device_type),
                dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
            proposal = self.free_proposal(encoded, proposal_noise, charge)
        active = ids != PAD
        if not active.any(-1).all():
            raise ValueError('empty target')
        clean = self.vp.target(ids); angle = t[:, None, None]*math.pi/2
        a, s = angle.cos(), angle.sin(); z = a*clean+s*objective_noise
        target_v = a*objective_noise-s*clean
        charge('denoise'); velocity = self.vp.denoise(z, t, encoded).float()
        predicted = a*z-s*velocity
        weights = active.float(); norm = weights.sum()
        vmse = ((velocity-target_v).square().mean(-1)*weights).sum()/norm
        cmse = ((predicted-clean).square().mean(-1)*weights).sum()/norm
        ce_weights = weights * torch.where(ids == 256, self.policy.training_eos_weight, 1.)
        byte_ce = (F.cross_entropy(self.vp.readout(predicted).transpose(1, 2), ids,
                                  reduction='none')*ce_weights).sum()/norm
        prefix = torch.cat((torch.full_like(ids[:, :1], BOS), ids[:, :-1]), 1)
        context = self.decoder.context(encoded[0], encoded[1], proposal)
        charge('decoder'); logits = self.decoder(prefix, context).float()
        labels = ids.masked_fill(~active, -100)
        ar_ce = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100,
                                reduction='none')
        ar_ce = (ar_ce*ce_weights).sum()/norm
        total = vmse + .25*cmse + .5*byte_ce + ar_ce
        return total, {'velocity_mse': vmse, 'clean_mse': cmse, 'VP_byte_ce': byte_ce,
            'AR_byte_ce': ar_ce, 'free_proposal': proposal, 'AR_logits': logits}

    @torch.no_grad()
    def generate(self, tokens, proposal_noise, charge=lambda kind: None, zero_latent=False):
        charge('encode'); encoded = self.vp.encode(tokens)
        proposal = self.free_proposal(encoded, proposal_noise, charge)
        context = self.decoder.context(encoded[0], encoded[1], proposal, zero_latent=zero_latent)
        return self.decoder.generate(context, charge), proposal

    def specification(self):
        names = [name for name, p in self.named_parameters() if p.requires_grad]
        return {'schema': 'fdt_live_language_continuation_v5_cpu_candidate', 'policy': asdict(self.policy),
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'trainable_language_parameters': sum(p.numel() for p in self.vp.language.parameters() if p.requires_grad),
            'trainable_language_blocks': list(range(self.core_config.lang_layers-self.policy.train_language_tail_blocks, self.core_config.lang_layers)),
            'trainable_names': names, 'training_calls_per_microbatch': self.policy.diffusion_steps+3,
            'AR_training_proposal_contains_gold': False, 'VP_objective_uses_gold_targets': True,
            'gradient_through_free_proposal': False, 'language_gradient_from': ['VP supervised denoising', 'AR raw-context crossattention'],
            'runtime_gold_roles': False, 'runtime_templates': False, 'broad_language_capability_measured': False}
