"""Two broad finite epochs; unchanged neural codec and ALL91 Adam states."""
import copy
import math
import torch
from contextlib import nullcontext
from research.text_diffusion_v1 import pack_prompts, pack_answers
from owt_fuzzy_engine_v17_base import amp_stop_reason
from three_memory_codec_engine_v33 import CodecEngine,CONTRACT,OWNERSHIP
from three_memory_codec_engine_v34 import BroadCodecEngine

SCHEMA='fdt_three_memory_language_codec_v43_fullstate'
CONTINUATION={'neural_function':'unchanged_V33_three_memory_codec','all_populated_Adam_states':91,
    'new_parameters':0,'old_allocation_reopened':False,'purpose':'two_broad_finite_TRAIN_epochs',
    'learning_rate_schedule':'constant_source_terminal_group_LRs_no_restart_no_moment_or_step_reset',
    'source_epoch_completed':True,'new_finite_epochs':2}

class TwoEpochCodecEngine(BroadCodecEngine):
    def import_completed_v42(self,state,source_plan,source_sha):
        if state['schema']!='fdt_three_memory_language_codec_v42_fullstate':raise ValueError('completed V42 source schema')
        if state['worker_stage']!='final_commit' or state['attempts']!=len(state['orders']) or state['cursor']!=sum(map(len,state['orders'])):
            raise ValueError('source must be terminal and all its finite suffix consumed')
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        super().import_codec_parent(adapted,source_plan,source_sha)
        self.ancestry['source_schema']=state['schema']
        self.ancestry['continuation']=copy.deepcopy(state['continuation'])
        self.continuation['policy']=copy.deepcopy(CONTINUATION)
        self.continuation['source_terminal_group_lrs']=[float(g['lr']) for g in state['optimizer']['param_groups']]
        self._terminal_lrs()
        return copy.deepcopy(self.continuation)

    def _terminal_lrs(self):
        values=self.continuation.get('source_terminal_group_lrs')
        if (not isinstance(values,list) or len(values)!=len(self.optimizer.param_groups)
            or any(type(v) is not float or not math.isfinite(v) or v<=0 for v in values)
            or any(g['lr']!=v for g,v in zip(self.optimizer.param_groups,values))):
            raise ValueError('exact imported/recovered constant group LR schedule')
        return values

    def step(self):
        self.budget.check(); indices = self.next_indices(); self.attempts += 1; self.cursor += len(indices)
        self.model.set_training_attempt(self.attempts)
        self.exposure['selected_rows'] += len(indices)
        byte_count = sum(len(self.targets[i]['target'].encode('utf-8')) for i in indices)
        self.exposure['selected_target_UTF8_bytes'] += byte_count
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        for group, lr in zip(self.optimizer.param_groups, self._terminal_lrs()):
            group['lr'] = lr
        x = pack_prompts([self.features[i]['prompt'] for i in indices], 128, self.device)
        y = pack_answers([self.targets[i]['target'] for i in indices], 129, self.device)
        def charge(kind):
            self.budget.charge(kind)
            if kind == 'encode':
                self.exposure['processed_rows'] += len(indices)
                self.exposure['target_UTF8_bytes'] += byte_count
                self.exposure['source_sentence_boundaries'] += len(indices)
        amp = torch.autocast('cuda', dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
        try:
            with amp:
                loss, episode = self.model.reconstruction_loss(x, y, source_ids=tuple(self.features[i]['id'] for i in indices),
                    source_versions=(self.data_revision,)*len(indices), model_epoch=self.memory_epoch(), charge=charge)
            if not torch.isfinite(loss) or float(loss.detach()) > 1000: raise FloatingPointError('codec nonfinite/loss explosion')
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None: raise RuntimeError('missing codec gradient '+name)
                if not parameter.requires_grad and parameter.grad is not None: raise RuntimeError('frozen gradient '+name)
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.)
            before = self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update()
            skip = self.scaler.get_scale() < before; self.skips += int(skip); self.updates += int(not skip)
            rec = {'attempt': self.attempts, 'updates': self.updates, 'rows': list(indices), 'loss': float(loss.detach()),
                'AMP_skip': skip, 'grad_norm': float(norm) if torch.isfinite(norm) else None, 'scale': self.scaler.get_scale(),
                'mean_fuzzy_degree': float(episode.diagnostics['memberships'].detach().mean()),
                'optimizer_group_lrs': [g['lr'] for g in self.optimizer.param_groups],
                'outcome': 'AMP_SKIP' if skip else 'UPDATED', 'mask_probability': .5, **self.model.last_objective_stats}
            self.history.append(rec); self.optimizer.zero_grad(set_to_none=True)
            for key in ('masked_prefix_bytes', 'eligible_prefix_bytes', 'decoder_supervised_symbol_occurrences'):
                self.exposure[key] = self.exposure.get(key, 0)+rec[key]
            reason = amp_stop_reason(self.history, self.ancestry.get('history', []), self.skips, skip, bool(torch.isfinite(norm)))
            if reason: raise FloatingPointError('codec persistent AMP/gradient stop: '+reason)
            if len(self.history) > 10 and rec['loss'] > 20*sorted(r['loss'] for r in self.history[-11:-1] if 'loss' in r)[5]:
                raise FloatingPointError('20x codec loss explosion')
            return rec
        except BaseException as exc:
            if len(self.history) < self.attempts:
                self.history.append({'attempt': self.attempts, 'updates': self.updates, 'rows': list(indices),
                    'outcome': 'FAILED_ATTEMPT', 'error': repr(exc)})
            raise

    def export(self):
        state=CodecEngine.export(self)
        state.update(schema=SCHEMA,continuation=copy.deepcopy(self.continuation))
        return state

    def restore(self,state):
        if state['schema']!=SCHEMA or state['continuation']['policy']!=CONTINUATION:raise ValueError('two_epoch finite epoch recovery contract')
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        CodecEngine.restore(self,adapted);self.continuation=copy.deepcopy(state['continuation'])
        self._terminal_lrs()

