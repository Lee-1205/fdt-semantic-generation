"""New bounded remainder purpose; original finite order/cosine position retained."""
import copy,math
from contextlib import nullcontext
import torch
from research.text_diffusion_v1 import pack_prompts,pack_answers
from three_memory_codec_engine_v33 import CodecEngine,CONTRACT,OWNERSHIP
from three_memory_codec_engine_v34 import BroadCodecEngine
from owt_fuzzy_engine_v17_base import amp_stop_reason

SCHEMA='fdt_three_memory_language_codec_v35_fullstate'
CONTINUATION={'neural_function':'unchanged_V33_three_memory_codec','all_populated_Adam_states':91,
    'new_parameters':0,'old_allocation_reopened':False,'purpose':'complete_unconsumed_suffix_of_existing_finite_epoch',
    'learning_rate_schedule':'original_full_epoch_cosine_position_no_restart_or_moment_reset','replayed_consumed_rows':0}

class RemainderCodecEngine(BroadCodecEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.epoch_schedule=None

    def import_v34_parent(self,state,source_plan,source_sha):
        if state['schema']!='fdt_three_memory_language_codec_v34_fullstate':raise ValueError('actual partial V34 schema')
        offset=state['attempts'];total=len(state['orders'])
        if not 0<offset<total or self.orders!=state['orders'][offset:]:raise ValueError('only exact unconsumed suffix')
        # Reuse verified ALL91 component import, preserving the actual schema and
        # prior continuation record explicitly after its schema adapter.
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        super().import_codec_parent(adapted,source_plan,source_sha)
        self.ancestry['source_schema']=state['schema'];self.ancestry['continuation']=copy.deepcopy(state['continuation'])
        self.epoch_schedule={'origin_plan_sha256':source_plan,'total_batches':total,'offset_batches':offset,
            'source_cursor':state['cursor'],'replayed_consumed_rows':0}
        self.continuation['policy']=copy.deepcopy(CONTINUATION)
        self.continuation['epoch_schedule']=copy.deepcopy(self.epoch_schedule)
        self.model.set_training_attempt(offset)
        return copy.deepcopy(self.continuation)

    def step(self):
        if self.epoch_schedule is None:raise ValueError('source finite schedule not imported')
        self.budget.check();indices=self.next_indices();self.attempts+=1;self.cursor+=len(indices)
        epoch_attempt=self.epoch_schedule['offset_batches']+self.attempts
        self.model.set_training_attempt(epoch_attempt)
        self.exposure['selected_rows']+=len(indices)
        byte_count=sum(len(self.targets[i]['target'].encode('utf-8')) for i in indices)
        self.exposure['selected_target_UTF8_bytes']+=byte_count
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        factor=.1+.9*(1+math.cos(math.pi*(epoch_attempt-1)/self.epoch_schedule['total_batches']))/2
        for group in self.optimizer.param_groups:group['lr']=group['base_lr']*factor
        x=pack_prompts([self.features[i]['prompt'] for i in indices],128,self.device)
        y=pack_answers([self.targets[i]['target'] for i in indices],129,self.device)
        def charge(kind):
            self.budget.charge(kind)
            if kind=='encode':
                self.exposure['processed_rows']+=len(indices);self.exposure['target_UTF8_bytes']+=byte_count
                self.exposure['source_sentence_boundaries']+=len(indices)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
        try:
            with amp:loss,episode=self.model.reconstruction_loss(x,y,source_ids=tuple(self.features[i]['id'] for i in indices),
                source_versions=(self.data_revision,)*len(indices),model_epoch=self.memory_epoch(),charge=charge)
            if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('codec nonfinite/loss explosion')
            self.scaler.scale(loss).backward();self.scaler.unscale_(self.optimizer)
            for name,parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:raise RuntimeError('missing codec gradient '+name)
                if not parameter.requires_grad and parameter.grad is not None:raise RuntimeError('frozen gradient '+name)
            norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
            before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
            skip=self.scaler.get_scale()<before;self.skips+=int(skip);self.updates+=int(not skip)
            rec={'attempt':self.attempts,'updates':self.updates,'epoch_attempt':epoch_attempt,'rows':list(indices),'loss':float(loss.detach()),
                'AMP_skip':skip,'grad_norm':float(norm) if torch.isfinite(norm) else None,'scale':self.scaler.get_scale(),
                'mean_fuzzy_degree':float(episode.diagnostics['memberships'].detach().mean()),'outcome':'AMP_SKIP' if skip else 'UPDATED',
                'mask_probability':.5,'schedule_factor':factor,**self.model.last_objective_stats}
            self.history.append(rec);self.optimizer.zero_grad(set_to_none=True)
            for key in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):
                self.exposure[key]=self.exposure.get(key,0)+rec[key]
            reason=amp_stop_reason(self.history,self.ancestry.get('history',[]),self.skips,skip,bool(torch.isfinite(norm)))
            if reason:raise FloatingPointError('codec persistent AMP/gradient stop: '+reason)
            if len(self.history)>10 and rec['loss']>20*sorted(r['loss'] for r in self.history[-11:-1] if 'loss' in r)[5]:raise FloatingPointError('20x codec loss explosion')
            return rec
        except BaseException as exc:
            if len(self.history)<self.attempts:self.history.append({'attempt':self.attempts,'updates':self.updates,'epoch_attempt':epoch_attempt,
                'rows':list(indices),'outcome':'FAILED_ATTEMPT','error':repr(exc)})
            raise

    def export(self):
        state=CodecEngine.export(self)
        state.update(schema=SCHEMA,continuation=copy.deepcopy(self.continuation),epoch_schedule=copy.deepcopy(self.epoch_schedule))
        return state

    def restore(self,state):
        if state['schema']!=SCHEMA or state['continuation']['policy']!=CONTINUATION or state['epoch_schedule']!=state['continuation']['epoch_schedule']:
            raise ValueError('remainder recovery contract')
        offset=state['epoch_schedule']['offset_batches']
        if state['orders']!=state['ancestry']['orders'][offset:] or offset!=state['ancestry']['attempts']:
            raise ValueError('remainder order/cursor origin')
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        CodecEngine.restore(self,adapted)
        self.continuation=copy.deepcopy(state['continuation']);self.epoch_schedule=copy.deepcopy(state['epoch_schedule'])
        self.model.set_training_attempt(max(1,offset+self.attempts))
