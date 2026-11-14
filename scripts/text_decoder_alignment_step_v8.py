from contextlib import nullcontext
import math,torch
from research.text_diffusion_v1 import pack_prompts,PAD
from text_language_engine_v6 import row_noise

class StepMixin:
    def step(self):
        self.prepare_phase();self.budget.check();indices=self.next_indices();self.attempts+=1;self.cursor+=len(indices)
        def expose(prefix,ids):
            self.exposure[prefix+'_rows']+=len(ids)
            self.exposure[prefix+'_target_bytes']+=sum(len(self.targets[i]['target'].encode('utf-8')) for i in ids)
            self.exposure[prefix+'_EOF']+=sum(self.targets[i]['source_eof'] for i in ids)
            self.exposure[prefix+'_context_bytes']+=sum(len(self.features[i]['prompt'].encode('utf-8')) for i in ids)
        expose('selected',indices);self.model.train();self.optimizer.zero_grad(set_to_none=True)
        factor=.1+.9*(1+math.cos(math.pi*(self.attempts-1)/self.limit))/2
        for group in self.optimizer.param_groups:group['lr']=group['base_lr']*factor
        total_symbols=sum(len(self.targets[i]['target_ids']) for i in indices)
        total_negative_spans=sum(len(self.negatives[i]['negative_spans']) for i in indices)
        means={};outcome='FAILED_ATTEMPT'
        try:
            for start in range(0,len(indices),4):
                ids=indices[start:start+4];rows=[self.features[i] for i in ids]
                target=torch.full((len(ids),self.model.policy.target_bytes_with_eos),PAD,device=self.device,dtype=torch.long)
                for j,i in enumerate(ids):target[j,:len(self.targets[i]['target_ids'])]=torch.tensor(self.targets[i]['target_ids'],device=self.device)
                tokens=pack_prompts([r['prompt'] for r in rows],self.model.policy.prompt_bytes,self.device)
                t=torch.rand(len(ids),device=self.device)*.998+.001
                noise=torch.randn(len(ids),target.shape[1],self.model.core_config.z_dim,device=self.device)
                proposal=None  # paired reconstruction uses objective noise; no training free VP sampling
                weight=sum(len(self.targets[i]['target_ids']) for i in ids)/total_symbols
                def charge(kind):
                    self.budget.charge(kind)
                    if kind=='encode':expose('processed',ids)
                negative=self.negatives_for(ids)
                negative=(*negative,len(negative[1])/total_negative_spans/weight)
                self.model.training_negative_batch=negative
                amp=torch.autocast('cuda',dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
                with amp:loss,aux=self.model.training_loss(tokens,target,t,noise,proposal,charge)
                if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('numeric/loss explosion')
                self.scaler.scale(loss*weight).backward()
                for key in ('velocity_mse','clean_mse','VP_byte_ce','AR_byte_ce','span_UL'):
                    means[key]=means.get(key,0.)+float(aux[key].detach())*weight
                means['loss']=means.get('loss',0.)+float(loss.detach())*weight
                del aux,loss
            self.scaler.unscale_(self.optimizer)
            for name,param in self.model.named_parameters():
                if not param.requires_grad and param.grad is not None:raise RuntimeError('frozen gradient ownership '+name)
                if param.requires_grad and param.grad is None:raise RuntimeError('missing required trainable gradient '+name)
            norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
            before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
            skipped=self.scaler.get_scale()<before;self.skips+=int(skipped);self.updates+=int(not skipped)
            outcome='AMP_SKIP' if skipped else 'UPDATED'
            record={'phase':self.phase,'attempt':self.attempts,'updates':self.updates,'rows':indices,'outcome':outcome,
                **means,'grad_norm':float(norm) if torch.isfinite(norm) else None,
                'AMP_skip':skipped,'scale':self.scaler.get_scale(),'actual_calls':dict(self.budget.calls)}
            self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
            if self.skips>=3 or (not skipped and not torch.isfinite(norm)):raise FloatingPointError('AMP/gradient stop')
            if len(self.history)>=11:
                reference=sorted(r['loss'] for r in self.history[-11:-1] if 'loss' in r)
                if reference and means['loss']>20*reference[len(reference)//2]:raise FloatingPointError('20x loss explosion')
            return record
        except BaseException as exc:
            if len(self.history)<self.attempts:
                self.history.append({'attempt':self.attempts,'updates':self.updates,'rows':indices,
                    'outcome':outcome,'error':repr(exc),'actual_calls':dict(self.budget.calls)})
            raise

