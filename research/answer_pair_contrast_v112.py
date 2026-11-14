"""Actual-pair endpoint contrast; ONLY thought theta learns.

Two target-free trajectories use the same independently drawn epsilon. This
does not change either endpoint's marginal noise distribution. Gold enters
endpoint/VP/difference losses only. A two-row graph is an atomic recovery unit.
For equal body masks, endpoint mean = ||e_mean||² + .25||e_a-e_b||².
Adding .25||e_a-e_b||² doubles the contrast coefficient, not common error.
This coupled intervention is not an isolated test of common-noise alone.
"""
from dataclasses import asdict
import torch
from research.answer_meaning_thought_v111 import AnswerMeaningEngine
from research.runtime_aligned_thought_v107 import training_rollout


def contrast_error(a,b,pa,pb):
    mask=pa['valid_extent_mask'] & pb['valid_extent_mask']
    if not mask.any() or a.shape!=b.shape:raise ValueError('shared body geometry')
    residual=(a-b)-(pa['semantic_target']-pb['semantic_target']).to(a)
    return residual.square().mean(-1)[mask].mean()


class PairObjective:
    def __init__(self,engine):self.engine=engine
    def __call__(self,m,packet,t,noise,charge):
        e=self.engine
        if not e.thought.training:raise RuntimeError('TRAIN-only target packet')
        required={'schema','answer_SHA','source_codec_SHA','semantic_target','valid_extent_mask','labels','model_epoch','target_condition_features_never_used'}
        if (set(packet)!=required or packet['schema']!='LOSS_ONLY_answer_meaning_v95'
            or packet['source_codec_SHA']!=e.codec_SHA or packet['model_epoch']!=m.model_epoch
            or not packet['target_condition_features_never_used']
            or any(packet[k].requires_grad for k in ('semantic_target','valid_extent_mask','labels'))):raise ValueError('loss-only E(Y) provenance')
        z=training_rollout(e.thought,m,noise,e.latent_policy.steps,charge)
        y=packet['semantic_target'].to(z);mask=packet['valid_extent_mask'].to(z.device)
        if z.shape!=y.shape or mask.shape!=z.shape[:2] or mask.dtype!=torch.bool or not mask.any():raise ValueError('body/tail geometry')
        err=(z-y).square().mean(-1);body=err[mask].mean();tail=err[~mask].mean() if (~mask).any() else err.sum()*0
        charge('denoise');aux,_=e.thought.training_objective(m,y,mask,t,noise)
        c=e.latent_policy;loss=c.body_weight*body+c.tail_weight*tail+c.VP_aux_weight*aux
        if not torch.isfinite(loss):raise FloatingPointError('finite endpoint and VP')
        return loss,dict(latent=float(loss.detach()),byte_CE=0.,EOS_CE=0.,language_losses_NOT_used=True,
            runtime_body_MSE=float(body.detach()),runtime_tail_MSE=float(tail.detach()),aux_VP=float(aux.detach()),
            active_bytes=int((packet['labels']<256).sum()),EOS_count=int((packet['labels']==256).sum()),
            PAD_ignored=int((packet['labels']==257).sum()),source_encoder_calls0=True,language_forward_calls0=True),z


class PairContrastEngine(AnswerMeaningEngine):
    def __init__(self,*args,units,coefficient=.25,**kwargs):
        if coefficient!=.25:raise ValueError('prospective fixed contrast .25')
        self.units=[dict(ids=list(u['ids']),real_pair=bool(u['real_pair']),key=u['key']) for u in units]
        self.coefficient=coefficient
        metadata=dict(kwargs.get('source_metadata') or {})
        metadata['pair_contrast_v112']=dict(coefficient=coefficient,common_epsilon_actual_pairs_only=True,
            atomic_two_rows=True,units=self.units)
        kwargs['source_metadata']=metadata
        super().__init__(*args,**kwargs)
        if self.batch%2 or [i for u in self.units for i in u['ids']]!=self.orders or any(len(u['ids'])!=2 for u in self.units):
            raise ValueError('exact atomic two-row order and even batch')
        self.objective=PairObjective(self)

    def micro(self,*a,**kw):raise RuntimeError('use atomic pair unit, not single-row micro')

    def unit(self,identities,sources,packets,charge=lambda kind:None):
        r=self.runtime;start=r['cursor']
        if r['phase']!='MICROS' or start%2 or start>=len(self.orders):raise ValueError('no finished/aborted unit replay')
        u=self.units[start//2]
        if (list(identities)!=u['ids'] or len(sources)!=2 or len(packets)!=2
                or self.group_size()-len(r['pending_ids'])<2):raise ValueError('exact next complete pair/single scheduling unit')
        if u['real_pair'] and self.times[start]!=self.times[start+1]:raise ValueError('actual pair same auxiliary t')
        gradients=self._gradients();calls_before=dict(r['calls'])
        def count(kind):
            if kind not in r['calls']:raise ValueError('thought/D only')
            charge(kind);r['calls'][kind]+=1
        try:
            shape=packets[0]['semantic_target'].shape
            noise=torch.randn(shape,device=self.device)
            noises=[noise,noise] if u['real_pair'] else [noise,torch.randn(shape,device=self.device)]
            with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
                values=[self.objective(m,p,torch.tensor([self.times[start+j]],device=self.device),noises[j],count)
                    for j,(m,p) in enumerate(zip(sources,packets))]
                contrast=contrast_error(values[0][2],values[1][2],*packets) if u['real_pair'] else values[0][0].new_zeros(())
                loss=values[0][0]+values[1][0]+2*self.coefficient*contrast
            if not torch.isfinite(loss) or float(loss.detach())>2000:raise FloatingPointError('finite pair loss bound')
            self.scaler.scale(loss/self.group_size()).backward()
        except BaseException as error:
            self._put_gradients(gradients);r['phase']='ABORTED';r['aborted']=dict(source_ids=list(identities),error=repr(error),
                calls_delta={k:r['calls'][k]-calls_before[k] for k in r['calls']},atomic_two_rows_consumed=True,no_graph_replay=True)
            r['cursor']+=2;r['exposure_bytes']+=sum(int((p['labels']<256).sum()) for p in packets)
            raise
        for identity,(_,metrics,_) in zip(identities,values):
            metrics.update(pair_contrast_MSE=float(contrast.detach()),actual_pair=u['real_pair'],
                pair_coefficient=self.coefficient,VP_time=self.times[r['cursor']])
            metrics['latent']+=self.coefficient*metrics['pair_contrast_MSE']
            r['pending_ids'].append(identity);r['pending_losses'].append(metrics)
            r['cursor']+=1;r['exposure_bytes']+=metrics['active_bytes']
        if len(r['pending_ids'])==self.group_size():r['phase']='READY'
        self.update.assert_fast();return [v[1] for v in values]

    def finish(self,**kwargs):
        rows=list(self.runtime['pending_losses']);h=super().finish(**kwargs)
        actual=[x['pair_contrast_MSE'] for x in rows if x['actual_pair']]
        h.update(pair_contrast_MSE=None if not actual else sum(actual)/len(actual),actual_pair_rows=len(actual),
            unit_RNG_draws=sum(1 if self.units[j//2]['real_pair'] else 2
                for j in range(h['cursor']-h['rows'],h['cursor'],2)))
        return h

    def payload(self,*,audit=True):
        p=super().payload(audit=audit)
        p.update(schema='answer_pair_contrast_fullstate_v112',atomic_unit_rows=2,pair_coefficient=self.coefficient)
        return p

    def restore(self,p,*,audit=True):
        if (p['schema']!='answer_pair_contrast_fullstate_v112' or p['pair_coefficient']!=self.coefficient
                or p['atomic_unit_rows']!=2 or p['runtime']['cursor']%2):raise ValueError('exact atomic paired schema')
        super().restore(dict(p,schema='conditional_answer_meaning_fullstate_v111'),audit=audit)
