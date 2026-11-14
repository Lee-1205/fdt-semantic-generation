"""CPU candidate from observed TRAIN noise gap, not a new trained model.

Match the exact16 inference times (including pure-noise t=1), and supervise
fixed-capacity tail velocities which remain visible to self-attention. No GT
extent enters inference. No old encoder/memory/decoder parameter is added.
"""
import hashlib,math,random
import torch
from research.memory_thought_diffusion_v94 import MemoryThoughtDiffusion,vp
from research.thought_train_engine_v98 import ThoughtTrainEngine


def finite_curriculum(train_targets):
    pairs={}
    for row in train_targets:pairs.setdefault(row['pair_key'],[]).append(row['id'])
    if len(train_targets)!=384 or len(pairs)!=192 or any(len(v)!=2 for v in pairs.values()):raise ValueError('existing384TRAIN complete pairs only')
    output=[]
    for epoch in range(16):
        keys=sorted(pairs);random.Random(104000+epoch).shuffle(keys);rows=[]
        for key in keys:
            slot=(epoch+int(hashlib.sha256(key.encode()).hexdigest()[:8],16))%16
            # Public sampler grid. Pair hashing is offline order scheduling,
            # never a semantic label, condition feature or runtime oracle.
            rows.extend(dict(id=identity,t=1-slot/16) for identity in pairs[key])
        output.append(rows)
    return output


class GridTailThought(MemoryThoughtDiffusion):
    def __init__(self,c=None,tail_weight=.05):
        if c is None:super().__init__()
        else:super().__init__(c)
        if not math.isfinite(tail_weight) or not 0<=tail_weight<=1:raise ValueError('fixed bounded tail coefficient')
        self.tail_weight=float(tail_weight)

    def training_objective(self,m,target,mask,t,noise):
        if not self.training:raise RuntimeError('TRAIN-only target/extent/noisy gold')
        centre,scale=self.initial_prior(m)
        if (target.shape!=centre.shape or mask.shape!=target.shape[:2] or mask.dtype!=torch.bool or not mask.any()
                or noise.shape!=target.shape or not torch.isfinite(noise).all() or not torch.isfinite(target).all()):raise ValueError('fixed-capacity loss-side target')
        residual=(target.detach()-centre.detach())/scale.detach();a,s=vp(t);rt=a*residual+s*noise
        zt=centre.detach()+scale.detach()*rt;predicted=self(zt,t,m);gold_v=a*noise-s*residual
        predicted_residual=a*((zt-centre)/scale)-s*predicted;clean=centre+scale*predicted_residual
        error=(predicted-gold_v).square().mean(-1)+.25*(clean-target.detach()).square().mean(-1)
        body=(error*mask).sum()/mask.sum()
        tail=(error*(~mask)).sum()/(~mask).sum().clamp_min(1)
        # Separate means avoid30tail slots outvoting4or5 content slots. Weight
        # .05 is a prospective choice, not an optimum or proven improvement.
        return body+self.tail_weight*tail,clean


class GridThoughtTrainEngine(ThoughtTrainEngine):
    def __init__(self,*args,times,**kwargs):
        self.times=list(times)
        super().__init__(*args,**kwargs)
        if len(self.times)!=len(self.orders) or any(t not in [1-i/16 for i in range(16)] for t in self.times):raise ValueError('finite exact inference-grid training times')

    def micro(self,identity,source,packet,charge=lambda kind:None):
        r=self.runtime
        if r['phase']!='MICROS' or r['cursor']>=len(self.orders) or identity!=self.orders[r['cursor']]:raise ValueError('exact next finite source')
        gradients=self._gradients();calls_before=dict(r['calls'])
        def count(kind):
            if kind not in r['calls']:raise ValueError('thought/D only')
            charge(kind);r['calls'][kind]+=1
        try:
            shape=packet['semantic_target'].shape;t=torch.tensor([self.times[r['cursor']]],device=self.device);noise=torch.randn(shape,device=self.device)
            with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
                loss,metrics=self.objective(source,packet,t,noise,count)
            if float(loss.detach())>1000:raise FloatingPointError('fixed gross loss bound1000')
            self.scaler.scale(loss/self.group_size()).backward()
        except BaseException as e:
            self._put_gradients(gradients);r['phase']='ABORTED';r['aborted']=dict(source_id=identity,error=repr(e),
                calls_delta={k:r['calls'][k]-calls_before[k] for k in r['calls']},no_graph_replay=True,previous_complete_micro_gradients_restored=True)
            r['cursor']+=1;r['exposure_bytes']+=int((packet['labels']<256).sum());raise
        metrics['VP_time']=float(t);r['pending_ids'].append(identity);r['pending_losses'].append(metrics);r['cursor']+=1;r['exposure_bytes']+=metrics['active_bytes']
        if len(r['pending_ids'])==self.group_size():r['phase']='READY'
        self.update.assert_fast();return metrics

    def payload(self,*,audit=True):
        p=super().payload(audit=audit);p.update(schema='grid_tail_thought_fullstate_v104',fixed_times=list(self.times),tail_weight=self.thought.tail_weight);return p

    def restore(self,p,*,audit=True):
        if p['schema']!='grid_tail_thought_fullstate_v104' or p['fixed_times']!=self.times or p['tail_weight']!=self.thought.tail_weight:raise ValueError('exact grid/tail policy restore')
        super().restore(dict(p,schema='thought_only_fullstate_v98'),audit=audit)
