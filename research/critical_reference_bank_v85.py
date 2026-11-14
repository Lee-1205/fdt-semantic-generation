"""TRAIN-only FP32 broad-top1 + all-weak-critical reference bank component.

This is a training-driver component, not a runnable optimizer/encoder or an
inference memory. Labels/spans are never neural features. Feature/target files
stay separate. The future driver captures prefix activations during its one
fresh full-FP32 teacher pass, under the same parent and precision as inference.
Historical BF16 activations, even cast to float, are not that fresh pass.
"""
import copy,math
import torch
from research.readout_precision_policy_v81 import POLICY
from research.train_critical_frontier_v84 import select
from research.train_margin_retention_v81 import margins
from research.actual_update_retention_v80 import TrainDecision

SCHEMA='TRAIN_critical_reference_bank_component_v85'
FKEYS={'hidden','source_id','source_SHA','embedding_SHA','prefix_SHA','parent_SHA','precision_policy'}
TKEYS={'source_id','source_SHA','source_bytes','labels','critical_mask','teacher_correct','teacher_margin','precision_policy'}

class CriticalReferenceBank:
    def __init__(self,sources,broad_indices,*,embedding_SHA,prefix_SHA,parent_SHA,max_patches=8192,max_weak_margin=1.,fraction=.5,floor_cap=1.):
        if (not sources or len({x['id'] for x in sources})!=len(sources) or any(set(x)!={'id','SHA'} or not x['id'] or len(x['SHA'])!=64 for x in sources)
            or len(set(broad_indices))!=len(broad_indices) or any(type(i)!=int or not 0<=i<len(sources) for i in broad_indices)):
            raise ValueError('explicit TRAIN allowlist and broad source indices')
        if (type(max_patches)!=int or max_patches<1 or not all(math.isfinite(v) for v in (max_weak_margin,fraction,floor_cap))
            or not 0<max_weak_margin or not 0<fraction<=1 or not 0<floor_cap):raise ValueError('fixed prospective bank limits')
        if any(not isinstance(v,str) or len(v)!=64 for v in (embedding_SHA,prefix_SHA,parent_SHA)):raise ValueError('actual frozen model identity')
        self.config=dict(sources=copy.deepcopy(sources),broad_indices=list(broad_indices),embedding_SHA=embedding_SHA,prefix_SHA=prefix_SHA,parent_SHA=parent_SHA,
            max_patches=max_patches,max_weak_margin=float(max_weak_margin),fraction=float(fraction),floor_cap=float(floor_cap),precision_policy=POLICY)
        self.cursor=0;self.parts=[];self.patch_count=0;self.weak_count=0;self.calibration=None

    def ingest(self,source_index,feature,target):
        if self.calibration is not None or source_index!=self.cursor or source_index>=len(self.config['sources']):raise ValueError('one finite ordered teacher pass, no replay')
        if set(feature)!=FKEYS or set(target)!=TKEYS:raise ValueError('strict separate feature and loss-only target schemas')
        identity=self.config['sources'][source_index]
        for packet in (feature,target):
            if packet['source_id']!=identity['id'] or packet['source_SHA']!=identity['SHA'] or packet['precision_policy']!=POLICY:raise ValueError('TRAIN identity/FP32 policy')
        for key in ('embedding_SHA','prefix_SHA','parent_SHA'):
            if feature[key]!=self.config[key]:raise ValueError('stale frozen-prefix/model epoch')
        h=feature['hidden'];y=target['labels'];critical=target['critical_mask'];correct=target['teacher_correct'];margin=target['teacher_margin'];source_bytes=target['source_bytes']
        if (not torch.is_tensor(h) or h.device.type!='cpu' or h.dtype!=torch.float32 or h.requires_grad or h.ndim!=2 or h.shape[-1]!=512
            or not torch.isfinite(h).all() or y.device.type!='cpu' or y.shape!=(len(h),16) or y.dtype!=torch.long or len(h)!=(source_bytes+15)//16+1):raise ValueError('fresh detached CPU FP32 prefix and byte/EOS/PAD shape')
        if (any(v.device.type!='cpu' or v.shape!=y.shape for v in (critical,correct,margin)) or critical.dtype!=torch.bool or correct.dtype!=torch.bool
            or margin.dtype!=torch.float32 or not torch.isfinite(margin).all() or source_bytes<0 or source_bytes>=y.numel()):raise ValueError('loss-only target schema')
        flat=y.flatten()
        if ((flat[:source_bytes]<0)|(flat[:source_bytes]>255)).any() or int(flat[source_bytes])!=256 or (flat[source_bytes+1:]!=257).any():raise ValueError('one true EOS then PAD, no gold length as NN feature')
        if (correct&(y==257)).any() or (critical&(y>=256)).any() or (correct&(margin<0)).any():raise ValueError('valid teacher/critical masks')
        packet=dict(target)
        selection=select(packet,identity['id'],{r['id'] for r in self.config['sources']},precision_policy=POLICY,expected_precision_policy=POLICY,
            max_margin=self.config['max_weak_margin'],max_patches=self.config['max_patches'])
        weak=torch.zeros_like(critical)
        for patch,role in selection['critical_positions']:weak[patch,role]=True
        keep=list(range(len(h))) if source_index in self.config['broad_indices'] else selection['patch_indices']
        if self.patch_count+len(keep)>self.config['max_patches']:raise OverflowError('whole finite bank cap, no truncation/growth')
        idx=torch.tensor(keep,dtype=torch.long)
        # Clone only after all checks, so a rejected input does not advance the
        # bank cursor or leave an oversized shared tensor storage in a checkpoint.
        part=dict(source_index=source_index,source_id=identity['id'],source_SHA=identity['SHA'],source_patch_indices=keep,
            hidden=h[idx].clone(),labels=y[idx].clone(),critical_mask=critical[idx].clone(),teacher_correct=correct[idx].clone(),
            teacher_margin=margin[idx].clone(),weak_critical_mask=weak[idx].clone())
        self.parts.append(part);self.patch_count+=len(keep);self.weak_count+=int(weak.sum());self.cursor+=1
        return dict(source_index=source_index,kept_patches=len(keep),weak_critical_bytes=int(weak.sum()),cursor=self.cursor,bank_patches=self.patch_count)

    def components(self):
        if self.cursor!=len(self.config['sources']) or not self.patch_count:raise ValueError('complete finite teacher pass required')
        metadata={k:copy.deepcopy(v) for k,v in self.config.items() if k!='sources'}
        ranges=[];start=0
        for p in self.parts:
            ranges.append(dict(source_index=p['source_index'],source_id=p['source_id'],source_SHA=p['source_SHA'],source_patch_indices=p['source_patch_indices'],start=start,end=start+len(p['hidden'])))
            start+=len(p['hidden'])
        features=dict(schema=SCHEMA+'_features',hidden=torch.cat([p['hidden'] for p in self.parts]),source_ranges=ranges,metadata=metadata)
        targets=dict(schema=SCHEMA+'_loss_only',**{key:torch.cat([p[key] for p in self.parts]) for key in ('labels','critical_mask','teacher_correct','teacher_margin','weak_critical_mask')})
        return features,targets

    @torch.no_grad()
    def calibrate(self,initial_logits,*,max_row_batch_margin_difference=1e-3):
        if self.calibration is not None:raise ValueError('one fixed reference calibration')
        if max_row_batch_margin_difference!=1e-3:raise ValueError('fixed row/batch numeric tolerance; no silent expansion')
        _,t=self.components();y=t['labels'];z=initial_logits
        if z.device.type!='cpu' or z.shape!=(*y.shape,258):raise ValueError('actual initial batched reference shape')
        m=margins(z,y);valid=y!=257;pred=z.argmax(-1);correct=(pred==y)&valid
        disagreement=int(((correct!=t['teacher_correct'])&valid).sum());drift=float((m[valid]-t['teacher_margin'][valid]).abs().max())
        if disagreement or drift>max_row_batch_margin_difference:raise ValueError('rowwise/batched reference precision/kernel parity outside fixed tolerance')
        floor_mask=t['weak_critical_mask']&correct
        floor=m.clamp(min=0,max=self.config['floor_cap'])*self.config['fraction']
        self.calibration=dict(labels=y.clone(),critical_mask=t['critical_mask'].clone(),top1_mask=correct,weak_floor_mask=floor_mask,
            initial_argmax=pred.to(torch.int16),initial_margin=m,required_margin=floor,max_row_batch_margin_difference=drift,
            teacher_correct_disagreements=disagreement,precision_policy=POLICY)
        return dict(protected_top1=int(correct.sum()),weak_critical_floors=int(floor_mask.sum()),bank_patches=self.patch_count,row_batch_max_margin_difference=drift)

    @torch.no_grad()
    def decision(self,candidate_logits):
        if self.calibration is None:raise ValueError('initial reference calibration required')
        c=self.calibration;y=c['labels'];z=candidate_logits
        if z.device.type!='cpu':raise ValueError('component decision CPU contract; future GPU driver must make explicit equivalent device transfer')
        m=margins(z,y);flipped=c['top1_mask']&(z.argmax(-1)!=y);weak_erosion=c['weak_floor_mask']&(m<c['required_margin']);critical=c['critical_mask']
        return TrainDecision(not bool((flipped|weak_erosion).any()),dict(new_critical_flips=int((flipped&critical).sum()),new_noncritical_EOS_flips=int((flipped&~critical).sum()),
            weak_critical_margin_violations=int(weak_erosion.sum()),protected_top1=int(c['top1_mask'].sum()),protected_weak_critical=int(c['weak_floor_mask'].sum()),bank_patches=self.patch_count))

    def export(self):
        return copy.deepcopy(dict(schema=SCHEMA,config=self.config,cursor=self.cursor,parts=self.parts,patch_count=self.patch_count,weak_count=self.weak_count,calibration=self.calibration))

    def restore(self,state):
        if set(state)!={'schema','config','cursor','parts','patch_count','weak_count','calibration'} or state['schema']!=SCHEMA or state['config']!=self.config:raise ValueError('bank schema/config/epoch')
        if not 0<=state['cursor']<=len(self.config['sources']) or len(state['parts'])!=state['cursor'] or state['patch_count']>self.config['max_patches']:raise ValueError('bank finite cursor/cap')
        if [p['source_index'] for p in state['parts']]!=list(range(state['cursor'])) or sum(len(p['hidden']) for p in state['parts'])!=state['patch_count'] or sum(int(p['weak_critical_mask'].sum()) for p in state['parts'])!=state['weak_count']:raise ValueError('partial bank exact source/patch cursor')
        for i,p in enumerate(state['parts']):
            identity=self.config['sources'][i]
            if p['source_id']!=identity['id'] or p['source_SHA']!=identity['SHA'] or p['hidden'].dtype!=torch.float32 or p['hidden'].shape!=(len(p['source_patch_indices']),512) or len(set(p['source_patch_indices']))!=len(p['source_patch_indices']):raise ValueError('bank immutable prefix/source identity')
            if any(p[k].shape!=(len(p['hidden']),16) for k in ('labels','critical_mask','teacher_correct','teacher_margin','weak_critical_mask')):raise ValueError('bank saved loss-only tensor shape')
        if state['calibration'] is not None and state['cursor']!=len(self.config['sources']):raise ValueError('calibrated partial source pass')
        if state['calibration'] is not None:
            c=state['calibration'];labels=torch.cat([p['labels'] for p in state['parts']]);crit=torch.cat([p['critical_mask'] for p in state['parts']]);weak=torch.cat([p['weak_critical_mask'] for p in state['parts']]);teacher=torch.cat([p['teacher_correct'] for p in state['parts']]);valid=labels!=257
            if (c['precision_policy']!=POLICY or c['initial_margin'].dtype!=torch.float32 or not torch.isfinite(c['initial_margin']).all()
                or not torch.equal(c['labels'],labels) or not torch.equal(c['critical_mask'],crit) or not torch.equal(c['top1_mask'],(c['initial_argmax'].long()==labels)&valid)
                or not torch.equal(c['top1_mask'],teacher) or not torch.equal(c['weak_floor_mask'],weak&teacher)
                or not torch.equal(c['required_margin'],c['initial_margin'].clamp(min=0,max=self.config['floor_cap'])*self.config['fraction'])
                or c['teacher_correct_disagreements']!=0 or not 0<=c['max_row_batch_margin_difference']<=1e-3):raise ValueError('saved reference mask/margin/precision contract')
        value=copy.deepcopy(state)
        self.cursor=value['cursor'];self.parts=value['parts'];self.patch_count=value['patch_count'];self.weak_count=value['weak_count'];self.calibration=value['calibration']
