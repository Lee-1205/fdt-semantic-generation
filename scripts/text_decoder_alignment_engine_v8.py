"""Decoder reconstruction then joint VP/paired-realizer finite curriculum."""
import copy
import torch
from text_language_engine_v7 import CurriculumEngine
from text_decoder_alignment_step_v8 import StepMixin
import text_generation_support_v1 as support


class AlignmentEngine(StepMixin,CurriculumEngine):
    def __init__(self,model,features,targets,negatives,budget,plan_sha,device='cuda',seeds=(71,72,73,74),reconstruction_epochs=4):
        self.permanent_names=[n for n,p in model.named_parameters() if not p.requires_grad]
        super().__init__(model,features,targets,budget,plan_sha,device,seeds)
        if len(negatives)!=len(features) or [r['id'] for r in negatives]!=[r['id'] for r in features]:raise ValueError('negative row identity')
        if not 1<=reconstruction_epochs<=len(seeds):raise ValueError('finite reconstruction epochs')
        self.negatives=negatives;self.reconstruct_limit=len(self.batches)//len(seeds)*reconstruction_epochs
        self.phase='reconstruction';self.model.set_phase(self.phase)
        self.phase_vp_sha=support.state_sha(self.model.vp.state_dict())

    def frozen_state(self):
        permanent=set(self.permanent_names)
        values={n:p for n,p in self.model.named_parameters() if n in permanent}
        values.update(dict(self.model.named_buffers()));return values

    def prepare_phase(self):
        phase='reconstruction' if self.attempts<self.reconstruct_limit else 'joint'
        if phase!=self.phase:
            if support.state_sha(self.model.vp.state_dict())!=self.phase_vp_sha:raise ValueError('VP changed during decoder-only reconstruction')
            self.phase=phase;self.model.set_phase(phase)
        for i,group in enumerate(self.optimizer.param_groups):group['base_lr']=1e-5 if i==0 else 3e-5

    def verify_frozen(self):
        super().verify_frozen()
        if getattr(self,'phase',None)=='reconstruction' and hasattr(self,'phase_vp_sha'):
            if support.state_sha(self.model.vp.state_dict())!=self.phase_vp_sha:raise ValueError('frozen VP changed')

    def negatives_for(self,indices):
        ids=torch.tensor([self.negatives[i]['negative_ids'] for i in indices],device=self.device,dtype=torch.long)
        spans=[(j,a,b) for j,i in enumerate(indices) for a,b in self.negatives[i]['negative_spans']]
        return ids,spans

    def export(self):
        value=super().export();value.update(schema='fdt_text_decoder_alignment_v8_fullstate',phase=self.phase,
            reconstruct_attempt_limit=self.reconstruct_limit,permanent_names=self.permanent_names,phase_vp_sha=self.phase_vp_sha)
        return value

    def restore(self,state):
        if state['schema']!='fdt_text_decoder_alignment_v8_fullstate' or state['reconstruct_attempt_limit']!=self.reconstruct_limit or state['permanent_names']!=self.permanent_names:raise ValueError('phase schema/ownership')
        self.phase=state['phase'];self.model.set_phase(self.phase);self.phase_vp_sha=state['phase_vp_sha']
        adapted=dict(state);adapted['schema']='fdt_text_language_v7_fullstate';super().restore(adapted)

    def import_parent(self,state,source_plan):
        if state['plan_sha256']!=source_plan or state['gradients']:raise ValueError('source identity/gradient')
        self.model.import_weights(state['model'])
        current=self.optimizer.state_dict();new_state={};groups=[]
        for gi,key in enumerate(('language','other')):
            old_names=state['optimizer_parameter_names'][key];new_names=self.parameter_names[key]
            expected=old_names+(['decoder.memory_positions.weight'] if key=='other' else [])
            if new_names!=expected:raise ValueError('only new memory parameter may be appended')
            old_group=state['optimizer']['param_groups'][gi];new_ids=current['param_groups'][gi]['params']
            for name,old_id in zip(old_names,old_group['params']):
                if old_id in state['optimizer']['state']:new_state[new_ids[new_names.index(name)]]=copy.deepcopy(state['optimizer']['state'][old_id])
            group=copy.deepcopy(old_group);group['params']=new_ids;groups.append(group)
        self.optimizer.load_state_dict({'state':new_state,'param_groups':groups});self.scaler.load_state_dict(state['scaler'])
        self.frozen_sha=state['frozen_sha'];self.budget.calls=dict(state['actual_calls']);self.restore_rng(state)
        self.phase_vp_sha=support.state_sha(self.model.vp.state_dict())
        self.verify_frozen()
        def components(model,optimizer,names,meta):
            moments={}
            for gi,key in enumerate(('language','other')):
                for name,index in zip(names[key],optimizer['param_groups'][gi]['params']):
                    if name in state['optimizer_parameter_names'][key] and index in optimizer['state']:moments[name]=optimizer['state'][index]
            return {'model':{k:model[k] for k in state['model']},'moments':moments,
                'groups':[{k:v for k,v in g.items() if k!='params'} for g in optimizer['param_groups']],
                **{k:meta[k] for k in ('scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')}}
        before=support.state_sha(components(state['model'],state['optimizer'],state['optimizer_parameter_names'],state))
        after=self.export()
        if before!=support.state_sha(components(after['model'],after['optimizer'],after['optimizer_parameter_names'],after)):raise ValueError('old model/moments/scaler/allRNG not preserved')
        self.ancestry={k:copy.deepcopy(state[k]) for k in ('plan_sha256','attempts','updates','cursor','history','exposure','order','evaluations','actual_calls','scheduler','ancestry') if k in state}
        self.ancestry['old_component_sha256']=before
        return before
