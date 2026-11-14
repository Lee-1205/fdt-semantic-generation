"""Explicit device callback equivalent to the V85 CPU reference criterion."""
import torch
from research.train_margin_retention_v81 import margins
from research.actual_update_retention_v80 import TrainDecision
from research.readout_precision_policy_v81 import POLICY

class DeviceReference:
    def __init__(self,features,calibration,device):
        if calibration['precision_policy']!=POLICY or features['hidden'].dtype!=torch.float32:raise ValueError('fresh fixed FP32 reference required')
        self.hidden=features['hidden'].to(device);self.c={k:v.to(device) if torch.is_tensor(v) else v for k,v in calibration.items()}

    @torch.no_grad()
    def decision(self,z):
        c=self.c;y=c['labels']
        if z.device!=y.device or z.shape!=(*y.shape,258):raise ValueError('fixed reference shape/device')
        gap=margins(z,y);flipped=c['top1_mask']&(z.argmax(-1)!=y);erosion=c['weak_floor_mask']&(gap<c['required_margin']);critical=c['critical_mask']
        return TrainDecision(not bool((flipped|erosion).any()),dict(new_critical_flips=int((flipped&critical).sum()),new_noncritical_EOS_flips=int((flipped&~critical).sum()),
            weak_critical_margin_violations=int(erosion.sum()),protected_top1=int(c['top1_mask'].sum()),protected_weak_critical=int(c['weak_floor_mask'].sum()),bank_patches=len(z)))
