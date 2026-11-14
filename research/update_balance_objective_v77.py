"""V77 uses unchanged V76 capped losses with frozen V75 TRAIN symbol weights."""
import copy
from research.critical_retention_objective_v76 import decoder_losses,COEFFICIENTS,MAX_POSITION_AMPLIFICATION
OBJECTIVE='V76_capped_correction+frozen_V75_TRAIN_symbol_weights+finite40_40_20_head_order'

def frozen_objective_report(current_report,profile):
    previous=profile['parent_objective_report'];report=copy.deepcopy(current_report)
    report['current_unique_pool_class_weights_NOT_used']=list(current_report['class_weights'])
    for key in ('mean_per_window_byte_mass','class_weights','EOS_weight','PAD_weight','normalization','TRAIN_only'):
        report[key]=copy.deepcopy(previous[key])
    for key in ('critical_CE_coefficient','critical_masks_TRAIN_only','anchor_coefficient'):report.pop(key,None)
    report.update(objective=OBJECTIVE,coefficients=COEFFICIENTS,decoder_LR=1e-5,
        critical_groups_independently_normalized=True,wrong_critical_max_position_amplification=8.,
        wrong_critical_denominator='max(W,C/8)',teacher_cache='loss-only; no feature cache; live frozen source reencoded for head',
        base_symbol_weight_policy='frozen_V75_TRAIN_profile',base_symbol_weight_source_SHA=profile['source_SHA'])
    return report
