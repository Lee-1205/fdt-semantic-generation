"""V79 changes optimizer scope ONLY; V78 loss and five-row grouping unchanged."""
from research.competing_margin_objective_v78 import (
    decoder_losses,gold_margin,GROUP_ROW_WEIGHTS,MARGIN_COEFFICIENTS,
    frozen_objective_report as previous_report,
)

OBJECTIVE='V78_same_loss_same_groups+frozen_readout_features+last_Linear2_ONLY'
LAST_LINEAR_NAMES=('readout.readout.net.2.weight','readout.readout.net.2.bias')


def freeze_to_last_linear(model):
    model.requires_grad_(False);model.eval();model.readout.readout.net[2].requires_grad_(True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if tuple(names)!=LAST_LINEAR_NAMES:raise ValueError('exact last linear weight/bias optimizer scope')
    return names


def frozen_objective_report(current,profile,group_steps=384):
    report=previous_report(current,profile,group_steps)
    report.update(objective=OBJECTIVE,trainable_scope=list(LAST_LINEAR_NAMES),
        frozen_readout_features='terminal/LayerNorm/firstLinear/SiLU and all encoder-memory-binding weights fixed',
        experiment='V78 loss/teacher-margin eligibility/coefficients/groups/LR/exposures kept; only trainability scope changes')
    return report
