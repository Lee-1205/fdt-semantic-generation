"""Language repair only: same V78 losses, new TRAIN coverage and actual-delta guard."""
from research.last_linear_objective_v79 import decoder_losses,gold_margin,GROUP_ROW_WEIGHTS,freeze_to_last_linear,LAST_LINEAR_NAMES
from research.last_linear_objective_v79 import frozen_objective_report as previous_report
OBJECTIVE='V80_numeric_coverage_quartets+V78_loss+lastLinear2_actual_Adam_delta_TRAIN_guard'


def frozen_objective_report(current,profile,group_steps=384):
    report=previous_report(current,profile,group_steps)
    report.pop('optimizer_steps',None)
    report.update(objective=OBJECTIVE,optimizer_attempts=group_steps,optimizer_steps_max=group_steps,accepted_steps_can_be_fewer=True,
        teacher_cache='loss-only teacher packets; separate selected TRAIN frozen-prefix reference features; head source and inference reencoded live',
        group_gradient='sum(.2*row_loss gradients) at SAME parameters, one unscale/clip/Adam PROPOSAL, bounded actual-delta acceptance or exact Adam rollback',
        group_order='authored quartet4 + OWT1',
        actual_update_guard='fixed TRAIN-only initial-reference-correct nonPAD top1; actual Adam delta alphas1,.5,.25,.125; all rejected consumes source/attempt/cursor/RNG/calls/scaler but restores weights/Adam',
        experiment='TRAIN content, group order and actual-update safeguard change jointly; no isolated causal attribution')
    return report
