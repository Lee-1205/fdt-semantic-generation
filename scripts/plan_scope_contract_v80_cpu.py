"""Prospective scope guard; rejects inherited contradictory plan metadata.

This is not an execution plan, worker, or permission boundary. Ownership and
resource claims must be derived from the actual active tensors and cost table.
"""


def validate_scope(plan, active_weights):
    obsolete = [k for k in plan if k.startswith('output_estimate_including')]
    if obsolete or 'actual_trainable_parameters' in plan:
        raise ValueError('obsolete inherited scope/estimate fields must be removed explicitly')
    required = {'tensors', 'parameters', 'names'}
    scope = plan['model_scope']
    if set(scope) != required:
        raise ValueError('explicit model scope keys')
    if scope != dict(tensors=len(active_weights), parameters=sum(t.numel() for t in active_weights.values()), names=sorted(active_weights)):
        raise ValueError('scope must equal actual optimizer-owned tensors')
    estimates = plan['resource_estimate']
    if set(estimates) != {'components', 'total_bytes'} or not estimates['components']:
        raise ValueError('one unambiguous output estimate')
    values = list(estimates['components'].values())
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('resource bytes must be nonnegative integers')
    if sum(values) != estimates['total_bytes'] or estimates['total_bytes'] >= plan['max_output_bytes']:
        raise ValueError('estimated owned output must fit fixed hard cap')
    if plan['start_free_bytes'] < plan['protected_free_bytes'] + plan['max_output_bytes'] + 8 * 1024**3:
        raise ValueError('100GiB protection plus hard output including STOP/temp plus 8GiB headroom')
    detail = plan['cost_detail']
    if sum(detail[k] for k in ('encode_total', 'readout_total', 'stream_chunk_total', 'reference_linear_total')) != plan['cost'] or plan['cost'] > plan['max_calls']:
        raise ValueError('one exact encode/readout/chunk call total')
    return True
