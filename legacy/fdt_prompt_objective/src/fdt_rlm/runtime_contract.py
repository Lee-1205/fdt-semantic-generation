"""Declared runtime limits; capability validation is reported separately."""
def runtime_capabilities(config):
    alpha = float(getattr(config, 'rope_transition_alpha', 1.0))
    legacy = int(getattr(config, 'legacy_position_transition_max_len', 0))
    scale = float(getattr(config, 'legacy_position_scale', -1.0))
    if scale < 0:
        scale = 1.0-alpha
    active_legacy = config.model_type == 'fdt_v4' and legacy > 0 and scale > 0
    maximum = min(config.max_seq_len, legacy) if active_legacy else config.max_seq_len
    cached = not (config.model_type == 'fdt_v4' and getattr(config,'rope_transition_mode','lerp') == 'output_blend' and 0 < alpha < 1)
    return {'full_forward_limit': maximum, 'full_recompute_limit': maximum,
            'incremental_limit': maximum if cached else None,
            'incremental_status': 'SUPPORTED_BY_CODE_UNVERIFIED' if cached else 'UNSUPPORTED',
            'legacy_table_required': active_legacy or config.model_type == 'fdt_v3',
            'rope_active': config.model_type == 'fdt_v4' and alpha > 0,
            'validation_status': 'UNVERIFIED'}


def validate_generation_budget(config, prompt_length, max_new_tokens):
    limit=runtime_capabilities(config)['full_recompute_limit']
    if prompt_length < 1 or max_new_tokens < 1 or prompt_length+max_new_tokens > limit:
        raise ValueError(f'prompt + generation budget exceeds effective context {limit}')
    return limit
