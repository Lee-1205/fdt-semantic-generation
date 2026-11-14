"""Fixed final language readout, live fuzzy extraction and role binding."""
from research.fuzzy_roundtrip_v64 import enable_cycle_training

def enable_fixed_readout_extraction(model):
    enable_cycle_training(model)
    model.readout.readout.requires_grad_(False)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if len(names)!=23 or 'readout.role_projection.weight' not in names or any(n.startswith('readout.readout.') for n in names):
        raise ValueError('live upstream23, final readout7 fixed')
    return names
