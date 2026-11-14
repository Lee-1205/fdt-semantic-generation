"""Prospective optimizer-scope isolation; reuse data, loss, groups, and LR."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.last_linear_objective_v79 import LAST_LINEAR_NAMES,decoder_losses
from research.competing_margin_objective_v78 import decoder_losses as previous_loss


def main():
    d=ROOT/'research/artifacts/last_linear_20261113_v79_cpu_v1';assert not (d/'manifest.json').exists()
    previous=io.load_json(ROOT/'runs/competing_margin_20261113_v78/plan.json')
    selected=io.load_json(ROOT/'configs/research_base_selection_v78_review_20261113.json');assert selected['selected_version']=='V75'
    source=torch.load(selected['selected_runtime'],map_location='cpu',weights_only=True)
    stored=source['active_names'];active={k:source['active_weights'][k] for k in LAST_LINEAR_NAMES};prefix={k:v for k,v in source['active_weights'].items() if k not in active}
    assert len(active)==2 and len(prefix)==5 and sum(v.numel() for v in active.values())==2117664 and decoder_losses is previous_loss
    assert support.sha(selected['selected_runtime'])==selected['selected_SHA'] and not torch.cuda.is_initialized()
    refs={key:previous[key] for key in ('data','balance_data','symbol_profile_data','eval_data','old_eval_data')}
    report=dict(schema='V79_SCOPE_ISOLATION_CPU',new_TRAIN_rows=0,new_eval_rows=0,previous_V78_objective_math_unchanged=True,
        previous_V78_teacher_weak_eligibility_and_coefficients_unchanged=True,previous_groups384_head_rows1920_LR1e5_seed77113_unchanged=True,
        active_optimizer_scope=dict(names=list(LAST_LINEAR_NAMES),tensors=2,parameters=sum(v.numel() for v in active.values())),
        storage_delta_scope=dict(names=stored,tensors=7,parameters=sum(v.numel() for v in source['active_weights'].values()),frozen_overrides=list(prefix),
            frozen_prefix_SHA=support.state_sha(prefix),legacy_envelope_active_names_means_stored_deltas_NOT_optimizer=True),
        frozen_scope='all encoder/memory/binding plus readout terminal/LayerNorm/firstLinear; two finalLinear tensors only trainable',
        input_paths=refs,input_SHA={str(Path(v)/name):support.sha(Path(v)/name) for key,v,name in (
            ('data',previous['data'],'train_features.json'),('data',previous['data'],'train_targets.json'),('balance_data',previous['balance_data'],'orders.json'),
            ('symbol_profile_data',previous['symbol_profile_data'],'symbol_weight_source.json'),('eval_data',previous['eval_data'],'features.json'),('eval_data',previous['eval_data'],'targets.json'))},
        source_SHA=selected['selected_SHA'],mandatory_standalone_base_SHA=selected['mandatory_base_SHA'],
        checkpoint_contract='direct standalone V74 + seven readout storage deltas, of which five frozen V75 prefix values; full runtime owns ONLY2Adam; no recursive parent or standalone copy',
        actual_worker_preflight_or_GPU_so_far=False,denoise0=True,Exact_content0=True,copy0=True,
        limits='Current language repair only. Future diffusion still freezes whole codec and trains separate thought module only. Changing trainability changes gradient clipping/Adam subspace too; no claim of individual prefix-component causality.')
    io.atomic_json(d/'manifest.json',report);io.atomic_json(d/'builder_SHA.json',dict(file=str(Path(__file__)),SHA=support.sha(Path(__file__))))
    print(json.dumps(report))


if __name__=='__main__':main()
