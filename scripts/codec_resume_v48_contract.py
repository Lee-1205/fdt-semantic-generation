"""V48's shared limits and immutable input validation. No Torch imports."""
import json
import shutil
from pathlib import Path

from supervise_querycount_partial9_20261027_v3 import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/three_memory_codec_20261111_v48'
PARENT = ROOT / 'runs/three_memory_codec_20261111_v47'
SOURCE = PARENT / 'model/checkpoints/state_2048.pt'
TRAIN_DATA = ROOT / 'research/artifacts/codec_boundary_repair_20261111_v47'
ORDERS = ROOT / 'research/artifacts/three_memory_codec_20261111_v45/orders.json'
ARTIFACT = ROOT / 'research/artifacts/three_memory_codec_20261111_v48'
SOURCE_SHA = '2c79076246164ec80c01b68b8d711f5fe0759cc1342159fb2423380523d320cf'
SOURCE_PLAN = '6bd38dc13d5972d2f5bce93a3bda5145dcd93fdf5fda9c1cc554508a9c7bf67e'
SOURCE_ALGORITHM = 'b66abf560e15237defdcb4d8a00f75c654d96070b2d31638b644a304a6f69f5a'
GLOBAL_ATTEMPTS, GLOBAL_UPDATES = 169956, 169893
SCHEMA = 'fdt_three_memory_language_codec_v48'
GIB = 1024**3
PROTECTED = 100 * GIB
STOP_FREE = PROTECTED + GIB
SEED = 45111
FRESH = PRIOR = 4096
ORIGINAL = 8192
REPLAY = PRIOR+ORIGINAL
LIMITS = dict(attempts=(FRESH + REPLAY)//8, batch=8, wall=900, reserve=120,
              max_calls=10800, max_output_bytes=GIB, start_free_bytes=110*GIB,
              min_gpu_free_mib=12288)
COST = dict(training=3*LIMITS['attempts'], focused_final_teacher=32,
            focused_final_BOS_max=2080, held_final_teacher=128,
            held_final_BOS_max=2080, held16_three_memory_ablations=12, repaired_BOS_rows_max=130)
COST['total_max'] = sum(COST.values())
STORAGE = dict(protected_free_bytes=PROTECTED, cooperative_stop_free_bytes=STOP_FREE,
               maximum_owned_output_bytes=GIB, minimum_start_bytes=LIMITS['start_free_bytes'])
POLICY = dict(purpose='eight_ordered_contextual_summaries_alongside_global_commit', new_parameters=4098,
              local_slots=8, input_bytes_per_local_slot=16,
              early_bytes=16, early_weight=4.0, utf8_guard=True, all_populated_Adam_states=92, old_allocation_reopened=False,
              learning_rate_schedule='constant_source_terminal_group_LRs',
              fresh_spans=FRESH, prior_replay_spans=PRIOR, original_replay_spans=ORIGINAL, seed=SEED, new_finite_epochs=1)
STATUS = 'THREE_MEMORY_CODEC_V48_COMPLETE'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate(path, output, invoked_script):
    plan = read(path)
    required = dict(schema=SCHEMA, root=str(ROOT), output=str(RUN), source=str(SOURCE),
                    source_sha=SOURCE_SHA, source_plan_sha=SOURCE_PLAN,
                    source_algorithm=SOURCE_ALGORITHM, source_global_attempts=GLOBAL_ATTEMPTS,
                    source_global_updates=GLOBAL_UPDATES, limits=LIMITS, cost_contract=COST,
                    storage_contract=STORAGE, continuation_policy=POLICY,
                    reasoning_diffusion_training=False, automatic_source_copy=False,
                    E2_final_accessed=False, generation_indices=[4*i for i in range(16)], full_BOS_indices=[4*i for i in range(64)],
                    training_artifact=str(TRAIN_DATA))
    for name, expected in required.items():
        if plan.get(name) != expected:
            raise ValueError('V48 plan field mismatch: ' + name)
    if Path(output).resolve() != RUN or type(plan['authorized']) is not bool:
        raise ValueError('V48 output or authorization identity')
    clock = tuple(plan[key] for key in ('allocation_start', 'cutoff', 'deadline'))
    if plan['authorized']:
        start, cutoff, deadline = clock
        if deadline-start != LIMITS['wall'] or deadline-cutoff != LIMITS['reserve']:
            raise ValueError('V48 fixed clock')
    elif any(value is not None for value in clock):
        raise ValueError('unsealed plan has a clock')
    for file, digest in plan['files'].items():
        if sha256(Path(file)) != digest:
            raise ValueError('input pin mismatch: ' + file)
    mandatory = (SOURCE, PARENT/'plan.json', TRAIN_DATA/'train_features.json', TRAIN_DATA/'train_targets.json',
                 ORDERS, TRAIN_DATA/'manifest.json', TRAIN_DATA/'independent_review.json', Path(invoked_script).resolve())
    if any(str(file) not in plan['files'] for file in mandatory):
        raise ValueError('V48 required input pin missing')
    if plan['files'][str(SOURCE)] != SOURCE_SHA or plan['files'][str(PARENT/'plan.json')] != SOURCE_PLAN:
        raise ValueError('V47 terminal identity')
    return plan
