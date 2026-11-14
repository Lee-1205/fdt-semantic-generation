"""V46's shared limits and immutable input validation. No Torch imports."""
import json
import shutil
from pathlib import Path

from supervise_querycount_partial9_20261027_v3 import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/three_memory_codec_20261111_v46'
PARENT = ROOT / 'runs/three_memory_codec_20261111_v45'
SOURCE = PARENT / 'model/checkpoints/state_2048.pt'
TRAIN_DATA = ROOT / 'research/artifacts/three_memory_codec_20261111_v45'
ARTIFACT = ROOT / 'research/artifacts/three_memory_codec_20261111_v46'
SOURCE_SHA = 'c34b45e074517192a531dfc2b942559e8c87bae6461baf4b0bc735b30f917740'
SOURCE_PLAN = 'ade4de9dcb99bd725d56b31851cede37aa20afc5298c1e86672bb8c2e0a77c9a'
SOURCE_ALGORITHM = 'a5032d47ddef81099466288b52b53622eadecb326bd7c8a5c445308a0d94d9bd'
GLOBAL_ATTEMPTS, GLOBAL_UPDATES = 165860, 165798
SCHEMA = 'fdt_three_memory_language_codec_v46'
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
            held_final_BOS_max=2080, held16_three_memory_ablations=12)
COST['total_max'] = sum(COST.values())
STORAGE = dict(protected_free_bytes=PROTECTED, cooperative_stop_free_bytes=STOP_FREE,
               maximum_owned_output_bytes=GIB, minimum_start_bytes=LIMITS['start_free_bytes'])
POLICY = dict(purpose='position_sensitive_contextual_reads_and_decoder_slots', new_parameters=1026,
              all_populated_Adam_states=91, old_allocation_reopened=False,
              learning_rate_schedule='constant_source_terminal_group_LRs',
              fresh_spans=FRESH, prior_replay_spans=PRIOR, original_replay_spans=ORIGINAL, seed=SEED, new_finite_epochs=1)
STATUS = 'THREE_MEMORY_CODEC_V46_COMPLETE'


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
            raise ValueError('V46 plan field mismatch: ' + name)
    if Path(output).resolve() != RUN or type(plan['authorized']) is not bool:
        raise ValueError('V46 output or authorization identity')
    clock = tuple(plan[key] for key in ('allocation_start', 'cutoff', 'deadline'))
    if plan['authorized']:
        start, cutoff, deadline = clock
        if deadline-start != LIMITS['wall'] or deadline-cutoff != LIMITS['reserve']:
            raise ValueError('V46 fixed clock')
    elif any(value is not None for value in clock):
        raise ValueError('unsealed plan has a clock')
    for file, digest in plan['files'].items():
        if sha256(Path(file)) != digest:
            raise ValueError('input pin mismatch: ' + file)
    mandatory = (SOURCE, PARENT/'plan.json', TRAIN_DATA/'features.json', TRAIN_DATA/'targets.json',
                 TRAIN_DATA/'orders.json', TRAIN_DATA/'manifest.json', Path(invoked_script).resolve())
    if any(str(file) not in plan['files'] for file in mandatory):
        raise ValueError('V46 required input pin missing')
    if plan['files'][str(SOURCE)] != SOURCE_SHA or plan['files'][str(PARENT/'plan.json')] != SOURCE_PLAN:
        raise ValueError('V45 terminal identity')
    return plan
