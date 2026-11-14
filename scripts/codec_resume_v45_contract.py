"""V45's shared limits and immutable input validation. No Torch imports."""
import json
import shutil
from pathlib import Path

from supervise_querycount_partial9_20261027_v3 import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/three_memory_codec_20261111_v45'
PARENT = ROOT / 'runs/three_memory_codec_20261111_v44'
SOURCE = PARENT / 'model/checkpoints/state_2048.pt'
ARTIFACT = ROOT / 'research/artifacts/three_memory_codec_20261111_v45'
SOURCE_SHA = '9e7eb0c686da7d82d2ccfb3957ac5fd34d3fd6309f420118684e0cd295213021'
SOURCE_PLAN = 'e559f3b49e9cf458dcd6e651c80dd798f2ff9b5a5732f771305efad326d40527'
SOURCE_ALGORITHM = '8973919382c4a13eca3053ab6c47ceed5d6b77c76092033f801f1f6aeb2fe929'
GLOBAL_ATTEMPTS, GLOBAL_UPDATES = 163812, 163751
SCHEMA = 'fdt_three_memory_language_codec_v45'
GIB = 1024**3
PROTECTED = 100 * GIB
STOP_FREE = PROTECTED + GIB
SEED = 45111
FRESH = PRIOR = 4096
ORIGINAL = 8192
REPLAY = PRIOR+ORIGINAL
LIMITS = dict(attempts=(FRESH + REPLAY)//8, batch=8, wall=900, reserve=120,
              max_calls=9000, max_output_bytes=GIB, start_free_bytes=110*GIB,
              min_gpu_free_mib=12288)
COST = dict(training=3*LIMITS['attempts'], focused_final_teacher=32,
            focused_final_BOS_max=2080, held_final_teacher=128,
            held_final_BOS_max=520, held16_three_memory_ablations=12)
COST['total_max'] = sum(COST.values())
STORAGE = dict(protected_free_bytes=PROTECTED, cooperative_stop_free_bytes=STOP_FREE,
               maximum_owned_output_bytes=GIB, minimum_start_bytes=LIMITS['start_free_bytes'])
POLICY = dict(purpose='fresh_train_sentence_spans_with_replay', new_parameters=0,
              all_populated_Adam_states=91, old_allocation_reopened=False,
              learning_rate_schedule='constant_source_terminal_group_LRs',
              fresh_spans=FRESH, prior_replay_spans=PRIOR, original_replay_spans=ORIGINAL, seed=SEED, new_finite_epochs=1)
STATUS = 'THREE_MEMORY_CODEC_V45_COMPLETE'


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
                    E2_final_accessed=False, generation_indices=[4*i for i in range(16)])
    for name, expected in required.items():
        if plan.get(name) != expected:
            raise ValueError('V45 plan field mismatch: ' + name)
    if Path(output).resolve() != RUN or type(plan['authorized']) is not bool:
        raise ValueError('V45 output or authorization identity')
    clock = tuple(plan[key] for key in ('allocation_start', 'cutoff', 'deadline'))
    if plan['authorized']:
        start, cutoff, deadline = clock
        if deadline-start != LIMITS['wall'] or deadline-cutoff != LIMITS['reserve']:
            raise ValueError('V45 fixed clock')
    elif any(value is not None for value in clock):
        raise ValueError('unsealed plan has a clock')
    for file, digest in plan['files'].items():
        if sha256(Path(file)) != digest:
            raise ValueError('input pin mismatch: ' + file)
    mandatory = (SOURCE, PARENT/'plan.json', ARTIFACT/'features.json', ARTIFACT/'targets.json',
                 ARTIFACT/'orders.json', ARTIFACT/'manifest.json', Path(invoked_script).resolve())
    if any(str(file) not in plan['files'] for file in mandatory):
        raise ValueError('V45 required input pin missing')
    if plan['files'][str(SOURCE)] != SOURCE_SHA or plan['files'][str(PARENT/'plan.json')] != SOURCE_PLAN:
        raise ValueError('V44 terminal identity')
    return plan
