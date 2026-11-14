"""Create V44 entry points from the previously reviewed worker and OS owner.

Keep the learning and Windows cleanup loops intact except for the new
allocation, corpus, storage reserve, and reporting fields.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
WORKER_NAME = 'run_three_memory_codec_20261111_v44.py'
OWNER_NAME = 'supervise_three_memory_codec_20261111_v44.py'

WORKER_HEADER = '''"""V44 same-input codec training on new TRAIN spans and replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, ctypes, json, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.three_memory_codec_v33 import ThreeMemoryCodec
from run_three_memory_codec_20261029_v33 import (
    CORPUS_MANIFEST, FOCUSED_MANIFEST, teacher_assessment, reconstruct_inputs,
)
from three_memory_codec_engine_v44 import CodecBudget, FreshSpanCodecEngine
from codec_resume_v44_contract import (
    ARTIFACT, FRESH, REPLAY, GLOBAL_ATTEMPTS, GLOBAL_UPDATES, LIMITS,
    SOURCE_ALGORITHM, SOURCE_PLAN, SOURCE_SHA, STATUS, validate as validate_pins,
)
import codec_initial_reference_v42_gzip as checkpoint_storage
import codec_single_snapshot_v42 as snapshot_storage
import text_generation_support_v1 as support
import learning_io_v2 as io


def validate(path, output, worker=None):
    plan = validate_pins(path, output, worker or __file__)
    if support.core_sha(ROOT) != support.CORE_SHA:
        raise ValueError('original reasoning core changed')
    corpus, focus = Path(plan['artifact']), Path(plan['focused_artifact'])
    if support.sha(corpus/'manifest.json') != CORPUS_MANIFEST or support.sha(focus/'manifest.json') != FOCUSED_MANIFEST:
        raise ValueError('original evaluation panel changed')
    features = {'train': io.load_json(ARTIFACT/'features.json'),
                'dev': io.load_json(corpus/'features.json')['dev'],
                'focused': io.load_json(focus/'features.json')}
    targets = {'train': io.load_json(ARTIFACT/'targets.json'),
               'dev': io.load_json(corpus/'targets.json')['dev'],
               'focused': io.load_json(focus/'targets.json')}
    orders = io.load_json(ARTIFACT/'orders.json')
    if sorted(i for batch in orders for i in batch) != list(range(FRESH+REPLAY)):
        raise ValueError('finite row coverage')
    if len(orders) != LIMITS['attempts'] or any(len(batch) != 8 or sum(i < FRESH for i in batch) != 4 for batch in orders):
        raise ValueError('exact balanced minibatches')
    for name, count in [('train', FRESH+REPLAY), ('dev', 256), ('focused', 64)]:
        if len(features[name]) != count or len(targets[name]) != count:
            raise ValueError('actual row count for '+name)
        for x, y in zip(features[name], targets[name]):
            if (set(x) != {'id','prompt'} or x['id'] != y['id'] or x['prompt'] != y['target']
                    or y['target_ids'] != list(x['prompt'].encode())+[256]
                    or not 1 <= len(x['prompt'].encode()) <= 128):
                raise ValueError('SAME-input feature/target contract')
    source = io.load_json(Path(plan['source_plan_path']))
    if plan['external_closed_assessment_calls'] != source['external_closed_assessment_calls']:
        raise ValueError('closed call ledger')
    return plan, features, targets, orders


'''
OWNER_HEADER = '''"""V44 OS-only owner with the user-requested C: free-space reserve."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from filelock import FileLock

import learning_io_v2 as io
import supervise_querycount_partial9_20261027_v3 as native
from codec_resume_v44_contract import (
    ROOT, LIMITS, STATUS, STOP_FREE, validate as validate_pins,
)


def validate(plan_path, output):
    plan = validate_pins(plan_path, output, __file__)
    if output.exists():
        allowed = {'plan.json', 'readiness.json', 'code_snapshot', 'supervisor.stdout.log',
                   'supervisor.stderr.log', 'supervisor'}
        if {path.name for path in output.iterdir()} - allowed:
            raise FileExistsError('V44 execution output already exists')
    return plan


'''


def change(text, before, after):
    if text.count(before) != 1:
        raise ValueError('expected one source match: '+before)
    return text.replace(before, after)


def write(name, text):
    with (SCRIPTS/name).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    old_worker = (SCRIPTS/'run_three_memory_codec_20261030_v43.py').read_text(encoding='utf-8')
    body = old_worker[old_worker.index('def load_import('):]
    body = change(body, 'TwoEpochCodecEngine(model', 'FreshSpanCodecEngine(model')
    body = change(body, 'data_revision=CORPUS_MANIFEST)', "data_revision=support.sha(ARTIFACT/'manifest.json'))")
    body = change(body, 'engine.import_completed_v42(state,SOURCE_PLAN,SOURCE_SHA)', 'engine.import_completed_v43(state, SOURCE_PLAN, SOURCE_SHA, CORPUS_MANIFEST)')
    body = change(body, 'torch.manual_seed(843)', 'torch.manual_seed(844)')
    body = body.replace('RecoveryBudget(p)', 'CodecBudget(p)')
    body = change(body, "snapshot_storage.check_checkpoint_write(budget,(32 if kind=='import' else 800)*1024**2,emergency=kind=='emergency')",
                  "budget.check_write((32 if kind=='import' else 800)*1024**2, emergency=kind=='emergency')")
    body = change(body, '# Saved V42 final is the unchanged-function baseline; no initial GPU repeat.',
                  '# The V43 results give the baseline on the same assessment panels.')
    body = change(body, 'two_epoch_broad_codec_training', 'fresh_train_span_codec_training')
    body = change(body, "'status':'THREE_MEMORY_CODEC_V43_COMPLETE'", "'status':STATUS")
    body = change(body, '153572+engine.attempts', 'GLOBAL_ATTEMPTS+engine.attempts')
    body = change(body, '153515+engine.updates', 'GLOBAL_UPDATES+engine.updates')
    body = change(body, "'elapsed':time.time()-began,'latest':latest", "'elapsed':time.time()-began,'minimum_C_free_bytes':budget.minimum_free_bytes,'latest':latest")
    body = change(body, "'seconds':time.time()-began,'resource_scans'", "'seconds':time.time()-began,'minimum_C_free_bytes':budget.minimum_free_bytes,'resource_scans'")
    body = change(body, "TWO finite broad TRAIN epochs, same three-memory codec and ALL91Adam; input reconstruction only, no full vNext/general perfection claim.",
                  'One finite balanced TRAIN pass with fresh sentence spans and replay. Same-input reconstruction.')
    write(WORKER_NAME, WORKER_HEADER+body)

    old_owner = (SCRIPTS/'supervise_three_memory_codec_20261030_v43.py').read_text(encoding='utf-8')
    resource_helper = old_owner[old_owner.index('def owned_bytes('):old_owner.index('def validate(')]
    body = old_owner[old_owner.index('def preflight('):]
    body = change(body, "new C30GiB=20protected+1output+1temporary+8headroom start floor",
                  'C: free space below the 110GiB launch reserve')
    body = change(body, "if shutil.disk_usage(ROOT).free < 22*1024**3: reason = 'disk reserve'; break",
                  "if shutil.disk_usage(ROOT).free < STOP_FREE: reason = '100GiB protected disk reserve'; break")
    body = change(body, "'THREE_MEMORY_CODEC_V43_COMPLETE'", 'STATUS')
    write(OWNER_NAME, OWNER_HEADER+resource_helper+body)


if __name__ == '__main__':
    main()
