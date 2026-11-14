"""Prepare the next finite codec run, preserving V44's sealed files.

Each batch has four original TRAIN spans, two spans learned in V44,
and two previously unselected TRAIN spans. The additional replay
addresses V44's regression on the original TRAIN reference panel.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'


def substitute(text, old, new):
    if old not in text:
        raise ValueError('missing source fragment: '+old)
    return text.replace(old, new)


def create(source, destination, edits):
    text = (SCRIPTS/source).read_text(encoding='utf-8')
    for old, new in edits:
        text = substitute(text, old, new)
    text = text.replace('v44', 'v45').replace('V44', 'V45')
    for protected in ('@V44@', '@v44@'):
        text = text.replace(protected.replace('44','45'), protected.strip('@'))
    with (SCRIPTS/destination).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    create('codec_resume_v44_contract.py', 'codec_resume_v45_contract.py', [
        ("PARENT = ROOT / 'runs/three_memory_codec_20261030_v43'",
         "PARENT = ROOT / 'runs/three_memory_codec_20261111_@v44@'"),
        ("SOURCE = PARENT / 'model/checkpoints/state_8192.pt'",
         "SOURCE = PARENT / 'model/checkpoints/state_2048.pt'"),
        ('a8cf6db9caf5599ee6118db070e4c6214c8055ba08917c5697ab441025aaaf9f',
         '9e7eb0c686da7d82d2ccfb3957ac5fd34d3fd6309f420118684e0cd295213021'),
        ('aa9cb1ebf2b8a2df1ea884943959173e0775db1d7b35ba719dac04f26cf727eb',
         'e559f3b49e9cf458dcd6e651c80dd798f2ff9b5a5732f771305efad326d40527'),
        ('84e8f47a5189be852c6593ad3efa589f955110f676f9d4cafe28542719bd62fd',
         '8973919382c4a13eca3053ab6c47ceed5d6b77c76092033f801f1f6aeb2fe929'),
        ('161764, 161704', '163812, 163751'),
        ('SEED = 44111\nFRESH = REPLAY = 8192',
         'SEED = 45111\nFRESH = PRIOR = 4096\nORIGINAL = 8192\nREPLAY = PRIOR+ORIGINAL'),
        ('fresh_spans=FRESH, replay_spans=REPLAY, seed=SEED',
         'fresh_spans=FRESH, prior_replay_spans=PRIOR, original_replay_spans=ORIGINAL, seed=SEED'),
        ('V43 terminal identity', '@V44@ terminal identity'),
    ])
    create('prepare_codec_resume_v44.py', 'prepare_codec_resume_v45.py', [
        ("startswith('V43_COMPLETE_') or not previous['all_owned_proxy_gone_before_analysis']",
         "startswith('@V44@_COMPLETE_') or not previous['all_owned_processes_gone']"),
        ("raise ValueError('V43 terminal review required')",
         "raise ValueError('@V44@ terminal review required')"),
        ("    for doc in docs['dev']:",
         "    prior_features = c.read(Path(original['training_artifact'])/'features.json')[:8192]\n"
         "    excluded.update(key(row['prompt']) for row in prior_features)\n"
         "    for doc in docs['dev']:"),
        ("len(docs['train']) != c.REPLAY", "len(docs['train']) != c.ORIGINAL"),
        ("replay = [original_rows['train'][4*i+rng.randrange(4)] for i in range(c.REPLAY)]",
         "old_replay = [original_rows['train'][4*i+rng.randrange(4)] for i in range(c.ORIGINAL)]\n"
         "    replay = rng.sample(prior_features, c.PRIOR) + old_replay"),
        ("""    fresh_order = list(range(c.FRESH))
    replay_order = list(range(c.FRESH, c.FRESH+c.REPLAY))
    rng.shuffle(fresh_order)
    rng.shuffle(replay_order)
    orders = []
    for offset in range(0, c.FRESH, 4):
        batch = fresh_order[offset:offset+4]+replay_order[offset:offset+4]""",
         """    fresh_order = list(range(c.FRESH))
    prior_order = list(range(c.FRESH, c.FRESH+c.PRIOR))
    old_order = list(range(c.FRESH+c.PRIOR, len(features)))
    for order in (fresh_order, prior_order, old_order):
        rng.shuffle(order)
    orders = []
    for offset in range(0, c.FRESH, 2):
        batch = fresh_order[offset:offset+2]+prior_order[offset:offset+2]+old_order[2*offset:2*offset+4]"""),
        ('learning_epoch_number=15', 'learning_epoch_number=16'),
    ])
    create('three_memory_codec_engine_v44.py', 'three_memory_codec_engine_v45.py', [
        ('from three_memory_codec_engine_v34 import BroadCodecEngine',
         'from three_memory_codec_engine_v34 import BroadCodecEngine\n'
         'from three_memory_codec_engine_@v44@ import SCHEMA as SOURCE_SCHEMA\n'
         'from codec_resume_@v44@_contract import POLICY as SOURCE_POLICY'),
        ('def import_completed_v43', 'def import_completed_@v44@'),
        ("state['schema'] != PARENT_SCHEMA or state['continuation']['policy'] != PARENT_POLICY",
         "state['schema'] != SOURCE_SCHEMA or state['continuation']['policy'] != SOURCE_POLICY"),
        ("'complete V43 source contract'", "'complete @V44@ source contract'"),
    ])
    create('run_three_memory_codec_20261111_v44.py', 'run_three_memory_codec_20261111_v45.py', [
        ('ARTIFACT, FRESH, REPLAY, GLOBAL_ATTEMPTS', 'ARTIFACT, FRESH, PRIOR, REPLAY, GLOBAL_ATTEMPTS'),
        ('sum(i < FRESH for i in batch) != 4',
         'sum(i < FRESH for i in batch) != 2 or sum(FRESH <= i < FRESH+PRIOR for i in batch) != 2'),
        ("(8192,8189,65536)", "(2048,2047,16384)"),
        ('torch.manual_seed(844)', 'torch.manual_seed(845)'),
        ('engine.import_completed_v43(state, SOURCE_PLAN, SOURCE_SHA, CORPUS_MANIFEST)',
         "engine.import_completed_@v44@(state, SOURCE_PLAN, SOURCE_SHA,\n"
         "        support.sha(ROOT/'research/artifacts/three_memory_codec_20261111_@v44@/manifest.json'))"),
        ('The V43 results', 'The @V44@ results'),
    ])
    create('supervise_three_memory_codec_20261111_v44.py', 'supervise_three_memory_codec_20261111_v45.py', [])
    create('seal_codec_resume_v44.py', 'seal_codec_resume_v45.py', [
        ('actual V43 full-state', 'actual @V44@ full-state'),
    ])
    create('review_codec_resume_v44.py', 'review_codec_resume_v45.py', [
        ("previous['saved_logit_metrics']['held_final']['normal']",
         "previous['saved_logits']['held256_teacher']['normal']"),
    ])


if __name__ == '__main__':
    main()
