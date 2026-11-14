"""Build the V47 boundary-repaired, early-byte-weighted codec allocation."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT/'scripts'


def write(path, content):
    with path.open('x',encoding='utf-8') as stream:
        stream.write(content)


def edit(source, dest, changes=()):
    text = (S/source).read_text(encoding='utf-8')
    for before, after in changes:
        if before not in text:
            raise ValueError('missing source fragment: '+before)
        text = text.replace(before,after)
    text = text.replace('v46','v47').replace('V46','V47')
    text = text.replace('@v47@','v46').replace('@V47@','V46')
    write(S/dest,text)


def build_objective():
    old = (ROOT/'research/three_memory_codec_v33.py').read_text(encoding='utf-8')
    start = old.index('    def reconstruction_loss(')
    end = old.index('    @torch.no_grad()',start)
    body = old[start:end]
    body = body.replace('ignore_index=-100, reduction=\'none\').sum()/count',
                        'ignore_index=-100, reduction=\'none\')')
    start = body.index('        self.last_objective_stats')
    body = body[:start]+'''        clean = early_byte_average(clean, active)
        corrupt = early_byte_average(corrupt, active)
'''+body[start:]
    header = '''"""Guarded reconstruction with additional supervision on the first 16 bytes."""
import torch
from torch.nn import functional as F
from research.text_diffusion_v1 import PAD
from research.utf8_codec_decode import Utf8PositionCodec

EARLY_BYTES = 16
EARLY_WEIGHT = 4.0


def early_byte_average(per_byte, active):
    positions = torch.arange(per_byte.shape[-1], device=per_byte.device)
    weight = torch.where(positions < EARLY_BYTES, EARLY_WEIGHT, 1.0) * active
    return (per_byte * weight).sum() / weight.sum()


class EarlyByteCodec(Utf8PositionCodec):
'''
    write(ROOT/'research/early_byte_codec_v47.py',header+body)


def main():
    build_objective()
    edit('codec_resume_v46_contract.py','codec_resume_v47_contract.py',[
        ("PARENT = ROOT / 'runs/three_memory_codec_20261111_v45'","PARENT = ROOT / 'runs/three_memory_codec_20261111_@v46@'"),
        ("TRAIN_DATA = ROOT / 'research/artifacts/three_memory_codec_20261111_v45'",
         "TRAIN_DATA = ROOT / 'research/artifacts/codec_boundary_repair_20261111_v47'\n"
         "ORDERS = ROOT / 'research/artifacts/three_memory_codec_20261111_v45/orders.json'"),
        ('c34b45e074517192a531dfc2b942559e8c87bae6461baf4b0bc735b30f917740',
         '8455119c5bef1301e87a95e76acf5c02a54b85232ffed7d95c977af5495f4996'),
        ('ade4de9dcb99bd725d56b31851cede37aa20afc5298c1e86672bb8c2e0a77c9a',
         'b97129e68471f27424d0b33f0fac1721414dfbbb18fbc9f1ff99dbc9d8bbf68a'),
        ('a5032d47ddef81099466288b52b53622eadecb326bd7c8a5c445308a0d94d9bd',
         '84787c8adce3d5cb09ddc1f7ad978cf398d5567d29d277b87bcd1e03c4ddba8a'),
        ('165860, 165798','167908, 167845'),
        ('held16_three_memory_ablations=12)', 'held16_three_memory_ablations=12, repaired_BOS_rows_max=130)'),
        ("purpose='position_sensitive_contextual_reads_and_decoder_slots', new_parameters=1026,\n              all_populated_Adam_states=91",
         "purpose='verified_boundaries_and_early_byte_loss', new_parameters=0,\n              early_bytes=16, early_weight=4.0, utf8_guard=True, all_populated_Adam_states=92"),
        ("TRAIN_DATA/'features.json', TRAIN_DATA/'targets.json',\n                 TRAIN_DATA/'orders.json', TRAIN_DATA/'manifest.json'",
         "TRAIN_DATA/'train_features.json', TRAIN_DATA/'train_targets.json',\n                 ORDERS, TRAIN_DATA/'manifest.json', TRAIN_DATA/'independent_review.json'"),
        ('V45 terminal identity','@V46@ terminal identity'),
    ])
    broad = (S/'three_memory_codec_engine_v34.py').read_text(encoding='utf-8')
    exact = broad[broad.index('    def import_codec_parent('):broad.index('    def export(')]
    exact = exact.replace('def import_codec_parent(', 'def import_unchanged_parameters(')
    exact = exact.replace('91','92').replace('after=super().export()','after=CodecEngine.export(self)')
    exact = exact.replace('copy.deepcopy(CONTINUATION)', 'copy.deepcopy(POLICY)')
    edit('three_memory_codec_engine_v46.py','three_memory_codec_engine_v47.py',[
        ('import copy','import copy\nimport random\nimport numpy as np'),
        ('from research.position_sensitive_codec_v46 import ADAPTER_NAME',
         'from research.position_sensitive_codec_@v46@ import ADAPTER_NAME'),
        ('from three_memory_codec_engine_v34 import BroadCodecEngine',
         'import text_generation_support_v1 as support\n'
         'from three_memory_codec_engine_v33 import CodecEngine, CONTRACT, OWNERSHIP'),
        ('from three_memory_codec_engine_v45 import CodecBudget, SCHEMA as SOURCE_SCHEMA',
         'from three_memory_codec_engine_v45 import CodecBudget\n'
         'from three_memory_codec_engine_@v46@ import SCHEMA as SOURCE_SCHEMA'),
        ('from codec_resume_v45_contract import POLICY as SOURCE_POLICY',
         'from codec_resume_@v46@_contract import POLICY as SOURCE_POLICY, TRAIN_DATA as OLD_DATA'),
        ('len(named) != 91', 'len(named) != 92'),
        ('    def import_completed_v45(',exact+'    def import_completed_@v46@('),
        ("state['data_revision'] != self.data_revision",
         "state['data_revision'] != support.sha(OLD_DATA/'manifest.json')"),
        ("adapted = dict(state, schema='fdt_three_memory_language_codec_v33_fullstate')\n        BroadCodecEngine.import_codec_parent(self, adapted, source_plan, source_sha)",
         "adapted = dict(state, schema='fdt_three_memory_language_codec_v33_fullstate',data_revision=self.data_revision)\n"
         "        self.import_unchanged_parameters(adapted, source_plan, source_sha)"),
    ])
    worker = (S/'run_three_memory_codec_20261111_v46.py').read_text(encoding='utf-8')
    parity = worker[worker.index('def verify_position_path('):worker.index('def load_import(')]
    edit('run_three_memory_codec_20261111_v46.py','run_three_memory_codec_20261111_v47.py',[
        ('from research.position_sensitive_codec_v46 import ADAPTER_SCALE, PositionSensitiveCodec',
         'from research.position_sensitive_codec_@v46@ import ADAPTER_SCALE\n'
         'from research.early_byte_codec_v47 import EarlyByteCodec'),
        ('TRAIN_DATA, FRESH, PRIOR, REPLAY, GLOBAL_ATTEMPTS', 'TRAIN_DATA, ORDERS, FRESH, PRIOR, REPLAY, GLOBAL_ATTEMPTS'),
        ("io.load_json(TRAIN_DATA/'features.json')","io.load_json(TRAIN_DATA/'train_features.json')"),
        ("io.load_json(TRAIN_DATA/'targets.json')","io.load_json(TRAIN_DATA/'train_targets.json')"),
        ("io.load_json(TRAIN_DATA/'orders.json')","io.load_json(ORDERS)"),
        (parity,''),
        ('torch.manual_seed(846);model=PositionSensitiveCodec(Config(**state[\'core\']))',
         "torch.manual_seed(847);model=EarlyByteCodec(Config(**state['core']))\n"
         "        model.enable_position_adapter()"),
        ('engine.import_completed_v45(', 'engine.import_completed_@v46@('),
        ('!=111732306 or len(engine.optimizer.state)!=91','!=111733332 or len(engine.optimizer.state)!=92'),
        ("        report.update(verify_position_path(engine, features['train'][0]))\n", ''),
        ("\n        engine.enable_position_adapter()", ''),
        ('The V45 results','The @V46@ results'),
        ('position_sensitive_codec_training','boundary_repaired_early_byte_training'),
        ("        engine.evaluations['held64_reconstruction.json']=support.sha(output/'held64_reconstruction.json')",
         """        engine.evaluations['held64_reconstruction.json']=support.sha(output/'held64_reconstruction.json')
        repair = io.load_json(TRAIN_DATA/'manifest.json')['held64_edits']
        repaired_features = io.load_json(TRAIN_DATA/'held64_features.json')
        corrected = list(full_rows)
        new_rows = reconstruct_inputs(engine, [repaired_features[item['index']] for item in repair], budget, 'cuda')
        for item, row in zip(repair, new_rows, strict=True):
            corrected[item['index']] = row
        io.atomic_json(output/'held64_boundary_repaired.json', corrected)
        engine.evaluations['held64_boundary_repaired.json']=support.sha(output/'held64_boundary_repaired.json')"""),
        ('One finite TRAIN pass on the same spans with position-sensitive contextual reads and slot tags.',
         'Boundary-repaired training rows, first 16 positions weighted fourfold, and UTF-8 guarded reconstruction.'),
    ])
    edit('supervise_three_memory_codec_20261111_v46.py','supervise_three_memory_codec_20261111_v47.py')
    edit('seal_codec_resume_v46.py','seal_codec_resume_v47.py',[
        ("\n            or not preflight['report']['zero_adapter_teacher_parity']", ''),
        ("'actual V45 full-state import not verified'", "'actual @V46@ full-state import not verified'"),
        ('research/position_sensitive_codec_v46.py','research/early_byte_codec_v47.py'),
    ])
    edit('review_codec_resume_v46.py','review_codec_resume_v47.py',[
        ("    expected_names = [list(group) for group in source['optimizer_parameter_names']]\n"
         "    expected_names[-1].append(ADAPTER)\n    assert state['optimizer_parameter_names'] == expected_names",
         "    assert state['optimizer_parameter_names'] == source['optimizer_parameter_names']"),
        ('len(original_adam) == 91 and len(final_adam) == 92 and original_adam.keys() < final_adam.keys()',
         'len(original_adam) == len(final_adam) == 92 and original_adam.keys() == final_adam.keys()'),
        ("    added = next(iter(final_adam.keys()-original_adam.keys()))\n    assert int(final_adam[added]['step']) == state['updates']\n", ''),
        ("c.read(c.TRAIN_DATA/'orders.json')","c.read(c.ORDERS)"),
        ('preserved_adam_states=91','preserved_adam_states=92'),
        ("    ablations = {}","""    raw_panels['held64_repaired'], scores['held64_repaired'] = raw_rows(
        c.read(c.RUN/'held64_boundary_repaired.json'), c.read(c.TRAIN_DATA/'held64_features.json'))
    assert all(s['UTF8']==s['rows'] for s in scores.values())
    ablations = {}"""),
    ])
    import codec_resume_v47_contract as c
    old = c.read(c.PARENT/'plan.json')
    review = c.read(c.PARENT/'root_result.json')
    if not review['all_owned_processes_gone'] or review['status'] != 'V46_COMPLETE_CPU_REVIEWED':
        raise ValueError('verified V46 source required')
    repairs = c.read(c.TRAIN_DATA/'independent_review.json')
    if (repairs['verified_train_replacements'], repairs['verified_held_replacements'],
        repairs['exact_normalized_split_overlap'], repairs['normalized_train_duplicates']) != (154,3,0,0):
        raise ValueError('verified split repairs required')
    if c.RUN.exists() or c.ARTIFACT.exists():
        raise FileExistsError('new V47 allocation only')
    c.ARTIFACT.mkdir()
    files = {name:digest for name,digest in old['files'].items() if Path(name).suffix != '.pt'}
    paths = [c.SOURCE,c.PARENT/'plan.json',c.PARENT/'root_result.json',
             c.PARENT/'root_utf8_decoder_repair.json',c.PARENT/'root_position_boundary_audit.json',
             ROOT/'research/early_byte_codec_v47.py',ROOT/'research/utf8_codec_decode.py',
             Path(__file__).resolve(),*c.TRAIN_DATA.glob('*.json')]
    paths += [S/name for name in ('codec_resume_v47_contract.py','three_memory_codec_engine_v47.py',
               'run_three_memory_codec_20261111_v47.py','supervise_three_memory_codec_20261111_v47.py',
               'seal_codec_resume_v47.py')]
    for path in paths:
        files[str(path)] = c.sha256(path)
    train = c.read(c.TRAIN_DATA/'train_features.json')
    exposure = dict(old['training_exposure'], target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train),
                    boundary_and_duplicate_repairs=154)
    plan = dict(old, schema=c.SCHEMA, authorized=False, output=str(c.RUN), source=str(c.SOURCE),
        source_sha=c.SOURCE_SHA, source_plan_sha=c.SOURCE_PLAN, source_algorithm=c.SOURCE_ALGORITHM,
        source_plan_path=str(c.PARENT/'plan.json'), source_global_attempts=c.GLOBAL_ATTEMPTS,
        source_global_updates=c.GLOBAL_UPDATES, limits=c.LIMITS, cost_contract=c.COST,
        storage_contract=c.STORAGE, continuation_policy=c.POLICY, training_artifact=str(c.TRAIN_DATA),
        training_exposure=exposure, allocation_start=None, cutoff=None, deadline=None,
        purpose='Verified boundary repairs and guarded decoder with first-16-byte training emphasis',
        files=files, source_baseline_reference=dict(run=str(c.PARENT), same_model_at_initialization=True,
            root_result_sha256=c.sha256(c.PARENT/'root_result.json'), repeat_initial_GPU_baseline=False))
    write(c.ARTIFACT/'plan_unsealed.json',json.dumps(plan,ensure_ascii=False,indent=2))
    print('V47 CPU validation input ready, '+str(c.COST['total_max'])+' max calculated GPU calls')


if __name__ == '__main__':
    main()
