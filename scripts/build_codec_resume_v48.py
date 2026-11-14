"""Build V48 with eight local fuzzy summaries alongside the two global Commit slots."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT/'scripts'


def edit(source, destination, changes=()):
    text = (S/source).read_text(encoding='utf-8')
    for old,new in changes:
        if old not in text:
            raise ValueError('missing source fragment: '+old)
        text = text.replace(old,new)
    text = text.replace('v47','v48').replace('V47','V48')
    text = text.replace('@v48@','v47').replace('@V48@','V47')
    with (S/destination).open('x',encoding='utf-8') as stream:
        stream.write(text)


def main():
    edit('codec_resume_v47_contract.py','codec_resume_v48_contract.py',[
        ("PARENT = ROOT / 'runs/three_memory_codec_20261111_v46'",
         "PARENT = ROOT / 'runs/three_memory_codec_20261111_@v47@'"),
        ("codec_boundary_repair_20261111_v47", "codec_boundary_repair_20261111_@v47@"),
        ('8455119c5bef1301e87a95e76acf5c02a54b85232ffed7d95c977af5495f4996',
         '2c79076246164ec80c01b68b8d711f5fe0759cc1342159fb2423380523d320cf'),
        ('b97129e68471f27424d0b33f0fac1721414dfbbb18fbc9f1ff99dbc9d8bbf68a',
         '6bd38dc13d5972d2f5bce93a3bda5145dcd93fdf5fda9c1cc554508a9c7bf67e'),
        ('84787c8adce3d5cb09ddc1f7ad978cf398d5567d29d277b87bcd1e03c4ddba8a',
         'b66abf560e15237defdcb4d8a00f75c654d96070b2d31638b644a304a6f69f5a'),
        ('167908, 167845','169956, 169893'),
        ("purpose='verified_boundaries_and_early_byte_loss', new_parameters=0,",
         "purpose='eight_ordered_contextual_summaries_alongside_global_commit', new_parameters=4098,\n"
         "              local_slots=8, input_bytes_per_local_slot=16,"),
        ('V46 terminal identity','@V47@ terminal identity'),
    ])
    edit('run_three_memory_codec_20261111_v47.py','run_three_memory_codec_20261111_v48.py',[
        ('from research.early_byte_codec_v47 import EarlyByteCodec',
         'from research.wide_context_codec_v48 import WideContextCodec, GAIN_SCALE'),
        ('PositionCodecEngine','WideCodecEngine'),
        ("(2048,2047,16384)", "(2048,2048,16384)"),
        ('torch.manual_seed(847);model=EarlyByteCodec','torch.manual_seed(848);model=WideContextCodec'),
        ('engine.import_completed_v46(', 'engine.import_completed_@v47@('),
        ('unchanged model/ALL91 Adam','unchanged model/ALL92 Adam'),
        ("phase='initial_commit';progress();commit('import')",
         "phase='initial_commit';progress();commit('import')\n        engine.enable_local_read()"),
        ('The V46 results','The @V47@ results'),
        ('boundary_repaired_early_byte_training','wide_context_codec_training'),
        ("'interpretation':'Boundary-repaired training rows, first 16 positions weighted fourfold, and UTF-8 guarded reconstruction.'",
         "'local_read_gains':(GAIN_SCALE*engine.model.decoder.local_read_gain).detach().tanh().cpu().tolist(),\n"
         "            'interpretation':'Eight ordered fuzzy contextual summaries plus two original global slots. Same corrected data and loss as @V47@.'"),
    ])
    edit('supervise_three_memory_codec_20261111_v47.py','supervise_three_memory_codec_20261111_v48.py')
    edit('seal_codec_resume_v47.py','seal_codec_resume_v48.py',[
        ('actual V46 full-state','actual @V47@ full-state'),
        ('research/early_byte_codec_v47.py','research/wide_context_codec_v48.py'),
    ])
    edit('review_codec_resume_v47.py','review_codec_resume_v48.py',[
        ("    assert state['optimizer_parameter_names'] == source['optimizer_parameter_names']",
         "    names = [list(group) for group in source['optimizer_parameter_names']]\n"
         "    names[-1].extend(('decoder.local_read_gain','decoder.local_slot_tag'))\n"
         "    assert state['optimizer_parameter_names'] == names"),
        ('len(original_adam) == len(final_adam) == 92 and original_adam.keys() == final_adam.keys()',
         'len(original_adam) == 92 and len(final_adam) == 94 and original_adam.keys() < final_adam.keys()'),
        ("    assert torch.isfinite(state['model'][ADAPTER]).all()",
         "    assert torch.isfinite(state['model'][ADAPTER]).all()\n"
         "    assert all(int(final_adam[i]['step'])==state['updates'] for i in final_adam.keys()-original_adam.keys())"),
    ])
    edit('review_early_byte_v47.py','review_early_byte_v48.py',[
        ('V46/V47','@V47@/V48'),
        ('V47_teacher=compare_teacher(current)','V48_teacher=compare_teacher(current)'),
        ('V46_teacher=compare_teacher(parent)','@V47@_teacher=compare_teacher(parent)'),
        ('V46_BOS64=before','@V47@_BOS64=before'),
        ('V47_BOS64=after','V48_BOS64=after'),
        ('V47_repaired_BOS64','V48_repaired_BOS64'),
    ])
    import codec_resume_v48_contract as c
    source_plan = c.read(c.PARENT/'plan.json')
    review = c.read(c.PARENT/'root_result.json')
    audit_path = c.PARENT/'root_v48_cpu_candidate.json'
    audit = c.read(audit_path)
    model = ROOT/'research/wide_context_codec_v48.py'
    if (review['status'] != 'V47_COMPLETE_CPU_REVIEWED' or not review['all_owned_processes_gone']
            or not audit['CPU_zero_gain_exact_logits'] or audit['code_sha256'] != c.sha256(model)
            or audit['source_plan'] != c.SOURCE_PLAN):
        raise ValueError('verified V47 source and ordered local candidate required')
    if c.RUN.exists() or c.ARTIFACT.exists():
        raise FileExistsError('new V48 allocation only')
    c.ARTIFACT.mkdir()
    files = {name:digest for name,digest in source_plan['files'].items() if Path(name).suffix != '.pt'}
    paths = [c.SOURCE,c.PARENT/'plan.json',c.PARENT/'root_result.json',audit_path,model,Path(__file__).resolve()]
    paths += [S/name for name in ('codec_resume_v48_contract.py','three_memory_codec_engine_v48.py',
               'run_three_memory_codec_20261111_v48.py','supervise_three_memory_codec_20261111_v48.py',
               'seal_codec_resume_v48.py')]
    for path in paths:
        files[str(path)] = c.sha256(path)
    plan = dict(source_plan, schema=c.SCHEMA, authorized=False, output=str(c.RUN), source=str(c.SOURCE),
        source_sha=c.SOURCE_SHA, source_plan_sha=c.SOURCE_PLAN, source_algorithm=c.SOURCE_ALGORITHM,
        source_plan_path=str(c.PARENT/'plan.json'), source_global_attempts=c.GLOBAL_ATTEMPTS,
        source_global_updates=c.GLOBAL_UPDATES, limits=c.LIMITS, cost_contract=c.COST,
        storage_contract=c.STORAGE, continuation_policy=c.POLICY, training_artifact=str(c.TRAIN_DATA),
        allocation_start=None, cutoff=None, deadline=None, preflight_revision=1,
        purpose='Preserve the two global views and learn reads from eight ordered local fuzzy summaries',
        files=files, source_baseline_reference=dict(run=str(c.PARENT), same_model_at_initialization=True,
            root_result_sha256=c.sha256(c.PARENT/'root_result.json'), repeat_initial_GPU_baseline=False))
    with (c.ARTIFACT/'plan_unsealed.json').open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,ensure_ascii=False,indent=2)
    print('V48 CPU validation input ready, '+str(c.COST['total_max'])+' max calculated GPU calls')


if __name__=='__main__':
    main()
