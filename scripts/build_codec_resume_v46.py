"""Build one V46 allocation from the proven V45 worker/owner interfaces."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'


def edit(old_name, new_name, changes):
    text = (SCRIPTS/old_name).read_text(encoding='utf-8')
    for before, after in changes:
        if before not in text:
            raise ValueError('missing source fragment: '+before)
        text = text.replace(before, after)
    text = text.replace('v45','v46').replace('V45','V46')
    text = text.replace('@v46@','v45').replace('@V46@','V45')
    with (SCRIPTS/new_name).open('x',encoding='utf-8') as f:
        f.write(text)


PARITY = '''def verify_position_path(engine, feature):
    model = engine.model
    model.eval()
    x = pack_prompts([feature['prompt']], 128, 'cpu')
    y = pack_answers([feature['prompt']], 129, 'cpu')
    args = dict(source_ids=(feature['id'],), source_versions=(engine.data_revision,),
                model_epoch=engine.memory_epoch())
    with torch.no_grad():
        original = model.teacher_logits(model.encode_input(x, **args).commit, y)
    adapter = engine.enable_position_adapter()
    active = model.teacher_logits(model.encode_input(x, **args).commit, y)
    if not torch.equal(original, active):
        raise ValueError('zero-adapter teacher logits must match exactly')
    grad, = torch.autograd.grad(active[0,0,y[0,0]], adapter)
    gains, tags = grad[:,0], grad[:,1:]
    if not torch.isfinite(grad).all() or gains.abs().max() <= 0 or tags.abs().max() <= 0:
        raise ValueError('BOS prediction must reach segment gains and position tags')
    return {'zero_adapter_teacher_parity':True, 'first_byte_segment_grad':float(gains.abs().max()),
            'first_byte_position_grad':float(tags.abs().max()), 'new_Adam_initial_state':0}


'''


def main():
    edit('codec_resume_v45_contract.py','codec_resume_v46_contract.py',[
        ("PARENT = ROOT / 'runs/three_memory_codec_20261111_v44'",
         "PARENT = ROOT / 'runs/three_memory_codec_20261111_@v45@'"),
        ("SOURCE = PARENT / 'model/checkpoints/state_2048.pt'",
         "SOURCE = PARENT / 'model/checkpoints/state_2048.pt'\n"
         "TRAIN_DATA = ROOT / 'research/artifacts/three_memory_codec_20261111_@v45@'"),
        ('9e7eb0c686da7d82d2ccfb3957ac5fd34d3fd6309f420118684e0cd295213021',
         'c34b45e074517192a531dfc2b942559e8c87bae6461baf4b0bc735b30f917740'),
        ('e559f3b49e9cf458dcd6e651c80dd798f2ff9b5a5732f771305efad326d40527',
         'ade4de9dcb99bd725d56b31851cede37aa20afc5298c1e86672bb8c2e0a77c9a'),
        ('8973919382c4a13eca3053ab6c47ceed5d6b77c76092033f801f1f6aeb2fe929',
         'a5032d47ddef81099466288b52b53622eadecb326bd7c8a5c445308a0d94d9bd'),
        ('163812, 163751','165860, 165798'),
        ('max_calls=9000','max_calls=10800'),
        ('held_final_BOS_max=520','held_final_BOS_max=2080'),
        ("purpose='fresh_train_sentence_spans_with_replay', new_parameters=0,",
         "purpose='position_sensitive_contextual_reads_and_decoder_slots', new_parameters=1026,"),
        ('generation_indices=[4*i for i in range(16)])',
         'generation_indices=[4*i for i in range(16)], full_BOS_indices=[4*i for i in range(64)],\n'
         '                    training_artifact=str(TRAIN_DATA))'),
        ("ARTIFACT/'features.json', ARTIFACT/'targets.json',\n                 ARTIFACT/'orders.json', ARTIFACT/'manifest.json'",
         "TRAIN_DATA/'features.json', TRAIN_DATA/'targets.json',\n                 TRAIN_DATA/'orders.json', TRAIN_DATA/'manifest.json'"),
        ('V44 terminal identity','@V45@ terminal identity'),
    ])
    edit('run_three_memory_codec_20261111_v45.py','run_three_memory_codec_20261111_v46.py',[
        ('from research.three_memory_codec_v33 import ThreeMemoryCodec',
         'from research.position_sensitive_codec_v46 import ADAPTER_SCALE, PositionSensitiveCodec\n'
         'from research.text_diffusion_v1 import pack_prompts, pack_answers'),
        ('FreshSpanCodecEngine','PositionCodecEngine'),
        ('ARTIFACT','TRAIN_DATA'),
        ('torch.manual_seed(845);model=ThreeMemoryCodec','torch.manual_seed(846);model=PositionSensitiveCodec'),
        ("source_parameter_groups=state['optimizer_parameter_names'][:3]", "source_parameter_groups=state['optimizer_parameter_names']"),
        ("""report=engine.import_completed_v44(state, SOURCE_PLAN, SOURCE_SHA,
        support.sha(ROOT/'research/artifacts/three_memory_codec_20261111_v44/manifest.json'))""",
         "report=engine.import_completed_@v45@(state, SOURCE_PLAN, SOURCE_SHA)"),
        ('def load_import(', PARITY+'def load_import('),
        ('    return engine,report','        report.update(verify_position_path(engine, features[\'train\'][0]))\n    return engine,report'),
        ("phase='initial_commit';progress();commit('import')",
         "phase='initial_commit';progress();commit('import')\n        engine.enable_position_adapter()"),
        ('The V44 results','The @V45@ results'),
        ('fresh_train_span_codec_training','position_sensitive_codec_training'),
        ("""        io.atomic_json(output/'held_final_reconstruction.json',reconstruct_inputs(engine,sf,budget,'cuda'))""",
         """        full_rows = reconstruct_inputs(engine, [f['dev'][i] for i in p['full_BOS_indices']], budget, 'cuda')
        io.atomic_json(output/'held_final_reconstruction.json', full_rows[:len(sf)])
        io.atomic_json(output/'held64_reconstruction.json', full_rows)
        engine.evaluations['held64_reconstruction.json']=support.sha(output/'held64_reconstruction.json')"""),
        ("'interpretation':'One finite balanced TRAIN pass with fresh sentence spans and replay. Same-input reconstruction.'",
         "'position_adapter_gains':(ADAPTER_SCALE*engine.model.memory_codec.position_adapter[:,0]).detach().tanh().cpu().tolist(),\n"
         "            'interpretation':'One finite TRAIN pass on the same spans with position-sensitive contextual reads and slot tags.'"),
    ])
    edit('supervise_three_memory_codec_20261111_v45.py','supervise_three_memory_codec_20261111_v46.py',[])
    edit('seal_codec_resume_v45.py','seal_codec_resume_v46.py',[
        ('actual V44 full-state','actual @V45@ full-state'),
        ("or preflight['report']['source_sha256'] != c.SOURCE_SHA)",
         "or preflight['report']['source_sha256'] != c.SOURCE_SHA\n"
         "            or not preflight['report']['zero_adapter_teacher_parity'])"),
        ("c.ROOT/'scripts/three_memory_codec_engine_v45.py')",
         "c.ROOT/'scripts/three_memory_codec_engine_v46.py', c.ROOT/'research/position_sensitive_codec_v46.py')"),
    ])
    edit('review_codec_resume_v45.py','review_codec_resume_v46.py',[
        ('import learning_io_v2 as io','import learning_io_v2 as io\nADAPTER = "memory_codec.position_adapter"'),
        ("assert state['optimizer_parameter_names'] == source['optimizer_parameter_names']",
         "expected_names = [list(group) for group in source['optimizer_parameter_names']]\n"
         "    expected_names[-1].append(ADAPTER)\n"
         "    assert state['optimizer_parameter_names'] == expected_names"),
        ('assert len(original_adam) == len(final_adam) == 91 and original_adam.keys() == final_adam.keys()',
         'assert len(original_adam) == 91 and len(final_adam) == 92 and original_adam.keys() < final_adam.keys()'),
        ("assert all(int(v['step']) == int(original_adam[i]['step'])+state['updates']\n               for i,v in final_adam.items())",
         "assert all(int(final_adam[i]['step']) == int(v['step'])+state['updates'] for i,v in original_adam.items())\n"
         "    added = next(iter(final_adam.keys()-original_adam.keys()))\n"
         "    assert int(final_adam[added]['step']) == state['updates']\n"
         "    assert torch.isfinite(state['model'][ADAPTER]).all()"),
        ("c.read(c.ARTIFACT/'orders.json')", "c.read(c.TRAIN_DATA/'orders.json')"),
        ("    ablations = {}", """    full_panel = [dev[i] for i in plan['full_BOS_indices']]
    raw_panels['held64'], scores['held64'] = raw_rows(c.read(c.RUN/'held64_reconstruction.json'), full_panel)
    assert raw_panels['held64'][:16] == raw_panels['held16']
    ablations = {}"""),
    ])
    import codec_resume_v46_contract as c
    old = c.read(c.PARENT/'plan.json')
    review = c.read(c.PARENT/'root_result.json')
    audit = c.PARENT/'root_code_data_audit_20261111.json'
    if (review['status'] != 'V45_COMPLETE_CPU_REVIEWED' or not review['all_owned_processes_gone']
            or c.read(audit)['provenance_verified'] != c.FRESH+c.REPLAY):
        raise ValueError('V45 verified terminal and corpus audit required')
    if c.RUN.exists() or c.ARTIFACT.exists():
        raise FileExistsError('new V46 allocation only')
    c.ARTIFACT.mkdir()
    files = {name:digest for name,digest in old['files'].items() if Path(name).suffix != '.pt'}
    paths = [c.SOURCE,c.PARENT/'plan.json',c.PARENT/'root_result.json',audit,
             ROOT/'research/position_sensitive_codec_v46.py',Path(__file__).resolve()]
    paths += [SCRIPTS/name for name in ('codec_resume_v46_contract.py','three_memory_codec_engine_v46.py',
               'run_three_memory_codec_20261111_v46.py','supervise_three_memory_codec_20261111_v46.py',
               'seal_codec_resume_v46.py')]
    for path in paths:
        files[str(path)] = c.sha256(path)
    plan = dict(old, schema=c.SCHEMA, authorized=False, output=str(c.RUN), source=str(c.SOURCE),
        source_sha=c.SOURCE_SHA, source_plan_sha=c.SOURCE_PLAN, source_algorithm=c.SOURCE_ALGORITHM,
        source_plan_path=str(c.PARENT/'plan.json'), source_global_attempts=c.GLOBAL_ATTEMPTS,
        source_global_updates=c.GLOBAL_UPDATES, limits=c.LIMITS, cost_contract=c.COST,
        storage_contract=c.STORAGE, continuation_policy=c.POLICY, training_artifact=str(c.TRAIN_DATA),
        full_BOS_indices=[4*i for i in range(64)], allocation_start=None, cutoff=None, deadline=None,
        research_target_KST='2026-11-12T00:00:00+09:00',
        purpose='Code and dataset audit followed by a zero-initialized position adapter, same TRAIN schedule',
        files=files, source_baseline_reference=dict(run=str(c.PARENT), same_function_at_initialization=True,
            root_result_sha256=c.sha256(c.PARENT/'root_result.json'), repeat_initial_GPU_baseline=False))
    with (c.ARTIFACT/'plan_unsealed.json').open('x',encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,indent=2)
    print('V46 CPU validation input ready, '+str(c.COST['total_max'])+' max calculated GPU calls')


if __name__=='__main__':
    main()
