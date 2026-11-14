"""Create-only V43 delta: preserve terminal group LR instead of cosine restart."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'scripts'

def common(text):
    for a,b in (
        ('three_memory_codec_20261030_v42','three_memory_codec_20261030_v43'),
        ('three_memory_codec_20261030_v41','three_memory_codec_20261030_v42'),
        ('three_memory_codec_engine_v42','three_memory_codec_engine_v43'),
        ('three_memory_worker_v42','three_memory_worker_v43'),
        ('three_memory_owner_v42_windows','three_memory_owner_v43_windows'),
        ('finalize_v42_cpu_review','finalize_v43_cpu_review'),
        ('build_v42_delta_20261030','build_v43_low_rate_delta_20261030'),
        ('fdt_three_memory_language_codec_v42','fdt_three_memory_language_codec_v43'),
        ('fdt_three_memory_language_codec_v41_fullstate','fdt_three_memory_language_codec_v42_fullstate'),
        ('THREE_MEMORY_CODEC_V42_COMPLETE','THREE_MEMORY_CODEC_V43_COMPLETE'),
        ('f242fc902bd65f331cde625378bf832cbb54c5e26ef4842cf6352ec64fb48887','41f0fc13626adb5ebdef9fe2e02842f64a0b78de715172e0d51c2079f04c0ad3'),
        ('7205572abeff36b18d202d652749ae0ef636e476f877593e3372265b119584d8','6211513e03f367ca0be8193525f6c52155382950241f3245b72f3cc2f7c06b4b'),
        ('b74eb3f6f7137c5967fe1d6a53d54e49ea3f97ca9fd10c28568cdc26ef6d201e','129f2b79eae7af34251d2892c69c60951f845af4eadc48c5efee13d8247d2734'),
        ('145380','153572'),('145326','153515'),
        ('import_completed_v41','import_completed_v42'),('completed V41','completed V42'),('V41 source','V42 source'),
        ('same_group_base_LRs_new_finite_cosine_no_moment_or_step_reset','constant_source_terminal_group_LRs_no_restart_no_moment_or_step_reset'),
        ('V41_COMPLETE_NINTH_TENTH_BROAD_EPOCHS_ROOT_REVIEWED','V42_COMPLETE_ELEVENTH_TWELFTH_BROAD_EPOCHS_ROOT_REVIEWED'),
        ('[41001,41002]','[42001,42002]'),('[41001, 41002]','[42001, 42002]'),
        ('learning_epoch_number=11','learning_epoch_number=13'),('torch.manual_seed(842)','torch.manual_seed(843)'),
    ):text=text.replace(a,b)
    return text

def write(path,text):
    with path.open('x',encoding='utf-8') as f:f.write(text)

engine=common((S/'three_memory_codec_engine_v42.py').read_text())
engine=engine.replace('import copy','import copy\nimport torch\nfrom contextlib import nullcontext\nfrom research.text_diffusion_v1 import pack_prompts, pack_answers\nfrom owt_fuzzy_engine_v17_base import amp_stop_reason')
needle='        self.continuation[\'policy\']=copy.deepcopy(CONTINUATION)'
assert needle in engine
engine=engine.replace(needle,needle+"\n        self.continuation['source_terminal_group_lrs']=[float(g['lr']) for g in state['optimizer']['param_groups']]")
base=(S/'three_memory_codec_engine_v33.py').read_text()
step=base[base.index('    def step(self):'):base.index('    def export(self):')]
old="        factor = .1+.9*(1+math.cos(math.pi*(self.attempts-1)/len(self.orders)))/2\n        for group in self.optimizer.param_groups: group['lr'] = group['base_lr']*factor"
assert old in step
step=step.replace(old,"        for group, lr in zip(self.optimizer.param_groups, self.continuation['source_terminal_group_lrs']):\n            group['lr'] = lr")
step=step.replace("'outcome': 'AMP_SKIP' if skip else 'UPDATED',", "'optimizer_group_lrs': [g['lr'] for g in self.optimizer.param_groups],\n                'outcome': 'AMP_SKIP' if skip else 'UPDATED',")
engine=engine.replace('    def export(self):',step+'    def export(self):')
engine += "\n"  # existing export/restore preserve the schedule values in continuation.
write(S/'three_memory_codec_engine_v43.py',engine)

worker=common((S/'run_three_memory_codec_20261030_v42.py').read_text())
worker=worker.replace('(8192,8188,65536)','(8192,8189,65536)')
worker=worker.replace('Two finite broad TRAIN epochs, same three-memory codec and ALL91Adam;',
                      'Two finite broad TRAIN epochs at preserved source-terminal group LRs, same codec and ALL91Adam;')
write(S/'run_three_memory_codec_20261030_v43.py',worker)
for name in ('supervise_three_memory_codec_20261030_v42.py','prepare_three_memory_worker_v42_cpu.py',
             'review_three_memory_owner_v42_windows_cpu.py','seal_three_memory_codec_20261030_v42.py',
             'finalize_v42_cpu_review_20261030.py'):
    text=common((S/name).read_text());dest=name.replace('v42','v43')
    text=text.replace('V42_NEW_TWO_EPOCH_EPOCH_PROSPECTIVE','V43_LOW_RATE_TWO_EPOCH_PROSPECTIVE')
    text=text.replace('V42_ONE_NEW_SEAL_NOT_EXECUTION','V43_ONE_NEW_SEAL_NOT_EXECUTION')
    text=text.replace('v42_CPU_contract_evidence','v43_CPU_contract_evidence')
    text=text.replace('ROOT_V42_CPU_OS_DELTA_REVIEW_COMPLETE_NOT_GPU','ROOT_V43_CPU_OS_DELTA_REVIEW_COMPLETE_NOT_GPU')
    text=text.replace('V41_FINAL_NOT_CPU_PARITY_REFERENCE','V42_FINAL_NOT_OLDER_PARITY_REFERENCE')
    text=text.replace("'source_version':'V41_FINAL_NOT_CPU_PARITY_REFERENCE','epochs':[11,12],'seeds':[41001,41002]",
                      "'source_version':'V42_FINAL_NOT_OLDER_PARITY_REFERENCE','epochs':[13,14],'seeds':[42001,42002]")
    text=text.replace("'epochs':[11,12]","'epochs':[13,14]")
    if name.startswith('prepare_'):
        text=text.replace('prepare_three_memory_worker_v42_cpu','prepare_three_memory_worker_v43_cpu')
        text=text.replace('Two finite broad SAME-input three-memory language codec epochs;',
                          'Two finite SAME-input codec epochs at constant source-terminal LRs without restart;')
    if name.startswith('seal_'):
        text=text.replace("pretests=ET.parse(a/'preflight_cpu.xml')", "pretests=ET.parse(ROOT/'research/artifacts/three_memory_codec_20261030_v42_worker_cpu/preflight_cpu.xml')")
        text=text.replace("int(suite.attrib['tests'])!=12","int(suite.attrib['tests'])!=13")
        text=text.replace("'CPU_tests':18","'CPU_tests':13,'unchanged_owned_preflight_tests_reused':6")
    if name.startswith('finalize_'):
        text=text.replace("[('cpu_delta.xml',12),('preflight_cpu.xml',6)]","[('cpu_delta.xml',13)]")
        text=text.replace("'CPU_tests':18","'CPU_tests':13,'unchanged_owned_preflight_tests_reused':6")
    write(S/dest,text)

test=common((ROOT/'tests/test_three_memory_worker_v42.py').read_text())
test=test.replace('from three_memory_codec_engine_v41 import CONTINUATION as SOURCE_POLICY',
                  'from three_memory_codec_engine_v42 import CONTINUATION as SOURCE_POLICY')
test=test.replace("all(g['lr']==g['base_lr'] for g in e.optimizer.param_groups)",
                  "[g['lr'] for g in e.optimizer.param_groups]==[g['lr'] for g in s['optimizer']['param_groups']]")
test += '''\n\ndef test_constant_imported_terminal_LRs_no_restart_and_recovery(tmp_path):
    e,s=new_epoch(tmp_path)
    lrs=[g['lr'] for g in s['optimizer']['param_groups']]
    bases=[g['base_lr'] for g in s['optimizer']['param_groups']]
    assert e.continuation['source_terminal_group_lrs']==lrs
    for _ in range(3):
        row=e.step()
        assert row['optimizer_group_lrs']==lrs
        assert [g['lr'] for g in e.optimizer.param_groups]==lrs
        assert [g['base_lr'] for g in e.optimizer.param_groups]==bases
    state=copy.deepcopy(e.export());expected=e.step();final=copy.deepcopy(e.export())
    restored,_=new_epoch(tmp_path);restored.restore(state)
    assert restored.step()==expected
    assert support.state_sha(restored.export())==support.state_sha(final)
'''
write(ROOT/'tests/test_three_memory_worker_v43.py',test)
print('V43 create-only low-rate code generated; no tests/preflight/seal/GPU yet')
