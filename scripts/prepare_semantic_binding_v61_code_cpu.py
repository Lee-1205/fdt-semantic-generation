"""Create NEW V61 driver/owner/CPU fixtures; never modify sealed older files."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(name,text):
    path=ROOT/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text);path.write_text(text,encoding='utf8')


def main():
    engine=(ROOT/'scripts/semantic_binding_engine_v60.py').read_text(encoding='utf-8-sig')
    engine=engine.replace('semantic_binding_v60','semantic_binding_v61').replace('fourrole_memory_only_fact_readout','coupled_spans_fourrole_memory_readout')
    write('scripts/semantic_binding_engine_v61.py',engine)
    worker=(ROOT/'scripts/run_semantic_binding_20261111_v60.py').read_text(encoding='utf-8-sig')
    for a,b in [('SemanticBinderV60','SemanticBinderV61'),('semantic_binding_v60','semantic_binding_v61'),('semantic_binding_engine_v60','semantic_binding_engine_v61'),
        ('V60','V61'),('v60','v61'),('66ec25782dd2f3a9ea1a84f31971ca5a8140d2d3f58fa665b29b3202bc089f3d','21dc84c862990fb75fafcdfcca6e473ebfb6c924105edf70b7fc833a340d0043'),
        ('f0381ce21cadfe984d9798927604a89f23c16ca3b6587385c1828afc3063e163','f1d568e3ab1a22fdbb0daed40e972e9f8d4b3842c01afd615e9aef1c47d1cd02'),
        ('950','1500'),('3*1024**3','5*1024**3'),("p['attempts']!=576","p['attempts']!=1024"),("p['cost']!=816","p['cost']!=1340"),
        ('len(train)!=2880','len(train)!=4096'),('len(orders)!=576','len(orders)!=1024'),("warm['updates']!=576","warm['updates']!=576"),
        ('torch.manual_seed(60111)','torch.manual_seed(61111)'),('random.seed(60111)','random.seed(61111)'),('np.random.seed(60111)','np.random.seed(61111)'),
        ('torch.cuda.manual_seed_all(60111)','torch.cuda.manual_seed_all(61111)'),('engine.attempts in (1,288)','engine.attempts in (1,512)'),
        ('fourrole_memory_only_fact_readout','coupled_spans_fourrole_memory_readout')]:worker=worker.replace(a,b)
    # BoundMeaningState is retained from V60; V61 adds routing, not a new type.
    worker=worker.replace('from research.semantic_binding_v61 import SemanticBinderV61,BoundMeaningState',
        'from research.semantic_binding_v61 import SemanticBinderV61\nfrom research.semantic_binding_v60 import BoundMeaningState')
    old="if set(copied.missing_keys)!={'role_start_gain','role_end_gain'} or copied.unexpected_keys:raise ValueError('only6 new context gains')"
    if old not in worker:raise ValueError('warm-copy contract anchor changed')
    worker=worker.replace(old,"expected_new={k for k in model.state_dict() if k.startswith(('sequence.','sequence_out.','entity_inside.'))}\n    if set(copied.missing_keys)!=expected_new or copied.unexpected_keys:raise ValueError('only new sequence and coupled-interior tensors')")
    worker=worker.replace('all_V59_binder_tensors_exact=True,new_parameters6_zero=True',
        'all_V60_binder_tensors_exact=True,new_residual_and_interior_zero=True')
    worker=worker.replace("not model.role_start_gain.count_nonzero() and not model.role_end_gain.count_nonzero()",
        "not model.sequence_out.weight.count_nonzero() and not model.sequence_out.bias.count_nonzero() and not model.entity_inside.weight.count_nonzero() and not model.entity_inside.bias.count_nonzero()")
    write('scripts/run_semantic_binding_20261111_v61.py',worker)
    owner=(ROOT/'scripts/supervise_semantic_binding_v60.py').read_text(encoding='utf-8-sig')
    owner=owner.replace('v60','v61').replace('950','1500').replace("p['attempts']!=576","p['attempts']!=1024").replace('3*1024**3','5*1024**3')
    write('scripts/supervise_semantic_binding_v61.py',owner)
    test=(ROOT/'scripts/test_semantic_binding_v60_cpu.py').read_text(encoding='utf-8-sig')
    test=test.replace('SemanticBinderV60','SemanticBinderV61').replace('semantic_binding_v60','semantic_binding_v61').replace('semantic_binding_engine_v60','semantic_binding_engine_v61')
    test=test.replace('from research.semantic_binding_v61 import SemanticBinderV61,BoundMeaningState',
        'from research.semantic_binding_v61 import SemanticBinderV61\nfrom research.semantic_binding_v60 import BoundMeaningState')
    test=test.replace("semantic_binding_20261111_v60_cpu","semantic_binding_20261111_v61_cpu")
    write('scripts/test_semantic_binding_v61_cpu.py',test)


if __name__=='__main__':main()
