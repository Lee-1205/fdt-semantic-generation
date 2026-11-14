"""Create NEW profile from preserved V112; never edit sealed old source."""
from pathlib import Path
P=Path(__file__).resolve().parents[1]


def swap(text,old,new):
    if old not in text:raise ValueError('expected reviewed source clause absent: '+old[:90])
    return text.replace(old,new)


def main():
    worker=(P/'scripts/run_answer_pair_contrast_v112_cpu.py').read_text(encoding='utf-8')
    substitutions={
        'from research.answer_pair_contrast_v112 import PairContrastEngine':'from research.answer_codec_fidelity_v113 import PairCodecEngine as PairContrastEngine',
        "SCHEMA='fdt_answer_pair_contrast_v112'":"SCHEMA='fdt_answer_codec_fidelity_v113'",
        'max_calls=41000,expected_calls=40358':'max_calls=43000,expected_calls=42502',
        'seed=1111412':'seed=1111413',
        "e['schema']!='conditional_answer_meaning_fullstate_v111'":"e['schema']!='answer_pair_contrast_fullstate_v112'",
        'preserved V111 parent':'preserved V112 parent',
        'actual V111 conditional parent architecture':'actual V112 paired parent architecture',
        '+cost>41000':'+cost>43000',">=41000":'>=43000','new41000callcap':'new43000callcap',
        'V112_final_answer_meaning_frozen_D':'V113_final_answer_meaning_frozen_D',
        'budget.before(34)':'budget.before(36)',
        "dict(denoise=36448,readout=0)":"dict(denoise=36448,readout=2144)",
        'dict(denoise=40128,readout=230,encode=0)':'dict(denoise=40128,readout=2374,encode=0)',
        "schema='answer_pair_contrast_complete_v112'":"schema='answer_codec_fidelity_complete_v113'",
        'ANSWER_PAIR_CONTRAST_TRIAL_COMPLETE':'ANSWER_CODEC_FIDELITY_TRIAL_COMPLETE',
        'warm_V111_ONLYtheta=True':'warm_V112_ONLYtheta=True',
        'training_D0=True':'training_D_frozen_gradient=True',
        "assert sum(count.values())==388 and count['readout']==2":"assert sum(count.values())==424 and count['readout']==21",
    }
    for old,new in substitutions.items():worker=swap(worker,old,new)
    worker=swap(worker,"    original=codec.verbalize\n    def forbidden(*a,**kw):raise AssertionError('D forbidden during theta latent training')\n    codec.verbalize=forbidden;thought.train();e.begin()", """    thought.train()
    loss,metrics,semantic=e.objective(m,d.target(d.train[orders[0]]),torch.ones(1),noise,charge)
    grads=torch.autograd.grad(e.objective.last_byte_CE,tuple(thought.parameters()),allow_unused=True)
    CE_gradient_norm=sum(float(g.square().sum()) for g in grads if g is not None)**.5
    assert CE_gradient_norm>0 and all(v.grad is None and not v.requires_grad for v in codec.parameters())
    del loss,metrics,semantic,grads
    e.begin()""")
    worker=swap(worker,'    codec.verbalize=original;e.update.audit_endpoint()','    e.update.audit_endpoint()')
    worker=swap(worker,'actual_real_and_unpaired_units=True,seconds=',
        'actual_real_and_unpaired_units=True,actual_CE_gradient_to_theta_norm=CE_gradient_norm,training_D_calls=18,seconds=')
    target=P/'scripts/run_answer_codec_fidelity_v113_cpu.py'
    with target.open('x',encoding='utf-8') as f:f.write(worker)

    prep=(P/'scripts/prepare_answer_pair_contrast_v112_cpu.py').read_text(encoding='utf-8')
    substitutions={
        'answer_pair_contrast_20261114_v112_C_cpu_v1':'answer_codec_fidelity_20261114_v113_C_cpu_v1',
        'runs/answer_pair_contrast_20261114_v112':'runs/answer_codec_fidelity_20261114_v113',
        'run_answer_pair_contrast_v112_cpu':'run_answer_codec_fidelity_v113_cpu',
        'supervise_answer_pair_contrast_v112_cpu':'supervise_answer_codec_fidelity_v113_cpu',
        "oldrun=P/'runs/answer_meaning_20261114_v111'":"oldrun=P/'runs/answer_pair_contrast_20261114_v112'",
        "p.update(schema='fdt_answer_pair_contrast_v112'":"p.update(schema='fdt_answer_codec_fidelity_v113',max_calls=43000",
        "'fdt_answer_pair_contrast_v112'":"'fdt_answer_codec_fidelity_v113'",
        'seed=1111412':'seed=1111413','expected_calls=40358':'expected_calls=42502',
        "baseline_TRAIN_assessment=str(pairrun/'assessment.json'),baseline_DEV_assessment=str(oldrun/'assessment.json')":
        "baseline_TRAIN_assessment=str(A/'inherited_TRAIN134.json'),baseline_DEV_assessment=str(A/'inherited_DEV96.json')",
        'pair_difference=.25,training_D0=True':'pair_difference=.25,frozen_D_content_CE=.5,training_D_weights_frozen=True',
        "P/'research/answer_pair_contrast_v112.py',P/'tests/test_answer_pair_contrast_v112.py'":
        "P/'research/answer_codec_fidelity_v113.py',P/'tests/test_answer_codec_fidelity_v113.py',A/'inherited_TRAIN134.json',A/'inherited_DEV96.json'",
        "baseline=io.load_json(pairrun/'assessment.json')+[r for r in io.load_json(oldrun/'assessment.json') if r['phase']=='final' and r['split'].startswith('dev_')]":
        "baseline=[r for r in io.load_json(oldrun/'assessment.json') if r['phase']=='final']",
        "r['training_D0']":"r['training_D_frozen_gradient'] and r['actual_CE_gradient_to_theta_norm']>0",
        "sum(r['NN_calls'].values())==388":"sum(r['NN_calls'].values())==424",
        "v['schema']=='answer_pair_contrast_fullstate_v112'":"v['schema']=='answer_codec_fidelity_fullstate_v113' and v['language_losses_used']",
        'ROOT_ACTUAL_PAIR_V112_READY':'ROOT_CODEC_CONTENT_V113_READY',
        "calls40358":'calls42502',
    }
    # The last optional documentation token isn't present in all preserved files.
    substitutions.pop('calls40358')
    for old,new in substitutions.items():prep=swap(prep,old,new)
    prep=swap(prep,"        warm=parent['checkpoints'][-1]", """        warm=parent['checkpoints'][-1]
        oldrows=io.load_json(oldrun/'assessment.json')
        io.atomic_json(A/'inherited_TRAIN134.json',[r for r in oldrows if r['phase']=='final' and r['split']=='TRAIN_all'])
        io.atomic_json(A/'inherited_DEV96.json',[r for r in oldrows if r['phase']=='final' and r['split'].startswith('dev_')])""")
    prep=swap(prep,"        with OWNER.open('x',encoding='utf-8') as f:f.write(text)",
        "        text=text.replace('41000','43000')\n        with OWNER.open('x',encoding='utf-8') as f:f.write(text)")
    prep=prep.replace('expected_calls40358=True','expected_calls42502=True')
    prep=prep.replace('focused_tests5_PASS=True','focused_fidelity_tests3_PASS=True')
    prep=prep.replace("coupled_intervention_NOT_separate_causal_estimates=True,", "coupled_intervention_NOT_separate_causal_estimates=True,added_frozen_D_content_CE=.5,D_weights_trainable=False,")
    prep=prep.replace("task='conditional E(Y), same-noise actual-pair endpoint-difference repair'", "task='conditional E(Y), fixedD content feedback to theta only'")
    # Snapshot the NEW objective, not a stale source name copied from the parent.
    prep=prep.replace("for file in (WORKER,OWNER,Path(__file__).resolve(),P/'research/answer_pair_contrast_v112.py')", "for file in (WORKER,OWNER,Path(__file__).resolve(),P/'research/answer_codec_fidelity_v113.py')")
    with (P/'scripts/prepare_answer_codec_fidelity_v113_cpu.py').open('x',encoding='utf-8') as f:f.write(prep)
    print('NEW V113 worker/preparer created; no run or plan launched.')


if __name__=='__main__':main()
