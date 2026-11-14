"""Pin an unsealed NEW read-only cause diagnostic; never launches a model."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import unpack
from research.precision_readout_v81 import MODES

def main():
    oldrun=ROOT/'runs/update_retention_20261113_v80';old=io.load_json(oldrun/'plan.json');artifact=ROOT/'research/artifacts/readout_cause_20261113_v81_cpu_v1';artifact.mkdir(exist_ok=False)
    output=ROOT/'runs/readout_cause_20261113_v81';cases=[];inputs={};extras=[]
    def info(panel,index,rows):
        raw=oldrun/f'raw_{panel}_{index:02d}.pt';packet=unpack(torch.load(raw,map_location='cpu',weights_only=True));extras.append(raw)
        if 'fuzzy_commit_reference' in packet:extras.append(Path(packet['fuzzy_commit_reference']['path']))
        return dict(raw=str(raw),row=rows[index])
    d=Path(old['eval_data']);rows=io.load_json(d/'features.json');labels=io.load_json(d/'targets.json');extras.extend([d/'features.json',d/'targets.json'])
    for i in (0,8,28,29,40,49,16,53):
        key=f'new{i}';own=info('new_English',i,rows);inputs[key]=own;partner=info('new_English',i^2,rows)
        span=labels[i]['B_span'];other=labels[i^2]['B_span'];assert span==other and len(own['row']['prompt'].encode())==len(partner['row']['prompt'].encode())
        off=span['start']+int(i==49)
        cases.append(dict(name=key,input=key,offset=off,gold=own['row']['prompt'].encode()[off],fact_span=span,partner=partner,
            intervention='Other A changes, target B fixed; bound local vs globals separated; source offset only diagnostic selector'))
    d=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu';rows=io.load_json(d/'dev_features.json');labels=io.load_json(d/'dev_targets.json');extras.extend([d/'dev_features.json',d/'dev_targets.json'])
    # Select fixed B second quantity using its authored target, not an NN feature.
    s={k:labels[56]['critical_spans'][-1][k] for k in ('start','end','text')};assert s=={k:labels[57]['critical_spans'][-1][k] for k in ('start','end','text')}
    inputs['obs56']=info('observed_V75_DEV',56,rows);cases.append(dict(name='obs56',input='obs56',offset=s['start'],gold=rows[56]['prompt'].encode()[s['start']],fact_span=s,partner=info('observed_V75_DEV',57,rows)))
    d=Path(old['old_eval_data']);rows=io.load_json(d/'long_features.json');labels=io.load_json(d/'long_targets.json');extras.extend([d/'long_features.json',d/'long_targets.json'])
    for i,event,field in ((0,1174,'quantity'),(1,1299,'quantity'),(1,1336,'actor')):
        key=f'long{i}'
        if key not in inputs:inputs[key]=info('long',i,rows)
        e=labels[i]['events'][event-1];source=rows[i]['prompt'].encode();segment=source[e['byte_start']:e['byte_end']]
        relative=segment.index((str(e['quantity'])+' '+e['item']).encode()) if field=='quantity' else segment.index(e['actor'].encode())
        offset=e['byte_start']+relative;cases.append(dict(name=f'{key}_{event}_{field}',input=key,offset=offset,gold=source[offset],episode=event,field=field))
    io.atomic_json(artifact/'cases.json',cases);io.atomic_json(artifact/'inputs.json',inputs)
    # Reuse the fully reviewed Torch-free postPopen owned-tree cleanup unchanged.
    owner=(ROOT/'scripts/supervise_update_retention_v80.py').read_text(encoding='utf-8')
    owner=owner.replace('fixed-clock Windows owner for one actual-update V80 readout trial','fixed-clock Windows owner for one READONLY V81 cause diagnostic')
    start=owner.index("    if (p['schema']");end=owner.index('    for name,digest',start)
    owner=owner[:start]+"    if (p['schema']!='fdt_readonly_readout_cause_v81' or p['wall_seconds']!=120 or p['max_calls']!=160\n            or p['attempts']!=0 or p['epochs']!=0 or p['max_output_bytes']!=32*1024**2):\n        raise ValueError('fixed readonly diagnostic contract')\n"+owner[end:]
    owner=owner.replace('run_update_retention_20261113_v80.py','run_readout_cause_20261113_v81.py').replace("p['deadline']-p['start']!=900","p['deadline']-p['start']!=120").replace("p['deadline']-120","p['deadline']-30")
    owner=owner.replace('108*1024**3+31*1024**3//32','108*1024**3+32*1024**2').replace('108.96875GiB start:100 protected+0.96875 owned incl temp+8 headroom','108.03125GiB start:100 protected+0.03125 owned+8 headroom')
    ownerpath=ROOT/'scripts/supervise_readout_cause_v81.py'
    with ownerpath.open('x',encoding='utf-8',newline='\n') as f:f.write(owner)
    oldbody=(ROOT/'scripts/supervise_update_retention_v80.py').read_text(encoding='utf-8').split("    state=dict(status='PREPARING'",1)[1]
    assert owner.split("    state=dict(status='PREPARING'",1)[1]==oldbody
    extras += [Path(old['storage_parent']),Path(old['cycle_source']),oldrun/'model/runtime.pt',artifact/'cases.json',artifact/'inputs.json',ownerpath,
        ROOT/'scripts/run_readout_cause_20261113_v81.py',ROOT/'scripts/prepare_readout_cause_v81.py',ROOT/'scripts/supervise_querycount_partial9_20261027_v3.py',
        ROOT/'scripts/owned_cpu_preflight_v50.py',ROOT/'scripts/owned_output_size_v52.py',ROOT/'scripts/learning_io_v2.py',ROOT/'scripts/text_generation_support_v1.py',ROOT/'scripts/saved_commit_reference_v76_v2.py',
        ROOT/'research/lossless_raw_tensor_v80.py',ROOT/'research/precision_readout_v81.py',ROOT/'tests/test_precision_readout_v81.py']
    # Include every local Python source reachable through the existing research
    # branch, plus original core; files are small and no historical raw expansion.
    extras += list((ROOT/'research').glob('*.py'))+list((ROOT/'src/fdt_local').glob('*.py'))
    files={str(x.resolve()):support.sha(x) for x in sorted(set(extras))}
    plan=dict(schema='fdt_readonly_readout_cause_v81',authorized=False,root=str(ROOT),output=str(output),wall_seconds=120,max_calls=160,max_output_bytes=32*1024**2,attempts=0,epochs=0,
        device='cuda',precision_modes=list(MODES),expected_calls=133,storage_parent=old['storage_parent'],cases=str(artifact/'cases.json'),inputs=str(artifact/'inputs.json'),
        heads={'V75':dict(path=old['cycle_source'],algorithm_SHA=old['cycle_algorithm_sha']),'V80':dict(path=str(oldrun/'model/runtime.pt'),algorithm_SHA='1c67b342a0df81aa993f97043f50b57d1d057858ed70c0750f3dfa1bb11b727c')},
        files=files,optimizer_steps=0,encoder_calls=0,denoise_calls=0,checkpoints=0,source_old_budgets_closed=True,e2_final_accessed=False,
        scope='Already observed purposive cases; no training and no all-language completion claim. Cached bound Commit includes upstream binding; diagnostics do not separately identify E vs binding.')
    io.atomic_json(artifact/'plan_unsealed.json',plan)
    print(json.dumps(dict(plan=str(artifact/'plan_unsealed.json'),pins=len(files),cases=len(cases),inputs=len(inputs),max_calls=160,expected_calls=133,CUDA_initialized=torch.cuda.is_initialized())))

if __name__=='__main__':main()
