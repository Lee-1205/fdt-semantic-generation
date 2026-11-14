"""Four fresh CPU calls only, eight preselected stored-Commit contrasts."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_commit_cross_20261112_v71 import validate

def main(seal):
    data=ROOT/'research/artifacts/commit_cross_20261112_v71_cpu_v2';review=data/'execution_review';file=review/'plan_unsealed.json'
    if seal:
        p=json.loads(file.read_text());output=Path(p['output']);validate(file,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';s=json.loads(proof.read_text());r=s['report']
        assert s['status']=='CPU_PREFLIGHT_PASS' and s['exit']['all_owned_processes_gone'] and not s['owner_imported_Torch']
        assert r['fullsize_verbalizer_restore'] and r['all_parent_tensors_exact'] and r['all_old_Adam_exact'] and r['parent_Adam_scaler_ALLRNG_unchanged']
        assert not r['CUDA_initialized'] and r['actual_parent_encoder_forward']==r['actual_verbalizer_forward']==r['optimizer_steps']==0
        independent=review/'root_independent_validate.json';assert json.loads(independent.read_text(encoding='utf-8-sig'))['status']=='ROOT_INDEPENDENT_V71_READY'
        p['files'][str(proof)]=support.sha(proof);p['files'][str(independent)]=support.sha(independent);now=time.time();p.update(authorized=True,start=now,cutoff=now+90,deadline=now+120)
        output.mkdir();sealed=output/'plan.json';sealed.write_text(json.dumps(p,indent=2)+'\n');validate(sealed,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        print(json.dumps(dict(plan_sha=support.sha(sealed),CPU_verbalizer_calls=4,case_patches=8,encoder0=True,CUDA0=True,wall120reserve30=True)));return
    assert not file.exists();oldrun=ROOT/'runs/hard_fuzzy_cycle_20261112_v67';newrun=ROOT/'runs/factual_cycle_20261112_v70'
    parent=json.loads((newrun/'plan.json').read_text());manifest=json.loads((data/'manifest.json').read_text());p={}
    p['files']={k:v for k,v in parent['files'].items() if k.endswith('.py')}
    names=['scripts/prepare_commit_cross_v71_cpu.py','scripts/test_commit_cross_v71_cpu.py','scripts/prepare_commit_cross_plan_v71.py',
        'scripts/run_commit_cross_20261112_v71.py','scripts/supervise_commit_cross_v71.py','scripts/owned_cpu_preflight_v71.py']
    p.update(schema='fdt_commit_cross_v71',root=str(ROOT),output=str(ROOT/'runs/commit_cross_20261112_v71'),data=str(data),authorized=False,
        start=None,cutoff=None,deadline=None,wall_seconds=120,max_calls=4,cases=8,attempts=0,epochs=0,max_output_bytes=16*1024**2,device='cpu',amp_dtype='float32',capacity=16384,
        old_source=str(oldrun/'model/checkpoints/state_1024.pt'),old_algorithm_sha='d8b2ba555605913d38415bf7dc427b16f50245e34b542e5f91c831f059b6cd20',
        new_source=str(newrun/'model/checkpoints/state_0512.pt'),new_algorithm_sha='f2baf2af63d1e47201206c87e43d53140c856b63022e54084443446b785a6e2f',
        purpose='distinguish saved bound Commit+binding drift from final verbalizer drift for eight observed correction/regression/control patches before further training',
        no_encoder_forward=True,no_CUDA=True,no_optimizer=True,source_labels_read_after_all_four_calls=True,
        diagonal_policy='interpret each focus only if both CPU-FP32 diagonals reproduce corresponding native focus argmax; retain all cases including parity failures',
        prior_research=['https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html','https://arxiv.org/abs/2404.15255'],
        limitations=manifest['limitation'],new_code_files=names)
    files=[ROOT/name for name in names]+[Path(p['old_source']),Path(p['new_source']),newrun/'root_saved_review.json',newrun/'root_meaning_review.json',
        oldrun/'root_saved_review.json',data/'features.pt',data/'reference_logits.pt',data/'cases.json',data/'manifest.json',data/'root_CPU_contract.json']
    files+=list(map(Path,manifest['source_raw_SHA']))
    for f in files:p['files'][str(f)]=support.sha(f)
    for name,sha in manifest['source_raw_SHA'].items():assert p['files'][name]==sha
    for name,digest in p['files'].items():
        assert support.sha(name)==digest
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();file.write_text(json.dumps(p,indent=2)+'\n');validate(file,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),fullsize_CPU_preflight_next=True,actual_calls_so_far=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');main(p.parse_args().seal)
