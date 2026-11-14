"""Reuse the previously verified exact-file lossless storage method on older CLOSED runs."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'scripts/compact_owned_ae_checkpoints_20261028.py').read_text(encoding='utf-8')
s=s.replace("OUT=ROOT/'research/artifacts/owt_checkpoint_lossless_space_20261028_v1'","OUT=ROOT/'research/artifacts/owt_checkpoint_lossless_space_20261029_v3'")
s=s.replace('MAX_FILES=32','MAX_FILES=24').replace('MAX_LOGICAL=24*1024**3','MAX_LOGICAL=18*1024**3').replace('WALL=480','WALL=300')
start=s.index('    allowed=');end=s.index('    OUT.mkdir()',start)
s=s[:start]+'''    run_names=['owt_fuzzy_reconstruction_20261028_v20','owt_fuzzy_reconstruction_20261028_v21','owt_fuzzy_reconstruction_20261028_v22','owt_fuzzy_reconstruction_20261028_v23_final']+[f'owt_fuzzy_reconstruction_20261029_v{v}' for v in (24,25,26)]
    allowed=set(run_names)
    for run_name in run_names:
        run=ROOT/'runs'/run_name
        terminal=json.loads((run/'supervisor/final_status.json').read_text(encoding='utf-8-sig'))
        if not terminal.get('exit',{}).get('all_owned_processes_gone') or terminal['status'] not in ('COMPLETE','FAILED','STOPPED'):raise RuntimeError('closed generated run only')
        for stage in ('model','import','initial','emergency'):
            for path in sorted((run/stage/'checkpoints').glob('*.pt')):
                resolved=path.resolve(strict=True)
                if path.is_symlink() or resolved!=path or resolved.relative_to(ROOT/'runs').parts[0] not in allowed:raise ValueError('exact older owned path')
                if len(candidates)>=MAX_FILES or total+path.stat().st_size>MAX_LOGICAL:break
                candidates.append(path);total+=path.stat().st_size
'''+s[end:]
s=s.replace("'protected_latest_runs':[20,21,22]","'protected_latest_runs':[27,28,29]")
s=s.replace('no active/latest V20-V22 runs','no active/latest V27-V29 runs')
with (R/'scripts/compact_owned_ae_checkpoints_20261029_v3.py').open('x',encoding='utf-8') as f:f.write(s)
