"""Bounded new storage scope: CLOSED generated V1-V8 checkpoints, logical bytes preserved."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'scripts/compact_owned_ae_checkpoints_20261029_v3.py').read_text(encoding='utf-8')
s=s.replace('owt_checkpoint_lossless_space_20261029_v3','owt_checkpoint_lossless_space_20261029_v4')
s=s.replace('MAX_FILES=24','MAX_FILES=40').replace('MAX_LOGICAL=18*1024**3','MAX_LOGICAL=28*1024**3').replace('WALL=300','WALL=480')
start=s.index('    run_names=');end=s.index('    OUT.mkdir()',start)
s=s[:start]+'''    run_names=['text_generation_20261028_v1','text_context_decoder_20261028_v2','text_semantic_alignment_20261028_v3','text_span_alignment_20261028_v4','text_language_continuation_20261028_v5','text_language_continuation_20261028_v5_deterministic_v2','text_language_continuation_20261028_v6','text_language_continuation_20261028_v7','text_decoder_alignment_20261028_v8']
    allowed=set(run_names)
    for run_name in run_names:
        run=ROOT/'runs'/run_name
        terminal=json.loads((run/'supervisor/final_status.json').read_text(encoding='utf-8-sig'))
        if not terminal.get('exit',{}).get('all_owned_processes_gone') or terminal['status'] not in ('COMPLETE','FAILED','STOPPED'):raise RuntimeError('closed generated run only')
        for stage in ('checkpoints','model/checkpoints','import/checkpoints','initial/checkpoints','reader/checkpoints','realizer/checkpoints'):
            for path in sorted((run/stage).glob('*.pt')):
                resolved=path.resolve(strict=True)
                if path.is_symlink() or resolved!=path or resolved.relative_to(ROOT/'runs').parts[0] not in allowed:raise ValueError('exact older generated path only')
                if not path.stem.startswith(('state_','payload_')):raise ValueError('fullstate names only, no raw tensor cache')
                if len(candidates)>=MAX_FILES or total+path.stat().st_size>MAX_LOGICAL:break
                candidates.append(path);total+=path.stat().st_size
'''+s[end:]
s=s.replace("'all_candidates_generated_closed_AE_only':True,'protected_latest_runs':[27,28,29]","'all_candidates_generated_closed_language_branches_only':True,'protected_source_and_latest_runs':['typed','V9','AE27','AE28','AE29','AE30']")
s=s.replace('CLOSED, generated AE checkpoints','CLOSED, generated old language-branch checkpoints').replace('no active/latest V27-V29 runs','no V9/original typed/source documents/raw corpus or latest AE27-AE30 runs')
with (R/'scripts/compact_owned_language_checkpoints_20261029_v4.py').open('x',encoding='utf-8') as f:f.write(s)
