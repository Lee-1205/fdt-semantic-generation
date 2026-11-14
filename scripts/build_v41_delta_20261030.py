"""Metadata/source-bound continuation of the unchanged V40 finite codec driver.

New run9th/10th epochs; no edits to any prior sealed source or allocation.
Initial reference39 and single updated fullstate40 storage helpers are reused.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLACE = {
    'three_memory_codec_20261029_v40': 'three_memory_codec_20261030_v41',
    'three_memory_codec_20261029_v39': 'three_memory_codec_20261029_v40',
    'build_v40_delta_20261029': 'build_v41_delta_20261030',
    'V39_COMPLETE_FIFTH_SIXTH_BROAD_EPOCHS_ROOT_REVIEWED':
        'V40_COMPLETE_SEVENTH_EIGHTH_BROAD_EPOCHS_ROOT_REVIEWED',
    'v40': 'v41', 'V40': 'V41', 'v39': 'v40', 'V39': 'V40',
    '39001': '40001', '39002': '40002', '38001': '39001',
    '128996': '137188', '128949': '137138',
    '1b698b47e9ad386f2713b5532b5f604166f448ec9f8d561c97b830a9462ce3a3':
        'fbaf29eca2c5c9e1df12e3ea71702aeee40c0bda5c7f9ec73cba339acd4cd677',
    '3d9563a081a80ed6c9be52e4c221bccd7eb9de3d83f9094818f9fd5a2183a68b':
        '00aca0d669b2228c49fd075ca45a9425ed655d7ce40857bc0a171c1905635a70',
    'fbc825d7e678b61c516019dd8323c704a857767a5601488463ddd6f413ea0bf9':
        '3262ad6fd5e6f9f38552c4ee1207735e76f42a9f61cc8c005d01179b65aa4199',
    '(8192,8190,65536)': '(8192,8189,65536)',
    'torch.manual_seed(840)': 'torch.manual_seed(841)',
    'learning_epoch_number=7': 'learning_epoch_number=9',
    'plan_unsealed_v2.json': 'plan_unsealed.json',
}


def main():
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(REPLACE, key=len, reverse=True)))
    sources = (
        'scripts/three_memory_codec_engine_v40.py',
        'scripts/run_three_memory_codec_20261029_v40.py',
        'scripts/supervise_three_memory_codec_20261029_v40.py',
        'scripts/prepare_three_memory_worker_v40_cpu.py',
        'scripts/review_three_memory_owner_v40_windows_cpu.py',
        'scripts/seal_three_memory_codec_20261029_v40.py',
        'tests/test_three_memory_worker_v40.py',
    )
    for source in sources:
        dest = pattern.sub(lambda m: REPLACE[m.group()], source)
        text = pattern.sub(lambda m: REPLACE[m.group()], (ROOT/source).read_text(encoding='utf-8-sig'))
        # These formats/functions are frozen, shared, and behaviorally unchanged.
        text = text.replace('codec_initial_reference_v40', 'codec_initial_reference_v39')
        text = text.replace('fdt_initial_parent_reference_v40', 'fdt_initial_parent_reference_v39')
        text = text.replace('codec_single_snapshot_v41', 'codec_single_snapshot_v40')
        text = text.replace('unchanged_parent_reference_v40', 'unchanged_parent_reference_v39')
        text = text.replace('Fresh V41 one-epoch seal', 'Fresh V41 two-epoch seal')
        if source.endswith('prepare_three_memory_worker_v40_cpu.py'):
            text = text.replace("files=dict(p['files'])", "files={name:sha for name,sha in p['files'].items() if Path(name).suffix!='.pt'}")
        with (ROOT/dest).open('x', encoding='utf-8', newline='\n') as file:
            file.write(text)
    print('V41_NEW_METADATA_SOURCE_BOUND_FILES_ONLY_NO_MODEL_OR_GPU')


if __name__ == '__main__':
    main()
