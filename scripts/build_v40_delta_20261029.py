"""NEW two-epoch worker with one final-or-emergency fullstate, no rotation."""
from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]
M={'v39':'v40','V39':'V40','v38':'v39','V38':'V39','38001':'39001','38002':'39002','37001':'38001',
   '120804':'128996','120759':'128949',
   '06f26d363c4ade4f0c70f043bc2506da4024f06644ff156ded0aabc29da74137':'1b698b47e9ad386f2713b5532b5f604166f448ec9f8d561c97b830a9462ce3a3',
   '0fdf36a73f9f81dda3b97c06dbbe7c92f4303cc104a65be490e1b5d4d8d5e4a6':'3d9563a081a80ed6c9be52e4c221bccd7eb9de3d83f9094818f9fd5a2183a68b',
   'c4351d80c8c5cce0cdda11f6f64d3266feaf24c02a55e69ecd261464fbb412e0':'fbc825d7e678b61c516019dd8323c704a857767a5601488463ddd6f413ea0bf9',
   '(4096,4094,32768)':'(8192,8190,65536)','torch.manual_seed(839)':'torch.manual_seed(840)',
   'V38_COMPLETE_FOURTH_BROAD_EPOCH_ROOT_REVIEWED':'V39_COMPLETE_FIFTH_SIXTH_BROAD_EPOCHS_ROOT_REVIEWED'}
pat=re.compile('|'.join(re.escape(k) for k in sorted(M,key=len,reverse=True)))
names=['scripts/three_memory_codec_engine_v39.py','scripts/run_three_memory_codec_20261029_v39.py',
       'scripts/supervise_three_memory_codec_20261029_v39.py','scripts/prepare_three_memory_worker_v39_cpu.py',
       'scripts/review_three_memory_owner_v39_windows_cpu.py','scripts/seal_three_memory_codec_20261029_v39.py',
       'tests/test_three_memory_worker_v39.py']
for name in names:
    t=pat.sub(lambda m:M[m.group()],(R/name).read_text(encoding='utf-8-sig'))
    # The existing initial-reference format is reused without editing it.
    t=t.replace('codec_initial_reference_v40','codec_initial_reference_v39')
    t=t.replace('fdt_initial_parent_reference_v40','fdt_initial_parent_reference_v39')
    if 'run_three_memory_' in name or 'supervise_three_memory_' in name:
        t=t.replace('3*1024**3','1024**3').replace('32*1024**3','30*1024**3').replace('C32GiB','C30GiB').replace('3output','1output')
    if 'prepare_three_memory_worker' in name:
        t=t.replace('learning_epoch_number=5','learning_epoch_number=7').replace('model/checkpoints/state_4096.pt','model/checkpoints/state_8192.pt')
        t=t.replace("'scripts/build_v40_delta_20261029.py'", "'scripts/build_v40_delta_20261029.py','scripts/codec_single_snapshot_v40.py'")
    if 'seal_three_memory_' in name:
        t=t.replace('!=9','!=12').replace("'CPU_tests':9","'CPU_tests':12")
    with (R/name.replace('v39','v40')).open('x',encoding='utf-8',newline='\n') as f:f.write(t)
print('V40_NEW_SOURCE_FILES_ONLY_NO_GPU')
