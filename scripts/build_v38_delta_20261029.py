"""Create NEW V38 continuation files; never edit earlier sealed code."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mapping = {
    'v37':'v38', 'V37':'V38', 'v36':'v37', 'V36':'V37',
    'Third':'Fourth', 'third':'fourth', 'THIRD':'FOURTH',
    '36001':'37001', '35001':'36001',
    '112612':'116708', '112570':'116665',
    '53a4486871a0e94df78cf8426dd786c5fdb86536c3b06d07469db742c8d40adc':
      '9ae9837e4d6c2324473e8a69653d16786dea70b671c9f402725538c8aedd2ab9',
    'd63383e4d8b4bcaf4ed0427d7fa14fa0c08b478e16bdd9c224bd8e44cd25c639':
      '2da0351ed2de70511b0c0dca3044dab6bb28f1df0811a3f248e17945b5f707eb',
    '800ff9820a44d9f88bda4d894963b778ebebf084113fa34adeab014fcd7dbb36':
      'bac7db67fc3e53e1621df3b429456a306ea0ec8f0e74a4c1ae516c3ce3b427bf',
    '(4096,4092,32768)':'(4096,4095,32768)',
    'torch.manual_seed(837)':'torch.manual_seed(838)',
    'learning_epoch_number=3':'learning_epoch_number=4',
    'V36_COMPLETE_SECOND_BROAD_EPOCH_ROOT_REVIEWED':'V37_COMPLETE_THIRD_BROAD_EPOCH_ROOT_REVIEWED',
}
pattern = re.compile('|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
names = [
    'scripts/three_memory_codec_engine_v37.py',
    'scripts/run_three_memory_codec_20261029_v37.py',
    'scripts/supervise_three_memory_codec_20261029_v37.py',
    'scripts/prepare_three_memory_worker_v37_cpu.py',
    'scripts/review_three_memory_owner_v37_windows_cpu.py',
    'scripts/seal_three_memory_codec_20261029_v37.py',
    'tests/test_three_memory_worker_v37.py',
]
for name in names:
    dest = ROOT / name.replace('v37', 'v38')
    text = pattern.sub(lambda m: mapping[m.group()], (ROOT/name).read_text(encoding='utf-8-sig'))
    with dest.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
print('NEW_V38_DELTA_FILES_CREATED_NO_MODEL_CALLS')
