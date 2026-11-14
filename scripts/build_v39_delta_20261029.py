"""NEW bounded two-epoch codec and reduced-output initial-reference contract."""
import re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
M={'v38':'v39','V38':'V39','v37':'v38','V37':'V38',
   'Fourth':'TwoEpoch','fourth':'two_epoch','FOURTH':'TWO_EPOCH',
   '37001':'38001','36001':'37001','116708':'120804','116665':'120759',
   '9ae9837e4d6c2324473e8a69653d16786dea70b671c9f402725538c8aedd2ab9':'06f26d363c4ade4f0c70f043bc2506da4024f06644ff156ded0aabc29da74137',
   '2da0351ed2de70511b0c0dca3044dab6bb28f1df0811a3f248e17945b5f707eb':'0fdf36a73f9f81dda3b97c06dbbe7c92f4303cc104a65be490e1b5d4d8d5e4a6',
   'bac7db67fc3e53e1621df3b429456a306ea0ec8f0e74a4c1ae516c3ce3b427bf':'c4351d80c8c5cce0cdda11f6f64d3266feaf24c02a55e69ecd261464fbb412e0',
   '(4096,4095,32768)':'(4096,4094,32768)',
   'V38_COMPLETE_FOURTH_BROAD_EPOCH_ROOT_REVIEWED':'V38_COMPLETE_FOURTH_BROAD_EPOCH_ROOT_REVIEWED',
   'torch.manual_seed(838)':'torch.manual_seed(839)',
}
pat=re.compile('|'.join(re.escape(k) for k in sorted(M,key=len,reverse=True)))
N=['scripts/three_memory_codec_engine_v38.py','scripts/run_three_memory_codec_20261029_v38.py',
   'scripts/supervise_three_memory_codec_20261029_v38.py','scripts/prepare_three_memory_worker_v38_cpu.py',
   'scripts/review_three_memory_owner_v38_windows_cpu.py','scripts/seal_three_memory_codec_20261029_v38.py',
   'tests/test_three_memory_worker_v38.py']
for name in N:
    t=pat.sub(lambda m:M[m.group()],(R/name).read_text(encoding='utf-8-sig'))
    if name.endswith('engine_v38.py'):
        t=t.replace("'new_finite_epochs':1", "'new_finite_epochs':2")
        t=t.replace('two_epoch_broad_finite_TRAIN_epoch','two_broad_finite_TRAIN_epochs')
    if 'run_three_memory_' in name or 'supervise_three_memory_' in name:
        for a,b in [("'attempts':4096","'attempts':8192"),("'attempts': 4096","'attempts': 8192"),
                    ("'wall':1200","'wall':2100"),("'wall': 1200","'wall': 2100"),
                    ("'reserve':120","'reserve':180"),("'reserve': 120","'reserve': 180"),
                    ("'max_calls':17000","'max_calls':30000"),("'max_calls': 17000","'max_calls': 30000"),
                    ('6*1024**3','3*1024**3'),('35*1024**3','32*1024**3'),('C35GiB','C32GiB'),('6output','3output'),
                    ("'training':12288","'training':24576"),("'training': 12288","'training': 24576"),
                    ('15060','27348'),("'training_seconds_ceiling':820","'training_seconds_ceiling':1639"),
                    ("'startup_seconds':60","'startup_seconds':90"),
                    ("'shutdown_reserve_seconds':120","'shutdown_reserve_seconds':180"),
                    ("'calculated_total_seconds':1120","'calculated_total_seconds':2029"),
                    ("'fixed_new_allocation_seconds':1200","'fixed_new_allocation_seconds':2100"),
                    ('820+60+60+60+120!=1120','1639+90+60+60+180!=2029'),
                    ("LIMITS['wall']!=1200","LIMITS['wall']!=2100"),
                    ("!= 1200","!= 2100"),("!=1200","!=2100"),
                    ("plan['deadline']-120","plan['deadline']-180"),("p['deadline']-120","p['deadline']-180"),
                    ("plan.get('epochs') != 1","plan.get('epochs') != 2"),("p.get('epochs')!=1","p.get('epochs')!=2"),
                    ('[38001,38001]','[38001,38002]'),('[38001, 38001]','[38001, 38002]'),
                    ('finite_orders(32768,[38001])','finite_orders(32768,[38001,38002])'),
                    ("'selected_rows':32768,'target_UTF8_bytes':2926478,'finite_epochs':1", "'selected_rows':65536,'target_UTF8_bytes':5852956,'finite_epochs':2"),
                    ("'new_finite_epochs': 1","'new_finite_epochs': 2"),
                    ('two_epoch_broad_finite_TRAIN_epoch','two_broad_finite_TRAIN_epochs'),
                    ('engine.attempts==2048','engine.attempts==4096'),
                    ('min(45, plan[\'cutoff\']-time.time())','min(90, plan[\'cutoff\']-time.time())'),
                    ('args.worker, 60)','args.worker, 90)')]: t=t.replace(a,b)
    if 'prepare_three_memory_worker' in name:
        t=t.replace('epochs=1,learning_epoch_number=4,order_seed_range=[38001,38001]',
                    'epochs=2,learning_epoch_number=5,order_seed_range=[38001,38002]')
        t=t.replace("'selected_rows':32768,'target_UTF8_bytes':2926478,'finite_epochs':1",
                    "'selected_rows':65536,'target_UTF8_bytes':5852956,'finite_epochs':2")
        t=t.replace('cost_max\':15060','cost_max\':27348')
        t=t.replace('Third finite broad','Two finite broad').replace('Fourth finite broad','Two finite broad').replace('measured1200sec','measured2100sec')
    if 'seal_three_memory_' in name or 'test_three_memory_' in name or 'review_three_memory_owner' in name:
        t=t.replace('+1200','+2100').replace('+1080','+1920').replace('!=7','!=9').replace("'CPU_tests':7","'CPU_tests':9")
    if 'test_three_memory_' in name:
        t=t.replace('len(orders)==4096','len(orders)==8192').replace('==list(range(32768))','==sorted(list(range(32768))*2)')
        t=t.replace('820+60+60+60+120==1120<1200','1639+90+60+60+180==2029<2100').replace('==15060','==27348')
    dest=R/name.replace('v38','v39')
    with dest.open('x',encoding='utf-8',newline='\n') as f:f.write(t)
print('V39_NEW_FILES_CREATED_UNSEALED_NO_GPU')
