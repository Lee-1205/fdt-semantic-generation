import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()
if a.preflight:
 print('CPU_PREFLIGHT_PASS');raise SystemExit(0)
out=Path(a.output)
(out/'worker_progress.json').write_text(json.dumps({'worker_pid':os.getpid(),'stage':'terminal','actual_model_calls':0}),encoding='utf-8')
(out/'diagnostic_result.json').write_text(json.dumps({'cpu_only':True,'actual_model_calls':0}),encoding='utf-8')
time.sleep(.15)
