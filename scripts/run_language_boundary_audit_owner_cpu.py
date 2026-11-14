"""Use established bounded Windows OS owner for one CPU boundary fixture."""
import argparse,json
from pathlib import Path
from owned_cpu_preflight_v50 import preflight

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);a=parser.parse_args()
    p=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    result=preflight(a.plan,p['output'],p['worker'],120)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],all_owned_gone=result['exit']['all_owned_processes_gone'],owner_imported_Torch=result['owner_imported_Torch'])))
