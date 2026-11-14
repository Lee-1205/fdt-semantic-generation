"""Seal one reviewed V44 allocation, without launching the worker."""
import shutil
import time
from pathlib import Path

import codec_resume_v44_contract as c
import learning_io_v2 as io
import supervise_three_memory_codec_20261111_v44 as owner

WORKER = c.ROOT/'scripts/run_three_memory_codec_20261111_v44.py'


def main():
    path = c.ARTIFACT/'plan_unsealed.json'
    plan = owner.validate(path, c.RUN)
    evidence = c.ARTIFACT/'cpu_preflight_owned/final_status.json'
    preflight = c.read(evidence)
    if (preflight['status'] != 'CPU_PREFLIGHT_PASS' or preflight['worker_exit_code'] != 0
            or not preflight['exit']['all_owned_processes_gone']
            or preflight['report']['source_sha256'] != c.SOURCE_SHA):
        raise ValueError('actual V43 full-state import not verified')
    if c.RUN.exists() or owner.duplicates(WORKER) or owner.native.process_inventory(c.ROOT):
        raise FileExistsError('V44 must have a fresh output and no competing model worker')
    resources = owner.resources(plan)
    plan['files'][str(evidence)] = c.sha256(evidence)
    plan['files'][str(Path(__file__).resolve())] = c.sha256(Path(__file__))
    c.RUN.mkdir()
    snapshot = c.RUN/'code_snapshot'
    snapshot.mkdir()
    for path in (WORKER, Path(owner.__file__), Path(c.__file__), Path(__file__).resolve(),
                 c.ROOT/'scripts/three_memory_codec_engine_v44.py'):
        shutil.copyfile(path, snapshot/path.name)
    io.atomic_json(c.RUN/'readiness.json', dict(
        status='CPU_AND_STORAGE_VERIFIED', preflight=preflight['report'],
        resources=resources, source=c.SOURCE_SHA, model_launches=0))
    start = time.time()
    plan.update(authorized=True, allocation_start=start, deadline=start+c.LIMITS['wall'],
                cutoff=start+c.LIMITS['wall']-c.LIMITS['reserve'])
    sealed = c.RUN/'plan.json'
    io.atomic_json(sealed, plan)
    owner.validate(sealed, c.RUN)
    print('V44 sealed '+c.sha256(sealed))


if __name__ == '__main__':
    main()
