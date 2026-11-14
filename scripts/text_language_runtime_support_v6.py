"""Bounded owned-directory accounting, tolerating only vanished atomic temp files."""
from pathlib import Path
import shutil,time
import text_generation_support_v1 as original


def owned_bytes(output):
    size=0
    for path in Path(output).rglob('*'):
        try:
            if path.is_file():size+=path.stat().st_size
        except FileNotFoundError:
            if not path.name.endswith('.tmp'):raise
    return size


class RecoveryBudget:
    def __init__(self,plan):
        self.plan=plan;self.output=Path(plan['output']);self.calls={'encode':0,'denoise':0,'decoder':0}
        self.allocation_calls=dict(self.calls);self.tick=None;self.last_tick=0.
    def check(self):
        if (self.output/'STOP').exists():raise InterruptedError('cooperative STOP')
        if time.time()>=self.plan['cutoff']:raise TimeoutError('new fixed worker cutoff')
        if shutil.disk_usage(self.output).free<22*1024**3:raise OSError('20GiB plus2temporary reserve')
        if owned_bytes(self.output)>=self.plan['limits']['max_output_bytes']:raise OSError('new owned output cap')
    def charge(self,kind):
        self.check()
        if sum(self.allocation_calls.values())>=self.plan['limits']['max_calls']:raise RuntimeError('new actual call cap')
        self.calls[kind]+=1;self.allocation_calls[kind]+=1
        if self.tick and time.time()-self.last_tick>=5:self.last_tick=time.time();self.tick()
