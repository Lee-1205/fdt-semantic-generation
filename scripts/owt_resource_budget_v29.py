"""Immediate control checks; bounded resource scans and forced large-write checks."""
from pathlib import Path
import shutil,time
from text_language_runtime_support_v6 import owned_bytes

RESOURCE_POLICY={'filesystem_interval_seconds':1.0,'force_on_start':True,
    'invalidate_on_restore':True,'force_before_large_write':True,
    'worker_disk_reserve_bytes':22*1024**3,'checkpoint_write_reserve_bytes':1024**3,
    'STOP_clock_callcap_immediate':True,'emergency_resource_only_check':True,
    'independent_owner_poll_seconds':1.0}

class RecoveryBudget:
    def __init__(self,plan):
        if plan.get('resource_policy')!=RESOURCE_POLICY:raise ValueError('fixed resource policy')
        self.plan=plan;self.output=Path(plan['output'])
        self.calls={'encode':0,'denoise':0,'decoder':0};self.allocation_calls=dict(self.calls)
        self.tick=None;self.last_tick=0.;self.last_resource_check=None;self.resource_scans=0

    def invalidate_resources(self):
        # Transient OS timing is not an algorithm/RNG checkpoint component.
        self.last_resource_check=None

    def check_resources(self,force=False,upcoming_bytes=0):
        if upcoming_bytes<0:raise ValueError('negative write estimate')
        now=time.monotonic()
        if not force and not upcoming_bytes and self.last_resource_check is not None and now-self.last_resource_check<1.0:return
        free=shutil.disk_usage(self.output).free;size=owned_bytes(self.output)
        self.resource_scans+=1
        if free<RESOURCE_POLICY['worker_disk_reserve_bytes']+upcoming_bytes:raise OSError('disk reserve including upcoming write')
        if size+upcoming_bytes>=self.plan['limits']['max_output_bytes']:raise OSError('owned output cap including upcoming write')
        # Measure from scan START, so a slow scan cannot extend the interval.
        self.last_resource_check=now

    def check(self,force=False,upcoming_bytes=0):
        if (self.output/'STOP').exists():raise InterruptedError('cooperative STOP')
        if time.time()>=self.plan['cutoff']:raise TimeoutError('new fixed worker cutoff')
        if sum(self.allocation_calls.values())>=self.plan['limits']['max_calls']:raise RuntimeError('new actual call cap')
        self.check_resources(force,upcoming_bytes)

    def charge(self,kind):
        if kind not in self.calls:raise ValueError('unknown call kind')
        self.check();self.calls[kind]+=1;self.allocation_calls[kind]+=1
        if self.tick and time.time()-self.last_tick>=5:self.last_tick=time.time();self.tick()

