"""One preserved updated fullstate: final OR emergency, without rotation.

The immutable parent plus the initial reference remain a separate recovery
point. No midpoint is saved. A failed full write is preserved and reported
unconfirmed; it never triggers a second full write into the same allocation.
"""
from pathlib import Path
import shutil
import text_generation_support_v1 as support
from text_language_runtime_support_v6 import owned_bytes

POLICY={'intermediate_full_checkpoint':False,'maximum_updated_full_files':1,
        'initial_checkpoint':'unchanged_parent_reference_v39',
        'after_final_exception':'reuse_only_if_entire_saved_state_matches_except_worker_stage',
        'incomplete_full_write':'preserve_candidates_report_unconfirmed_no_second_write',
        'maximum_uncommitted_attempts':8192,'parent_and_initial_reference_preserved':True}
STORAGE_BOUND={'maximum_full_files':1,'maximum_full_file_bytes':800*1024**2,
    'maximum_initial_reference_bytes':32*1024**2,'assessment_bytes':128*1024**2,
    'other_bytes':32*1024**2,'total_bound_bytes':992*1024**2,'output_cap_bytes':1024**3}

def check_checkpoint_write(budget, upcoming_bytes, emergency=False):
    if upcoming_bytes not in (32*1024**2,800*1024**2):raise ValueError('fixed write-size bound')
    if emergency:budget.check_resources(force=True)
    else:budget.check(force=True)
    # Keep the existing 22GiB + 1GiB disk write reserve. The owned-output
    # limit accounts for the bounded actual file, rather than counting that
    # free-disk reserve as another output file.
    if shutil.disk_usage(budget.output).free<23*1024**3:raise OSError('unchanged 23GiB checkpoint disk reserve')
    if owned_bytes(budget.output)+upcoming_bytes>=budget.plan['limits']['max_output_bytes']:
        raise OSError('single-snapshot output cap including upcoming file')

class SingleSnapshot:
    def __init__(self):
        self.full_write_started=False;self.meta=None;self.path=None;self.saved_stage=None

    def save(self,directory,payload):
        if self.full_write_started:raise RuntimeError('one updated fullstate slot already attempted')
        self.full_write_started=True;self.saved_stage=payload['worker_stage']
        meta=support.save_full(directory,payload)
        if meta['bytes']>STORAGE_BOUND['maximum_full_file_bytes']:
            raise ValueError('updated fullstate size bound; preserve file without a second write')
        self.meta=meta;self.path=Path(directory)/'checkpoints'/meta['file']
        return meta

    def verify_current(self,payload):
        if self.meta is None:
            raise RuntimeError('updated full write incomplete; current recovery UNCONFIRMED; candidates preserved')
        if support.sha(self.path)!=self.meta['sha256']:
            raise ValueError('existing fullstate file identity changed')
        # A later status-publication error can change this descriptive field.
        # Every model/Adam/scaler/RNG/cursor/order/history/call field must match.
        normalized=dict(payload,worker_stage=self.saved_stage)
        if support.state_sha(normalized)!=self.meta['algorithm_sha256']:
            raise ValueError('existing fullstate is not the entire current state')
        return {'status':'CURRENT_FULLSTATE_VERIFIED_NO_DUPLICATE_WRITE',
                'path':str(self.path.resolve()),'sha256':self.meta['sha256'],
                'algorithm_sha256':self.meta['algorithm_sha256'],
                'only_worker_stage_normalized':True,'second_full_write':False}
