"""Same one updated fullstate; only the INITIAL reference format is new."""
import codec_single_snapshot_v40 as base

SingleSnapshot = base.SingleSnapshot
support = base.support
check_checkpoint_write = base.check_checkpoint_write
STORAGE_BOUND = dict(base.STORAGE_BOUND)
POLICY = dict(base.POLICY, initial_checkpoint='lossless_gzip_parent_reference_v42')
