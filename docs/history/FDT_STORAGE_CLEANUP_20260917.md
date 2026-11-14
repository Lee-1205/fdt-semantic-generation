# FDT storage cleanup — 2026-09-17

Status: **BLOCKED_BEFORE_DELETE**

The user approved the exact cleanup scope in the latest request. No deletion was performed because the command runner reached its account usage limit before the pre-delete command could run. The limitation is recorded in `FDT_STORAGE_CLEANUP_RESUME_20260917.md`.

Last read-only measurement before the approval:

- C: free: 36.83GB
- C: used: 893.87GB
- C: total: 930.70GB
- FDT root: 136.72GB; `runs` 130.11GB; `prepared_data` 6.32GB

Deleted files: **0**

Critical checkpoints preserved: **YES**

Phase-2 evidence preserved: **YES**

Prepared data modified: **NO**

The approved deletion list and candidate paths are in `STORAGE_AUDIT_2026-09-17_KR.md`. This file must be updated only after the deletion command actually runs; the blocked status must not be replaced with a completion claim based on estimates.
