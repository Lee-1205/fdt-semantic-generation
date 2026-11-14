"""Wall-clock and safe commit order, testable without PyTorch or training."""
import math
import time


class Budget:
    def __init__(self, seconds, clock=time.monotonic):
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('wall_seconds는 유한한 0 이상의 초 단위 값이어야 합니다.')
        self.clock = clock
        self.deadline = clock() + seconds

    def expired(self):
        return self.clock() >= self.deadline


def run_with_budget(trainer, wall_seconds=21600, max_updates=None, clock=time.monotonic):
    """The deadline is never reset by checkpoint uploads or phase transitions.

    A currently executing GPU step, upload, or evaluation completes before the
    next boundary check. This budget cannot enforce Colab VM lifetime.
    """
    budget = Budget(wall_seconds, clock)
    accounted = clock()
    first_update = trainer.updates
    reason = 'time_budget'

    def account_time():
        nonlocal accounted
        now = clock()
        trainer.elapsed += now - accounted
        accounted = now

    try:
        while trainer.phase < 3:
            if budget.expired():
                break
            if max_updates is not None and trainer.updates - first_update >= max_updates:
                reason = 'update_budget'
                break
            if trainer.phase_step >= trainer.plan.phase_steps[trainer.phase]:
                if not trainer.validate_phase():
                    reason = 'gate_failed'
                    break
                account_time()
                trainer.save()
                account_time()
                continue
            losses, successful = trainer._step()
            if successful:
                trainer.record_step(losses)
            account_time()
            # The history and exposure counters belong to the same committed step.
            # No following optimizer attempt is permitted before this returns.
            trainer.save()
            account_time()
        account_time()
        trainer.save()
        account_time()
        if trainer.phase == 3:
            reason = 'complete' if trainer.finish(budget) else 'evaluation_pending'
            account_time()
    except BaseException:
        account_time()
        # The wrapper refuses this save for a partial optimizer step or a failed upload.
        trainer.save()
        raise
    return {**trainer.state(), 'stop_reason': reason,
            'colab_runtime_end_prevention': False}
