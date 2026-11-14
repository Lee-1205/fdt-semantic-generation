"""Control-flow failure injection only; no tensor operations or training."""
from types import SimpleNamespace
import unittest
from run_control import Budget, run_with_budget


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakeTrainer:
    def __init__(self, clock):
        self.clock = clock
        self.phase = self.phase_step = self.updates = self.attempts = 0
        self.elapsed = 0.
        self.plan = SimpleNamespace(phase_steps=(100, 100, 100))
        self.events = []
        self.history = []
        self.saved_updates = []
        self.unsafe = False
        self.fail_save = self.fail_step = False
        self.pass_gate = True

    def _step(self):
        self.events.append('step')
        self.unsafe = True
        self.clock.now += 1
        if self.fail_step:
            raise RuntimeError('partial optimizer step')
        self.updates += 1
        self.attempts += 1
        self.phase_step += 1
        self.unsafe = False
        return {'loss': 1.}, True

    def record_step(self, losses):
        self.events.append('history')
        self.history.append({'update': self.updates, **losses})

    def save(self):
        self.events.append('save')
        if self.unsafe:
            return
        if self.fail_save:
            self.unsafe = True  # The real wrapper marks persistence failure.
            raise RuntimeError('Drive upload failed')
        if not self.saved_updates or self.saved_updates[-1] != self.updates:
            self.clock.now += 3
            self.saved_updates.append(self.updates)
            assert not self.history or self.history[-1]['update'] == self.updates

    def validate_phase(self):
        self.events.append('gate')
        if not self.pass_gate:
            return False
        self.phase += 1
        self.phase_step = 0
        return True

    def finish(self, budget):
        self.events.append('finish')
        return not budget.expired()

    def state(self):
        return {'optimizer_updates': self.updates, 'phase': self.phase}


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.trainer = FakeTrainer(self.clock)

    def test_upload_time_counts_toward_unchanged_deadline(self):
        status = run_with_budget(self.trainer, 10, clock=self.clock)
        self.assertEqual(status['optimizer_updates'], 3)
        self.assertEqual(self.trainer.elapsed, 12)
        self.assertEqual(status['stop_reason'], 'time_budget')

    def test_history_is_committed_before_next_step(self):
        run_with_budget(self.trainer, 100, max_updates=2, clock=self.clock)
        self.assertEqual(self.trainer.events[:6], ['step', 'history', 'save'] * 2)

    def test_save_failure_prevents_next_update(self):
        self.trainer.fail_save = True
        with self.assertRaises(RuntimeError):
            run_with_budget(self.trainer, 100, clock=self.clock)
        self.assertEqual(self.trainer.events.count('step'), 1)
        self.assertEqual(self.trainer.saved_updates, [])

    def test_partial_step_is_never_saved(self):
        self.trainer.fail_step = True
        with self.assertRaises(RuntimeError):
            run_with_budget(self.trainer, 100, clock=self.clock)
        self.assertEqual(self.trainer.saved_updates, [])

    def test_phase_transition_saved_before_new_phase_step(self):
        self.trainer.plan.phase_steps = (1, 1, 1)
        status = run_with_budget(self.trainer, 100, clock=self.clock)
        self.assertEqual(status['stop_reason'], 'complete')
        for index, event in enumerate(self.trainer.events):
            if event == 'gate':
                self.assertEqual(self.trainer.events[index + 1], 'save')
        self.assertEqual(self.trainer.updates, 3)

    def test_failed_gate_stops_training(self):
        self.trainer.plan.phase_steps = (1, 1, 1)
        self.trainer.pass_gate = False
        status = run_with_budget(self.trainer, 100, clock=self.clock)
        self.assertEqual(status['stop_reason'], 'gate_failed')
        self.assertEqual(self.trainer.updates, 1)

    def test_complete_checkpoint_runs_only_remaining_evaluation(self):
        self.trainer.phase = 3
        status = run_with_budget(self.trainer, 100, clock=self.clock)
        self.assertEqual(status['stop_reason'], 'complete')
        self.assertNotIn('step', self.trainer.events)

    def test_zero_budget_starts_no_step(self):
        status = run_with_budget(self.trainer, 0, clock=self.clock)
        self.assertEqual(status['optimizer_updates'], 0)

    def test_invalid_time_budget_is_rejected(self):
        for value in (-1, float('inf'), float('nan'), '6'):
            with self.assertRaises(ValueError):
                Budget(value, self.clock)


if __name__ == '__main__':
    unittest.main()
