"""Tests for the scoring code, so the results table can be trusted."""

from __future__ import annotations

import unittest

import numpy as np

from cartpole.dynamics import CartPoleParams
from cartpole.metrics import evaluate, settling_time, to_markdown_table
from cartpole.simulate import SimResult


def make_result(time, angle, position=None, force=None) -> SimResult:
    states = np.zeros((len(time), 4))
    states[:, 0] = np.zeros(len(time)) if position is None else position
    states[:, 2] = angle
    forces = np.zeros(len(time)) if force is None else force
    return SimResult(
        time=np.asarray(time, dtype=float),
        states=states,
        commanded_force=forces,
        applied_force=forces,
        disturbance=np.zeros(len(time)),
        reference=np.zeros(len(time)),
        modes=["test"] * len(time),
        controller_name="test",
        scenario_name="test",
        params=CartPoleParams(),
    )


class TestSettlingTime(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 10.0, 1001)

    def test_zero_when_never_outside_the_band(self):
        self.assertEqual(settling_time(self.time, np.zeros_like(self.time), 0.1), 0.0)

    def test_finds_the_last_band_exit(self):
        signal = np.where(self.time < 4.0, 1.0, 0.0)
        self.assertAlmostEqual(settling_time(self.time, signal, 0.1), 4.0, places=2)

    def test_nan_when_still_outside_at_the_end(self):
        self.assertTrue(np.isnan(settling_time(self.time, np.ones_like(self.time), 0.1)))

    def test_ignores_early_returns_into_the_band(self):
        """A signal that dips back inside then leaves again is not settled early."""
        signal = np.zeros_like(self.time)
        signal[(self.time > 1.0) & (self.time < 2.0)] = 1.0
        signal[(self.time > 6.0) & (self.time < 7.0)] = 1.0
        self.assertGreater(settling_time(self.time, signal, 0.1), 6.9)


class TestEvaluate(unittest.TestCase):
    def test_decaying_run_is_a_success(self):
        time = np.linspace(0.0, 6.0, 601)
        result = make_result(time, 0.2 * np.exp(-2.0 * time))
        metrics = evaluate(result)

        self.assertTrue(metrics.success)
        self.assertAlmostEqual(metrics.peak_angle_deg, np.rad2deg(0.2), places=3)
        self.assertLess(metrics.settling_time_s, 3.0)

    def test_fallen_pole_is_a_failure(self):
        time = np.linspace(0.0, 6.0, 601)
        result = make_result(time, np.linspace(0.0, np.pi, 601))
        self.assertFalse(evaluate(result).success)

    def test_runaway_cart_is_a_failure(self):
        time = np.linspace(0.0, 6.0, 601)
        result = make_result(time, np.zeros(601), position=np.linspace(0.0, 5.0, 601))
        self.assertFalse(evaluate(result).success)

    def test_control_effort_integrates_the_squared_force(self):
        time = np.linspace(0.0, 2.0, 2001)
        result = make_result(time, np.zeros_like(time), force=np.full_like(time, 3.0))
        self.assertAlmostEqual(evaluate(result).control_effort_n2s, 18.0, places=1)

    def test_settle_from_ignores_an_early_transient(self):
        time = np.linspace(0.0, 8.0, 801)
        angle = np.where(time < 3.0, 1.5, 0.0)
        self.assertTrue(evaluate(make_result(time, angle), settle_from=3.5).success)
        self.assertFalse(evaluate(make_result(time, angle)).success)


class TestMarkdownTable(unittest.TestCase):
    def test_renders_one_row_per_metric(self):
        time = np.linspace(0.0, 6.0, 601)
        rows = [evaluate(make_result(time, 0.2 * np.exp(-2.0 * time)))]
        table = to_markdown_table(rows)

        self.assertIn("| Scenario |", table)
        self.assertEqual(len(table.splitlines()), 3)

    def test_unsettled_runs_are_labelled(self):
        time = np.linspace(0.0, 6.0, 601)
        rows = [evaluate(make_result(time, np.full_like(time, 0.5)))]
        self.assertIn("not settled", to_markdown_table(rows))


if __name__ == "__main__":
    unittest.main()
