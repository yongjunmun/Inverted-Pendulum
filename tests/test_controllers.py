"""Closed-loop tests: every controller must actually stabilise the plant."""

from __future__ import annotations

import unittest

import numpy as np

from cartpole.controllers import CascadedPID, LQRController, MPCController, SwingUpLQR
from cartpole.dynamics import CartPoleParams
from cartpole.metrics import evaluate
from cartpole.scenarios import swing_up_scenario
from cartpole.simulate import SimConfig, simulate, step_reference

PARAMS = CartPoleParams()
TILTED = np.array([0.0, 0.0, 0.2, 0.0])


def build(name: str):
    if name == "PID":
        return CascadedPID(control_dt=0.01, force_limit=PARAMS.force_limit)
    if name == "LQR":
        return LQRController(PARAMS)
    return MPCController(PARAMS, control_dt=0.01)


class TestBalancingControllers(unittest.TestCase):
    def test_all_recover_from_a_tilt(self):
        for name in ("PID", "LQR", "MPC"):
            with self.subTest(controller=name):
                result = simulate(build(name), PARAMS, config=SimConfig(duration=5.0), initial_state=TILTED)
                metrics = evaluate(result)
                self.assertTrue(metrics.success, f"{name} failed to stabilise")
                self.assertLess(metrics.rms_angle_deg, 5.0)

    def test_all_track_a_cart_step(self):
        for name in ("PID", "LQR", "MPC"):
            with self.subTest(controller=name):
                result = simulate(
                    build(name),
                    PARAMS,
                    config=SimConfig(duration=6.0),
                    initial_state=np.zeros(4),
                    reference=step_reference(1.0, 0.5),
                )
                self.assertLess(abs(evaluate(result).final_position_error_m), 0.1)

    def test_applied_force_never_exceeds_the_actuator_limit(self):
        for name in ("PID", "LQR", "MPC"):
            with self.subTest(controller=name):
                result = simulate(build(name), PARAMS, config=SimConfig(duration=3.0), initial_state=TILTED)
                self.assertLessEqual(float(np.max(np.abs(result.applied_force))), PARAMS.force_limit + 1e-9)

    def test_runs_are_reproducible(self):
        config = SimConfig(duration=2.0, angle_noise_std=0.01, seed=3)
        first = simulate(build("LQR"), PARAMS, config=config, initial_state=TILTED)
        second = simulate(build("LQR"), PARAMS, config=config, initial_state=TILTED)
        np.testing.assert_allclose(first.states, second.states)


class TestLQR(unittest.TestCase):
    def test_closed_loop_poles_are_stable(self):
        self.assertLess(float(np.max(LQRController(PARAMS).closed_loop_poles.real)), 0.0)

    def test_heavier_angle_weight_gives_a_larger_angle_gain(self):
        relaxed = LQRController(PARAMS, state_cost=np.diag([6.0, 1.0, 10.0, 2.0]))
        strict = LQRController(PARAMS, state_cost=np.diag([6.0, 1.0, 200.0, 2.0]))
        self.assertGreater(abs(strict.gain[0, 2]), abs(relaxed.gain[0, 2]))


class TestMPC(unittest.TestCase):
    def test_planned_input_honours_the_hard_constraint(self):
        """The QP must return commands inside the box, not rely on clipping."""
        controller = MPCController(PARAMS, control_dt=0.01)
        result = simulate(controller, PARAMS, config=SimConfig(duration=3.0), initial_state=np.array([0.0, 0.0, 0.4, 0.0]))
        self.assertLessEqual(float(np.max(np.abs(result.commanded_force))), PARAMS.force_limit + 1e-6)

    def test_tighter_force_limit_reduces_peak_force(self):
        weak_plant = CartPoleParams(force_limit=4.0)
        controller = MPCController(weak_plant, control_dt=0.01)
        result = simulate(controller, weak_plant, config=SimConfig(duration=4.0), initial_state=TILTED)
        self.assertLessEqual(float(np.max(np.abs(result.commanded_force))), 4.0 + 1e-6)
        self.assertTrue(evaluate(result).success)

    def test_a_short_horizon_still_stabilises_thanks_to_the_terminal_cost(self):
        controller = MPCController(PARAMS, horizon=8, control_dt=0.01)
        result = simulate(controller, PARAMS, config=SimConfig(duration=5.0), initial_state=TILTED)
        self.assertTrue(evaluate(result).success)


class TestSwingUp(unittest.TestCase):
    def test_swings_up_from_hanging_and_catches(self):
        scenario = swing_up_scenario()
        controller = SwingUpLQR(PARAMS)
        result = simulate(
            controller,
            PARAMS,
            config=scenario.config,
            initial_state=scenario.initial_state,
            scenario_name=scenario.name,
        )
        metrics = evaluate(result, settle_from=scenario.settle_from)

        self.assertTrue(metrics.success)
        self.assertIn("LQR catch", result.modes)
        self.assertLess(float(np.max(np.abs(result.states[:, 0]))), PARAMS.rail_limit)

    def test_energy_error_shrinks_during_the_swing(self):
        from cartpole.dynamics import pole_energy, target_energy

        scenario = swing_up_scenario()
        result = simulate(
            SwingUpLQR(PARAMS),
            PARAMS,
            config=scenario.config,
            initial_state=scenario.initial_state,
        )
        errors = np.array([abs(pole_energy(state, PARAMS) - target_energy(PARAMS)) for state in result.states])
        early = float(np.mean(errors[result.time < 0.5]))
        late = float(np.mean(errors[(result.time > 2.0) & (result.time < 2.7)]))
        self.assertLess(late, early)


if __name__ == "__main__":
    unittest.main()
