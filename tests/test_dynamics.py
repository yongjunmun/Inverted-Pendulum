"""Tests for the plant model: linearisation, energy conservation, equilibria."""

from __future__ import annotations

import unittest

import numpy as np

from cartpole.dynamics import (
    CartPoleParams,
    derivative,
    linearize_upright,
    numeric_jacobian,
    rk4_step,
    total_energy,
    wrap_angle,
)


class TestWrapAngle(unittest.TestCase):
    def test_maps_into_pi_interval(self):
        self.assertAlmostEqual(float(wrap_angle(0.0)), 0.0)
        self.assertAlmostEqual(float(wrap_angle(2.0 * np.pi + 0.3)), 0.3)
        self.assertAlmostEqual(float(wrap_angle(-2.0 * np.pi - 0.3)), -0.3)

    def test_interval_is_half_open_at_plus_pi(self):
        """``pi`` and its aliases wrap to ``-pi``, so the interval is ``[-pi, pi)``."""
        for angle in (np.pi, 3.0 * np.pi, -np.pi, -3.0 * np.pi):
            self.assertAlmostEqual(float(wrap_angle(angle)), -np.pi)

    def test_stays_inside_the_interval_for_a_wide_sweep(self):
        wrapped = wrap_angle(np.linspace(-40.0, 40.0, 20001))
        self.assertTrue(np.all(wrapped >= -np.pi - 1e-12))
        self.assertTrue(np.all(wrapped < np.pi))

    def test_is_vectorised(self):
        wrapped = wrap_angle(np.array([0.1, 2.0 * np.pi + 0.1, -2.0 * np.pi + 0.1]))
        np.testing.assert_allclose(wrapped, 0.1, atol=1e-12)


class TestEquilibria(unittest.TestCase):
    def setUp(self):
        self.params = CartPoleParams()

    def test_upright_is_an_equilibrium(self):
        rate = derivative(np.zeros(4), 0.0, self.params)
        np.testing.assert_allclose(rate, np.zeros(4), atol=1e-12)

    def test_hanging_is_an_equilibrium(self):
        rate = derivative(np.array([0.0, 0.0, np.pi, 0.0]), 0.0, self.params)
        np.testing.assert_allclose(rate, np.zeros(4), atol=1e-12)

    def test_upright_is_unstable_without_control(self):
        state = np.array([0.0, 0.0, 0.01, 0.0])
        for _ in range(2000):
            state = rk4_step(state, 0.0, 0.001, self.params)
        self.assertGreater(abs(state[2]), 0.5, "pole should fall when uncontrolled")


class TestLinearisation(unittest.TestCase):
    def test_analytic_matches_finite_differences(self):
        """The hand-derived Jacobians must agree with a numerical Jacobian."""
        params = CartPoleParams()
        analytic_a, analytic_b = linearize_upright(params)
        numeric_a, numeric_b = numeric_jacobian(np.zeros(4), 0.0, params)

        np.testing.assert_allclose(analytic_a, numeric_a, atol=1e-6)
        np.testing.assert_allclose(analytic_b, numeric_b, atol=1e-9)

    def test_holds_for_several_parameter_sets(self):
        for cart_mass, pole_mass, length in [(0.5, 0.2, 0.6), (1.0, 0.1, 1.2), (0.3, 0.9, 0.35)]:
            params = CartPoleParams(cart_mass=cart_mass, pole_mass=pole_mass, pole_length=length)
            analytic_a, analytic_b = linearize_upright(params)
            numeric_a, numeric_b = numeric_jacobian(np.zeros(4), 0.0, params)
            np.testing.assert_allclose(analytic_a, numeric_a, atol=1e-6)
            np.testing.assert_allclose(analytic_b, numeric_b, atol=1e-9)

    def test_has_exactly_one_unstable_pole(self):
        state_matrix, _ = linearize_upright(CartPoleParams())
        unstable = [value for value in np.linalg.eigvals(state_matrix) if value.real > 1e-9]
        self.assertEqual(len(unstable), 1)


class TestIntegrator(unittest.TestCase):
    def test_energy_is_conserved_without_damping_or_force(self):
        """RK4 accuracy check: a frictionless, unforced cart-pole conserves energy."""
        params = CartPoleParams(cart_damping=0.0, pole_damping=0.0)
        state = np.array([0.0, 0.0, 2.5, 0.0])

        initial = total_energy(state, params)
        for _ in range(5000):
            state = rk4_step(state, 0.0, 0.001, params)
        drift = abs(total_energy(state, params) - initial) / max(abs(initial), 1e-9)

        self.assertLess(drift, 1e-8, f"energy drifted by {drift:.2e} over 5 s")

    def test_step_size_convergence_is_fourth_order(self):
        """Halving the step should cut the error by roughly 2^4."""
        params = CartPoleParams(cart_damping=0.0, pole_damping=0.0)
        start = np.array([0.0, 0.0, 1.0, 0.5])

        def integrate(dt: float, horizon: float = 0.5) -> np.ndarray:
            state = start.copy()
            for _ in range(int(round(horizon / dt))):
                state = rk4_step(state, 1.0, dt, params)
            return state

        reference = integrate(1e-5)
        coarse = np.max(np.abs(integrate(4e-3) - reference))
        fine = np.max(np.abs(integrate(2e-3) - reference))

        self.assertGreater(coarse / max(fine, 1e-18), 8.0)


if __name__ == "__main__":
    unittest.main()
