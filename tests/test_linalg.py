"""Tests for the from-scratch numerical linear algebra."""

from __future__ import annotations

import unittest

import numpy as np

from cartpole.dynamics import CartPoleParams, linearize_upright
from cartpole.linalg import discretize, expm, lqr_gain, solve_box_qp, solve_care, solve_dare


class TestMatrixExponential(unittest.TestCase):
    def test_diagonal_case(self):
        matrix = np.diag([0.5, -2.0, 3.0])
        np.testing.assert_allclose(expm(matrix), np.diag(np.exp([0.5, -2.0, 3.0])), atol=1e-10)

    def test_rotation_generator(self):
        angle = 1.3
        generator = np.array([[0.0, -angle], [angle, 0.0]])
        expected = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        np.testing.assert_allclose(expm(generator), expected, atol=1e-10)

    def test_large_norm_uses_scaling_and_squaring(self):
        matrix = np.diag([12.0, -9.0])
        np.testing.assert_allclose(expm(matrix), np.diag(np.exp([12.0, -9.0])), rtol=1e-9)


class TestDiscretisation(unittest.TestCase):
    def test_matches_fine_grained_integration(self):
        """Zero-order-hold discretisation must match a finely integrated response."""
        state_matrix, input_matrix = linearize_upright(CartPoleParams())
        dt = 0.01
        discrete_a, discrete_b = discretize(state_matrix, input_matrix, dt)

        state = np.array([0.1, -0.2, 0.05, 0.3])
        force = 2.5

        fine = state.copy()
        steps = 20000
        for _ in range(steps):
            fine = fine + (dt / steps) * (state_matrix @ fine + input_matrix[:, 0] * force)

        predicted = discrete_a @ state + discrete_b[:, 0] * force
        np.testing.assert_allclose(predicted, fine, atol=1e-5)


class TestRiccati(unittest.TestCase):
    def setUp(self):
        self.state_matrix, self.input_matrix = linearize_upright(CartPoleParams())
        self.state_cost = np.diag([6.0, 1.0, 60.0, 2.0])
        self.input_cost = np.array([[0.08]])

    def test_care_residual_is_zero(self):
        riccati = solve_care(self.state_matrix, self.input_matrix, self.state_cost, self.input_cost)
        residual = (
            self.state_matrix.T @ riccati
            + riccati @ self.state_matrix
            - riccati @ self.input_matrix @ np.linalg.inv(self.input_cost) @ self.input_matrix.T @ riccati
            + self.state_cost
        )
        self.assertLess(np.max(np.abs(residual)), 1e-8)

    def test_care_solution_is_symmetric_positive_definite(self):
        riccati = solve_care(self.state_matrix, self.input_matrix, self.state_cost, self.input_cost)
        np.testing.assert_allclose(riccati, riccati.T, atol=1e-10)
        self.assertGreater(np.min(np.linalg.eigvalsh(riccati)), 0.0)

    def test_lqr_closed_loop_is_stable(self):
        gain, _ = lqr_gain(self.state_matrix, self.input_matrix, self.state_cost, self.input_cost)
        poles = np.linalg.eigvals(self.state_matrix - self.input_matrix @ gain)
        self.assertLess(np.max(poles.real), 0.0)

    def test_dare_residual_is_zero_and_closed_loop_is_stable(self):
        discrete_a, discrete_b = discretize(self.state_matrix, self.input_matrix, 0.01)
        riccati = solve_dare(discrete_a, discrete_b, self.state_cost, self.input_cost)

        weighted = self.input_cost + discrete_b.T @ riccati @ discrete_b
        gain = np.linalg.solve(weighted, discrete_b.T @ riccati @ discrete_a)
        residual = (
            discrete_a.T @ riccati @ discrete_a
            - discrete_a.T @ riccati @ discrete_b @ gain
            + self.state_cost
            - riccati
        )
        self.assertLess(np.max(np.abs(residual)), 1e-8)

        poles = np.linalg.eigvals(discrete_a - discrete_b @ gain)
        self.assertLess(np.max(np.abs(poles)), 1.0)


class TestBoxQP(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        root = rng.normal(size=(6, 6))
        self.hessian = root @ root.T + 0.5 * np.eye(6)
        self.gradient = rng.normal(size=6)

    def test_matches_unconstrained_solution_when_bounds_are_loose(self):
        expected = np.linalg.solve(self.hessian, -self.gradient)
        solution = solve_box_qp(self.hessian, self.gradient, -1e3, 1e3, max_iterations=5000, tolerance=1e-14)
        np.testing.assert_allclose(solution, expected, atol=1e-6)

    def test_respects_tight_bounds(self):
        solution = solve_box_qp(self.hessian, self.gradient, -0.05, 0.05, max_iterations=2000)
        self.assertLessEqual(float(np.max(solution)), 0.05 + 1e-12)
        self.assertGreaterEqual(float(np.min(solution)), -0.05 - 1e-12)

    def test_constrained_solution_satisfies_kkt(self):
        lower, upper = -0.2, 0.2
        solution = solve_box_qp(self.hessian, self.gradient, lower, upper, max_iterations=20000, tolerance=1e-15)
        gradient = self.hessian @ solution + self.gradient

        interior = (solution > lower + 1e-7) & (solution < upper - 1e-7)
        self.assertLess(np.max(np.abs(gradient[interior])) if np.any(interior) else 0.0, 1e-5)
        self.assertTrue(np.all(gradient[solution <= lower + 1e-7] >= -1e-5))
        self.assertTrue(np.all(gradient[solution >= upper - 1e-7] <= 1e-5))


if __name__ == "__main__":
    unittest.main()
