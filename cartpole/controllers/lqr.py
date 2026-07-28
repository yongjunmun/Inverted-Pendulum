"""Infinite-horizon LQR designed on the upright linearisation.

The plant is linearised analytically about ``theta = 0``, the continuous-time
algebraic Riccati equation is solved from scratch (see :mod:`cartpole.linalg`),
and the resulting gain is applied to the *nonlinear* plant as ``u = -K (s - s_ref)``.

Because cart position enters the model as a pure integrator, no integral action
is needed to remove steady-state position error: the only equilibrium with
``theta = 0`` and zero velocities is ``x = x_ref`` with ``u = 0``.
"""

from __future__ import annotations

import numpy as np

from cartpole.controllers.base import Controller
from cartpole.dynamics import ANGLE, CartPoleParams, linearize_upright, wrap_angle
from cartpole.linalg import lqr_gain

DEFAULT_STATE_COST = np.diag([6.0, 1.0, 60.0, 2.0])
"""Weights on ``[x, x_dot, theta, theta_dot]``; angle is punished hardest."""

DEFAULT_INPUT_COST = np.array([[0.08]])
"""Weight on force. Lower = more aggressive, more likely to saturate."""


class LQRController(Controller):
    """Linear quadratic regulator about the upright equilibrium."""

    name = "LQR"

    def __init__(
        self,
        params: CartPoleParams,
        state_cost: np.ndarray | None = None,
        input_cost: np.ndarray | None = None,
    ):
        self.params = params
        self.state_cost = DEFAULT_STATE_COST if state_cost is None else np.asarray(state_cost, dtype=float)
        self.input_cost = DEFAULT_INPUT_COST if input_cost is None else np.asarray(input_cost, dtype=float)

        state_matrix, input_matrix = linearize_upright(params)
        self.state_matrix = state_matrix
        self.input_matrix = input_matrix
        self.gain, self.riccati = lqr_gain(state_matrix, input_matrix, self.state_cost, self.input_cost)

    @property
    def closed_loop_poles(self) -> np.ndarray:
        """Eigenvalues of ``A - BK``; all should sit in the left half plane."""
        return np.linalg.eigvals(self.state_matrix - self.input_matrix @ self.gain)

    def error_state(self, state: np.ndarray, reference: float) -> np.ndarray:
        """State error with the pole angle wrapped into ``(-pi, pi]``."""
        error = np.asarray(state, dtype=float).copy()
        error[0] -= reference
        error[ANGLE] = wrap_angle(error[ANGLE])
        return error

    def compute(self, state: np.ndarray, time: float, reference: float) -> float:
        return float(-(self.gain @ self.error_state(state, reference))[0])
