"""Linear MPC with a condensed QP and hard input constraints.

The upright linearisation is discretised with a zero-order hold, the finite
horizon cost is *condensed* onto the input sequence only

``J(U) = (Sx s0 + Su U - Xref)' Qbar (Sx s0 + Su U - Xref) + U' Rbar U``

and the resulting box-constrained QP is solved with FISTA. The terminal weight
is the discrete Riccati solution, which lets a short horizon inherit
infinite-horizon stability instead of being greedy.

The key difference from LQR: ``|u| <= force_limit`` is enforced *inside* the
optimisation, so the controller plans around saturation rather than being
clipped after the fact.
"""

from __future__ import annotations

import numpy as np

from cartpole.controllers.base import Controller
from cartpole.dynamics import ANGLE, CartPoleParams, linearize_upright, wrap_angle
from cartpole.linalg import discretize, solve_box_qp, solve_dare

DEFAULT_STATE_COST = np.diag([6.0, 1.0, 60.0, 2.0])
DEFAULT_INPUT_COST = np.array([[0.08]])


class MPCController(Controller):
    """Receding-horizon controller for the linearised cart-pole."""

    name = "MPC"

    def __init__(
        self,
        params: CartPoleParams,
        horizon: int = 30,
        control_dt: float = 0.01,
        state_cost: np.ndarray | None = None,
        input_cost: np.ndarray | None = None,
        qp_iterations: int = 120,
    ):
        self.params = params
        self.horizon = int(horizon)
        self.control_dt = float(control_dt)
        self.qp_iterations = int(qp_iterations)
        self.state_cost = DEFAULT_STATE_COST if state_cost is None else np.asarray(state_cost, dtype=float)
        self.input_cost = DEFAULT_INPUT_COST if input_cost is None else np.asarray(input_cost, dtype=float)

        continuous_a, continuous_b = linearize_upright(params)
        self.discrete_a, self.discrete_b = discretize(continuous_a, continuous_b, self.control_dt)
        self.terminal_cost = solve_dare(
            self.discrete_a, self.discrete_b, self.state_cost * self.control_dt, self.input_cost * self.control_dt
        )

        self._build_prediction_matrices()
        self.reset()

    def _build_prediction_matrices(self) -> None:
        """Pre-compute the condensed prediction and Hessian matrices (time invariant)."""
        n_states, horizon = 4, self.horizon

        free_response = np.zeros((n_states * horizon, n_states))
        forced_response = np.zeros((n_states * horizon, horizon))

        power = np.eye(n_states)
        for step in range(horizon):
            power = power @ self.discrete_a if step else self.discrete_a.copy()
            free_response[step * n_states : (step + 1) * n_states, :] = power

        for row in range(horizon):
            for col in range(row + 1):
                transition = np.linalg.matrix_power(self.discrete_a, row - col)
                block = (transition @ self.discrete_b).reshape(n_states)
                forced_response[row * n_states : (row + 1) * n_states, col] = block

        stage_weights = np.zeros((n_states * horizon, n_states * horizon))
        for step in range(horizon - 1):
            stage_weights[step * n_states : (step + 1) * n_states, step * n_states : (step + 1) * n_states] = (
                self.state_cost * self.control_dt
            )
        stage_weights[(horizon - 1) * n_states :, (horizon - 1) * n_states :] = self.terminal_cost

        input_weights = np.eye(horizon) * float(self.input_cost[0, 0]) * self.control_dt

        self.free_response = free_response
        self.forced_response = forced_response
        self.stage_weights = stage_weights
        self.hessian = 2.0 * (forced_response.T @ stage_weights @ forced_response + input_weights)
        self._weighted_forced = 2.0 * forced_response.T @ stage_weights

    def reset(self) -> None:
        self._previous_plan = np.zeros(self.horizon)

    def error_state(self, state: np.ndarray, reference: float) -> np.ndarray:
        error = np.asarray(state, dtype=float).copy()
        error[0] -= reference
        error[ANGLE] = wrap_angle(error[ANGLE])
        return error

    def compute(self, state: np.ndarray, time: float, reference: float) -> float:
        error = self.error_state(state, reference)
        gradient_offset = self._weighted_forced @ (self.free_response @ error)

        limit = self.params.force_limit
        warm_start = np.concatenate([self._previous_plan[1:], self._previous_plan[-1:]])
        plan = solve_box_qp(
            self.hessian,
            gradient_offset,
            lower=-limit,
            upper=limit,
            initial=warm_start,
            max_iterations=self.qp_iterations,
        )

        self._previous_plan = plan
        return float(plan[0])
