"""Adapter that lets a trained policy run in the same benchmark as PID/LQR/MPC.

Learned policies are scored by exactly the same harness, on exactly the same
scenarios, with the same actuator limit and control rate. Without this the
comparison would be meaningless: most published RL cart-pole results run on a
different plant, a different reward and a different time step from the classical
baselines they are implicitly compared against.
"""

from __future__ import annotations

import numpy as np

from cartpole.controllers.base import Controller
from cartpole.dynamics import ANGLE, CartPoleParams, wrap_angle
from cartpole.learning.policies import Policy


class LearnedController(Controller):
    """Wrap a trained :class:`Policy` in the :class:`Controller` interface."""

    def __init__(self, policy: Policy, params: CartPoleParams, name: str | None = None):
        self.policy = policy
        self.params = params
        self.name = name or f"{policy.name}-learned"

    def error_state(self, state: np.ndarray, reference: float) -> np.ndarray:
        """Same error convention as the LQR controller, so tracking works too."""
        error = np.asarray(state, dtype=float).copy()
        error[0] -= reference
        error[ANGLE] = wrap_angle(error[ANGLE])
        return error

    def compute(self, state: np.ndarray, time: float, reference: float) -> float:
        return float(self.policy.act(self.error_state(state, reference)))
