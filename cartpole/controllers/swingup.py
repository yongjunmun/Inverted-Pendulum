"""Energy-shaping swing-up with an LQR catch.

Starting from hanging down, no linear controller can help: the state is nowhere
near the operating point the linearisation is valid at. The standard solution is
energy shaping. For a pole with energy ``E = 0.5 J th_dot^2 + m g l cos(th)``,
differentiating along the dynamics gives

``E_dot = -m l a th_dot cos(th)``   (``a`` is the cart acceleration)

so choosing ``a = k (E - E_target) th_dot cos(th)`` makes
``E_dot = -m l k (E - E_target) (th_dot cos(th))^2``, which drives the energy
monotonically toward the energy of the upright equilibrium.

The requested cart acceleration is then turned into a force by *collocated
partial feedback linearisation* (inverting the cart row of the equations of
motion exactly), and control is handed over to LQR once the state enters the
region where the linear design is trustworthy.
"""

from __future__ import annotations

import numpy as np

from cartpole.controllers.base import Controller
from cartpole.controllers.lqr import LQRController
from cartpole.dynamics import (
    ANGLE,
    POSITION,
    RATE,
    VELOCITY,
    CartPoleParams,
    pole_energy,
    target_energy,
    wrap_angle,
)


class SwingUpLQR(Controller):
    """Hybrid controller: energy pumping, then LQR capture near upright."""

    name = "SwingUp+LQR"

    def __init__(
        self,
        params: CartPoleParams,
        lqr: LQRController | None = None,
        energy_gain: float = 14.0,
        centering_p: float = 1.6,
        centering_d: float = 2.2,
        acceleration_limit: float = 12.0,
        catch_angle: float = 0.35,
        catch_value: float = 8.0,
    ):
        self.params = params
        self.lqr = lqr or LQRController(params)
        self.energy_gain = energy_gain
        self.centering_p = centering_p
        self.centering_d = centering_d
        self.acceleration_limit = acceleration_limit
        self.catch_angle = catch_angle
        self.catch_value = catch_value
        self.reset()

    def reset(self) -> None:
        self._captured = False
        self.lqr.reset()

    def mode(self) -> str:
        return "LQR catch" if self._captured else "energy swing-up"

    def _should_capture(self, state: np.ndarray, reference: float) -> bool:
        """Switch to LQR inside its region of attraction.

        The test combines a plain angle window with the LQR value function
        ``e' P e``, which is a Lyapunov function for the linear closed loop and
        therefore a far better proxy for "the catch will succeed" than angle alone.
        """
        if abs(wrap_angle(state[ANGLE])) > self.catch_angle:
            return False
        error = self.lqr.error_state(state, reference)
        return float(error @ self.lqr.riccati @ error) < self.catch_value

    def _swing_up_force(self, state: np.ndarray, reference: float) -> float:
        params = self.params
        _, velocity, angle, rate = state
        sin_angle, cos_angle = np.sin(angle), np.cos(angle)

        energy_error = pole_energy(state, params) - target_energy(params)
        desired_accel = self.energy_gain * energy_error * rate * cos_angle

        # Keep the cart near the reference so it does not run off the rail while pumping.
        desired_accel += -self.centering_p * (state[POSITION] - reference) - self.centering_d * velocity
        desired_accel = float(np.clip(desired_accel, -self.acceleration_limit, self.acceleration_limit))

        # Break away from the hanging equilibrium, where th_dot * cos(th) == 0
        # makes the energy law produce exactly zero effort.
        if abs(rate) < 1e-2 and energy_error < -1e-3:
            desired_accel += self.acceleration_limit * (1.0 if cos_angle < 0 else -1.0) * 0.25

        coupling = params.pole_mass * params.half_length
        pole_accel = (
            params.pole_mass * params.gravity * params.half_length * sin_angle
            - coupling * cos_angle * desired_accel
            - params.pole_damping * rate
        ) / params.pivot_inertia

        return float(
            (params.cart_mass + params.pole_mass) * desired_accel
            + coupling * cos_angle * pole_accel
            - coupling * sin_angle * rate**2
            + params.cart_damping * velocity
        )

    def compute(self, state: np.ndarray, time: float, reference: float) -> float:
        if not self._captured and self._should_capture(state, reference):
            self._captured = True
        if self._captured:
            return self.lqr.compute(state, time, reference)
        return self._swing_up_force(np.asarray(state, dtype=float), reference)
