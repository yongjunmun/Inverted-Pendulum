"""Cascaded PID: an outer cart-position loop driving an inner angle loop.

A single PID on the pole angle can hold the pole upright but cannot hold the
cart still, because the cart position is invisible to that loop. The classic
industrial fix is a cascade: a slow outer loop converts cart-position error into
a small *tilt setpoint*, and a fast inner loop tracks that tilt.

The outer-loop sign is the subtle part, and it is where the cart-pole earns its
reputation. Setting ``theta_ddot = 0`` in the pole equation gives
``x_ddot = g tan(theta)``: while the inner loop *holds* a lean, the cart must
chase it, so a **positive** tilt setpoint moves the cart in **+x**. That is the
opposite of the instantaneous open-loop response ``dx_ddot/dtheta < 0`` at fixed
force, because the plant is **non-minimum phase**: to end up moving right the
cart first has to duck left. The outer loop must therefore be much slower than
the inner loop (here about 1.0 rad/s against 12.3 rad/s) or the two fight and
the cart limit-cycles against the tilt saturation limit.

The integral term is deliberately small. Cart position enters the plant as a
free integrator, so there is no steady-state position error for it to remove; it
is kept only to reject a constant actuator bias, and it costs a little
steady-state wander in the bias-free benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cartpole.controllers.base import Controller
from cartpole.dynamics import ANGLE, POSITION, RATE, VELOCITY, wrap_angle


@dataclass
class PIDGains:
    """Gains for the cascaded loop.

    Tuned by grid search over the whole benchmark suite. The ratios that matter
    are ``angle_d / angle_p ~ 0.18`` and ``position_d / position_p ~ 1.3``;
    pushing the outer loop faster than about 1 rad/s makes the two loops fight
    over the non-minimum-phase response and the cart limit-cycles.
    """

    angle_p: float = 40.0
    angle_i: float = 6.0
    angle_d: float = 7.0
    position_p: float = 0.10
    position_d: float = 0.13
    tilt_limit: float = 0.20
    """Maximum tilt setpoint the outer loop may request [rad]."""
    integral_limit: float = 0.3
    """Clamp on the integrated angle error [rad.s], for anti-windup."""
    setpoint_filter: float = 0.04
    """First-order filter time constant on the tilt setpoint [s]."""


class CascadedPID(Controller):
    """Outer position loop plus inner angle loop with anti-windup."""

    name = "PID"

    def __init__(self, gains: PIDGains | None = None, control_dt: float = 0.01, force_limit: float = 10.0):
        self.gains = gains or PIDGains()
        self.control_dt = control_dt
        self.force_limit = force_limit
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._tilt_setpoint = 0.0
        self._previous_tilt = 0.0

    def compute(self, state: np.ndarray, time: float, reference: float) -> float:
        gains = self.gains
        dt = self.control_dt

        # --- Outer loop: cart position error -> tilt setpoint -------------
        # Lean *toward* the target and let the inner loop chase the lean.
        position_error = reference - state[POSITION]
        raw_tilt = gains.position_p * position_error - gains.position_d * state[VELOCITY]
        raw_tilt = float(np.clip(raw_tilt, -gains.tilt_limit, gains.tilt_limit))

        # Filter the setpoint so the inner loop is not kicked by steps.
        blend = dt / (gains.setpoint_filter + dt)
        self._previous_tilt = self._tilt_setpoint
        self._tilt_setpoint += blend * (raw_tilt - self._tilt_setpoint)
        setpoint_rate = (self._tilt_setpoint - self._previous_tilt) / dt

        # --- Inner loop: track the tilt setpoint ---------------------------
        angle_error = wrap_angle(state[ANGLE] - self._tilt_setpoint)
        rate_error = state[RATE] - setpoint_rate

        unsaturated = (
            gains.angle_p * angle_error + gains.angle_i * self._integral + gains.angle_d * rate_error
        )
        force = float(np.clip(unsaturated, -self.force_limit, self.force_limit))

        # Conditional integration: stop winding up while the actuator is pinned
        # and the error would push it further into saturation.
        saturated = abs(unsaturated - force) > 1e-9
        if not (saturated and np.sign(angle_error) == np.sign(unsaturated)):
            self._integral = float(
                np.clip(self._integral + angle_error * dt, -gains.integral_limit, gains.integral_limit)
            )

        return force
