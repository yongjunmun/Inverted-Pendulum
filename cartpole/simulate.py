"""Closed-loop simulation harness.

The plant integrates at a fine step (default 1 kHz) while the controller runs on
a slower clock (default 100 Hz) with a zero-order hold, which is how a real
embedded controller behaves. Sensor noise, actuator saturation and external
disturbance pushes are all optional and reproducible from a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cartpole.controllers.base import Controller
from cartpole.dynamics import ANGLE, POSITION, CartPoleParams, rk4_step, wrap_angle

ReferenceFn = Callable[[float], float]


def constant_reference(value: float = 0.0) -> ReferenceFn:
    """Reference generator holding the cart at a fixed position."""

    def reference(time: float) -> float:
        return value

    return reference


def step_reference(value: float, step_time: float) -> ReferenceFn:
    """Reference generator that steps the cart target at ``step_time``."""

    def reference(time: float) -> float:
        return value if time >= step_time else 0.0

    return reference


@dataclass
class SimConfig:
    """Everything that defines a run apart from the controller and the plant."""

    duration: float = 6.0
    sim_dt: float = 0.001
    control_dt: float = 0.01

    angle_noise_std: float = 0.0
    """Standard deviation of the pole angle measurement noise [rad]."""
    position_noise_std: float = 0.0
    """Standard deviation of the cart position measurement noise [m]."""
    rate_noise_std: float = 0.0
    """Standard deviation of the velocity/rate measurement noise."""

    disturbance_force: float = 0.0
    """Amplitude of an external push on the cart [N]."""
    disturbance_time: float = 2.0
    """When the push starts [s]."""
    disturbance_duration: float = 0.05
    """How long the push lasts [s]."""

    seed: int = 0

    def __post_init__(self) -> None:
        steps_per_control = self.control_dt / self.sim_dt
        if abs(steps_per_control - round(steps_per_control)) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of sim_dt")


@dataclass
class SimResult:
    """Time histories produced by :func:`simulate`."""

    time: np.ndarray
    states: np.ndarray
    """Shape ``(N, 4)``: ``[x, x_dot, theta, theta_dot]`` at every logged step."""
    commanded_force: np.ndarray
    """Controller output *before* saturation [N]."""
    applied_force: np.ndarray
    """Force actually applied to the cart, after saturation [N]."""
    disturbance: np.ndarray
    reference: np.ndarray
    modes: list[str] = field(default_factory=list)
    controller_name: str = ""
    scenario_name: str = ""
    params: CartPoleParams = field(default_factory=CartPoleParams)

    @property
    def angle(self) -> np.ndarray:
        """Pole angle wrapped into ``[-pi, pi)`` [rad]."""
        return np.asarray(wrap_angle(self.states[:, ANGLE]), dtype=float)

    @property
    def position_error(self) -> np.ndarray:
        return self.states[:, POSITION] - self.reference

    @property
    def saturated_fraction(self) -> float:
        """Fraction of the run spent against the force limit."""
        limit = self.params.force_limit
        return float(np.mean(np.abs(self.commanded_force) >= limit - 1e-9))


def simulate(
    controller: Controller,
    params: CartPoleParams,
    config: SimConfig | None = None,
    initial_state: np.ndarray | None = None,
    reference: ReferenceFn | None = None,
    scenario_name: str = "",
    log_every: int = 10,
) -> SimResult:
    """Run one closed-loop experiment and return the logged trajectories.

    Args:
        controller: control law under test; :meth:`Controller.reset` is called first.
        params: the *true* plant, which may differ from the controller's model.
        config: timing, noise and disturbance settings.
        initial_state: ``[x, x_dot, theta, theta_dot]``; defaults to a 0.2 rad tilt.
        reference: cart position setpoint as a function of time.
        scenario_name: label carried through to plots and the results table.
        log_every: log one sample every ``log_every`` integration steps.
    """
    config = config or SimConfig()
    reference = reference or constant_reference(0.0)
    state = np.array([0.0, 0.0, 0.2, 0.0]) if initial_state is None else np.asarray(initial_state, dtype=float).copy()

    controller.reset()
    rng = np.random.default_rng(config.seed)

    total_steps = int(round(config.duration / config.sim_dt))
    control_interval = int(round(config.control_dt / config.sim_dt))

    times, states, commanded, applied, disturbances, references, modes = [], [], [], [], [], [], []
    command = 0.0

    for step in range(total_steps + 1):
        time = step * config.sim_dt
        setpoint = reference(time)

        if step % control_interval == 0:
            measurement = state.copy()
            if config.position_noise_std:
                measurement[0] += rng.normal(0.0, config.position_noise_std)
            if config.rate_noise_std:
                measurement[1] += rng.normal(0.0, config.rate_noise_std)
                measurement[3] += rng.normal(0.0, config.rate_noise_std)
            if config.angle_noise_std:
                measurement[2] += rng.normal(0.0, config.angle_noise_std)
            command = float(controller.compute(measurement, time, setpoint))

        saturated = float(np.clip(command, -params.force_limit, params.force_limit))
        push = (
            config.disturbance_force
            if config.disturbance_time <= time < config.disturbance_time + config.disturbance_duration
            else 0.0
        )

        if step % log_every == 0 or step == total_steps:
            times.append(time)
            states.append(state.copy())
            commanded.append(command)
            applied.append(saturated)
            disturbances.append(push)
            references.append(setpoint)
            modes.append(controller.mode())

        if step < total_steps:
            state = rk4_step(state, saturated + push, config.sim_dt, params)

    return SimResult(
        time=np.array(times),
        states=np.array(states),
        commanded_force=np.array(commanded),
        applied_force=np.array(applied),
        disturbance=np.array(disturbances),
        reference=np.array(references),
        modes=modes,
        controller_name=controller.name,
        scenario_name=scenario_name,
        params=params,
    )
