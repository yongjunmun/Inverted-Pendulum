"""Named benchmark scenarios.

Every balancing controller is run on exactly the same set of scenarios, with the
same seed, the same actuator limit and the same control rate, so the results
table is a fair comparison rather than four separately tuned demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cartpole.simulate import ReferenceFn, SimConfig, constant_reference, step_reference


@dataclass
class Scenario:
    """One reproducible experiment definition."""

    name: str
    description: str
    initial_state: np.ndarray
    config: SimConfig
    reference: ReferenceFn = field(default_factory=lambda: constant_reference(0.0))
    settle_from: float = 0.0
    requires_swing_up: bool = False


def balancing_scenarios() -> list[Scenario]:
    """Scenarios every balancing controller must handle."""
    return [
        Scenario(
            name="regulation",
            description="Recover from an initial 11.5 deg tilt and hold the cart at x = 0.",
            initial_state=np.array([0.0, 0.0, 0.2, 0.0]),
            config=SimConfig(duration=5.0),
        ),
        Scenario(
            name="disturbance",
            description="Balanced upright, then hit with a 12 N impulse push for 50 ms at t = 1 s.",
            initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
            config=SimConfig(
                duration=5.0,
                disturbance_force=12.0,
                disturbance_time=1.0,
                disturbance_duration=0.05,
            ),
        ),
        Scenario(
            name="tracking",
            description="Move the cart 1 m to the right at t = 0.5 s without dropping the pole.",
            initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
            config=SimConfig(duration=6.0),
            reference=step_reference(1.0, 0.5),
        ),
        Scenario(
            name="noisy-sensors",
            description="Regulation with 0.5 deg angle noise, 2 mm encoder noise and noisy rates.",
            initial_state=np.array([0.0, 0.0, 0.15, 0.0]),
            config=SimConfig(
                duration=6.0,
                angle_noise_std=np.deg2rad(0.5),
                position_noise_std=0.002,
                rate_noise_std=0.05,
                seed=7,
            ),
        ),
    ]


def swing_up_scenario() -> Scenario:
    """Start hanging straight down; swing up and capture."""
    return Scenario(
        name="swing-up",
        description="Start hanging down at 180 deg, pump energy, then catch and balance.",
        initial_state=np.array([0.0, 0.0, np.pi, 0.0]),
        config=SimConfig(duration=8.0),
        settle_from=3.0,
        requires_swing_up=True,
    )


def all_scenarios() -> list[Scenario]:
    return [*balancing_scenarios(), swing_up_scenario()]
