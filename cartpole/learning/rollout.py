"""Episode rollouts and the reward function used for training.

**The reward is the LQR cost, negated.** Training minimises exactly the quantity
the Riccati equation minimises in closed form::

    r_t = -(e' Q e + R u^2) dt

with the same ``Q`` and ``R`` used to design the LQR controller. Any difference
in the results is therefore attributable to the *method*, not to the two
controllers chasing different objectives - which is what makes the comparison in
the README meaningful rather than decorative.

**One trap worth naming.** Every reward here is negative, so an agent that lets
the cart hit the rail and ends the episode early accumulates *less* negative
reward and scores *better* than one that balances for the full duration. Early
termination must therefore be paid for: :func:`rollout` charges the remaining
steps at a fixed failure cost. Without that, random search reliably learns to
crash on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cartpole.controllers.lqr import DEFAULT_INPUT_COST, DEFAULT_STATE_COST
from cartpole.dynamics import ANGLE, CartPoleParams, rk4_step, wrap_angle
from cartpole.learning.policies import Policy


@dataclass
class TrainingConfig:
    """Settings for the fast training simulator.

    Training integrates at the control rate (one RK4 step per action) rather
    than the 1 kHz used by :func:`cartpole.simulate.simulate`. That is roughly
    ten times faster and accurate enough to learn from; every trained policy is
    then re-scored on the full-fidelity simulator, so the numbers reported in
    the benchmark never come from the coarse model.
    """

    episode_seconds: float = 10.0
    """Episode length [s].

    Not an arbitrary choice. A policy trained on 4 s episodes scored near-optimal
    while hiding a closed-loop pole at +0.093 - a 10.8 s time constant that the
    reward could not see, and which drifted the cart 35.8 m over a 60 s run.
    The episode must outlast the slowest mode you care about.
    """
    dt: float = 0.02
    initial_angle: float = 0.2
    """Episodes start from a uniformly random tilt in +/- this value [rad]."""
    initial_position: float = 1.5
    """...and a uniformly random cart offset [m].

    Also measured rather than guessed. Trained at 0.1 m the policy failed the
    1 m tracking scenario outright; it had never seen a cart error that large.
    Widening to 1.5 m fixed it with no other change.
    """
    failure_cost: float = 50.0
    """Per-second cost charged for the remainder of an episode after a crash."""
    state_cost: np.ndarray = field(default_factory=lambda: DEFAULT_STATE_COST.copy())
    input_cost: np.ndarray = field(default_factory=lambda: DEFAULT_INPUT_COST.copy())

    @property
    def n_steps(self) -> int:
        return int(round(self.episode_seconds / self.dt))


def sample_initial_state(config: TrainingConfig, rng: np.random.Generator) -> np.ndarray:
    """Random start state, so the policy cannot overfit a single trajectory."""
    return np.array(
        [
            rng.uniform(-config.initial_position, config.initial_position),
            0.0,
            rng.uniform(-config.initial_angle, config.initial_angle),
            0.0,
        ]
    )


def rollout(
    policy: Policy,
    params: CartPoleParams,
    config: TrainingConfig,
    rng: np.random.Generator,
    initial_state: np.ndarray | None = None,
) -> tuple[float, int]:
    """Run one episode and return ``(total_reward, steps_survived)``."""
    state = sample_initial_state(config, rng) if initial_state is None else np.asarray(initial_state, dtype=float).copy()

    state_cost = config.state_cost
    input_cost = float(config.input_cost[0, 0])
    total = 0.0

    for step in range(config.n_steps):
        observation = state.copy()
        observation[ANGLE] = wrap_angle(observation[ANGLE])

        force = float(np.clip(policy.act(observation), -params.force_limit, params.force_limit))
        total -= float(observation @ state_cost @ observation + input_cost * force**2) * config.dt

        state = rk4_step(state, force, config.dt, params)

        if abs(state[0]) > params.rail_limit or not np.all(np.isfinite(state)):
            remaining = (config.n_steps - step - 1) * config.dt
            return total - config.failure_cost * remaining, step + 1

    return total, config.n_steps


def evaluate(
    policy: Policy,
    params: CartPoleParams,
    config: TrainingConfig,
    rng: np.random.Generator,
    episodes: int = 5,
) -> float:
    """Mean return over several random starts."""
    return float(np.mean([rollout(policy, params, config, rng)[0] for _ in range(episodes)]))


def count_steps(config: TrainingConfig, episodes: int) -> int:
    """Environment interactions consumed, for reporting sample efficiency."""
    return config.n_steps * episodes
