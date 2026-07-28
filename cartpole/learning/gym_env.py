"""Optional Gymnasium adapter, so external RL libraries can use this plant.

Gymnasium is **not** a required dependency. The trainers in this package are
pure NumPy and run without it; this module exists so that anyone wanting to
point Stable-Baselines3, CleanRL or their own PPO at the same physics, the same
reward and the same actuator limit can do so and get a directly comparable
number.

    pip install gymnasium

    from cartpole.learning.gym_env import CartPoleContinuousEnv
    env = CartPoleContinuousEnv()
"""

from __future__ import annotations

import numpy as np

from cartpole.dynamics import ANGLE, CartPoleParams, rk4_step, wrap_angle
from cartpole.learning.rollout import TrainingConfig, sample_initial_state

try:  # pragma: no cover - exercised only when the optional dependency is present
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
    _Base = gym.Env
except ImportError:  # pragma: no cover
    GYMNASIUM_AVAILABLE = False
    _Base = object


class CartPoleContinuousEnv(_Base):
    """Continuous-force cart-pole with the LQR cost as the reward.

    Unlike ``Gymnasium``'s built-in ``CartPole-v1``, this environment uses a
    continuous force, the full nonlinear rod dynamics, and a quadratic cost
    rather than a +1-per-timestep survival bonus. That matters: a survival bonus
    cannot distinguish a controller that balances smoothly from one that
    thrashes, so it cannot be compared against LQR at all.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: CartPoleParams | None = None, config: TrainingConfig | None = None):
        if not GYMNASIUM_AVAILABLE:  # pragma: no cover
            raise ImportError("gymnasium is required for CartPoleContinuousEnv; pip install gymnasium")

        self.params = params or CartPoleParams()
        self.config = config or TrainingConfig()

        limit = self.params.force_limit
        self.action_space = spaces.Box(low=-limit, high=limit, shape=(1,), dtype=np.float32)
        bound = np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-bound, high=bound, dtype=np.float32)

        self._state = np.zeros(4)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self._state = sample_initial_state(self.config, rng)
        self._step = 0
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        observation = self._state.copy()
        observation[ANGLE] = wrap_angle(observation[ANGLE])
        return observation.astype(np.float32)

    def step(self, action):
        force = float(np.clip(np.asarray(action).reshape(-1)[0], -self.params.force_limit, self.params.force_limit))

        observation = self._observation()
        cost = float(
            observation @ self.config.state_cost @ observation + float(self.config.input_cost[0, 0]) * force**2
        )
        reward = -cost * self.config.dt

        self._state = rk4_step(self._state, force, self.config.dt, self.params)
        self._step += 1

        terminated = bool(abs(self._state[0]) > self.params.rail_limit)
        truncated = bool(self._step >= self.config.n_steps)
        if terminated:
            remaining = (self.config.n_steps - self._step) * self.config.dt
            reward -= self.config.failure_cost * remaining

        return self._observation(), reward, terminated, truncated, {}
