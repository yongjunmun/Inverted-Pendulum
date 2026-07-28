"""Parameterised policies for learning-based control.

Deliberately small and NumPy-only. Mania et al. (2018), *Simple random search
provides a competitive approach to reinforcement learning*, showed that linear
policies trained by random search match deep RL on continuous control
benchmarks, so a linear policy is not a toy here - it is the strong baseline.

Keeping the policy linear also buys something no neural network can: the learned
parameters live in the same space as the LQR gain, so the two can be compared
directly (see :mod:`cartpole.learning.analysis`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """A deterministic state-feedback policy with a flat parameter vector."""

    name: str = "policy"

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters."""

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return the parameters as a flat vector."""

    @abstractmethod
    def set_params(self, values: np.ndarray) -> None:
        """Load parameters from a flat vector."""

    @abstractmethod
    def act(self, observation: np.ndarray) -> float:
        """Return the commanded force [N] for an observation."""

    def copy_with(self, values: np.ndarray) -> "Policy":
        """Return a shallow copy carrying different parameters."""
        import copy

        clone = copy.deepcopy(self)
        clone.set_params(values)
        return clone


class LinearPolicy(Policy):
    """``u = clip(w . s, -limit, limit)``.

    Structurally identical to the LQR control law ``u = -K s``, which is the
    point: if training works, ``w`` should converge to roughly ``-K``.
    """

    name = "Linear"

    def __init__(self, n_observations: int = 4, force_limit: float = 10.0):
        self.n_observations = n_observations
        self.force_limit = force_limit
        self.weights = np.zeros(n_observations)

    @property
    def n_params(self) -> int:
        return self.n_observations

    def get_params(self) -> np.ndarray:
        return self.weights.copy()

    def set_params(self, values: np.ndarray) -> None:
        self.weights = np.asarray(values, dtype=float).reshape(self.n_observations)

    def act(self, observation: np.ndarray) -> float:
        raw = float(self.weights @ observation)
        return float(np.clip(raw, -self.force_limit, self.force_limit))

    @property
    def equivalent_gain(self) -> np.ndarray:
        """The policy expressed as an LQR-style gain, ``K = -w``."""
        return -self.weights


class MLPPolicy(Policy):
    """Small tanh multi-layer perceptron, for testing whether depth actually helps.

    Included as a control experiment rather than an upgrade. If a nonlinear
    policy does not beat the linear one on a task whose optimal solution is
    known to be linear, that is a result worth reporting, not a failure.
    """

    name = "MLP"

    def __init__(
        self,
        n_observations: int = 4,
        hidden_sizes: tuple[int, ...] = (16, 16),
        force_limit: float = 10.0,
    ):
        self.n_observations = n_observations
        self.hidden_sizes = hidden_sizes
        self.force_limit = force_limit

        self.shapes: list[tuple[int, ...]] = []
        sizes = (n_observations, *hidden_sizes, 1)
        for inputs, outputs in zip(sizes[:-1], sizes[1:]):
            self.shapes.append((inputs, outputs))
            self.shapes.append((outputs,))

        self._params = np.zeros(sum(int(np.prod(shape)) for shape in self.shapes))

    @property
    def n_params(self) -> int:
        return int(self._params.size)

    def get_params(self) -> np.ndarray:
        return self._params.copy()

    def set_params(self, values: np.ndarray) -> None:
        self._params = np.asarray(values, dtype=float).reshape(self._params.shape)

    def _layers(self):
        offset = 0
        for index in range(0, len(self.shapes), 2):
            weight_shape = self.shapes[index]
            bias_shape = self.shapes[index + 1]
            weight_size = int(np.prod(weight_shape))
            bias_size = int(np.prod(bias_shape))

            weight = self._params[offset : offset + weight_size].reshape(weight_shape)
            offset += weight_size
            bias = self._params[offset : offset + bias_size].reshape(bias_shape)
            offset += bias_size
            yield weight, bias

    def act(self, observation: np.ndarray) -> float:
        activation = np.asarray(observation, dtype=float)
        layers = list(self._layers())
        for weight, bias in layers[:-1]:
            activation = np.tanh(activation @ weight + bias)
        weight, bias = layers[-1]
        raw = float((activation @ weight + bias)[0])
        return float(np.clip(raw, -self.force_limit, self.force_limit))
