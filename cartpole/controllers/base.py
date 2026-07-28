"""Controller interface shared by every control law in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """A discrete-time feedback law evaluated on a fixed control clock.

    Implementations receive the (optionally noisy) measured state and return a
    force command in newtons. Saturation is applied by the simulator, not here,
    so that every controller is judged against the same actuator limit.
    """

    name: str = "controller"

    def reset(self) -> None:
        """Clear any internal state before a new run. Default: nothing to clear."""

    @abstractmethod
    def compute(self, state: np.ndarray, time: float, reference: float) -> float:
        """Return the commanded cart force [N].

        Args:
            state: measured ``[x, x_dot, theta, theta_dot]``.
            time: simulation time [s].
            reference: desired cart position [m].
        """

    def mode(self) -> str:
        """Label of the currently active internal mode, for plotting."""
        return self.name
