"""Cart-pole (inverted pendulum) simulation and control laboratory.

Everything in this package is pure NumPy: the plant, the linearisation, the
Riccati solvers and the model predictive controller are implemented from
scratch so the maths is visible rather than hidden behind a solver call.
"""

from cartpole.dynamics import (
    CartPoleParams,
    derivative,
    linearize_upright,
    numeric_jacobian,
    pole_energy,
    rk4_step,
    target_energy,
    total_energy,
    wrap_angle,
)

__all__ = [
    "CartPoleParams",
    "derivative",
    "linearize_upright",
    "numeric_jacobian",
    "pole_energy",
    "rk4_step",
    "target_energy",
    "total_energy",
    "wrap_angle",
]

__version__ = "1.0.0"
