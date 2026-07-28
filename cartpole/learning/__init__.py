"""Learning-based control for the cart-pole, implemented from scratch in NumPy.

The point of this package is not to show that reinforcement learning can balance
a pendulum - that has been done thousands of times. It is to measure a learned
controller against three tuned classical controllers on identical scenarios,
with identical metrics, optimising an identical objective, and to report what
learning costs as well as what it buys.
"""

from cartpole.learning.analysis import GainComparison, compare_to_lqr
from cartpole.learning.policies import LinearPolicy, MLPPolicy, Policy
from cartpole.learning.rollout import TrainingConfig, evaluate, rollout
from cartpole.learning.trainers import TrainingHistory, train_ars, train_cem

__all__ = [
    "GainComparison",
    "LinearPolicy",
    "MLPPolicy",
    "Policy",
    "TrainingConfig",
    "TrainingHistory",
    "compare_to_lqr",
    "evaluate",
    "rollout",
    "train_ars",
    "train_cem",
]
