"""Does random search rediscover optimal control?

A linear policy ``u = w . s`` and the LQR law ``u = -K s`` live in the same
parameter space, so a trained policy can be compared against the analytic
solution directly rather than only through episode returns.

Three questions are worth asking, and only the first is usually asked:

1. Does the learned policy score well?
2. Does it point in the same **direction** as the optimal gain? Actuator
   saturation means a policy can scale its weights up arbitrarily and behave
   almost identically, so magnitude is uninformative and cosine similarity is
   the meaningful comparison.
3. Is the learned gain **provably stabilising**? The closed-loop eigenvalues of
   ``A - B K`` answer that in closed form. A good episode return does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cartpole.controllers.lqr import LQRController
from cartpole.dynamics import CartPoleParams, linearize_upright
from cartpole.learning.policies import LinearPolicy


@dataclass
class GainComparison:
    """How a learned linear policy relates to the analytic LQR solution."""

    learned_gain: np.ndarray
    lqr_gain: np.ndarray
    cosine_similarity: float
    """1.0 means the learned policy points exactly along the optimal gain."""
    magnitude_ratio: float
    """``|learned| / |LQR|``. Large values usually mean a saturating policy."""
    closed_loop_poles: np.ndarray
    is_stabilising: bool
    """True when every eigenvalue of ``A - B K_learned`` is in the left half plane."""
    lqr_closed_loop_poles: np.ndarray

    def summary(self) -> str:
        lines = [
            f"learned gain      [{', '.join(f'{value:8.2f}' for value in self.learned_gain)}]",
            f"LQR gain          [{', '.join(f'{value:8.2f}' for value in self.lqr_gain)}]",
            f"cosine similarity {self.cosine_similarity: .4f}",
            f"magnitude ratio   {self.magnitude_ratio: .2f}x",
            f"stabilising       {'yes' if self.is_stabilising else 'NO'}",
            f"slowest pole      {max(self.closed_loop_poles.real): .3f}"
            f"   (LQR {max(self.lqr_closed_loop_poles.real): .3f})",
        ]
        return "\n".join(lines)


def compare_to_lqr(policy: LinearPolicy, params: CartPoleParams) -> GainComparison:
    """Compare a trained linear policy against the LQR gain for the same plant."""
    state_matrix, input_matrix = linearize_upright(params)
    lqr = LQRController(params)

    learned_gain = policy.equivalent_gain
    lqr_gain = lqr.gain[0]

    learned_norm = float(np.linalg.norm(learned_gain))
    lqr_norm = float(np.linalg.norm(lqr_gain))
    cosine = float(learned_gain @ lqr_gain / (learned_norm * lqr_norm)) if learned_norm > 0 else 0.0

    poles = np.linalg.eigvals(state_matrix - input_matrix @ learned_gain.reshape(1, -1))

    return GainComparison(
        learned_gain=learned_gain,
        lqr_gain=lqr_gain,
        cosine_similarity=cosine,
        magnitude_ratio=learned_norm / lqr_norm if lqr_norm > 0 else float("inf"),
        closed_loop_poles=poles,
        is_stabilising=bool(np.max(poles.real) < 0.0),
        lqr_closed_loop_poles=lqr.closed_loop_poles,
    )
