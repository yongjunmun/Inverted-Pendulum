"""Experiment drivers for the learning half of the project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cartpole.controllers.learned import LearnedController
from cartpole.dynamics import CartPoleParams
from cartpole.learning.analysis import GainComparison, compare_to_lqr
from cartpole.learning.policies import LinearPolicy, MLPPolicy, Policy
from cartpole.learning.rollout import TrainingConfig, evaluate
from cartpole.learning.trainers import TrainingHistory, train_ars, train_cem
from cartpole.metrics import Metrics
from cartpole.metrics import evaluate as score_run
from cartpole.scenarios import balancing_scenarios
from cartpole.simulate import SimConfig, simulate

TRAINERS = {"ARS": train_ars, "CEM": train_cem}


@dataclass
class LearningResult:
    """Everything measured about one trained policy."""

    algorithm: str
    policy: Policy
    history: TrainingHistory
    training_score: float
    comparison: GainComparison | None
    metrics: list[Metrics]
    long_run_drift_m: float
    """Cart displacement after a 60 s hold - catches slow modes a short run hides."""

    @property
    def scenarios_passed(self) -> int:
        return sum(1 for row in self.metrics if row.success)


def benchmark_policy(policy: Policy, params: CartPoleParams, label: str) -> tuple[list[Metrics], float]:
    """Score a trained policy on the standard scenarios plus a 60 s hold test."""
    controller = LearnedController(policy, params, name=label)

    rows = []
    for scenario in balancing_scenarios():
        result = simulate(
            controller,
            params,
            config=scenario.config,
            initial_state=scenario.initial_state,
            reference=scenario.reference,
            scenario_name=scenario.name,
        )
        rows.append(score_run(result, settle_from=scenario.settle_from))

    hold = simulate(
        controller,
        params,
        config=SimConfig(duration=60.0),
        initial_state=np.array([0.0, 0.0, 0.2, 0.0]),
        scenario_name="60s-hold",
    )
    return rows, float(abs(hold.states[-1, 0]))


def train_and_evaluate(
    algorithm: str = "ARS",
    params: CartPoleParams | None = None,
    config: TrainingConfig | None = None,
    policy: Policy | None = None,
    seed: int = 0,
    **trainer_kwargs,
) -> LearningResult:
    """Train one policy and measure it the same way every classical controller is."""
    params = params or CartPoleParams()
    config = config or TrainingConfig()
    policy = policy or LinearPolicy(force_limit=params.force_limit)

    trained, history = TRAINERS[algorithm](policy, params, config, seed=seed, **trainer_kwargs)
    rows, drift = benchmark_policy(trained, params, f"{algorithm}")

    return LearningResult(
        algorithm=algorithm,
        policy=trained,
        history=history,
        training_score=evaluate(trained, params, config, np.random.default_rng(seed + 1), episodes=20),
        comparison=compare_to_lqr(trained, params) if isinstance(trained, LinearPolicy) else None,
        metrics=rows,
        long_run_drift_m=drift,
    )


def run_horizon_study(
    params: CartPoleParams | None = None,
    episode_lengths: tuple[float, ...] = (4.0, 6.0, 10.0, 20.0),
    seed: int = 0,
) -> list[dict]:
    """Show that the training horizon bounds what the reward can possibly detect.

    A policy is only penalised for behaviour that occurs inside an episode, so a
    closed-loop mode slower than the episode is invisible to the return no matter
    how good that return looks.

    Two conditions are deliberately held fixed, because the effect needs both:

    * a **narrow start distribution** (``initial_position=0.1``), so the policy is
      never pushed far enough off centre to need real cart-position feedback;
    * a **greedy search** (``elite_fraction=0.2``), which overfits to whatever the
      episode length happens to reveal.

    Under the tuned defaults (1.5 m offsets, elite fraction 0.5) every horizon
    yields a stabilising gain and the effect disappears - which is itself the
    useful lesson. A short horizon is only dangerous when the rest of the training
    distribution lets the policy get away with ignoring the slow state.
    """
    params = params or CartPoleParams()
    rows = []

    for seconds in episode_lengths:
        config = TrainingConfig(episode_seconds=seconds, initial_position=0.1)
        policy, _ = train_cem(
            LinearPolicy(force_limit=params.force_limit),
            params,
            config,
            iterations=30,
            population=40,
            elite_fraction=0.2,
            noise_decay=0.95,
            seed=seed,
        )
        report = compare_to_lqr(policy, params)

        hold = simulate(
            LearnedController(policy, params),
            params,
            config=SimConfig(duration=60.0),
            initial_state=np.array([0.0, 0.0, 0.2, 0.0]),
        )
        worst = float(np.max(report.closed_loop_poles.real))

        rows.append(
            {
                "episode_seconds": seconds,
                "worst_pole": worst,
                "time_constant_s": float("inf") if abs(worst) < 1e-9 else 1.0 / abs(worst),
                "stabilising": report.is_stabilising,
                "cosine_similarity": report.cosine_similarity,
                "drift_60s_m": float(abs(hold.states[-1, 0])),
            }
        )
    return rows


def run_capacity_study(params: CartPoleParams | None = None, seed: int = 0) -> list[dict]:
    """Does a neural policy beat a linear one on a problem whose optimum is linear?

    Trained with the same algorithm, the same budget and the same reward, so the
    only variable is the function class.
    """
    params = params or CartPoleParams()
    config = TrainingConfig()
    rows = []

    candidates: list[tuple[str, Policy, dict]] = [
        ("Linear (4 params)", LinearPolicy(force_limit=params.force_limit), {}),
        (
            "MLP 16x16 (369 params)",
            MLPPolicy(hidden_sizes=(16, 16), force_limit=params.force_limit),
            dict(exploration_noise=1.0, step_size=0.5),
        ),
    ]

    for label, policy, overrides in candidates:
        trained, history = train_ars(policy, params, config, seed=seed, **overrides)
        rows_metrics, drift = benchmark_policy(trained, params, label)
        rows.append(
            {
                "policy": label,
                "n_params": trained.n_params,
                "score": evaluate(trained, params, config, np.random.default_rng(seed + 1), episodes=20),
                "scenarios_passed": sum(1 for row in rows_metrics if row.success),
                "env_steps": history.total_env_steps,
                "drift_60s_m": drift,
            }
        )
    return rows
