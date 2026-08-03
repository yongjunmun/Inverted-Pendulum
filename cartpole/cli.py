"""Command line entry point.

Examples::

    python -m cartpole.cli bench          # scenario suite + plots + results table
    python -m cartpole.cli robustness     # model-mismatch success maps
    python -m cartpole.cli animate        # swing-up GIF
    python -m cartpole.cli all            # everything, straight into results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cartpole.controllers import LQRController
from cartpole.dynamics import CartPoleParams
from cartpole.experiments import (
    run_benchmark,
    run_robustness,
    swing_up_result,
    write_csv,
    write_summary,
)
from cartpole.metrics import to_markdown_table
from cartpole.plotting import plot_effort, plot_phase, plot_robustness, plot_scenario
from cartpole.scenarios import all_scenarios

if TYPE_CHECKING:
    from cartpole.learning.experiments import LearningResult

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results"

# Regression floors for `learn --check`. The recorded run reaches cosine 0.9918
# (ARS) and 0.9908 (CEM), 4/4 scenarios and 0.000 m drift; these sit far enough
# below that a stochastic search does not go red on sampling noise alone.
MIN_COSINE_SIMILARITY = 0.95
MAX_LONG_RUN_DRIFT_M = 0.5


def _describe_design(params: CartPoleParams) -> str:
    """Short design report: open-loop instability and closed-loop poles."""
    lqr = LQRController(params)
    open_loop = np.linalg.eigvals(lqr.state_matrix)
    closed_loop = lqr.closed_loop_poles

    def fmt(values: np.ndarray) -> str:
        return ", ".join(f"{value.real:+.2f}{value.imag:+.2f}j" for value in sorted(values, key=lambda v: v.real))

    return (
        "## Design report\n\n"
        f"- Open-loop poles: `{fmt(open_loop)}` - one in the right half plane, so the plant is unstable.\n"
        f"- LQR gain `K = [{', '.join(f'{value:.2f}' for value in lqr.gain[0])}]`\n"
        f"- Closed-loop poles: `{fmt(closed_loop)}`\n"
    )


def command_bench(args: argparse.Namespace) -> None:
    params = CartPoleParams()
    results_dir = Path(args.output)

    trajectories, rows = run_benchmark(params)

    for scenario in all_scenarios():
        results = trajectories[scenario.name]
        plot_scenario(results, f"{scenario.name}: {scenario.description}", results_dir / f"{scenario.name}.png")
        plot_phase(results, f"Phase portrait - {scenario.name}", results_dir / f"{scenario.name}_phase.png")

    balancing = [row for row in rows if row.scenario != "swing-up"]
    plot_effort(balancing, results_dir / "effort.png")

    write_csv(rows, results_dir / "benchmark.csv")
    write_summary(rows, results_dir / "summary.md", extra=_describe_design(params))

    print(to_markdown_table(rows))
    print(f"\nWrote plots, benchmark.csv and summary.md to {results_dir}")

    failures = [f"{row.scenario}/{row.controller}" for row in rows if not row.success]
    print(f"{len(rows) - len(failures)}/{len(rows)} runs stabilised")
    if getattr(args, "check", False) and failures:
        raise SystemExit("regression: " + ", ".join(failures))


def command_robustness(args: argparse.Namespace) -> None:
    results_dir = Path(args.output)
    steps = args.grid
    scales = np.geomspace(0.4, 3.0, steps)

    grid, mass_scales, length_scales = run_robustness(mass_scales=scales, length_scales=scales)
    plot_robustness(grid, mass_scales, length_scales, results_dir / "robustness.png")

    print("Share of the mismatch grid stabilised:")
    for controller, values in grid.items():
        print(f"  {controller:<5s} {100.0 * float(np.mean(values)):5.1f} %")
    print(f"\nWrote robustness.png to {results_dir}")


def command_animate(args: argparse.Namespace) -> None:
    from cartpole.animate import animate

    results_dir = Path(args.output)
    result = swing_up_result()
    path = animate(result, results_dir / "swingup.gif", fps=args.fps)
    print(f"Wrote {path}")


def command_learn(args: argparse.Namespace) -> None:
    """Train policies, score them in the same benchmark, compare to the LQR gain."""
    from cartpole.controllers.lqr import LQRController
    from cartpole.learning.experiments import run_capacity_study, run_horizon_study, train_and_evaluate
    from cartpole.learning.plotting import plot_gain_comparison, plot_horizon_study, plot_learning_curves
    from cartpole.learning.policies import LinearPolicy
    from cartpole.learning.rollout import TrainingConfig, evaluate as score_policy
    from cartpole.metrics import to_markdown_table

    params = CartPoleParams()
    results_dir = Path(args.output)
    config = TrainingConfig()

    # The analytic gain, scored in the very same training environment.
    reference = LinearPolicy(force_limit=params.force_limit)
    reference.set_params(-LQRController(params).gain[0])
    lqr_score = score_policy(reference, params, config, np.random.default_rng(1), episodes=20)
    print(f"LQR reference score (0 training samples): {lqr_score:.2f}\n")

    results = []
    for algorithm in ("ARS", "CEM"):
        print(f"training {algorithm} ...")
        result = train_and_evaluate(algorithm, params, config, seed=args.seed)
        results.append(result)

        print(f"  score {result.training_score:8.2f}   vs LQR {lqr_score:.2f}")
        print(f"  {result.history.total_episodes} episodes / {result.history.total_env_steps:,} env steps")
        print(f"  benchmark {result.scenarios_passed}/4   60 s drift {result.long_run_drift_m:.3f} m")
        if result.comparison:
            print("  " + result.comparison.summary().replace("\n", "\n  "))
        print()

    plot_learning_curves(results, lqr_score, results_dir / "learning_curves.png")
    plot_gain_comparison(results, results_dir / "learned_gains.png")

    rows = [row for result in results for row in result.metrics]
    print(to_markdown_table(rows))

    if not args.quick:
        print("\nhorizon study (why a short episode hides a slow instability) ...")
        horizon = run_horizon_study(params, seed=args.seed)
        plot_horizon_study(horizon, results_dir / "horizon_study.png")
        for row in horizon:
            state = "stable" if row["stabilising"] else "UNSTABLE"
            print(
                f"  episode {row['episode_seconds']:5.1f} s -> {state:<8} "
                f"worst pole {row['worst_pole']:+.4f}  drift over 60 s {row['drift_60s_m']:7.2f} m"
            )

        print("\ncapacity study (does a neural policy beat a linear one?) ...")
        for row in run_capacity_study(params, seed=args.seed):
            print(
                f"  {row['policy']:<24} params={row['n_params']:>4}  score={row['score']:9.2f}  "
                f"benchmark={row['scenarios_passed']}/4"
            )

    print(f"\nWrote learning figures to {results_dir}")

    if getattr(args, "check", False):
        failures = _learning_regressions(results)
        if failures:
            raise SystemExit("regression: " + "; ".join(failures))


def _learning_regressions(results: list[LearningResult]) -> list[str]:
    """Every way a trained policy failed to reproduce the recorded headline result."""
    failures = []
    for result in results:
        name = result.algorithm
        comparison = result.comparison
        if comparison is None:
            failures.append(f"{name}: no gain comparison (policy is not linear)")
        else:
            if comparison.cosine_similarity < MIN_COSINE_SIMILARITY:
                failures.append(
                    f"{name}: cosine {comparison.cosine_similarity:.4f} < {MIN_COSINE_SIMILARITY}"
                )
            if not comparison.is_stabilising:
                failures.append(f"{name}: learned gain is not stabilising")
        expected = len(result.metrics)
        if result.scenarios_passed < expected:
            failures.append(f"{name}: {result.scenarios_passed}/{expected} scenarios")
        if result.long_run_drift_m > MAX_LONG_RUN_DRIFT_M:
            failures.append(
                f"{name}: 60 s drift {result.long_run_drift_m:.3f} m > {MAX_LONG_RUN_DRIFT_M} m"
            )
    return failures


def command_all(args: argparse.Namespace) -> None:
    command_bench(args)
    command_robustness(args)
    command_animate(args)


def build_parser() -> argparse.ArgumentParser:
    # --output is shared by every subcommand and must be accepted *after* the
    # subcommand name, which is the order people naturally type it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", default=str(DEFAULT_RESULTS), help="directory for figures and tables")

    parser = argparse.ArgumentParser(prog="cartpole", description="Inverted pendulum control laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser(
        "bench", parents=[common], help="run the scenario suite and score every controller"
    )
    bench.add_argument("--check", action="store_true", help="exit non-zero if any run fails to stabilise")
    bench.set_defaults(handler=command_bench)

    robustness = subparsers.add_parser("robustness", parents=[common], help="sweep plant/model mismatch")
    robustness.add_argument("--grid", type=int, default=13, help="grid resolution per axis")
    robustness.set_defaults(handler=command_robustness)

    animation = subparsers.add_parser("animate", parents=[common], help="render the swing-up as a GIF")
    animation.add_argument("--fps", type=int, default=30)
    animation.set_defaults(handler=command_animate)

    learn = subparsers.add_parser(
        "learn", parents=[common], help="train ARS/CEM policies and compare them to the LQR gain"
    )
    learn.add_argument("--seed", type=int, default=0)
    learn.add_argument("--quick", action="store_true", help="skip the horizon and capacity studies")
    learn.add_argument(
        "--check", action="store_true", help="exit non-zero if a learned policy regresses against the LQR gain"
    )
    learn.set_defaults(handler=command_learn)

    everything = subparsers.add_parser("all", parents=[common], help="bench + robustness + animate")
    everything.add_argument("--grid", type=int, default=13)
    everything.add_argument("--fps", type=int, default=30)
    everything.add_argument("--check", action="store_true")
    everything.set_defaults(handler=command_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
