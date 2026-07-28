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

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results"


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
