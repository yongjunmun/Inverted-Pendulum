"""Experiment drivers: the benchmark sweep and the robustness study."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cartpole.controllers import CascadedPID, Controller, LQRController, MPCController, SwingUpLQR
from cartpole.dynamics import CartPoleParams
from cartpole.metrics import Metrics, evaluate, to_markdown_table
from cartpole.scenarios import Scenario, all_scenarios, balancing_scenarios, swing_up_scenario
from cartpole.simulate import SimConfig, SimResult, simulate


def build_balancing_controllers(params: CartPoleParams, control_dt: float = 0.01) -> list[Controller]:
    """The three balancing controllers, all designed against the same nominal model."""
    return [
        CascadedPID(control_dt=control_dt, force_limit=params.force_limit),
        LQRController(params),
        MPCController(params, control_dt=control_dt),
    ]


def run_scenario(scenario: Scenario, params: CartPoleParams) -> tuple[list[SimResult], list[Metrics]]:
    """Run every applicable controller on one scenario."""
    if scenario.requires_swing_up:
        controllers: list[Controller] = [SwingUpLQR(params)]
    else:
        controllers = build_balancing_controllers(params, scenario.config.control_dt)

    results, rows = [], []
    for controller in controllers:
        result = simulate(
            controller,
            params,
            config=scenario.config,
            initial_state=scenario.initial_state,
            reference=scenario.reference,
            scenario_name=scenario.name,
        )
        results.append(result)
        rows.append(evaluate(result, settle_from=scenario.settle_from))
    return results, rows


def run_benchmark(
    params: CartPoleParams | None = None,
    include_swing_up: bool = True,
) -> tuple[dict[str, list[SimResult]], list[Metrics]]:
    """Run the full scenario suite and return trajectories plus scored metrics."""
    params = params or CartPoleParams()
    scenarios = all_scenarios() if include_swing_up else balancing_scenarios()

    trajectories: dict[str, list[SimResult]] = {}
    rows: list[Metrics] = []
    for scenario in scenarios:
        results, metrics = run_scenario(scenario, params)
        trajectories[scenario.name] = results
        rows.extend(metrics)
    return trajectories, rows


def run_robustness(
    params: CartPoleParams | None = None,
    mass_scales: np.ndarray | None = None,
    length_scales: np.ndarray | None = None,
    initial_angle: float = 0.2,
    duration: float = 4.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Design on the nominal model, then test against deliberately wrong plants.

    Each controller is designed once against ``params`` and then asked to
    stabilise plants whose pole mass and length are scaled away from nominal.
    This is the closest thing to a hardware reality check available in simulation:
    on a real rig you never know the true parameters.
    """
    params = params or CartPoleParams()
    mass_scales = np.geomspace(0.4, 3.0, 13) if mass_scales is None else mass_scales
    length_scales = np.geomspace(0.4, 3.0, 13) if length_scales is None else length_scales

    config = SimConfig(duration=duration)
    initial_state = np.array([0.0, 0.0, initial_angle, 0.0])

    controllers = build_balancing_controllers(params, config.control_dt)
    grid = {controller.name: np.zeros((len(mass_scales), len(length_scales))) for controller in controllers}

    for row, mass_scale in enumerate(mass_scales):
        for column, length_scale in enumerate(length_scales):
            true_plant = params.perturbed(mass_scale=mass_scale, length_scale=length_scale)
            for controller in controllers:
                result = simulate(
                    controller,
                    true_plant,
                    config=config,
                    initial_state=initial_state,
                    scenario_name="robustness",
                )
                grid[controller.name][row, column] = 1.0 if evaluate(result).success else 0.0

    return grid, mass_scales, length_scales


def write_csv(rows: list[Metrics], path: Path) -> Path:
    """Write the metrics table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_row()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
    return path


def write_summary(rows: list[Metrics], path: Path, extra: str = "") -> Path:
    """Write a human-readable markdown summary of the benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = {scenario.name: scenario.description for scenario in all_scenarios()}

    lines = ["# Benchmark results", "", "## Scenarios", ""]
    lines += [f"- **{name}** - {description}" for name, description in scenarios.items()]
    lines += ["", "## Scores", "", to_markdown_table(rows), ""]
    if extra:
        lines += [extra, ""]

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def swing_up_result(params: CartPoleParams | None = None) -> SimResult:
    """Convenience helper returning just the swing-up trajectory (used for the GIF)."""
    params = params or CartPoleParams()
    scenario = swing_up_scenario()
    results, _ = run_scenario(scenario, params)
    return results[0]
