"""Figures: time histories, phase portraits, effort comparison, robustness maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cartpole.metrics import Metrics  # noqa: E402
from cartpole.simulate import SimResult  # noqa: E402

PALETTE = {
    "PID": "#d1495b",
    "LQR": "#2a9d8f",
    "MPC": "#264653",
    "SwingUp+LQR": "#e07a5f",
}


def _colour(name: str) -> str:
    return PALETTE.get(name, "#6c757d")


def plot_scenario(results: list[SimResult], title: str, path: Path) -> Path:
    """Angle, cart position and force for every controller on one scenario."""
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    figure.suptitle(title, fontsize=13, fontweight="bold")

    limit = results[0].params.force_limit
    for result in results:
        colour = _colour(result.controller_name)
        axes[0].plot(result.time, np.rad2deg(result.angle), color=colour, label=result.controller_name, lw=1.6)
        axes[1].plot(result.time, result.states[:, 0], color=colour, lw=1.6)
        axes[2].plot(result.time, result.applied_force, color=colour, lw=1.2)

    axes[1].plot(results[0].time, results[0].reference, "k--", lw=1.0, label="reference")
    axes[2].axhline(limit, color="k", ls=":", lw=1.0)
    axes[2].axhline(-limit, color="k", ls=":", lw=1.0, label="force limit")

    # Mark hand-overs between internal controller modes (e.g. swing-up -> LQR catch).
    for result in results:
        for index in range(1, len(result.modes)):
            if result.modes[index] != result.modes[index - 1]:
                for axis in axes:
                    axis.axvline(result.time[index], color="#6a4c93", ls="--", lw=1.2)
                axes[0].annotate(
                    f"{result.modes[index]}\nt = {result.time[index]:.2f} s",
                    xy=(result.time[index], 0.0),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    color="#6a4c93",
                )

    if np.any(results[0].disturbance != 0.0):
        pushed = results[0].time[results[0].disturbance != 0.0]
        for axis in axes:
            axis.axvspan(pushed[0], pushed[-1], color="#ffb703", alpha=0.25)

    axes[0].set_ylabel("pole angle [deg]")
    axes[1].set_ylabel("cart position [m]")
    axes[2].set_ylabel("force [N]")
    axes[2].set_xlabel("time [s]")

    axes[0].legend(loc="upper right", ncol=len(results))
    axes[1].legend(loc="lower right")
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_phase(results: list[SimResult], title: str, path: Path) -> Path:
    """Pole angle versus angular rate, showing the trajectory into the origin."""
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    for result in results:
        axis.plot(
            np.rad2deg(result.angle),
            result.states[:, 3],
            color=_colour(result.controller_name),
            lw=1.4,
            label=result.controller_name,
        )
        axis.plot(np.rad2deg(result.angle[0]), result.states[0, 3], "o", color=_colour(result.controller_name), ms=6)

    axis.plot(0, 0, "k*", ms=12, label="upright equilibrium")
    axis.set_xlabel("pole angle [deg]")
    axis.set_ylabel("pole rate [rad/s]")
    axis.set_title(title, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_effort(rows: list[Metrics], path: Path) -> Path:
    """Grouped bars: control effort and RMS angle error per scenario."""
    scenarios = sorted({row.scenario for row in rows})
    controllers = sorted({row.controller for row in rows})
    positions = np.arange(len(scenarios))
    width = 0.8 / max(len(controllers), 1)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for index, controller in enumerate(controllers):
        effort, rms = [], []
        for scenario in scenarios:
            match = [row for row in rows if row.controller == controller and row.scenario == scenario]
            effort.append(match[0].control_effort_n2s if match else np.nan)
            rms.append(match[0].rms_angle_deg if match else np.nan)
        offset = positions + index * width - 0.4 + width / 2
        axes[0].bar(offset, effort, width, label=controller, color=_colour(controller))
        axes[1].bar(offset, rms, width, label=controller, color=_colour(controller))

    for axis, label in zip(axes, ("control effort  $\\int u^2 dt$  [N$^2$s]", "RMS pole angle [deg]")):
        axis.set_xticks(positions)
        axis.set_xticklabels(scenarios, rotation=12, ha="right")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, axis="y")
    axes[0].legend()

    figure.suptitle("Cost of control: effort versus accuracy", fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_robustness(
    grid: dict[str, np.ndarray],
    mass_scales: np.ndarray,
    length_scales: np.ndarray,
    path: Path,
) -> Path:
    """Success maps over pole mass and length model mismatch."""
    controllers = list(grid)
    figure, axes = plt.subplots(
        1,
        len(controllers),
        figsize=(4.4 * len(controllers), 4.4),
        squeeze=False,
        layout="constrained",
    )

    for axis, controller in zip(axes[0], controllers):
        image = axis.imshow(grid[controller], origin="lower", cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        rate = 100.0 * float(np.mean(grid[controller]))
        axis.set_title(f"{controller}\n{rate:.0f}% of grid stabilised", fontweight="bold")
        axis.set_xlabel("true pole length / model")
        axis.set_ylabel("true pole mass / model")

        ticks = np.arange(0, len(length_scales), max(1, len(length_scales) // 6))
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{length_scales[index]:.2f}" for index in ticks])
        ticks = np.arange(0, len(mass_scales), max(1, len(mass_scales) // 6))
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{mass_scales[index]:.2f}" for index in ticks])

        nominal_x = float(np.argmin(np.abs(length_scales - 1.0)))
        nominal_y = float(np.argmin(np.abs(mass_scales - 1.0)))
        axis.plot(nominal_x, nominal_y, "k*", ms=13, label="nominal model")
        axis.legend(loc="lower left", fontsize=8)

    figure.colorbar(image, ax=axes[0].tolist(), label="stabilised (1) / lost (0)")
    figure.suptitle("Robustness to plant/model mismatch", fontweight="bold", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
