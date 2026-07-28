"""Figures for the learning experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cartpole.learning.experiments import LearningResult  # noqa: E402

COLOURS = {"ARS": "#8338ec", "CEM": "#fb8500", "LQR": "#2a9d8f"}


def plot_learning_curves(results: list[LearningResult], lqr_score: float, path: Path) -> Path:
    """Return versus environment steps, with the analytic LQR score as the target."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    for result in results:
        colour = COLOURS.get(result.algorithm, "#6c757d")
        axes[0].plot(result.history.env_steps, result.history.mean_return, color=colour, lw=1.8, label=result.algorithm)
        axes[1].plot(result.history.env_steps, result.history.mean_return, color=colour, lw=1.8, label=result.algorithm)

    for axis in axes:
        axis.axhline(lqr_score, color=COLOURS["LQR"], ls="--", lw=1.6, label="LQR (0 samples)")
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("mean return")
    axes[0].set_title("Learning curve", fontweight="bold")
    axes[1].set_yscale("symlog")
    axes[1].set_ylabel("mean return (symlog)")
    axes[1].set_title("Same data, log scale", fontweight="bold")
    axes[0].legend(loc="lower right")

    figure.suptitle(
        "Learning has to pay for what the Riccati equation gets in closed form",
        fontweight="bold",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_gain_comparison(results: list[LearningResult], path: Path) -> Path:
    """Learned gains against the analytic LQR gain, element by element."""
    linear = [result for result in results if result.comparison is not None]
    if not linear:
        raise ValueError("no linear policies to compare")

    labels = ["cart position", "cart velocity", "pole angle", "pole rate"]
    positions = np.arange(len(labels))
    width = 0.8 / (len(linear) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    axes[0].bar(
        positions - 0.4 + width / 2,
        linear[0].comparison.lqr_gain,
        width,
        label="LQR (analytic)",
        color=COLOURS["LQR"],
    )
    for index, result in enumerate(linear, start=1):
        axes[0].bar(
            positions - 0.4 + width / 2 + index * width,
            result.comparison.learned_gain,
            width,
            label=f"{result.algorithm} (learned)",
            color=COLOURS.get(result.algorithm, "#6c757d"),
        )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].set_ylabel("gain")
    axes[0].set_title("Learned gain vs analytic LQR gain", fontweight="bold")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    similarities = [result.comparison.cosine_similarity for result in linear]
    names = [result.algorithm for result in linear]
    bars = axes[1].barh(names, similarities, color=[COLOURS.get(name, "#6c757d") for name in names])
    axes[1].axvline(1.0, color="k", ls=":", lw=1.2)
    axes[1].set_xlim(0.0, 1.08)
    axes[1].set_xlabel("cosine similarity with the LQR gain")
    axes[1].set_title("Direction, not magnitude\n(saturation makes scale uninformative)", fontweight="bold")
    axes[1].grid(alpha=0.25, axis="x")
    for bar, value in zip(bars, similarities):
        axes[1].text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_horizon_study(rows: list[dict], path: Path) -> Path:
    """Training horizon versus the slowest closed-loop mode it can detect."""
    seconds = [row["episode_seconds"] for row in rows]
    poles = [row["worst_pole"] for row in rows]
    drift = [row["drift_60s_m"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    colours = ["#d1495b" if pole > 0 else "#2a9d8f" for pole in poles]
    axes[0].bar([str(value) for value in seconds], poles, color=colours)
    axes[0].axhline(0.0, color="k", lw=1.2)
    axes[0].set_xlabel("training episode length [s]")
    axes[0].set_ylabel("worst closed-loop pole (real part)")
    axes[0].set_title("Above zero = unstable", fontweight="bold")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar([str(value) for value in seconds], drift, color=colours)
    axes[1].set_xlabel("training episode length [s]")
    axes[1].set_ylabel("cart drift after 60 s [m]")
    axes[1].set_title("What the short-horizon reward never saw", fontweight="bold")
    axes[1].grid(alpha=0.25, axis="y")
    for index, value in enumerate(drift):
        axes[1].text(index, value, f" {value:.2f} m", ha="center", va="bottom", fontsize=9)

    figure.suptitle(
        "A reward cannot penalise a mode slower than its own episode",
        fontweight="bold",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
