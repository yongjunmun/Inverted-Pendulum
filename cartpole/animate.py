"""Animate a cart-pole run and export it as a GIF for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from cartpole.simulate import SimResult  # noqa: E402


def animate(result: SimResult, path: Path, fps: int = 30, seconds_per_second: float = 1.0) -> Path:
    """Render ``result`` as an animated GIF.

    Args:
        result: trajectory to animate.
        path: output ``.gif`` path.
        fps: frames per second of the GIF.
        seconds_per_second: playback speed (``0.5`` renders in slow motion).
    """
    params = result.params
    sample_dt = float(np.mean(np.diff(result.time)))
    stride = max(1, int(round(seconds_per_second / (fps * sample_dt))))
    frames = range(0, len(result.time), stride)

    cart_width, cart_height = 0.28, 0.16
    span = max(0.9, float(np.max(np.abs(result.states[:, 0]))) + params.pole_length + 0.3)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.set_xlim(-span, span)
    axis.set_ylim(-params.pole_length - 0.35, params.pole_length + 0.35)
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.axhline(0.0, color="#adb5bd", lw=2.0)
    axis.set_xlabel("x [m]")
    axis.set_title(f"{result.controller_name} - {result.scenario_name}", fontweight="bold")

    cart = Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fc="#264653", ec="black", zorder=3)
    axis.add_patch(cart)
    (pole,) = axis.plot([], [], lw=5.0, color="#e07a5f", solid_capstyle="round", zorder=4)
    (bob,) = axis.plot([], [], "o", ms=9, color="#f4a261", zorder=5)
    (trail,) = axis.plot([], [], lw=1.0, color="#e9c46a", alpha=0.85, zorder=2)
    readout = axis.text(0.02, 0.94, "", transform=axis.transAxes, fontsize=10, family="monospace", va="top")

    tip_x: list[float] = []
    tip_y: list[float] = []

    def tip(index: int) -> tuple[float, float]:
        position = result.states[index, 0]
        angle = result.states[index, 2]
        return (
            position + params.pole_length * np.sin(angle),
            params.pole_length * np.cos(angle),
        )

    def update(index: int):
        position = result.states[index, 0]
        end_x, end_y = tip(index)

        cart.set_xy((position - cart_width / 2, -cart_height / 2))
        pole.set_data([position, end_x], [0.0, end_y])
        bob.set_data([end_x], [end_y])

        tip_x.append(end_x)
        tip_y.append(end_y)
        trail.set_data(tip_x[-140:], tip_y[-140:])

        readout.set_text(
            f"t     = {result.time[index]:5.2f} s\n"
            f"angle = {np.rad2deg(result.angle[index]):6.1f} deg\n"
            f"force = {result.applied_force[index]:6.2f} N\n"
            f"mode  = {result.modes[index]}"
        )
        return cart, pole, bob, trail, readout

    animation = FuncAnimation(figure, update, frames=frames, blit=True, interval=1000 / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return path
