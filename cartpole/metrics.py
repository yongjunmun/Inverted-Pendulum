"""Objective performance metrics used to score every controller run.

All metrics are computed from the logged trajectory only, so a controller cannot
score well by reporting on itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cartpole.simulate import SimResult

ANGLE_TOLERANCE = np.deg2rad(2.0)
"""Settling band on the pole angle [rad]."""
POSITION_TOLERANCE = 0.05
"""Settling band on the cart position [m]."""
FAILURE_ANGLE = np.deg2rad(60.0)
"""Beyond this tilt the pole is considered lost (after any swing-up phase)."""


@dataclass
class Metrics:
    """Scalar summary of one closed-loop run."""

    controller: str
    scenario: str
    success: bool
    settling_time_s: float
    peak_angle_deg: float
    rms_angle_deg: float
    final_position_error_m: float
    rms_position_error_m: float
    peak_force_n: float
    control_effort_n2s: float
    itae: float
    saturated_fraction: float

    def as_row(self) -> dict:
        return asdict(self)


def settling_time(time: np.ndarray, error: np.ndarray, tolerance: float) -> float:
    """Last instant the signal leaves ``+/- tolerance`` and never returns outside it.

    Returns ``nan`` if the signal is still outside the band at the end of the run.
    """
    outside = np.nonzero(np.abs(error) > tolerance)[0]
    if outside.size == 0:
        return 0.0
    last = int(outside[-1])
    if last >= len(time) - 1:
        return float("nan")
    return float(time[last + 1])


def evaluate(result: SimResult, settle_from: float = 0.0) -> Metrics:
    """Score a run.

    Args:
        result: trajectory produced by :func:`cartpole.simulate.simulate`.
        settle_from: ignore everything before this time when judging success and
            settling. Swing-up scenarios use it to exclude the swing itself.
    """
    time = result.time
    angle = result.angle
    position_error = result.position_error
    window = time >= settle_from

    dt = float(np.mean(np.diff(time))) if time.size > 1 else 0.0
    steady = time >= max(time[-1] - 1.0, settle_from)

    angle_settle = settling_time(time[window], angle[window], ANGLE_TOLERANCE)
    position_settle = settling_time(time[window], position_error[window], POSITION_TOLERANCE)
    settled = float(np.nanmax([angle_settle, position_settle])) if not (
        np.isnan(angle_settle) or np.isnan(position_settle)
    ) else float("nan")

    upright = bool(np.max(np.abs(angle[window])) < FAILURE_ANGLE) if np.any(window) else False
    steady_ok = bool(
        np.mean(np.abs(angle[steady])) < np.deg2rad(3.0)
        and np.mean(np.abs(position_error[steady])) < 0.1
    )
    on_rail = bool(np.max(np.abs(result.states[:, 0])) < result.params.rail_limit)

    return Metrics(
        controller=result.controller_name,
        scenario=result.scenario_name,
        success=upright and steady_ok and on_rail,
        settling_time_s=settled,
        peak_angle_deg=float(np.rad2deg(np.max(np.abs(angle[window])))) if np.any(window) else float("nan"),
        rms_angle_deg=float(np.rad2deg(np.sqrt(np.mean(angle[window] ** 2)))) if np.any(window) else float("nan"),
        final_position_error_m=float(np.mean(position_error[steady])),
        rms_position_error_m=float(np.sqrt(np.mean(position_error[window] ** 2))) if np.any(window) else float("nan"),
        peak_force_n=float(np.max(np.abs(result.applied_force))),
        control_effort_n2s=float(np.sum(result.applied_force**2) * dt),
        itae=float(np.sum(time * np.abs(angle)) * dt),
        saturated_fraction=result.saturated_fraction,
    )


def to_markdown_table(rows: list[Metrics]) -> str:
    """Render metrics as a GitHub-flavoured markdown table."""
    header = (
        "| Scenario | Controller | Success | Settling [s] | Peak angle [deg] | "
        "RMS angle [deg] | Peak force [N] | Effort [N^2 s] | ITAE |"
    )
    divider = "|---|---|:---:|---:|---:|---:|---:|---:|---:|"

    def fmt(value: float, digits: int = 2) -> str:
        return "not settled" if np.isnan(value) else f"{value:.{digits}f}"

    lines = [header, divider]
    for row in rows:
        lines.append(
            f"| {row.scenario} | {row.controller} | {'yes' if row.success else 'no'} | "
            f"{fmt(row.settling_time_s)} | {fmt(row.peak_angle_deg)} | {fmt(row.rms_angle_deg)} | "
            f"{fmt(row.peak_force_n)} | {fmt(row.control_effort_n2s)} | {fmt(row.itae, 3)} |"
        )
    return "\n".join(lines)
