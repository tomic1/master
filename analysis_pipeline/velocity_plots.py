"""Velocity-based plotting helpers for bead-analysis outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .comparison import comparison_style_context, save_comparison_dual_pdf


def _format_time_label(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:d}:{remainder:02d}"


def _prepare_velocity_time_series(vel_df: pd.DataFrame, fps: float | None) -> pd.DataFrame:
    if vel_df is None or vel_df.empty or "frame" not in vel_df.columns:
        return pd.DataFrame()

    speed_cols = [
        column
        for column in ("speed_um_s", "drift_speed_um_s", "speed_drift_corrected_um_s")
        if column in vel_df.columns
    ]
    if not speed_cols:
        return pd.DataFrame()

    grouped = vel_df.groupby("frame", sort=True)[speed_cols].mean(numeric_only=True).reset_index()
    fps_value = float(fps) if fps is not None and float(fps) > 0 else None
    if fps_value is None:
        grouped["time_s"] = grouped["frame"].astype(float)
    else:
        grouped["time_s"] = grouped["frame"].astype(float) / fps_value
    grouped["time_label"] = grouped["time_s"].map(_format_time_label)
    return grouped


def plot_velocity_over_time(
    ax,
    vel_df: pd.DataFrame,
    *,
    fps: float | None,
    title: str | None = None,
) -> None:
    """Plot mean bead speed, drift speed, and drift-corrected speed over time."""

    time_df = _prepare_velocity_time_series(vel_df, fps)
    if time_df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    curves = (
        ("speed_um_s", "speed", "#6baed6"),
        ("drift_speed_um_s", "drift speed", "#fd8d3c"),
        ("speed_drift_corrected_um_s", "drift-corrected speed", "#74c476"),
    )
    x = time_df["time_s"].to_numpy(dtype=float)
    plotted = False
    for column, label, color in curves:
        if column not in time_df.columns:
            continue
        y = time_df[column].to_numpy(dtype=float)
        ax.plot(x, y, lw=1.9, color=color, label=label)
        plotted = True

    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"speed ($\mu\mathrm{m}/\mathrm{s}$)")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    if plotted:
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    finite_x = x[np.isfinite(x)]
    if finite_x.size:
        x_min = float(np.nanmin(finite_x))
        x_max = float(np.nanmax(finite_x))
        if x_min == x_max:
            margin = 1.0 if x_min == 0.0 else abs(x_min) * 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
        else:
            ax.set_xlim(x_min, x_max)

    finite_y_parts = [time_df[column].to_numpy(dtype=float) for column, _, _ in curves if column in time_df.columns]
    finite_y_arrays = [values[np.isfinite(values)] for values in finite_y_parts if np.isfinite(values).any()]
    if finite_y_arrays:
        finite_y = np.concatenate(finite_y_arrays)
        y_min = float(np.nanmin(finite_y))
        y_max = float(np.nanmax(finite_y))
        if y_min == y_max:
            margin = 1.0 if y_min == 0.0 else abs(y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            pad = (y_max - y_min) * 0.06
            ax.set_ylim(y_min - pad, y_max + pad)


def save_velocity_over_time_dual_pdf(
    vel_df: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    fps: float | None,
    title: str | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Save a velocity-over-time plot as matching black and white PDFs."""

    with comparison_style_context("dark"):
        fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=dpi)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        plot_velocity_over_time(ax, vel_df, fps=fps, title=title)

        def _white(ax_white) -> None:
            plot_velocity_over_time(ax_white, vel_df, fps=fps, title=None)

        paths = save_comparison_dual_pdf(fig, output_dir, stem, white_plot_fn=_white, dpi=dpi)
        plt.close(fig)
        return paths
