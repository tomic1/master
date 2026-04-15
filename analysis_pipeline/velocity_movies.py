"""Velocity-based movie helpers for bead-analysis overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle

from .vector_correlation import _prepare_velocity_dataframe, _velocity_output_name, _velocity_source, _velocity_column_names


def build_velocity_artifact_name(base_name: str, vector_cfg: Dict[str, Any], distance_mode: str | None = None) -> str:
    """Return a filename with the velocity-source and outlier suffixes applied."""

    return _velocity_output_name(base_name, vector_cfg, distance_mode=distance_mode)


def build_velocity_artifact_stem(base_stem: str, vector_cfg: Dict[str, Any], distance_mode: str | None = None) -> str:
    """Return a plot stem with the velocity-source and outlier suffixes applied."""

    named = _velocity_output_name(f"{base_stem}.pdf", vector_cfg, distance_mode=distance_mode)
    return named[:-4] if named.endswith(".pdf") else named


def _format_mmss(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:d}:{remainder:02d}"


def _format_elapsed_time(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, remainder = divmod(total_seconds, 60)
    if minutes <= 0:
        return f"{remainder:d} s"
    return f"{minutes:d} min {remainder:02d} s"


def _extract_position_columns(df: pd.DataFrame) -> tuple[str, str, bool]:
    if {"x_um", "y_um"}.issubset(df.columns):
        return "x_um", "y_um", True
    if {"x", "y"}.issubset(df.columns):
        return "x", "y", False
    raise ValueError("tracks_vel_df must contain x/y or x_um/y_um columns")


def render_bead_displacement_overlay_movie(
    img,
    tracks_vel_df: pd.DataFrame,
    out_path: str | Path,
    *,
    px_per_micron: float,
    vector_cfg: Dict[str, Any] | None = None,
    bead_channel: int = 1,
    background_channel: int | None = None,
    scalebar_um: float | None = None,
    scalebar_loc: str = "lower right",
    scalebar_height_px: int = 6,
    scalebar_pad_px: int = 10,
    frame_step: int = 1,
    vector_scale: float = 1.0,
    max_vectors: int | None = 3000,
    fps: float = 10,
    dpi: int = 150,
    show_title: bool = True,
    verbose: bool = True,
    progress_every: int = 10,
) -> Path:
    """Render a bead-displacement overlay movie with mm:ss timestamps.

    The movie uses the velocity source and optional Westerweel-Scarano filtering
    defined in ``vector_cfg`` so the rendered arrows match the vector-correlation
    configuration.
    """

    if tracks_vel_df is None or len(tracks_vel_df) == 0:
        raise ValueError("tracks_vel_df is empty")
    if not px_per_micron or float(px_per_micron) <= 0:
        raise ValueError("px_per_micron must be a positive number")

    vector_cfg = dict(vector_cfg or {})
    velocity_source = _velocity_source(vector_cfg)
    dfv = _prepare_velocity_dataframe(tracks_vel_df, vector_cfg, velocity_source)
    if len(dfv) == 0:
        raise ValueError("No velocity rows remain after optional outlier filtering")

    arr_shape = getattr(img, "shape", None)
    if arr_shape is None or len(arr_shape) not in (3, 4, 5):
        raise ValueError(f"img must be shaped (T,Y,X), (T,Z,Y,X), or (T,C,Z,Y,X), got {arr_shape}")
    T = int(arr_shape[0])

    if not {"frame", "dt_s"}.issubset(dfv.columns):
        raise ValueError("tracks_vel_df must contain frame and dt_s columns")

    x_col, y_col, coords_in_um = _extract_position_columns(dfv)
    vx_col, vy_col, _ = _velocity_column_names(dfv, velocity_source)

    dfv = dfv[dfv["dt_s"].to_numpy(dtype=float) > 0].copy()
    if len(dfv) == 0:
        raise ValueError("No valid rows with dt_s > 0")

    dfv["frame"] = dfv["frame"].astype(int)
    if "particle" in dfv.columns:
        dfv["particle"] = dfv["particle"].astype(int)
    else:
        dfv["particle"] = -1
    dfv["x_px"] = dfv[x_col].astype(float) * float(px_per_micron) if coords_in_um else dfv[x_col].astype(float)
    dfv["y_px"] = dfv[y_col].astype(float) * float(px_per_micron) if coords_in_um else dfv[y_col].astype(float)
    dt = dfv["dt_s"].astype(float).to_numpy()
    dfv["u_px_pf"] = dfv[vx_col].astype(float).to_numpy() * dt * float(px_per_micron)
    dfv["v_px_pf"] = dfv[vy_col].astype(float).to_numpy() * dt * float(px_per_micron)
    dfv["speed_um_s"] = np.sqrt(dfv[vx_col].to_numpy(dtype=float) ** 2 + dfv[vy_col].to_numpy(dtype=float) ** 2)

    frames = sorted(set(int(f) for f in dfv["frame"].unique()))
    frames = [f for f in frames if 0 <= f < T]
    frames = frames[::max(1, int(frame_step))]
    if not frames:
        raise ValueError("No frame indices available after filtering")
    progress_every = max(1, int(progress_every))

    if verbose:
        print(
            f"Rendering displacement overlay animation: {len(frames)} frames "
            f"(step={max(1, int(frame_step))})",
            flush=True,
        )

    frame_speed = dfv.groupby("frame", sort=True)["speed_um_s"].mean().reindex(range(T)).to_numpy()

    def _bg_zmax(ti: int) -> np.ndarray:
        fr = img[int(ti)]
        if len(arr_shape) == 5:
            channel_idx = int(background_channel if background_channel is not None else bead_channel)
            fr = fr[channel_idx]
        if hasattr(fr, "compute"):
            fr = fr.compute()
        fr = np.asarray(fr)
        if fr.ndim == 3:
            return fr.max(axis=0)
        if fr.ndim == 2:
            return fr
        raise ValueError(f"Unsupported frame ndim {fr.ndim} for index {ti}")

    bg0 = _bg_zmax(frames[0])
    height, width = bg0.shape
    fig, ax = plt.subplots(figsize=(7, 7 * height / max(width, 1)), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    im = ax.imshow(bg0, cmap="gray")

    artists: dict[str, Any] = {"Q": None}
    scatter = ax.scatter([], [], s=6, c="cyan", alpha=0.6, edgecolors="none")

    def _nice_scalebar_um(target_um: float) -> float:
        if not np.isfinite(target_um) or target_um <= 0:
            return 1.0
        exp10 = 10 ** np.floor(np.log10(target_um))
        mant = target_um / exp10
        if mant < 1.5:
            nice = 1.0
        elif mant < 3.5:
            nice = 2.0
        elif mant < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * exp10)

    sb_um = scalebar_um
    if sb_um is None and px_per_micron and float(px_per_micron) > 0:
        fov_um = float(width) / float(px_per_micron)
        sb_um = _nice_scalebar_um(0.2 * fov_um)

    sb_len_px = None
    if sb_um is not None and px_per_micron and float(px_per_micron) > 0:
        sb_len_px = int(max(1, round(float(sb_um) * float(px_per_micron))))
        sb_len_px = min(sb_len_px, max(1, width - 2 * int(scalebar_pad_px)))

    if sb_len_px is not None:
        assert sb_um is not None
        w_frac = float(sb_len_px) / max(float(width), 1.0)
        h_frac = float(scalebar_height_px) / max(float(height), 1.0)
        pad_x = float(scalebar_pad_px) / max(float(width), 1.0)
        pad_y = float(scalebar_pad_px) / max(float(height), 1.0)
        loc = (scalebar_loc or "lower right").lower()

        if loc in ("lower left", "ll"):
            x0, y0 = pad_x, pad_y
            y_text = y0 + h_frac + 0.01
            va = "bottom"
        elif loc in ("upper right", "ur"):
            x0, y0 = 1.0 - pad_x - w_frac, 1.0 - pad_y - h_frac
            y_text = y0 - 0.01
            va = "top"
        elif loc in ("upper left", "ul"):
            x0, y0 = pad_x, 1.0 - pad_y - h_frac
            y_text = y0 - 0.01
            va = "top"
        else:
            x0, y0 = 1.0 - pad_x - w_frac, pad_y
            y_text = y0 + h_frac + 0.01
            va = "bottom"

        rect = Rectangle((x0, y0), w_frac, h_frac, transform=ax.transAxes, facecolor="white", edgecolor="none", zorder=6)
        ax.add_patch(rect)
        ax.text(
            x0 + 0.5 * w_frac,
            y_text,
            f"{float(sb_um):g} µm",
            transform=ax.transAxes,
            color="white",
            ha="center",
            va=va,
            fontsize=10,
            zorder=7,
        )

    def update(i):
        ii = int(i)
        ti = frames[ii]
        if verbose and ((ii % progress_every) == 0 or ii == (len(frames) - 1)):
            print(f"  rendering frame {ii + 1}/{len(frames)} (t={ti})", flush=True)
        im.set_data(_bg_zmax(ti))

        g = dfv[dfv["frame"] == int(ti)]
        if max_vectors is not None and len(g) > int(max_vectors):
            g = g.sample(int(max_vectors), random_state=0)

        xs = g["x_px"].to_numpy(dtype=float)
        ys = g["y_px"].to_numpy(dtype=float)
        us = g["u_px_pf"].to_numpy(dtype=float) * float(vector_scale)
        vs = g["v_px_pf"].to_numpy(dtype=float) * float(vector_scale)

        if artists["Q"] is not None:
            try:
                artists["Q"].remove()
            except Exception:
                pass
            artists["Q"] = None
        if xs.size:
            artists["Q"] = ax.quiver(xs, ys, us, vs, color="yellow", angles="xy", scale_units="xy", scale=1.0, width=0.002)

        scatter.set_offsets(np.c_[xs, ys] if xs.size else np.empty((0, 2)))

        if show_title:
            ms = frame_speed[int(ti)] if int(ti) < len(frame_speed) else np.nan
            if np.isfinite(ms):
                ax.set_title(f"t = {_format_elapsed_time(int(ti) / max(float(fps), 1e-12))} | mean speed = {ms:.3g} µm/s", fontsize=12, color="white")
            else:
                ax.set_title(f"t = {_format_elapsed_time(int(ti) / max(float(fps), 1e-12))}", fontsize=12, color="white")
        return [im, artists["Q"], scatter] if (artists["Q"] is not None) else [im, scatter]

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=200)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=int(round(float(fps))), bitrate=12000)
    ani.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    if verbose:
        print(f"Saved displacement-per-frame overlay animation to {out_path}")
    return out_path