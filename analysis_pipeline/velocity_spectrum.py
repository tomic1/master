from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .comparison import comparison_style_context, save_comparison_dual_pdf
from .beads_velocity import compute_velocity_from_tracks
from .vector_correlation import (
    _position_column_names,
    _prepare_velocity_dataframe,
    _velocity_column_names,
    _velocity_output_name,
    _velocity_source,
)


def velocity_spectrum_output_name(base_name: str, vector_cfg: Dict[str, Any]) -> str:
    return _velocity_output_name(base_name, vector_cfg)


def velocity_spectrum_artifact_stem(base_stem: str, vector_cfg: Dict[str, Any]) -> str:
    return Path(_velocity_output_name(f"{base_stem}.parquet", vector_cfg)).stem


def velocity_vorticity_output_name(base_name: str, vector_cfg: Dict[str, Any]) -> str:
    return _velocity_output_name(base_name, vector_cfg)


def velocity_vorticity_artifact_stem(base_stem: str, vector_cfg: Dict[str, Any]) -> str:
    return Path(_velocity_output_name(f"{base_stem}.parquet", vector_cfg)).stem

def velocity_vorticity_spectrum_output_name(base_name: str, vector_cfg: Dict[str, Any]) -> str:
    return _velocity_output_name(base_name, vector_cfg)

def velocity_vorticity_spectrum_artifact_stem(base_stem: str, vector_cfg: Dict[str, Any]) -> str:
    return Path(_velocity_output_name(f"{base_stem}.parquet", vector_cfg)).stem


def _velocity_grid_bounds(df: pd.DataFrame, *, padding_fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    pos_x, pos_y, pos_z = _position_column_names(df)
    points = df[[pos_x, pos_y, pos_z]].to_numpy(dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        raise ValueError("velocity dataframe contains no finite positions")

    lower = np.nanmin(points, axis=0)
    upper = np.nanmax(points, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    padding = span * float(padding_fraction)
    return lower - padding, upper + padding


def _xy_grid_bounds(df: pd.DataFrame, *, padding_fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    pos_x, pos_y, _ = _position_column_names(df)
    points = df[[pos_x, pos_y]].to_numpy(dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        raise ValueError("velocity dataframe contains no finite xy positions")

    lower = np.nanmin(points, axis=0)
    upper = np.nanmax(points, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    padding = span * float(padding_fraction)
    return lower - padding, upper + padding


def _grid_centers(lower: np.ndarray, upper: np.ndarray, grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    nx, ny, nz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    dx = float((upper[0] - lower[0]) / nx)
    dy = float((upper[1] - lower[1]) / ny)
    dz = float((upper[2] - lower[2]) / nz)
    x_centers = np.linspace(float(lower[0]) + dx / 2.0, float(upper[0]) - dx / 2.0, nx)
    y_centers = np.linspace(float(lower[1]) + dy / 2.0, float(upper[1]) - dy / 2.0, ny)
    z_centers = np.linspace(float(lower[2]) + dz / 2.0, float(upper[2]) - dz / 2.0, nz)
    return x_centers, y_centers, z_centers, (dx, dy, dz)


def _component_grid_from_samples(
    positions: np.ndarray,
    values: np.ndarray,
    *,
    grid_shape: tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    nx, ny, nz = grid_shape
    edges = [
        np.linspace(float(lower[axis]), float(upper[axis]), int(grid_shape[axis]) + 1)
        for axis in range(3)
    ]
    sums, _ = np.histogramdd(positions, bins=edges, weights=values)
    counts, _ = np.histogramdd(positions, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)

    x_centers, y_centers, z_centers, spacing = _grid_centers(lower, upper, grid_shape)
    occupied = np.argwhere(np.isfinite(grid))
    if occupied.size == 0:
        raise ValueError("cannot build a velocity field from an empty grid")

    sample_points = np.column_stack([x_centers[occupied[:, 0]], y_centers[occupied[:, 1]], z_centers[occupied[:, 2]]])
    target_mesh = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    target_points = np.column_stack([axis.ravel() for axis in target_mesh])
    try:
        filled = griddata(sample_points, grid[tuple(occupied.T)], target_points, method="linear")
    except Exception:
        filled = None
    if filled is None:
        filled = griddata(sample_points, grid[tuple(occupied.T)], target_points, method="nearest")
    elif np.isnan(filled).any():
        nearest = griddata(sample_points, grid[tuple(occupied.T)], target_points, method="nearest")
        filled = np.where(np.isnan(filled), nearest, filled)

    filled = np.asarray(filled, dtype=float).reshape(grid_shape)
    return filled, spacing


def _xy_grid_from_samples(
    positions: np.ndarray,
    values: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    interpolation_method: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Interpolate scattered xy samples to a regular grid."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    x_centers = np.linspace(float(lower[0]), float(upper[0]), nx)
    y_centers = np.linspace(float(lower[1]), float(upper[1]), ny)
    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers, indexing="xy")
    target_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    try:
        interpolated = griddata(positions, values, target_points, method=str(interpolation_method).strip().lower())
    except Exception:
        interpolated = None

    if interpolated is None:
        interpolated = griddata(positions, values, target_points, method="nearest")
    elif np.isnan(interpolated).any():
        nearest = griddata(positions, values, target_points, method="nearest")
        interpolated = np.where(np.isnan(interpolated), nearest, interpolated)

    field = np.asarray(interpolated, dtype=float).reshape(ny, nx)
    dx = float((x_centers[-1] - x_centers[0]) / max(nx - 1, 1)) if nx > 1 else 1.0
    dy = float((y_centers[-1] - y_centers[0]) / max(ny - 1, 1)) if ny > 1 else 1.0
    return field, x_centers, y_centers, (dx, dy)


def _select_frame_for_xy_vorticity(velocity_df: pd.DataFrame, frame_selection: Any) -> tuple[int, pd.DataFrame]:
    if velocity_df.empty or "frame" not in velocity_df.columns:
        return -1, pd.DataFrame()

    frame_values = np.asarray(sorted(pd.unique(velocity_df["frame"].dropna())), dtype=float)
    if frame_values.size == 0:
        return -1, pd.DataFrame()

    selection = frame_selection
    if selection is None:
        selection = "middle"

    if isinstance(selection, str):
        selection_key = selection.strip().lower()
        if selection_key in {"middle", "median", "center", "centre"}:
            selected_frame = int(frame_values[len(frame_values) // 2])
        elif selection_key in {"first", "start"}:
            selected_frame = int(frame_values[0])
        elif selection_key in {"last", "final"}:
            selected_frame = int(frame_values[-1])
        else:
            selected_frame = int(float(selection))
    else:
        selected_frame = int(selection)

    frame_df = velocity_df.loc[velocity_df["frame"].astype(int) == selected_frame].copy()
    if frame_df.empty:
        nearest_index = int(np.argmin(np.abs(frame_values - float(selected_frame))))
        selected_frame = int(frame_values[nearest_index])
        frame_df = velocity_df.loc[velocity_df["frame"].astype(int) == selected_frame].copy()
    return selected_frame, frame_df


def _compute_xy_vorticity_frame(
    frame_df: pd.DataFrame,
    *,
    grid_shape: tuple[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    interpolation_method: str,
    frame_index: int | None = None,
) -> pd.DataFrame:
    """Interpolate one frame to a 2D vector field and compute z-vorticity."""
    pos_x, pos_y, _ = _position_column_names(frame_df)
    try:
        vx_col, vy_col, _ = _velocity_column_names(frame_df, "drift_corrected")
    except Exception:
        vx_col, vy_col, _ = _velocity_column_names(frame_df, "raw")
    positions = frame_df[[pos_x, pos_y]].to_numpy(dtype=float)
    vx = frame_df[vx_col].to_numpy(dtype=float)
    vy = frame_df[vy_col].to_numpy(dtype=float)

    valid = np.all(np.isfinite(positions), axis=1) & np.isfinite(vx) & np.isfinite(vy)
    positions = positions[valid]
    vx = vx[valid]
    vy = vy[valid]
    if positions.size == 0:
        return pd.DataFrame()

    vx_grid, x_centers, y_centers, spacing = _xy_grid_from_samples(
        positions,
        vx,
        grid_shape=grid_shape,
        lower=lower,
        upper=upper,
        interpolation_method=interpolation_method,
    )
    vy_grid, _, _, _ = _xy_grid_from_samples(
        positions,
        vy,
        grid_shape=grid_shape,
        lower=lower,
        upper=upper,
        interpolation_method=interpolation_method,
    )

    dvy_dy, dvy_dx = np.gradient(vy_grid, y_centers, x_centers, edge_order=1)
    dvx_dy, dvx_dx = np.gradient(vx_grid, y_centers, x_centers, edge_order=1)
    vorticity = dvy_dx - dvx_dy

    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers, indexing="xy")
    frame_value = int(frame_index) if frame_index is not None else int(frame_df["frame"].iloc[0]) if "frame" in frame_df.columns and not frame_df.empty else -1
    return pd.DataFrame(
        {
            "frame": frame_value,
            "x_um": x_mesh.ravel(),
            "y_um": y_mesh.ravel(),
            "vx_um_s": vx_grid.ravel(),
            "vy_um_s": vy_grid.ravel(),
            "speed_um_s": np.sqrt(vx_grid**2 + vy_grid**2).ravel(),
            "vorticity_s_inv": vorticity.ravel(),
            "grid_nx": int(grid_shape[0]),
            "grid_ny": int(grid_shape[1]),
            "dx_um": float(spacing[0]),
            "dy_um": float(spacing[1]),
            "interpolation_method": str(interpolation_method),
        }
    )


def plot_xy_vorticity_overlay(
    ax,
    df: pd.DataFrame,
    title: str | None = None,
    *,
    quiver_stride: int = 3,
    quiver_scale: float | None = None,
    cmap: str = "RdBu_r",
    vorticity_limit: float | None = None,
    quiver_color: str = "black",
) -> None:
    """Plot an interpolated xy vector field over a vorticity heatmap."""
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    required = {"x_um", "y_um", "vx_um_s", "vy_um_s", "vorticity_s_inv"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required.difference(df.columns)))
        raise ValueError(f"vorticity dataframe missing required columns: {missing}")

    work = df.dropna(subset=["x_um", "y_um", "vx_um_s", "vy_um_s", "vorticity_s_inv"]).copy()
    if work.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no finite field", ha="center", va="center", transform=ax.transAxes)
        return

    x_values = np.sort(np.asarray(pd.unique(work["x_um"]), dtype=float))
    y_values = np.sort(np.asarray(pd.unique(work["y_um"]), dtype=float))
    x_mesh, y_mesh = np.meshgrid(x_values, y_values, indexing="xy")

    pivot_vorticity = work.pivot(index="y_um", columns="x_um", values="vorticity_s_inv").reindex(index=y_values, columns=x_values)
    pivot_vx = work.pivot(index="y_um", columns="x_um", values="vx_um_s").reindex(index=y_values, columns=x_values)
    pivot_vy = work.pivot(index="y_um", columns="x_um", values="vy_um_s").reindex(index=y_values, columns=x_values)

    vorticity = pivot_vorticity.to_numpy(dtype=float)
    vx = pivot_vx.to_numpy(dtype=float)
    vy = pivot_vy.to_numpy(dtype=float)

    finite = np.isfinite(vorticity)
    if not np.any(finite):
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no finite vorticity", ha="center", va="center", transform=ax.transAxes)
        return

    if vorticity_limit is None:
        vmax = float(np.nanmax(np.abs(vorticity[finite])))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        vorticity_limit = vmax
    vmin = -float(vorticity_limit)
    vmax = float(vorticity_limit)

    mesh = ax.pcolormesh(x_mesh, y_mesh, vorticity, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    stride = max(1, int(quiver_stride))
    quiver_kwargs: dict[str, Any] = {
        "color": quiver_color,
        "angles": "xy",
        "scale_units": "xy",
        "width": 0.0025,
        "pivot": "mid",
    }
    if quiver_scale is not None and np.isfinite(quiver_scale) and quiver_scale > 0.0:
        quiver_kwargs["scale"] = float(quiver_scale)
    ax.quiver(x_mesh[::stride, ::stride], y_mesh[::stride, ::stride], vx[::stride, ::stride], vy[::stride, ::stride], **quiver_kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"x ($\mu\mathrm{m}$)")
    ax.set_ylabel(r"y ($\mu\mathrm{m}$)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=r"$\omega_z$ ($\mathrm{s}^{-1}$)")


def _spatial_window_3d(grid_shape: tuple[int, int, int], window_function: str = "hann") -> np.ndarray:
    if str(window_function).strip().lower() not in {"hann", "hanning"}:
        return np.ones(tuple(int(size) for size in grid_shape), dtype=float)

    wx = np.hanning(int(grid_shape[0])) if int(grid_shape[0]) > 1 else np.ones(1, dtype=float)
    wy = np.hanning(int(grid_shape[1])) if int(grid_shape[1]) > 1 else np.ones(1, dtype=float)
    wz = np.hanning(int(grid_shape[2])) if int(grid_shape[2]) > 1 else np.ones(1, dtype=float)
    window = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    rms = float(np.sqrt(np.mean(window**2)))
    if np.isfinite(rms) and rms > 0.0:
        window = window / rms
    return window


def _spatial_window_2d(grid_shape: tuple[int, int], window_function: str = "hann") -> np.ndarray:
    if str(window_function).strip().lower() not in {"hann", "hanning"}:
        return np.ones(tuple(int(size) for size in grid_shape), dtype=float)

    wx = np.hanning(int(grid_shape[0])) if int(grid_shape[0]) > 1 else np.ones(1, dtype=float)
    wy = np.hanning(int(grid_shape[1])) if int(grid_shape[1]) > 1 else np.ones(1, dtype=float)
    window = wy[:, None] * wx[None, :]
    rms = float(np.sqrt(np.mean(window**2)))
    if np.isfinite(rms) and rms > 0.0:
        window = window / rms
    return window


def _shell_average_spectrum(
    power: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    *,
    k_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_mag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    finite = np.isfinite(power) & np.isfinite(k_mag)
    if not np.any(finite):
        raise ValueError("spectrum contains no finite values")

    power_valid = power[finite]
    k_valid = k_mag[finite]
    k_max = float(np.nanmax(k_valid))
    if not np.isfinite(k_max) or k_max <= 0:
        raise ValueError("invalid wavenumber range for spectrum")

    bins = np.linspace(0.0, k_max, max(2, int(k_bins)) + 1)
    counts, _ = np.histogram(k_valid, bins=bins)
    weighted, _ = np.histogram(k_valid, bins=bins, weights=power_valid)
    centers = 0.5 * (bins[:-1] + bins[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = np.divide(weighted, counts, out=np.full_like(centers, np.nan, dtype=float), where=counts > 0)
    return centers, spectrum, counts.astype(float)


def _shell_average_spectrum_2d(
    power: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    k_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_mag = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    finite = np.isfinite(power) & np.isfinite(k_mag)
    if not np.any(finite):
        raise ValueError("spectrum contains no finite values")

    power_valid = power[finite]
    k_valid = k_mag[finite]
    k_max = float(np.nanmax(k_valid))
    if not np.isfinite(k_max) or k_max <= 0:
        raise ValueError("invalid wavenumber range for spectrum")

    bins = np.linspace(0.0, k_max, max(2, int(k_bins)) + 1)
    counts, _ = np.histogram(k_valid, bins=bins)
    weighted, _ = np.histogram(k_valid, bins=bins, weights=power_valid)
    centers = 0.5 * (bins[:-1] + bins[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = np.divide(weighted, counts, out=np.full_like(centers, np.nan, dtype=float), where=counts > 0)
    return centers, spectrum, counts.astype(float)


def _compute_xy_vorticity_spectrum_from_field(
    vorticity_df: pd.DataFrame,
    *,
    k_bins: int,
    subtract_mean: bool,
    apply_window: bool,
    window_function: str,
    frame_index: int | None = None,
) -> pd.DataFrame:
    """Compute a 2D isotropic vorticity spectrum from a gridded vorticity field."""
    required = {"x_um", "y_um", "vorticity_s_inv"}
    if vorticity_df.empty or not required.issubset(vorticity_df.columns):
        return pd.DataFrame()

    work = vorticity_df.dropna(subset=["x_um", "y_um", "vorticity_s_inv"]).copy()
    if work.empty:
        return pd.DataFrame()

    x_values = np.sort(np.asarray(pd.unique(work["x_um"]), dtype=float))
    y_values = np.sort(np.asarray(pd.unique(work["y_um"]), dtype=float))
    if x_values.size < 2 or y_values.size < 2:
        return pd.DataFrame()

    omega_grid = (
        work.pivot(index="y_um", columns="x_um", values="vorticity_s_inv")
        .reindex(index=y_values, columns=x_values)
        .to_numpy(dtype=float)
    )
    if not np.any(np.isfinite(omega_grid)):
        return pd.DataFrame()

    if bool(subtract_mean):
        omega_grid = omega_grid - float(np.nanmean(omega_grid))

    if bool(apply_window):
        omega_grid = omega_grid * _spatial_window_2d((x_values.size, y_values.size), window_function=window_function)

    dx = float(np.nanmean(np.diff(x_values)))
    dy = float(np.nanmean(np.diff(y_values)))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("invalid xy grid spacing for vorticity spectrum")

    omega_hat = np.fft.fftn(omega_grid, norm="backward") * float(dx * dy)
    power = 0.5 * np.abs(omega_hat) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(x_values.size, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(y_values.size, d=dy)
    k_centers, spectrum, counts = _shell_average_spectrum_2d(power, kx, ky, k_bins=k_bins)
    if k_centers.size == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "k_rad_per_um": k_centers,
            "enstrophy": spectrum,
            "mode_count": counts,
            "frame": int(frame_index) if frame_index is not None else int(work["frame"].iloc[0]) if "frame" in work.columns and not work.empty else -1,
            "grid_nx": int(x_values.size),
            "grid_ny": int(y_values.size),
            "dx_um": float(dx),
            "dy_um": float(dy),
            "cell_area_um2": float(dx * dy),
        }
    )


def plot_vorticity_spectrum(
    ax,
    df: pd.DataFrame,
    title: str | None = None,
    *,
    label: str | None = None,
    color: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
) -> None:
    """Plot an isotropic vorticity spectrum on log-log axes."""
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    work = df.copy()
    if "k_rad_per_um" not in work.columns:
        raise ValueError("vorticity spectrum dataframe must contain k_rad_per_um")
    if "enstrophy" in work.columns:
        value_col = "enstrophy"
    elif "energy" in work.columns:
        value_col = "energy"
    else:
        value_col = work.columns[1]
    work = work.sort_values("k_rad_per_um")
    x = work["k_rad_per_um"].to_numpy(dtype=float)
    y = work[value_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if x_range is not None:
        lower, upper = x_range
        if lower is not None:
            mask &= x >= float(lower)
        if upper is not None:
            mask &= x <= float(upper)
    if y_range is not None:
        lower, upper = y_range
        if lower is not None:
            mask &= y >= float(lower)
        if upper is not None:
            mask &= y <= float(upper)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no finite spectrum", ha="center", va="center", transform=ax.transAxes)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, y, lw=2.0, color=color, label=label)
    if title:
        ax.set_title(title)
    ax.set_xlabel(r"wavenumber $k$ ($\mathrm{rad}\,\mu\mathrm{m}^{-1}$)")
    ax.set_ylabel(r"$\Omega(k)$")
    ax.grid(True, which="both", alpha=0.25)
    x_limits = _log_axis_limits([x])
    y_limits = _log_axis_limits([y])
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if label is not None:
        ax.legend(frameon=True)


def _smooth_positive_series(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window_size = max(1, int(window_size))
    if values.size == 0 or window_size <= 1:
        return values.copy()

    valid = np.isfinite(values) & (values > 0.0)
    if not np.any(valid):
        return values.copy()

    series = pd.Series(np.where(valid, values, np.nan), dtype=float)
    logged = pd.Series(np.log(series.clip(lower=1e-300).to_numpy(dtype=float)), index=series.index)
    smoothed = logged.rolling(window_size, center=True, min_periods=1).mean()
    return np.exp(smoothed.to_numpy(dtype=float))


def _compute_frame_spectrum(
    frame_df: pd.DataFrame,
    *,
    grid_shape: tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    k_bins: int,
    subtract_mean: bool,
    apply_window: bool,
    window_function: str,
    frame_index: int | None = None,
) -> pd.DataFrame:
    pos_x, pos_y, pos_z = _position_column_names(frame_df)
    vx_col, vy_col, vz_col = _velocity_column_names(frame_df, "drift_corrected")
    positions = frame_df[[pos_x, pos_y, pos_z]].to_numpy(dtype=float)
    vx = frame_df[vx_col].to_numpy(dtype=float)
    vy = frame_df[vy_col].to_numpy(dtype=float)
    vz = frame_df[vz_col].to_numpy(dtype=float)

    valid = np.all(np.isfinite(positions), axis=1) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
    positions = positions[valid]
    vx = vx[valid]
    vy = vy[valid]
    vz = vz[valid]
    if positions.size == 0:
        return pd.DataFrame()

    vx_grid, spacing = _component_grid_from_samples(positions, vx, grid_shape=grid_shape, lower=lower, upper=upper)
    vy_grid, _ = _component_grid_from_samples(positions, vy, grid_shape=grid_shape, lower=lower, upper=upper)
    vz_grid, _ = _component_grid_from_samples(positions, vz, grid_shape=grid_shape, lower=lower, upper=upper)

    if bool(subtract_mean):
        vx_grid = vx_grid - float(np.nanmean(vx_grid))
        vy_grid = vy_grid - float(np.nanmean(vy_grid))
        vz_grid = vz_grid - float(np.nanmean(vz_grid))

    if bool(apply_window):
        window = _spatial_window_3d(grid_shape, window_function=window_function)
        vx_grid = vx_grid * window
        vy_grid = vy_grid * window
        vz_grid = vz_grid * window

    cell_volume = float(spacing[0] * spacing[1] * spacing[2])
    vx_hat = np.fft.fftn(vx_grid, norm="backward") * cell_volume
    vy_hat = np.fft.fftn(vy_grid, norm="backward") * cell_volume
    vz_hat = np.fft.fftn(vz_grid, norm="backward") * cell_volume
    power = 0.5 * (np.abs(vx_hat) ** 2 + np.abs(vy_hat) ** 2 + np.abs(vz_hat) ** 2)

    nx, ny, nz = grid_shape
    dx, dy, dz = spacing
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)

    k_centers, energy, counts = _shell_average_spectrum(power, kx, ky, kz, k_bins=k_bins)
    if k_centers.size == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "k_rad_per_um": k_centers,
            "energy": energy,
            "mode_count": counts,
            "frame": int(frame_index) if frame_index is not None else int(frame_df["frame"].iloc[0]) if "frame" in frame_df.columns and not frame_df.empty else -1,
            "grid_nx": int(nx),
            "grid_ny": int(ny),
            "grid_nz": int(nz),
            "dx_um": float(dx),
            "dy_um": float(dy),
            "dz_um": float(dz),
            "cell_volume_um3": float(cell_volume),
        }
    )


def _aggregate_frame_spectra(frame_spectra: pd.DataFrame, *, plot_smoothing_bins: int = 5) -> pd.DataFrame:
    if frame_spectra.empty:
        return pd.DataFrame()

    grouped = (
        frame_spectra.dropna(subset=["k_rad_per_um", "energy"])
        .groupby("k_rad_per_um", as_index=False)
        .agg(
            energy_mean=("energy", "mean"),
            energy_std=("energy", "std"),
            energy_var=("energy", "var"),
            mode_count_mean=("mode_count", "mean"),
            n_frames=("frame", "nunique"),
        )
        .sort_values("k_rad_per_um")
        .reset_index(drop=True)
    )
    grouped["energy_std"] = grouped["energy_std"].fillna(0.0)
    grouped["energy_var"] = grouped["energy_var"].fillna(0.0)
    grouped["energy_sem"] = grouped["energy_std"] / np.sqrt(np.maximum(grouped["n_frames"].to_numpy(dtype=float), 1.0))
    grouped["energy_smooth"] = _smooth_positive_series(grouped["energy_mean"].to_numpy(dtype=float), plot_smoothing_bins)
    return grouped


def _log_axis_limits(values: list[np.ndarray], *, pad_fraction: float = 0.12) -> tuple[float, float] | None:
    finite_parts: list[np.ndarray] = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        array = array[np.isfinite(array) & (array > 0.0)]
        if array.size:
            finite_parts.append(array)
    if not finite_parts:
        return None

    combined = np.concatenate(finite_parts)
    lower = float(np.nanmin(combined))
    upper = float(np.nanmax(combined))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= 0.0:
        return None
    if lower == upper:
        return lower / 10.0, upper * 10.0

    pad = float(np.clip(pad_fraction, 0.0, 0.45))
    factor = float(np.exp(np.log(upper / lower) * pad)) if upper > lower else 1.0
    if not np.isfinite(factor) or factor <= 1.0:
        factor = 1.15
    return lower / factor, upper * factor


def _fit_power_law_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fit_range: tuple[float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if fit_range is not None:
        lower, upper = fit_range
        if lower is not None:
            mask &= x >= float(lower)
        if upper is not None:
            mask &= x <= float(upper)

    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 3:
        return None

    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    alpha = float(-slope)
    x_line = np.logspace(float(np.log10(np.nanmin(x_fit))), float(np.log10(np.nanmax(x_fit))), 240)
    y_line = 10.0 ** intercept * np.power(x_line, slope)
    return x_line, y_line, alpha


def plot_velocity_spectrum(
    ax,
    df: pd.DataFrame,
    title: str | None = None,
    *,
    label: str | None = None,
    color: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
) -> None:
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    work = df.copy()
    if "k_rad_per_um" not in work.columns:
        raise ValueError("spectrum dataframe must contain k_rad_per_um")
    if "energy_smooth" in work.columns:
        energy_col = "energy_smooth"
    elif "energy_mean" in work.columns:
        energy_col = "energy_mean"
    else:
        energy_col = "energy"
    sem_col = "energy_std" if "energy_std" in work.columns else ("energy_sem" if "energy_sem" in work.columns else None)
    work = work.sort_values("k_rad_per_um")
    x = work["k_rad_per_um"].to_numpy(dtype=float)
    y = work[energy_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if x_range is not None:
        lower, upper = x_range
        if lower is not None:
            mask &= x >= float(lower)
        if upper is not None:
            mask &= x <= float(upper)
    if y_range is not None:
        lower, upper = y_range
        if lower is not None:
            mask &= y >= float(lower)
        if upper is not None:
            mask &= y <= float(upper)
    x = x[mask]
    y = y[mask]
    if sem_col is not None and sem_col in work.columns:
        yerr = work[sem_col].to_numpy(dtype=float)[mask]
    else:
        yerr = None

    if x.size == 0:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no finite spectrum", ha="center", va="center", transform=ax.transAxes)
        return

    display_label = label
    x_fit = None
    y_fit = None
    fit_curve = _fit_power_law_curve(x, y, fit_range=x_range)
    if fit_curve is not None:
        x_fit, y_fit, alpha = fit_curve
        fit_label = rf"$\alpha={alpha:.2f}$"
        display_label = fit_label if label is None else f"{label} ({fit_label})"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, y, lw=2.0, color=color, label=display_label)
    if yerr is not None and np.any(np.isfinite(yerr)):
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        lower = np.clip(y - yerr, 1e-30, None)
        upper = np.clip(y + yerr, 1e-30, None)
        ax.fill_between(x, lower, upper, alpha=0.15, color=color, linewidth=0)

    if fit_curve is not None and x_fit is not None and y_fit is not None:
        ax.plot(x_fit, y_fit, ls="--", lw=1.6, color=color, alpha=0.95, label="_nolegend_")

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"wavenumber $k$ ($\mathrm{rad}\,\mu\mathrm{m}^{-1}$)")
    ax.set_ylabel(r"$E(k)$")
    ax.grid(True, which="both", alpha=0.25)
    x_limits = _log_axis_limits([x])
    y_limits = _log_axis_limits([y])
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if display_label is not None:
        ax.legend(frameon=True)


def save_velocity_spectrum_dual_pdf(
    fig,
    output_dir: str | Path,
    stem: str,
    *,
    white_plot_fn=None,
    dpi: int = 150,
) -> dict[str, Path]:
    return save_comparison_dual_pdf(fig, output_dir, stem, white_plot_fn=white_plot_fn, dpi=dpi)


def run_velocity_spectrum_core(
    config: Dict[str, Any],
    state: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    from .config import merge_overrides
    from .io_dataset import load_dataset_state

    cfg = config if overrides is None else merge_overrides(config, overrides)
    runtime = cfg.get("runtime", {})
    spectrum_cfg = dict(cfg.get("velocity_spectrum", {}))
    vector_cfg = dict(cfg.get("vector_corr", {}))
    xy_vorticity_cfg = dict(spectrum_cfg.get("xy_vorticity", {}))
    xy_vorticity_spectrum_cfg = dict(xy_vorticity_cfg.get("spectrum", {}))
    skip_existing = bool(runtime.get("skip_existing", True))

    if state is None:
        state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))

    derived_dir = Path(state["paths"]["derived_dir"])
    vel_path = derived_dir / "beads_tracks_with_velocity.parquet"
    if vel_path.exists() and skip_existing:
        vel_df = pd.read_parquet(vel_path)
    else:
        if vel_path.exists():
            vel_df = pd.read_parquet(vel_path)
        else:
            tracks_path = derived_dir / "beads_tracks.parquet"
            if tracks_path.exists():
                tracks_df = pd.read_parquet(tracks_path)
                vel_df = compute_velocity_from_tracks(state, tracks_df, skip_existing=False)
            else:
                vel_df = pd.DataFrame()
    if vel_df.empty:
        return {
            "velocity_spectrum_df": pd.DataFrame(),
            "velocity_spectrum_frames_df": pd.DataFrame(),
            "velocity_vorticity_df": pd.DataFrame(),
            "velocity_vorticity_spectrum_df": pd.DataFrame(),
        }

    velocity_source = _velocity_source(vector_cfg)
    vel_df = _prepare_velocity_dataframe(vel_df, vector_cfg, velocity_source)
    if vel_df.empty:
        return {
            "velocity_spectrum_df": pd.DataFrame(),
            "velocity_spectrum_frames_df": pd.DataFrame(),
            "velocity_vorticity_df": pd.DataFrame(),
            "velocity_vorticity_spectrum_df": pd.DataFrame(),
        }

    grid_shape_cfg = spectrum_cfg.get("grid_shape", (32, 32, 32))
    if isinstance(grid_shape_cfg, (list, tuple)) and len(grid_shape_cfg) == 3:
        grid_shape = (int(grid_shape_cfg[0]), int(grid_shape_cfg[1]), int(grid_shape_cfg[2]))
    else:
        raise ValueError("velocity_spectrum.grid_shape must be a 3-element sequence")
    if any(size <= 0 for size in grid_shape):
        raise ValueError("velocity_spectrum.grid_shape values must be positive")

    lower, upper = _velocity_grid_bounds(vel_df, padding_fraction=float(spectrum_cfg.get("box_padding_fraction", 0.05)))
    k_bins = int(spectrum_cfg.get("k_bins", 48))
    subtract_mean = bool(spectrum_cfg.get("subtract_mean", True))
    apply_window = bool(spectrum_cfg.get("apply_window", True))
    window_function = str(spectrum_cfg.get("window_function", "hann"))
    plot_smoothing_bins = int(spectrum_cfg.get("plot_smoothing_bins", 5))

    frame_rows: list[pd.DataFrame] = []
    frame_groups = vel_df.groupby("frame", sort=True)
    for frame_value, frame_df in frame_groups:
        frame_index = int(frame_df["frame"].iloc[0])
        spectrum_df = _compute_frame_spectrum(
            frame_df,
            grid_shape=grid_shape,
            lower=lower,
            upper=upper,
            k_bins=k_bins,
            subtract_mean=subtract_mean,
            apply_window=apply_window,
            window_function=window_function,
            frame_index=frame_index,
        )
        if spectrum_df.empty:
            continue
        spectrum_df = spectrum_df.copy()
        spectrum_df["n_points"] = int(len(frame_df))
        frame_rows.append(spectrum_df)

    frame_spectra = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame()
    aggregated = _aggregate_frame_spectra(frame_spectra, plot_smoothing_bins=plot_smoothing_bins)

    xy_vorticity_df = pd.DataFrame()
    xy_vorticity_spectrum_df = pd.DataFrame()
    xy_vorticity_enabled = bool(xy_vorticity_cfg.get("enabled", False))
    xy_vorticity_spectrum_enabled = bool(xy_vorticity_spectrum_cfg.get("enabled", False))
    if xy_vorticity_enabled or xy_vorticity_spectrum_enabled:
        xy_grid_shape_cfg = xy_vorticity_cfg.get("grid_shape", (64, 64))
        if isinstance(xy_grid_shape_cfg, (list, tuple)) and len(xy_grid_shape_cfg) == 2:
            xy_grid_shape = (int(xy_grid_shape_cfg[0]), int(xy_grid_shape_cfg[1]))
        else:
            raise ValueError("velocity_spectrum.xy_vorticity.grid_shape must be a 2-element sequence")
        if any(size <= 0 for size in xy_grid_shape):
            raise ValueError("velocity_spectrum.xy_vorticity.grid_shape values must be positive")

        xy_padding_fraction = float(xy_vorticity_cfg.get("box_padding_fraction", spectrum_cfg.get("box_padding_fraction", 0.05)))
        interpolation_method = str(xy_vorticity_cfg.get("interpolation_method", "linear"))
        frame_selection = xy_vorticity_cfg.get("frame", "middle")
        selected_frame, xy_frame_df = _select_frame_for_xy_vorticity(vel_df, frame_selection)
        if not xy_frame_df.empty:
            xy_lower, xy_upper = _xy_grid_bounds(xy_frame_df, padding_fraction=xy_padding_fraction)
            xy_vorticity_df = _compute_xy_vorticity_frame(
                xy_frame_df,
                grid_shape=xy_grid_shape,
                lower=xy_lower,
                upper=xy_upper,
                interpolation_method=interpolation_method,
                frame_index=selected_frame,
            )

            if xy_vorticity_spectrum_enabled:
                xy_vorticity_spectrum_df = _compute_xy_vorticity_spectrum_from_field(
                    xy_vorticity_df,
                    k_bins=int(xy_vorticity_spectrum_cfg.get("k_bins", spectrum_cfg.get("k_bins", 48))),
                    subtract_mean=bool(xy_vorticity_spectrum_cfg.get("subtract_mean", True)),
                    apply_window=bool(xy_vorticity_spectrum_cfg.get("apply_window", True)),
                    window_function=str(xy_vorticity_spectrum_cfg.get("window_function", "hann")),
                    frame_index=selected_frame,
                )

    spectrum_name = velocity_spectrum_output_name("beads_velocity_spectrum.parquet", vector_cfg)
    frame_name = velocity_spectrum_output_name("beads_velocity_spectrum_frames.parquet", vector_cfg)
    vorticity_name = velocity_vorticity_output_name("beads_velocity_vorticity_xy.parquet", vector_cfg)
    vorticity_spectrum_name = velocity_vorticity_spectrum_output_name("beads_velocity_vorticity_spectrum.parquet", vector_cfg)
    aggregated.to_parquet(derived_dir / spectrum_name, index=False)
    frame_spectra.to_parquet(derived_dir / frame_name, index=False)
    if not xy_vorticity_df.empty:
        xy_vorticity_df.to_parquet(derived_dir / vorticity_name, index=False)
    if not xy_vorticity_spectrum_df.empty:
        xy_vorticity_spectrum_df.to_parquet(derived_dir / vorticity_spectrum_name, index=False)

    return {
        "velocity_spectrum_df": aggregated,
        "velocity_spectrum_frames_df": frame_spectra,
        "velocity_vorticity_df": xy_vorticity_df,
        "velocity_vorticity_spectrum_df": xy_vorticity_spectrum_df,
    }