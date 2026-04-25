"""Notebook-like orchestration for the analysis pipeline.

Feature diagram
===============

analysis_unified.py
|-- beads
|   compute / plot / overwrite
|-- autocorr
|   compute / plot / overwrite
|-- image_corr
|   compute / plot / overwrite
|-- vector_corr
|   compute / plot / overwrite
|-- summary
|   compute / plot / overwrite

The heavy math stays in ``analysis_pipeline``. This module only loads config,
dispatches per-feature work, and keeps the workflow notebook-like.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis_pipeline.beads_track import detect_and_link_beads, preview_bead_detection
from analysis_pipeline.beads_velocity import compute_angular_speed_xy, compute_velocity_from_tracks
from analysis_pipeline.comparison import (
    ComparisonSpec,
    build_comparison_specs,
    comparison_output_dir,
    comparison_palette,
    comparison_style_context,
    comparison_registry_from_config,
    resolve_comparison_preset,
    save_comparison_dual_pdf,
)
from analysis_pipeline.config import load_analysis_config, merge_overrides
from analysis_pipeline.correlation_plots import (
    _exp_decay,
    _fit_signed_decay,
    plot_spatial_vector_correlation,
    _fit_temporal_decay_with_error,
    plot_temporal_vector_correlation,
    plot_vector_tensor_correlation,
    plot_vector_tensor_pair_decay,
    save_vector_correlation_dual_pdf,
)
from analysis_pipeline.image_correlation import compute_raw_time_image_correlation, fit_time_image_correlation
from analysis_pipeline.io_dataset import load_dataset_state, prepare_output_dirs
from analysis_pipeline.pipeline import run_autocorr_core, run_vector_correlation_core
from analysis_pipeline.velocity_spectrum import (
    _fit_power_law_curve as _velocity_spectrum_fit_power_law_curve,
    _log_axis_limits as _velocity_spectrum_log_axis_limits,
    plot_xy_vorticity_overlay,
    plot_vorticity_spectrum,
    plot_velocity_spectrum,
    run_velocity_spectrum_core,
    velocity_spectrum_artifact_stem,
    velocity_vorticity_artifact_stem,
    velocity_spectrum_output_name,
    velocity_vorticity_output_name,
    velocity_vorticity_spectrum_artifact_stem,
    velocity_vorticity_spectrum_output_name,
)
from analysis_pipeline.velocity_movies import build_velocity_artifact_name, build_velocity_artifact_stem, render_bead_displacement_overlay_movie
from analysis_pipeline.velocity_plots import save_velocity_over_time_dual_pdf
from analysis_pipeline.vector_correlation import _velocity_output_name


DEFAULT_CONFIG_PATH = Path("config/analysis_default.yaml")
FEATURE_ORDER = ("beads", "autocorr", "image_corr", "vector_corr", "velocity_spectrum", "summary")

# Edit these values directly when you want to run the file like a notebook.
#NOTEBOOK_DATASET_ID =  "AMF_108_002__C640_C470"  # "AMF_105_002__C640_C470"
NOTEBOOK_DATASET_ID = None
NOTEBOOK_BASE_DIR = "/Volumes/X9"  # dataserver_files/Group_Bausch/Tom_Dataserver/20260407
NOTEBOOK_VARIATION: str | None = None
NOTEBOOK_FEATURE_COMMANDS = [
    "beads:compute=0,plot=0,overwrite=0",
    "autocorr:compute=0,plot=0,overwrite=0",
    "image_corr:compute=0,plot=0,overwrite=0",
    "vector_corr:compute=0,plot=0,overwrite=0",
    "velocity_spectrum:compute=1,plot=1",
    "summary:compute=0,plot=0,overwrite=0",
]
NOTEBOOK_ENABLE: list[str] = []
NOTEBOOK_DISABLE: list[str] = []
NOTEBOOK_OVERWRITE: list[str] = []
NOTEBOOK_RUN_ORDER: list[str] | None = None
NOTEBOOK_BATCH_DATASET_IDS: list[str] = []
NOTEBOOK_BATCH_BASE_DIRS: list[str] = []
NOTEBOOK_COMPARISON_NAME: str | None = None

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "beads": "Detects and links beads, computes velocities and optional angular speed; can also render displacement overlay movies and velocity-over-time plots.",
    "autocorr": "Computes FFT-based 2D/3D autocorrelations and radial averages; saves autocorrelation plots.",
    "image_corr": "Computes time-image correlations and fits; saves raw and fitted correlation plots.",
    "vector_corr": "Computes temporal, spatial, and tensor bead-motion vector correlations; can optionally reject Westerweel-Scarano outliers before plotting.",
    "velocity_spectrum": "Computes velocity spectra from drift-corrected 3D velocity fields and also supports 2D xy vorticity maps and vorticity spectra.",
    "summary": "Writes a feature manifest summarizing which outputs were produced.",
}


@dataclass
class FeatureSwitch:
    compute: bool = True
    plot: bool = True
    overwrite: bool = False


def _feature_switches() -> dict[str, FeatureSwitch]:
    return {name: FeatureSwitch() for name in FEATURE_ORDER}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _result_frame(result: dict[str, Any], *keys: str) -> pd.DataFrame:
    if not isinstance(result, dict):
        return pd.DataFrame()
    for key in keys:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            return value
    return pd.DataFrame()


def _finite_limits(values: Sequence[np.ndarray | pd.Series | list[float]], *, pad_fraction: float = 0.06) -> tuple[float, float] | None:
    finite_parts = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return None

    combined = np.concatenate(finite_parts)
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        delta = abs(lo) * pad_fraction if lo != 0 else pad_fraction
        return lo - delta, hi + delta
    span = hi - lo
    pad = span * pad_fraction
    return lo - pad, hi + pad


def _config_bool(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_str_list(config: dict[str, Any], key: str, default: list[str] | None = None) -> list[str]:
    value = config.get(key, default if default is not None else [])
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _range_from_config(min_value: Any, max_value: Any) -> tuple[float | None, float | None] | None:
    lower = None if min_value is None else float(min_value)
    upper = None if max_value is None else float(max_value)
    if lower is None and upper is None:
        return None
    return lower, upper


def _format_math_uncertainty_label(symbol: str, value: float, uncertainty: float | None, unit: str) -> str:
    if uncertainty is None or not np.isfinite(uncertainty):
        return rf"${symbol}={value:.3g}\,{unit}$"
    return rf"${symbol}={value:.3g}\pm{uncertainty:.2g}\,{unit}$"


def _format_group_label(group_cols: list[str], group_key: object) -> str:
    if not group_cols:
        return str(group_key)

    if not isinstance(group_key, tuple):
        group_key = (group_key,)

    parts = [f"{column}={value}" for column, value in zip(group_cols, group_key)]
    return ", ".join(parts) if parts else str(group_key)


def _grouped_line_plot(
    ax: Axes,
    df: pd.DataFrame,
    x_col: str,
    *,
    y_col: str = "corr",
    title: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    label_col: str | None = None,
    band_col: str | None = None,
    color_col: str | None = "dataset_color",
    fit_value_col: str | None = None,
    fit_error_col: str | None = None,
    fit_amplitude_col: str | None = None,
    fit_offset_col: str | None = None,
    fit_symbol: str | None = None,
    fit_unit: str | None = None,
    fit_label_formatter: Callable[[str, float, float | None, str, str], str] | None = None,
) -> None:
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    group_cols = [column for column in ("component", "channel", "projection", "frame") if column in df.columns]
    if label_col is not None and label_col in df.columns and label_col not in group_cols:
        group_cols.append(label_col)
    grouped = list(df.groupby(group_cols, sort=True)) if group_cols else [("all", df)]

    cmap = plt.get_cmap("viridis")

    def _group_sort_key(item: tuple[object, pd.DataFrame]) -> tuple[float, str, str]:
        group_key, group_df = item
        order_value = float("inf")
        if "dataset_order" in group_df.columns:
            dataset_orders = group_df["dataset_order"].to_numpy(dtype=float)
            finite_orders = dataset_orders[np.isfinite(dataset_orders)]
            if finite_orders.size:
                order_value = float(np.nanmin(finite_orders))
        return order_value, _format_group_label(group_cols, group_key).lower(), str(group_key)

    grouped = sorted(grouped, key=_group_sort_key)
    colors = [cmap(value) for value in np.linspace(0.15, 0.95, max(1, len(grouped)))]
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []

    for (group_key, group_df), color in zip(grouped, colors):
        sub = group_df.sort_values(x_col)
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        if x_range is not None:
            lower, upper = x_range
            mask = np.isfinite(x) & np.isfinite(y)
            if lower is not None:
                mask &= x >= float(lower)
            if upper is not None:
                mask &= x <= float(upper)
            x = x[mask]
            y = y[mask]
            sub = sub.loc[mask]
        else:
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            sub = sub.loc[mask]

        if x.size == 0:
            continue

        x_series.append(x)
        y_series.append(y)

        if label_col is not None and label_col in sub.columns:
            label_values = sub[label_col].dropna().astype(str)
            group_label = str(label_values.iloc[0]) if not label_values.empty else _format_group_label(group_cols, group_key)
        else:
            group_label = _format_group_label(group_cols, group_key)

        line_color = color
        if color_col is not None and color_col in sub.columns:
            color_values = sub[color_col].dropna().astype(str)
            if not color_values.empty:
                color_value = color_values.iloc[0].strip()
                if color_value:
                    line_color = color_value

        ax.plot(x, y, lw=1.8, color=line_color, label=group_label)

        if band_col is not None and band_col in sub.columns:
            band = sub[band_col].to_numpy(dtype=float)
            band_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(band)
            if np.any(band_mask):
                ax.fill_between(x[band_mask], y[band_mask] - band[band_mask], y[band_mask] + band[band_mask], alpha=0.12, color=line_color)

        fit_column = fit_value_col or ("xi_um" if x_col == "r_um" else "tau_str_s" if x_col == "lag_s" else None)
        if fit_column and fit_column in sub.columns:
            fit_values = sub[fit_column].to_numpy(dtype=float)
            fit_value = float(fit_values[np.isfinite(fit_values)][0]) if np.isfinite(fit_values).any() else np.nan
            if np.isfinite(fit_value) and fit_value > 0.0 and np.isfinite(x).any() and np.isfinite(y).any():
                fit_err_column = fit_error_col or ("xi_err_um" if fit_column == "xi_um" else "tau_err_s" if fit_column == "tau_str_s" else None)
                fit_error = None
                if fit_err_column and fit_err_column in sub.columns:
                    fit_errors = sub[fit_err_column].to_numpy(dtype=float)
                    fit_error = float(fit_errors[np.isfinite(fit_errors)][0]) if np.isfinite(fit_errors).any() else None
                x_min = float(np.nanmin(x[np.isfinite(x)]))
                x_max = float(np.nanmax(x[np.isfinite(x)]))
                if x_range is not None:
                    lower, upper = x_range
                    if lower is not None:
                        x_min = max(x_min, float(lower))
                    if upper is not None:
                        x_max = min(x_max, float(upper))
                x_fit = np.linspace(x_min, x_max, 250)
                finite_y = y[np.isfinite(y)]
                if finite_y.size:
                    amplitude_column = fit_amplitude_col or ("amp" if "amp" in sub.columns else None)
                    offset_column = fit_offset_col or ("offset" if "offset" in sub.columns else None)

                    amplitude = np.nan
                    offset = np.nan
                    if amplitude_column and amplitude_column in sub.columns:
                        amp_values = sub[amplitude_column].to_numpy(dtype=float)
                        if np.isfinite(amp_values).any():
                            amplitude = float(amp_values[np.isfinite(amp_values)][0])
                    if offset_column and offset_column in sub.columns:
                        offset_values = sub[offset_column].to_numpy(dtype=float)
                        if np.isfinite(offset_values).any():
                            offset = float(offset_values[np.isfinite(offset_values)][0])

                    if not np.isfinite(amplitude):
                        amplitude = max(0.05, float(np.nanmax(finite_y) - np.nanmin(finite_y)))
                    if not np.isfinite(offset):
                        offset = float(np.nanmin(finite_y))

                    y_fit = _exp_decay(x_fit, amplitude, fit_value, offset)
                    if fit_column == "xi_um":
                        default_symbol, default_unit = r"\xi", r"\mu\mathrm{m}"
                    elif fit_column == "tau_str_s":
                        default_symbol, default_unit = r"\tau", r"\mathrm{s}"
                    else:
                        default_symbol, default_unit = "", ""
                    symbol = fit_symbol or default_symbol
                    unit = fit_unit or default_unit
                    if fit_label_formatter is not None:
                        fit_label = fit_label_formatter(group_label, fit_value, fit_error, symbol, unit)
                    else:
                        if fit_column == "xi_um":
                            if fit_error is not None and np.isfinite(fit_error):
                                fit_label = rf"{group_label} fit ($\xi={fit_value:.2g}\pm{fit_error:.2g}\,{unit}$)"
                            else:
                                fit_label = rf"{group_label} fit ($\xi={fit_value:.2g}\,{unit}$)"
                        else:
                            if fit_error is not None and np.isfinite(fit_error):
                                fit_label = rf"{group_label} fit ($\tau={fit_value:.2g}\pm{fit_error:.2g}\,{unit}$)"
                            else:
                                fit_label = rf"{group_label} fit ($\tau={fit_value:.2g}\,{unit}$)"
                    ax.plot(x_fit, y_fit, ls="--", lw=1.4, color=line_color, alpha=0.85, label=fit_label)

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"distance ($\mu\mathrm{m}$)" if x_col == "r_um" else x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"distance ($\mu\mathrm{m}$)" if x_col == "r_um" else x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _tensor_fit_summary_rows(
    tensor_df: pd.DataFrame,
    *,
    part: str,
    component_pairs: Sequence[str],
    fit_range: tuple[float | None, float | None] | None,
    min_points: int,
) -> pd.DataFrame:
    if tensor_df.empty or "part" not in tensor_df.columns or "component_pair" not in tensor_df.columns:
        return pd.DataFrame()

    part_df = tensor_df.loc[tensor_df["part"].astype(str) == str(part)].copy()
    if part_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for component_pair in component_pairs:
        pair_df = part_df.loc[part_df["component_pair"].astype(str) == str(component_pair)].copy()
        if pair_df.empty:
            continue
        grouped = pair_df.dropna(subset=["distance_um", "corr"]).groupby("distance_um", as_index=False).mean(numeric_only=True)
        if grouped.empty:
            continue
        yerr = grouped["corr_sem"].to_numpy(dtype=float) if "corr_sem" in grouped.columns else None
        popt, xi, xi_err = _fit_signed_decay(
            grouped["distance_um"].to_numpy(dtype=float),
            grouped["corr"].to_numpy(dtype=float),
            yerr=yerr,
            fit_range=fit_range,
            min_points=min_points,
        )
        if popt is None or xi is None:
            continue
        rows.append(
            {
                "component_pair": str(component_pair),
                "xi_um": float(xi),
                "xi_err_um": float(xi_err) if xi_err is not None and np.isfinite(xi_err) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _tensor_plot_sample_suffix(tensor_df: pd.DataFrame) -> str | None:
    if tensor_df.empty or "frame" not in tensor_df.columns:
        return None

    frame_values = sorted({int(frame) for frame in tensor_df["frame"].dropna().astype(int).tolist()})
    frame_counts: list[int] = []
    if "frame_count" in tensor_df.columns:
        frame_counts = [int(value) for value in tensor_df["frame_count"].dropna().astype(int).tolist() if int(value) > 0]

    if frame_counts:
        frame_count = max(frame_counts)
        if frame_count > 1:
            return f"mean_n{frame_count}"

    if len(frame_values) == 1 and frame_values[0] >= 0:
        return f"frame_{frame_values[0]}"

    if len(frame_values) == 1 and frame_values[0] < 0:
        return "mean"

    return None


def _tensor_plot_stem(
    dataset_id: str,
    base_name: str,
    vector_cfg: dict[str, Any],
    tensor_df: pd.DataFrame,
    *,
    distance_mode: str | None = None,
) -> str:
    stem = f"{dataset_id}_{base_name}"
    sample_suffix = _tensor_plot_sample_suffix(tensor_df)
    if sample_suffix:
        stem = f"{stem}_{sample_suffix}"
    return build_velocity_artifact_stem(stem, vector_cfg, distance_mode=distance_mode)


def _tensor_plot_sample_dir(tensor_dfs: Sequence[pd.DataFrame], base_dir: Path) -> Path:
    for tensor_df in tensor_dfs:
        sample_suffix = _tensor_plot_sample_suffix(tensor_df)
        if sample_suffix:
            return _ensure_dir(base_dir / sample_suffix)
    return _ensure_dir(base_dir)


def _tensor_component_pair_for_basis(component_pair: str, tensor_basis: str) -> str:
    pair = str(component_pair).strip()
    basis = str(tensor_basis).strip().lower()
    if basis != "spherical":
        return pair

    spherical_map = {
        "xx": "rr",
        "xy": "rtheta",
        "xz": "rphi",
        "yx": "thetar",
        "yy": "thetatheta",
        "yz": "thetaphi",
        "zx": "phir",
        "zy": "phitheta",
        "zz": "phiphi",
    }
    lower = pair.lower()
    return spherical_map.get(lower, pair)


def _tensor_fit_component_pairs_for_basis(component_pairs: Sequence[str], tensor_basis: str) -> list[str]:
    translated: list[str] = []
    seen: set[str] = set()
    for component_pair in component_pairs:
        basis_pair = _tensor_component_pair_for_basis(component_pair, tensor_basis)
        if not basis_pair or basis_pair in seen:
            continue
        translated.append(basis_pair)
        seen.add(basis_pair)
    return translated


def _tensor_comparison_output_keys(tensor_basis: str) -> tuple[str, str]:
    basis = str(tensor_basis).strip().lower()
    if basis == "spherical":
        return "tensor_spherical_vector_corr_df", "tensor_spherical_time_series_df"
    return "tensor_vector_corr_df", "tensor_time_series_df"


def _tensor_comparison_component_specs(tensor_basis: str, tensor_fit_enabled: bool) -> list[tuple[str, str, bool, str]]:
    basis = str(tensor_basis).strip().lower()
    if basis == "spherical":
        return [
            ("symmetric", "rr", tensor_fit_enabled, r"Spherical tensor $rr$ correlation"),
            ("symmetric", "thetatheta", tensor_fit_enabled, r"Spherical tensor $\theta\theta$ correlation"),
            ("symmetric", "phiphi", tensor_fit_enabled, r"Spherical tensor $\phi\phi$ correlation"),
            ("antisymmetric", "rtheta", False, r"Spherical tensor $r\,\theta$ correlation"),
            ("antisymmetric", "rphi", False, r"Spherical tensor $r\,\phi$ correlation"),
            ("antisymmetric", "thetaphi", False, r"Spherical tensor $\theta\,\phi$ correlation"),
        ]
    return [
        ("symmetric", "xx", tensor_fit_enabled, r"Tensor $xx$ correlation"),
        ("antisymmetric", "xy", False, r"Tensor $xy$ correlation"),
    ]


def _plot_tensor_fit_summary(
    ax: Axes,
    fit_rows: pd.DataFrame,
    *,
    title: str | None,
    dataset_labels: Sequence[str],
    component_pairs: Sequence[str],
    part_label: str,
) -> None:
    if fit_rows.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, f"no {part_label} fits", ha="center", va="center", transform=ax.transAxes)
        return

    cmap = plt.get_cmap("tab10")
    colors = [cmap(index % 10) for index in range(max(1, len(component_pairs)))]
    x_positions = np.arange(len(dataset_labels), dtype=float)
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []

    for index, component_pair in enumerate(component_pairs):
        pair_df = fit_rows.loc[fit_rows["component_pair"].astype(str) == str(component_pair)].copy()
        if pair_df.empty:
            continue
        pair_df = pair_df.sort_values("dataset_order")
        x = pair_df["dataset_order"].to_numpy(dtype=float)
        y = pair_df["xi_um"].to_numpy(dtype=float)
        yerr = pair_df["xi_err_um"].to_numpy(dtype=float) if "xi_err_um" in pair_df.columns else None
        x_series.append(x)
        y_series.append(y)
        color = colors[index % len(colors)]
        if yerr is not None and np.any(np.isfinite(yerr)):
            ax.errorbar(x, y, yerr=yerr, lw=1.6, marker="o", ms=5, capsize=3, color=color, label=component_pair)
        else:
            ax.plot(x, y, lw=1.8, marker="o", ms=5, color=color, label=component_pair)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_labels, rotation=20, ha="right")
    ax.set_xlabel("experiment")
    ax.set_ylabel(r"decay length ($\mu\mathrm{m}$)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, title=part_label)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(-0.3, max(float(x_positions.max()), x_limits[1]) + 0.3 if x_positions.size else x_limits[1])
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _plot_image_correlation(ax: Axes, raw_df: pd.DataFrame, fit_df: pd.DataFrame, *, title: str | None = None) -> None:
    if raw_df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    group_cols = [column for column in ("channel", "projection") if column in raw_df.columns]
    grouped = list(raw_df.groupby(group_cols, sort=True)) if group_cols else [("all", raw_df)]

    def _group_sort_key(item: tuple[object, pd.DataFrame]) -> tuple[float, str, str]:
        group_key, group_df = item
        order_value = float("inf")
        if "dataset_order" in group_df.columns:
            dataset_orders = group_df["dataset_order"].to_numpy(dtype=float)
            finite_orders = dataset_orders[np.isfinite(dataset_orders)]
            if finite_orders.size:
                order_value = float(np.nanmin(finite_orders))
        return order_value, str(group_key).lower(), str(group_key)

    grouped = sorted(grouped, key=_group_sort_key)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(value) for value in np.linspace(0.15, 0.95, max(1, len(grouped)))]
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []
    for (group_key, group_df), color in zip(grouped, colors):
        sub = group_df.sort_values("dt_s")
        x = sub["dt_s"].to_numpy(dtype=float)
        y = sub["corr"].to_numpy(dtype=float)
        x_series.append(x)
        y_series.append(y)
        line_color = color
        if "dataset_color" in sub.columns:
            color_values = sub["dataset_color"].dropna().astype(str)
            if not color_values.empty and color_values.iloc[0].strip():
                line_color = color_values.iloc[0].strip()
        ax.plot(x, y, lw=1.8, color=line_color, label=str(group_key))

        if "corr_sem" in sub.columns:
            yerr = sub["corr_sem"].to_numpy(dtype=float)
            if np.any(np.isfinite(yerr)):
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=line_color)

        if not fit_df.empty:
            if group_cols and all(column in fit_df.columns for column in group_cols):
                match = fit_df
                key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
                for column, value in zip(group_cols, key_tuple):
                    match = match.loc[match[column] == value]
            else:
                match = fit_df
            if not match.empty and {"tau_str_s", "beta"}.issubset(match.columns):
                row = match.iloc[0]
                tau = float(row["tau_str_s"])
                beta = float(row["beta"])
                finite_x = x[np.isfinite(x)]
                if finite_x.size:
                    dt = np.linspace(float(np.nanmin(finite_x)), float(np.nanmax(finite_x)), 250)
                    fit = np.exp(-np.power(np.clip(dt, 0.0, None) / max(tau, 1e-12), max(beta, 1e-12)))
                    ax.plot(
                        dt,
                        fit,
                        ls="--",
                        lw=1.4,
                        color=line_color,
                        alpha=0.8,
                        label=rf"{group_key} fit ($\tau={tau:.2g}\,\mathrm{{s}},\,\beta={beta:.2g}$)",
                    )

    if title:
        ax.set_title(title)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("correlation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _normalize_batch_inputs(
    dataset_ids: Sequence[str] | None,
    base_dirs: Sequence[str] | None,
    *,
    default_base_dir: str,
) -> list[tuple[str, str]]:
    ids = [str(dataset_id).strip() for dataset_id in (dataset_ids or []) if str(dataset_id).strip()]
    if not ids:
        return []

    dirs = [str(base_dir).strip() for base_dir in (base_dirs or []) if str(base_dir).strip()]
    if not dirs:
        dirs = [str(default_base_dir)] * len(ids)
    elif len(dirs) == 1 and len(ids) > 1:
        dirs = dirs * len(ids)
    elif len(dirs) != len(ids):
        raise ValueError(
            f"base_dirs length ({len(dirs)}) must be 1 or match dataset_ids length ({len(ids)})"
        )

    return list(zip(ids, dirs))


def _comparison_specs_for_batch(
    config: dict[str, Any],
    dataset_pairs: Sequence[tuple[str, str]],
    *,
    variation: str | None = None,
) -> list[ComparisonSpec]:
    comparison_cfg = config.get("comparison", {}) if isinstance(config.get("comparison", {}), dict) else {}
    comparison_cfg = resolve_comparison_preset(comparison_cfg)
    palette = str(comparison_cfg.get("palette", "atp"))
    registry_specs: list[ComparisonSpec] = []
    try:
        registry_specs = comparison_registry_from_config(comparison_cfg)
    except Exception:
        registry_specs = []

    registry_map = {spec.dataset_id: spec for spec in registry_specs}
    default_colors = comparison_palette(palette, len(dataset_pairs))
    default_group = str(comparison_cfg.get("name", "batch_comparison")).strip()
    resolved_variation = "" if variation is None else str(variation)

    specs: list[ComparisonSpec] = []
    for index, (dataset_id, base_dir) in enumerate(dataset_pairs):
        reg = registry_map.get(dataset_id)
        label = reg.label if reg else dataset_id
        color = reg.color if reg else default_colors[index]
        group = reg.group if reg else default_group
        replicate = reg.replicate if reg else ""
        spec_variation = resolved_variation if resolved_variation else (reg.variation if reg else "")
        specs.append(
            ComparisonSpec(
                dataset_id=dataset_id,
                label=label,
                color=color,
                group=group,
                replicate=replicate,
                variation=spec_variation,
                base_dir=base_dir,
                raw_base_dir=reg.raw_base_dir if reg is not None else None,
            )
        )
    return specs


def _comparison_dataset_pairs_from_config(
    config: dict[str, Any],
    *,
    default_base_dir: str,
) -> list[tuple[str, str]]:
    comparison_cfg = config.get("comparison", {}) if isinstance(config.get("comparison", {}), dict) else {}
    comparison_cfg = resolve_comparison_preset(comparison_cfg)
    registry = comparison_cfg.get("registry")
    if not isinstance(registry, list) or not registry:
        return []

    dataset_ids: list[str] = []
    base_dirs: list[str] = []
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        dataset_id = str(entry.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        base_dir = str(entry.get("base_dir", default_base_dir)).strip() or default_base_dir
        dataset_ids.append(dataset_id)
        base_dirs.append(base_dir)

    return _normalize_batch_inputs(dataset_ids, base_dirs, default_base_dir=default_base_dir)


def _prepare_velocity_time_series_for_comparison(
    vel_df: pd.DataFrame,
    fps: float | None,
    *,
    speed_columns: Sequence[str] = ("speed_um_s", "drift_speed_um_s", "speed_drift_corrected_um_s"),
) -> pd.DataFrame:
    if vel_df is None or vel_df.empty or "frame" not in vel_df.columns:
        return pd.DataFrame()

    speed_cols = [
        col
        for col in speed_columns
        if col in vel_df.columns
    ]
    if not speed_cols:
        return pd.DataFrame()

    speed_col = "speed_drift_corrected_um_s" if "speed_drift_corrected_um_s" in speed_cols else speed_cols[0]
    grouped = vel_df.groupby("frame", sort=True)[speed_col].agg(["mean", "std"]).reset_index()
    grouped = grouped.rename(columns={"mean": speed_col, "std": f"{speed_col}_std"})
    fps_value = float(fps) if fps is not None and float(fps) > 0 else None
    if fps_value is None:
        grouped["time_s"] = grouped["frame"].astype(float)
    else:
        grouped["time_s"] = grouped["frame"].astype(float) / fps_value
    return grouped


def _prepare_drift_velocity_time_series_for_comparison(vel_df: pd.DataFrame, fps: float | None) -> pd.DataFrame:
    return _prepare_velocity_time_series_for_comparison(vel_df, fps, speed_columns=("drift_speed_um_s",))


def _collapse_series_for_comparison(
    df: pd.DataFrame,
    x_col: str,
    *,
    sampled_frame_mode: str = "mean",
    sampled_frame: int | None = None,
) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or "corr" not in df.columns:
        return pd.DataFrame()

    base = df.dropna(subset=[x_col, "corr"]).copy()
    if base.empty:
        return pd.DataFrame()

    mode_norm = str(sampled_frame_mode).strip().lower()
    if mode_norm in {"sample", "frame", "single"} and "frame" in base.columns:
        frame_values = sorted({int(frame) for frame in base["frame"].dropna().astype(int).tolist()})
        if frame_values:
            selected_frame = frame_values[0] if sampled_frame is None else int(sampled_frame)
            if selected_frame not in frame_values:
                selected_frame = frame_values[0]
            base = base.loc[base["frame"].astype(int) == int(selected_frame)].copy()
        if base.empty:
            return pd.DataFrame()

    group_cols = [x_col]
    for col in ("channel", "projection"):
        if col in base.columns:
            group_cols.append(col)

    agg_mode = "mean" if "frame" in base.columns and mode_norm == "mean" else "median"
    agg: dict[str, str] = {"corr": agg_mode}
    for col in ("xi_um", "xi_err_um", "tau_str_s", "tau_err_s", "amp", "offset"):
        if col in base.columns:
            agg[col] = agg_mode
    if "corr_sem" in base.columns:
        agg["corr_sem"] = agg_mode
    if "corr_std" in base.columns:
        agg["corr_std"] = agg_mode

    return base.groupby(group_cols, as_index=False).agg(agg).sort_values(x_col)


def _plot_metric_bar(
    ax: Axes,
    table: pd.DataFrame,
    *,
    value_col: str,
    error_col: str | None,
    ylabel: str,
    title: str | None,
) -> None:
    if table.empty or value_col not in table.columns:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    ordered = table.sort_values("dataset_order").reset_index(drop=True)
    x = np.arange(len(ordered), dtype=float)
    y = ordered[value_col].to_numpy(dtype=float)
    labels = ordered["dataset_label"].astype(str).tolist() if "dataset_label" in ordered.columns else [str(index) for index in range(len(ordered))]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(value) for value in np.linspace(0.18, 0.9, max(1, len(ordered)))]
    facecolor = np.asarray(ax.get_facecolor()[:3], dtype=float)
    is_dark_background = float(facecolor.mean()) < 0.5
    error_color = "white" if is_dark_background else "black"

    bar_colors = colors
    if "dataset_color" in ordered.columns:
        dataset_colors = [str(value).strip() for value in ordered["dataset_color"].tolist()]
        if all(color for color in dataset_colors):
            bar_colors = dataset_colors

    bars = ax.bar(x, y, width=0.68, color=bar_colors, alpha=0.72, zorder=1, edgecolor="none")
    for bar in bars:
        bar.set_zorder(1)

    if error_col and error_col in ordered.columns:
        yerr = ordered[error_col].to_numpy(dtype=float)
        has_err = np.any(np.isfinite(yerr))
        if has_err:
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            error_container = ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=error_color,
                elinewidth=1.8,
                capsize=4,
                capthick=1.5,
                zorder=50,
                barsabove=True,
                clip_on=False,
            )
            for artist_group in getattr(error_container, "lines", []):
                if artist_group is None:
                    continue
                if isinstance(artist_group, (list, tuple)):
                    for artist in artist_group:
                        if artist is not None:
                            artist.set_zorder(50)
                            artist.set_clip_on(False)
                else:
                    artist_group.set_zorder(50)
                    artist_group.set_clip_on(False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("experiment")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if len(ordered) == 1:
        ax.set_xlim(-0.6, 0.6)
    elif len(ordered) > 1:
        ax.set_xlim(-0.6, float(len(ordered) - 1) + 0.6)

    y_limits = _finite_limits([y])
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _sampled_autocorr_length_time_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "frame" not in df.columns or "r_um" not in df.columns or "corr" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for frame_value, frame_df in df.dropna(subset=["frame"]).groupby("frame", sort=True):
        frame_df = frame_df.dropna(subset=["r_um", "corr"]).copy()
        if frame_df.empty:
            continue

        frame_index = int(frame_df["frame"].astype(int).iloc[0])

        if "time_s" in frame_df.columns and frame_df["time_s"].dropna().size:
            time_value = float(frame_df["time_s"].dropna().astype(float).iloc[0])
        else:
            time_value = float(frame_index)

        if "xi_um" in frame_df.columns and np.isfinite(frame_df["xi_um"]).any():
            xi_values = frame_df["xi_um"].to_numpy(dtype=float)
            xi_values = xi_values[np.isfinite(xi_values)]
            xi_value = float(np.nanmedian(xi_values)) if xi_values.size else np.nan
            xi_err_value = np.nan
            if "xi_err_um" in frame_df.columns:
                xi_err_values = frame_df["xi_err_um"].to_numpy(dtype=float)
                xi_err_values = xi_err_values[np.isfinite(xi_err_values)]
                xi_err_value = float(np.nanmedian(xi_err_values)) if xi_err_values.size else np.nan
        else:
            grouped = frame_df.groupby("r_um", as_index=False).mean(numeric_only=True)
            if grouped.empty:
                continue
            yerr = grouped["corr_sem"].to_numpy(dtype=float) if "corr_sem" in grouped.columns else None
            _, xi_value, xi_err_value = _fit_signed_decay(
                grouped["r_um"].to_numpy(dtype=float),
                grouped["corr"].to_numpy(dtype=float),
                yerr=yerr,
                fit_range=None,
                min_points=4,
            )
            if xi_value is None:
                continue

        rows.append(
            {
                "time_s": time_value,
                "frame": frame_index,
                "xi_um": float(xi_value),
                "xi_err_um": float(xi_err_value) if xi_err_value is not None and np.isfinite(xi_err_value) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)


def _tensor_decay_length_time_series(
    df: pd.DataFrame,
    *,
    part: str,
    component_pair: str,
    fit_range: tuple[float | None, float | None] | None = None,
    min_points: int = 4,
) -> pd.DataFrame:
    if df.empty or not {"frame", "distance_um", "corr", "part", "component_pair"}.issubset(df.columns):
        return pd.DataFrame()

    subset = df.loc[(df["part"].astype(str) == str(part)) & (df["component_pair"].astype(str) == str(component_pair))].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for frame_value, frame_df in subset.groupby("frame", sort=True):
        frame_df = frame_df.dropna(subset=["distance_um", "corr"]).copy()
        if frame_df.empty:
            continue

        frame_index = int(frame_df["frame"].astype(int).iloc[0])

        grouped = frame_df.groupby("distance_um", as_index=False).mean(numeric_only=True).sort_values("distance_um")
        if grouped.empty or "distance_um" not in grouped.columns or "corr" not in grouped.columns:
            continue

        fit_popt, fit_xi, fit_xi_err = _fit_signed_decay(
            grouped["distance_um"].to_numpy(dtype=float),
            grouped["corr"].to_numpy(dtype=float),
            yerr=grouped["corr_sem"].to_numpy(dtype=float) if "corr_sem" in grouped.columns else None,
            fit_range=fit_range,
            min_points=min_points,
        )
        if fit_popt is None or fit_xi is None:
            continue

        if "time_s" in frame_df.columns and frame_df["time_s"].dropna().size:
            time_value = float(frame_df["time_s"].dropna().astype(float).iloc[0])
        else:
            time_value = float(frame_index)

        rows.append(
            {
                "frame": frame_index,
                "time_s": time_value,
                "part": str(part),
                "component_pair": str(component_pair),
                "xi_um": float(fit_xi),
                "xi_err_um": float(fit_xi_err) if fit_xi_err is not None and np.isfinite(fit_xi_err) else np.nan,
                "amp": float(fit_popt[0]) if fit_popt is not None and len(fit_popt) > 0 else np.nan,
                "offset": float(fit_popt[2]) if fit_popt is not None and len(fit_popt) > 2 else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)


def _plot_autocorr_length_time_series(
    ax: Axes,
    table: pd.DataFrame,
    *,
    title: str | None = None,
    time_col: str = "time_s",
    time_unit: str = "s",
    use_log_x: bool = False,
) -> None:
    if table.empty or time_col not in table.columns or "xi_um" not in table.columns:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    group_col = "dataset_order" if "dataset_order" in table.columns else None
    groups = table.groupby(group_col, sort=True) if group_col is not None else [(None, table)]
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []

    for group_key, group_df in groups:
        group_df = group_df.sort_values(time_col)
        if group_df.empty:
            continue

        color = str(group_df["dataset_color"].iloc[0]) if "dataset_color" in group_df.columns else None
        label = (
            str(group_df["legend_label"].iloc[0])
            if "legend_label" in group_df.columns
            else str(group_df["dataset_label"].iloc[0])
            if "dataset_label" in group_df.columns
            else str(group_key) if group_key is not None else "sampled autocorr"
        )

        x = group_df[time_col].to_numpy(dtype=float)
        y = group_df["xi_um"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        x = x[mask]
        y = y[mask]
        if use_log_x:
            x = np.clip(x, np.finfo(float).tiny, None)

        ax.plot(x, y, lw=2.0, color=color, label=label)
        if "xi_err_um" in group_df.columns:
            yerr = group_df["xi_err_um"].to_numpy(dtype=float)
            yerr = yerr[mask]
            err_mask = np.isfinite(yerr)
            if np.any(err_mask):
                lower = np.maximum(y[err_mask] - yerr[err_mask], np.finfo(float).tiny)
                upper = y[err_mask] + yerr[err_mask]
                ax.fill_between(x[err_mask], lower, upper, alpha=0.14, color=color, linewidth=0)

        x_series.append(x)
        y_series.append(y)

    if title:
        ax.set_title(title)
    ax.set_xlabel(f"time ({time_unit})")
    ax.set_ylabel(r"autocorr length ($\mu\mathrm{m}$)")
    if use_log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _plot_autocorr_weighted_near0_profile(
    ax: Axes,
    df: pd.DataFrame,
    *,
    title: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
) -> None:
    x_col = "r_um" if "r_um" in df.columns else df.columns[0]
    y_col = "corr" if "corr" in df.columns else ("corr_mean" if "corr_mean" in df.columns else (df.columns[1] if len(df.columns) > 1 else x_col))
    _grouped_line_plot(ax, df, x_col, y_col=y_col, title=title, x_range=x_range)


def _comparison_sort_key(spec: ComparisonSpec) -> tuple[float, str, str]:
    label = str(spec.label).strip()
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*nM", label, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), label.lower(), spec.dataset_id
    numeric = re.search(r"-?\d+(?:\.\d+)?", label)
    if numeric:
        return float(numeric.group(0)), label.lower(), spec.dataset_id
    return float("inf"), label.lower(), spec.dataset_id


def _dataset_derived_dir(runner: "AnalysisNotebookRunner") -> Path:
    state = runner.load_state()
    return Path(state["paths"]["derived_dir"])


def _read_output_or_parquet(
    runner: "AnalysisNotebookRunner",
    feature: str,
    key: str,
    fallback_name: str,
) -> pd.DataFrame:
    feature_out = runner.outputs.get(feature, {}) if isinstance(runner.outputs.get(feature, {}), dict) else {}
    value = feature_out.get(key)
    if isinstance(value, pd.DataFrame):
        return value
    return _read_parquet(_dataset_derived_dir(runner) / fallback_name)


def run_batch_comparison(
    records: Sequence[tuple[ComparisonSpec, "AnalysisNotebookRunner"]],
    *,
    comparison_name: str,
    output_root: str | Path,
) -> dict[str, dict[str, Path]]:
    if not records:
        return {}

    reference_cfg = records[0][1].config
    variation = str(reference_cfg.get("dataset", {}).get("variation", ""))
    comparison_cfg = dict(reference_cfg.get("comparison", {})) if isinstance(reference_cfg.get("comparison", {}), dict) else {}
    vector_cfg = dict(reference_cfg.get("vector_corr", {})) if isinstance(reference_cfg.get("vector_corr", {}), dict) else {}
    plot_velocity_over_time_enabled = _config_bool(comparison_cfg, "plot_velocity_over_time_enabled", True)
    plot_drift_velocity_over_time_enabled = _config_bool(comparison_cfg, "drift_velocity_over_time_enabled", False)
    plot_mean_speed_summary_enabled = _config_bool(comparison_cfg, "plot_mean_speed_summary_enabled", True)
    autocorr_sample_mode_raw = str(comparison_cfg.get("autocorr_sample_mode", "mean")).strip().lower()
    autocorr_sample_mode = "frame" if autocorr_sample_mode_raw in {"sample", "frame", "single"} else "mean"
    autocorr_sample_frame_raw = comparison_cfg.get("autocorr_sample_frame")
    autocorr_sample_frame = None if autocorr_sample_frame_raw is None else int(autocorr_sample_frame_raw)
    out_dir = comparison_output_dir(output_root, comparison_name, variation=variation)
    saved: dict[str, dict[str, Path]] = {}
    sorted_records = sorted(records, key=lambda item: _comparison_sort_key(item[0]))
    autocorr_xi_rows: list[dict[str, Any]] = []

    tensor_sample_suffix = None
    for _, runner in sorted_records:
        tensor_probe = _read_output_or_parquet(
            runner,
            "vector_corr",
            "tensor_vector_corr_df",
            _velocity_output_name(
                "beads_vector_correlation_tensor_avg.parquet" if bool(reference_cfg.get("vector_corr", {}).get("multi_frame_average", False)) else "beads_vector_correlation_tensor.parquet",
                reference_cfg.get("vector_corr", {}) if isinstance(reference_cfg.get("vector_corr", {}), dict) else {},
                distance_mode=str((reference_cfg.get("vector_corr", {}) if isinstance(reference_cfg.get("vector_corr", {}), dict) else {}).get("tensor_distance_mode", "xyz")).strip().lower(),
            ),
        )
        if tensor_probe.empty:
            tensor_probe = _read_output_or_parquet(
                runner,
                "vector_corr",
                "tensor_spherical_vector_corr_df",
                _velocity_output_name(
                    "beads_vector_correlation_tensor_avg.parquet" if bool(reference_cfg.get("vector_corr", {}).get("multi_frame_average", False)) else "beads_vector_correlation_tensor.parquet",
                    reference_cfg.get("vector_corr", {}) if isinstance(reference_cfg.get("vector_corr", {}), dict) else {},
                    distance_mode=str((reference_cfg.get("vector_corr", {}) if isinstance(reference_cfg.get("vector_corr", {}), dict) else {}).get("tensor_distance_mode", "xyz")).strip().lower(),
                    tensor_basis="spherical",
                ),
            )
        tensor_sample_suffix = _tensor_plot_sample_suffix(tensor_probe)
        if tensor_sample_suffix:
            break
    if tensor_sample_suffix:
        out_dir = _ensure_dir(out_dir / tensor_sample_suffix)

    # Velocity-over-time comparison (dataset-level curves).
    velocity_parts: list[pd.DataFrame] = []
    speed_summary_rows: list[dict[str, Any]] = []
    drift_velocity_parts: list[pd.DataFrame] = []
    for dataset_order, (spec, runner) in enumerate(sorted_records):
        vel_df = _read_output_or_parquet(runner, "beads", "tracks_vel_df", "beads_tracks_with_velocity.parquet")
        fps = runner.load_state().get("calibration", {}).get("fps")
        series = _prepare_velocity_time_series_for_comparison(vel_df, fps=fps)
        if series.empty:
            continue
        speed_col = "speed_drift_corrected_um_s" if "speed_drift_corrected_um_s" in series.columns else next(
            (column for column in series.columns if column.endswith("_um_s") and not column.endswith("_std")),
            None,
        )
        if speed_col is None:
            continue
        speed_values = series[speed_col].to_numpy(dtype=float)
        mean_value = float(np.nanmean(speed_values))
        std_value = float(np.nanstd(speed_values, ddof=0))
        series = series.rename(columns={speed_col: "corr", f"{speed_col}_std": "corr_std"})
        velocity_label = _format_math_uncertainty_label(r"v_{\mathrm{mean}}", mean_value, std_value, r"\mu\mathrm{m}/\mathrm{s}")
        series["dataset_order"] = int(dataset_order)
        series["dataset_color"] = spec.color
        series["legend_label"] = f"{spec.label} {velocity_label}"
        velocity_parts.append(series[["time_s", "corr", "corr_std", "dataset_order", "dataset_color", "legend_label"]])
        speed_summary_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "dataset_label": spec.label,
                "dataset_order": int(dataset_order),
                "dataset_color": spec.color,
                "mean_speed_um_s": mean_value,
                "speed_std_um_s": std_value,
            }
        )

        drift_series = _prepare_drift_velocity_time_series_for_comparison(vel_df, fps=fps)
        if drift_series.empty or "drift_speed_um_s" not in drift_series.columns:
            continue
        drift_values = drift_series["drift_speed_um_s"].to_numpy(dtype=float)
        drift_mean_value = float(np.nanmean(drift_values))
        drift_std_value = float(np.nanstd(drift_values, ddof=0))
        drift_series = drift_series.rename(columns={"drift_speed_um_s": "corr", "drift_speed_um_s_std": "corr_std"})
        drift_velocity_label = _format_math_uncertainty_label(r"v_{\mathrm{drift}}", drift_mean_value, drift_std_value, r"\mu\mathrm{m}/\mathrm{s}")
        drift_series["dataset_order"] = int(dataset_order)
        drift_series["dataset_color"] = spec.color
        drift_series["legend_label"] = f"{spec.label} {drift_velocity_label}"
        drift_velocity_parts.append(drift_series[["time_s", "corr", "corr_std", "dataset_order", "dataset_color", "legend_label"]])

    if plot_velocity_over_time_enabled and velocity_parts:
        vel_cmp = pd.concat(velocity_parts, ignore_index=True)
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
            _grouped_line_plot(
                ax,
                vel_cmp,
                "time_s",
                title="Drift-corrected bead speed comparison",
                y_col="corr",
                label_col="legend_label",
                band_col="corr_std",
            )
            ax.set_ylabel(r"speed ($\mu\mathrm{m}/\mathrm{s}$)")
            ax.set_xlabel("time (s)")

            def _white_velocity(ax_white: Axes, table=vel_cmp) -> None:
                _grouped_line_plot(ax_white, table, "time_s", title=None, y_col="corr", label_col="legend_label", band_col="corr_std")
                ax_white.set_ylabel(r"speed ($\mu\mathrm{m}/\mathrm{s}$)")
                ax_white.set_xlabel("time (s)")

            stem = f"{comparison_name}_beads_velocity_over_time"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_velocity)
            plt.close(fig)

    if plot_drift_velocity_over_time_enabled and drift_velocity_parts:
        drift_vel_cmp = pd.concat(drift_velocity_parts, ignore_index=True)
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
            _grouped_line_plot(
                ax,
                drift_vel_cmp,
                "time_s",
                title="Drift speed comparison",
                y_col="corr",
                label_col="legend_label",
                band_col="corr_std",
            )
            ax.set_ylabel(r"drift speed ($\mu\mathrm{m}/\mathrm{s}$)")
            ax.set_xlabel("time (s)")

            def _white_drift_velocity(ax_white: Axes, table=drift_vel_cmp) -> None:
                _grouped_line_plot(ax_white, table, "time_s", title=None, y_col="corr", label_col="legend_label", band_col="corr_std")
                ax_white.set_ylabel(r"drift speed ($\mu\mathrm{m}/\mathrm{s}$)")
                ax_white.set_xlabel("time (s)")

            stem = f"{comparison_name}_beads_drift_velocity_over_time"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_drift_velocity)
            plt.close(fig)

    if plot_mean_speed_summary_enabled and speed_summary_rows:
        speed_summary_df = pd.DataFrame(speed_summary_rows).drop_duplicates(subset=["dataset_id"], keep="last")
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
            _plot_metric_bar(
                ax,
                speed_summary_df,
                value_col="mean_speed_um_s",
                error_col="speed_std_um_s",
                ylabel=r"mean speed ($\mu\mathrm{m}/\mathrm{s}$)",
                title="Mean bead speed summary",
            )

            def _white_speed_summary(ax_white: Axes, table=speed_summary_df) -> None:
                _plot_metric_bar(
                    ax_white,
                    table,
                    value_col="mean_speed_um_s",
                    error_col="speed_std_um_s",
                    ylabel=r"mean speed ($\mu\mathrm{m}/\mathrm{s}$)",
                    title=None,
                )

            stem = f"{comparison_name}_mean_speed_summary"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_speed_summary)
            plt.close(fig)

    if bool(vector_cfg.get("enabled", True)):
        comparison_vector_cfg = dict(vector_cfg)
        comparison_vector_overrides = comparison_cfg.get("vector_corr", {})
        if isinstance(comparison_vector_overrides, dict):
            comparison_vector_cfg.update(comparison_vector_overrides)
        plot_vector_temporal_enabled = _config_bool(comparison_vector_cfg, "plot_temporal_enabled", _config_bool(comparison_vector_cfg, "enabled", True))
        plot_vector_spatial_enabled = _config_bool(comparison_vector_cfg, "plot_spatial_enabled", _config_bool(comparison_vector_cfg, "enabled", True))
        plot_tensor_enabled = _config_bool(comparison_vector_cfg, "plot_tensor_enabled", _config_bool(comparison_vector_cfg, "tensor_enabled", True))
        plot_tensor_parts = set(_config_str_list(comparison_vector_cfg, "plot_tensor_parts", ["full", "symmetric", "antisymmetric"]))
        plot_tensor_pair_fits_enabled = _config_bool(comparison_vector_cfg, "plot_tensor_pair_fits_enabled", _config_bool(comparison_vector_cfg, "tensor_fit_enabled", False))
        plot_tensor_time_series_enabled = _config_bool(comparison_vector_cfg, "plot_tensor_time_series_enabled", _config_bool(comparison_vector_cfg, "tensor_time_series_enabled", False))
        temporal_range = _range_from_config(comparison_vector_cfg.get("temporal_plot_lag_s_min"), comparison_vector_cfg.get("temporal_plot_lag_s_max"))
        temporal_parts: list[pd.DataFrame] = []
        for dataset_order, (spec, runner) in enumerate(sorted_records):
            temporal_df = _read_output_or_parquet(
                runner,
                "vector_corr",
                "temporal_vector_corr_df",
                _velocity_output_name("beads_vector_correlation_temporal.parquet", vector_cfg),
            )
            if temporal_df.empty or not {"lag_s", "corr"}.issubset(temporal_df.columns):
                continue
            grouped = temporal_df.copy()
            grouped["dataset_order"] = int(dataset_order)
            grouped["dataset_color"] = spec.color
            grouped["legend_label"] = spec.label
            temporal_parts.append(grouped)

        if plot_vector_temporal_enabled and temporal_parts:
            temporal_cmp = pd.concat(temporal_parts, ignore_index=True)
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
                x_series: list[np.ndarray] = []
                y_series: list[np.ndarray] = []
                for dataset_order, dataset_df in temporal_cmp.groupby("dataset_order", sort=True):
                    dataset_df = dataset_df.sort_values("lag_s")
                    if dataset_df.empty:
                        continue
                    color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                    label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                    x = dataset_df["lag_s"].to_numpy(dtype=float)
                    y = dataset_df["corr"].to_numpy(dtype=float)
                    yerr = dataset_df["corr_std"].to_numpy(dtype=float) if "corr_std" in dataset_df.columns else None
                    mask = np.isfinite(x) & np.isfinite(y)
                    if temporal_range is not None:
                        lower, upper = temporal_range
                        if lower is not None:
                            mask &= x >= float(lower)
                        if upper is not None:
                            mask &= x <= float(upper)
                    x = x[mask]
                    y = y[mask]
                    if yerr is not None:
                        yerr = yerr[mask]
                    if x.size == 0:
                        continue
                    x_series.append(x)
                    y_series.append(y)

                    tau_values = dataset_df["tau_str_s"].to_numpy(dtype=float) if "tau_str_s" in dataset_df.columns else np.array([], dtype=float)
                    tau_values = tau_values[np.isfinite(tau_values)]
                    tau = float(np.nanmedian(tau_values)) if tau_values.size else np.nan
                    tau_err_values = dataset_df["tau_err_s"].to_numpy(dtype=float) if "tau_err_s" in dataset_df.columns else np.array([], dtype=float)
                    tau_err_values = tau_err_values[np.isfinite(tau_err_values)] if tau_err_values.size else np.array([], dtype=float)
                    tau_err = float(np.nanmedian(tau_err_values)) if tau_err_values.size else np.nan
                    amp_values = dataset_df["amp"].to_numpy(dtype=float) if "amp" in dataset_df.columns else np.array([], dtype=float)
                    amp_values = amp_values[np.isfinite(amp_values)] if amp_values.size else np.array([], dtype=float)
                    offset_values = dataset_df["offset"].to_numpy(dtype=float) if "offset" in dataset_df.columns else np.array([], dtype=float)
                    offset_values = offset_values[np.isfinite(offset_values)] if offset_values.size else np.array([], dtype=float)
                    if np.isfinite(tau) and tau > 0 and amp_values.size and offset_values.size:
                        tau_label = _format_math_uncertainty_label(r"\tau", tau, tau_err, r"\mathrm{s}")
                        label = f"{label} {tau_label}"
                        x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 250)
                        ax.plot(x_fit, _exp_decay(x_fit, float(np.nanmedian(amp_values)), tau, float(np.nanmedian(offset_values))), ls="--", lw=1.4, color=color, alpha=0.85, label="_nolegend_")

                    ax.plot(x, y, lw=2.0, color=color, label=label)
                    if yerr is not None and np.any(np.isfinite(yerr)):
                        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                        ax.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=color, linewidth=0)

                ax.set_title("Temporal vector correlation comparison")
                ax.set_xlabel("lag (s)")
                ax.set_ylabel("correlation")
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=True)

                x_limits = _finite_limits(x_series)
                y_limits = _finite_limits(y_series)
                if x_limits is not None:
                    ax.set_xlim(*x_limits)
                if y_limits is not None:
                    ax.set_ylim(*y_limits)

                def _white_temporal(ax_white: Axes, table=temporal_cmp) -> None:
                    x_series_white: list[np.ndarray] = []
                    y_series_white: list[np.ndarray] = []
                    for dataset_order, dataset_df in table.groupby("dataset_order", sort=True):
                        dataset_df = dataset_df.sort_values("lag_s")
                        if dataset_df.empty:
                            continue
                        color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                        label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                        x = dataset_df["lag_s"].to_numpy(dtype=float)
                        y = dataset_df["corr"].to_numpy(dtype=float)
                        yerr = dataset_df["corr_std"].to_numpy(dtype=float) if "corr_std" in dataset_df.columns else None
                        mask = np.isfinite(x) & np.isfinite(y)
                        if temporal_range is not None:
                            lower, upper = temporal_range
                            if lower is not None:
                                mask &= x >= float(lower)
                            if upper is not None:
                                mask &= x <= float(upper)
                        x = x[mask]
                        y = y[mask]
                        if yerr is not None:
                            yerr = yerr[mask]
                        if x.size == 0:
                            continue
                        x_series_white.append(x)
                        y_series_white.append(y)
                        ax_white.plot(x, y, lw=2.0, color=color, label=label)
                        if yerr is not None and np.any(np.isfinite(yerr)):
                            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                            ax_white.fill_between(x, np.clip(y - yerr, 1e-30, None), np.clip(y + yerr, 1e-30, None), alpha=0.14, color=color, linewidth=0)

                    ax_white.set_xlabel("lag (s)")
                    ax_white.set_ylabel("correlation")
                    ax_white.grid(True, alpha=0.25)
                    ax_white.legend(frameon=True)

                    x_limits = _finite_limits(x_series_white)
                    y_limits = _finite_limits(y_series_white)
                    if x_limits is not None:
                        ax_white.set_xlim(*x_limits)
                    if y_limits is not None:
                        ax_white.set_ylim(*y_limits)

                stem = f"{comparison_name}_vector_temporal_correlation"
                saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_temporal)
                plt.close(fig)

    if autocorr_xi_rows:
        xi_df = pd.DataFrame(autocorr_xi_rows).drop_duplicates(subset=["dataset_id"], keep="last")
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
            _plot_metric_bar(
                ax,
                xi_df,
                value_col="xi_um",
                error_col="xi_err_um",
                ylabel=r"$\xi$ ($\mu\mathrm{m}$)",
                title="Autocorrelation xi summary",
            )

            def _white_xi_summary(ax_white: Axes, table=xi_df) -> None:
                _plot_metric_bar(
                    ax_white,
                    table,
                    value_col="xi_um",
                    error_col="xi_err_um",
                    ylabel=r"$\xi$ ($\mu\mathrm{m}$)",
                    title=None,
                )

            stem = f"{comparison_name}_autocorr_xi_summary"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_xi_summary)
            plt.close(fig)

    # Image correlation comparison.
    raw_parts: list[pd.DataFrame] = []
    fit_parts: list[pd.DataFrame] = []
    for dataset_order, (spec, runner) in enumerate(sorted_records):
        raw_df = _read_output_or_parquet(runner, "image_corr", "image_corr_df", "image_time_correlation_raw.parquet")
        fit_df = _read_output_or_parquet(runner, "image_corr", "image_corr_fit_df", "image_time_correlation_fit.parquet")
        if not raw_df.empty and {"dt_s", "corr"}.issubset(raw_df.columns):
            raw_summary = raw_df.dropna(subset=["dt_s", "corr"]).groupby("dt_s", as_index=False).agg(corr=("corr", "mean"))
            raw_summary["channel"] = spec.label
            raw_summary["projection"] = "all"
            raw_summary["dataset_order"] = int(dataset_order)
            raw_summary["dataset_color"] = spec.color
            raw_parts.append(raw_summary)
        if not fit_df.empty and {"tau_str_s", "beta"}.issubset(fit_df.columns):
            row = fit_df.iloc[0]
            fit_parts.append(
                pd.DataFrame(
                    {
                        "channel": [spec.label],
                        "projection": ["all"],
                        "dataset_order": [int(dataset_order)],
                        "dataset_color": [spec.color],
                        "tau_str_s": [float(row["tau_str_s"])],
                        "beta": [float(row["beta"])],
                    }
                )
            )

    plot_image_corr_enabled = _config_bool(dict(reference_cfg.get("image_corr", {})), "enabled", False)
    if plot_image_corr_enabled and raw_parts:
        raw_cmp = pd.concat(raw_parts, ignore_index=True)
        fit_cmp = pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame()
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
            _plot_image_correlation(ax, raw_cmp, fit_cmp, title="Image-time correlation comparison")

            def _white_image(ax_white: Axes, raw=raw_cmp, fit=fit_cmp) -> None:
                _plot_image_correlation(ax_white, raw, fit, title=None)

            stem = f"{comparison_name}_image_correlation"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_image)
            plt.close(fig)

    vector_cfg = dict(reference_cfg.get("vector_corr", {}))
    spectrum_cfg = dict(reference_cfg.get("velocity_spectrum", {}))
    comparison_spectrum_overrides = comparison_cfg.get("velocity_spectrum", {})
    comparison_spectrum_cfg = dict(spectrum_cfg)
    if isinstance(comparison_spectrum_overrides, dict):
        comparison_spectrum_cfg.update(comparison_spectrum_overrides)
    plot_velocity_spectrum_enabled = _config_bool(comparison_spectrum_cfg, "plot_enabled", _config_bool(comparison_spectrum_cfg, "enabled", True))
    spectrum_plot_range = _range_from_config(comparison_spectrum_cfg.get("plot_k_min"), comparison_spectrum_cfg.get("plot_k_max"))

    spectrum_parts: list[pd.DataFrame] = []
    for dataset_order, (spec, runner) in enumerate(sorted_records):
        spectrum_df = _read_output_or_parquet(
            runner,
            "velocity_spectrum",
            "velocity_spectrum_df",
            velocity_spectrum_output_name("beads_velocity_spectrum.parquet", vector_cfg),
        )
        if spectrum_df.empty or not {"k_rad_per_um", "energy_mean"}.intersection(spectrum_df.columns):
            continue
        grouped = spectrum_df.copy()
        grouped["dataset_order"] = int(dataset_order)
        grouped["dataset_color"] = spec.color
        grouped["legend_label"] = spec.label
        spectrum_parts.append(grouped)

    if plot_velocity_spectrum_enabled and spectrum_parts:
        spectrum_cmp = pd.concat(spectrum_parts, ignore_index=True)
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=150)
            x_series: list[np.ndarray] = []
            y_series: list[np.ndarray] = []
            for dataset_order, dataset_df in spectrum_cmp.groupby("dataset_order", sort=True):
                dataset_df = dataset_df.sort_values("k_rad_per_um")
                if dataset_df.empty:
                    continue
                color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                x = dataset_df["k_rad_per_um"].to_numpy(dtype=float)
                if "energy_smooth" in dataset_df.columns:
                    energy_col = "energy_smooth"
                elif "energy_mean" in dataset_df.columns:
                    energy_col = "energy_mean"
                else:
                    energy_col = "energy"
                y = dataset_df[energy_col].to_numpy(dtype=float)
                yerr = dataset_df["energy_std"].to_numpy(dtype=float) if "energy_std" in dataset_df.columns else (dataset_df["energy_sem"].to_numpy(dtype=float) if "energy_sem" in dataset_df.columns else None)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if spectrum_plot_range is not None:
                    lower, upper = spectrum_plot_range
                    if lower is not None:
                        mask &= x >= float(lower)
                    if upper is not None:
                        mask &= x <= float(upper)
                x = x[mask]
                y = y[mask]
                if yerr is not None:
                    yerr = yerr[mask]
                if x.size == 0:
                    continue
                x_series.append(x)
                y_series.append(y)
                if yerr is not None and np.any(np.isfinite(yerr)):
                    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                    ax.fill_between(x, np.clip(y - yerr, 1e-30, None), np.clip(y + yerr, 1e-30, None), alpha=0.14, color=color, linewidth=0)

                fit_curve = _velocity_spectrum_fit_power_law_curve(x, y, fit_range=spectrum_plot_range)
                display_label = label
                if fit_curve is not None:
                    x_fit, y_fit, alpha = fit_curve
                    display_label = f"{label} ($\\alpha={alpha:.2f}$)"
                    ax.plot(x_fit, y_fit, ls="--", lw=1.5, color=color, alpha=0.95, label="_nolegend_")

                ax.plot(x, y, lw=2.0, color=color, label=display_label)

            ax.set_title("Time-averaged 3D velocity spectrum comparison")
            ax.set_xlabel(r"wavenumber $k$ ($\mathrm{rad}\,\mu\mathrm{m}^{-1}$)")
            ax.set_ylabel(r"$E(k)$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

            x_limits = _velocity_spectrum_log_axis_limits(x_series)
            y_limits = _velocity_spectrum_log_axis_limits(y_series)
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

            def _white_spectrum(ax_white: Axes, table=spectrum_cmp) -> None:
                x_series_white: list[np.ndarray] = []
                y_series_white: list[np.ndarray] = []
                for dataset_order, dataset_df in table.groupby("dataset_order", sort=True):
                    dataset_df = dataset_df.sort_values("k_rad_per_um")
                    if dataset_df.empty:
                        continue
                    color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                    label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                    x = dataset_df["k_rad_per_um"].to_numpy(dtype=float)
                    if "energy_smooth" in dataset_df.columns:
                        energy_col = "energy_smooth"
                    elif "energy_mean" in dataset_df.columns:
                        energy_col = "energy_mean"
                    else:
                        energy_col = "energy"
                    y = dataset_df[energy_col].to_numpy(dtype=float)
                    yerr = dataset_df["energy_std"].to_numpy(dtype=float) if "energy_std" in dataset_df.columns else (dataset_df["energy_sem"].to_numpy(dtype=float) if "energy_sem" in dataset_df.columns else None)
                    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                    if spectrum_plot_range is not None:
                        lower, upper = spectrum_plot_range
                        if lower is not None:
                            mask &= x >= float(lower)
                        if upper is not None:
                            mask &= x <= float(upper)
                    x = x[mask]
                    y = y[mask]
                    if yerr is not None:
                        yerr = yerr[mask]
                    if x.size == 0:
                        continue
                    x_series_white.append(x)
                    y_series_white.append(y)
                    if yerr is not None and np.any(np.isfinite(yerr)):
                        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                        ax_white.fill_between(x, np.clip(y - yerr, 1e-30, None), np.clip(y + yerr, 1e-30, None), alpha=0.14, color=color, linewidth=0)

                    fit_curve = _velocity_spectrum_fit_power_law_curve(x, y, fit_range=spectrum_plot_range)
                    display_label = label
                    if fit_curve is not None:
                        x_fit, y_fit, alpha = fit_curve
                        display_label = f"{label} ($\\alpha={alpha:.2f}$)"
                        ax_white.plot(x_fit, y_fit, ls="--", lw=1.5, color=color, alpha=0.95, label="_nolegend_")

                    ax_white.plot(x, y, lw=2.0, color=color, label=display_label)

                ax_white.set_xlabel(r"wavenumber $k$ ($\mathrm{rad}\,\mu\mathrm{m}^{-1}$)")
                ax_white.set_ylabel(r"$E(k)$")
                ax_white.set_xscale("log")
                ax_white.set_yscale("log")
                ax_white.grid(True, which="both", alpha=0.25)
                ax_white.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
                x_limits = _velocity_spectrum_log_axis_limits(x_series_white)
                y_limits = _velocity_spectrum_log_axis_limits(y_series_white)
                if x_limits is not None:
                    ax_white.set_xlim(*x_limits)
                if y_limits is not None:
                    ax_white.set_ylim(*y_limits)

            stem = f"{comparison_name}_velocity_spectrum"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_spectrum)
            plt.close(fig)

    vorticity_spectrum_cfg = dict(comparison_cfg.get("vorticity_spectrum", {}))
    plot_vorticity_spectrum_enabled = _config_bool(vorticity_spectrum_cfg, "plot_enabled", _config_bool(vorticity_spectrum_cfg, "enabled", True))
    vorticity_spectrum_plot_range = _range_from_config(vorticity_spectrum_cfg.get("plot_k_min"), vorticity_spectrum_cfg.get("plot_k_max"))

    vorticity_spectrum_parts: list[pd.DataFrame] = []
    for dataset_order, (spec, runner) in enumerate(sorted_records):
        vorticity_spectrum_df = _read_output_or_parquet(
            runner,
            "velocity_spectrum",
            "velocity_vorticity_spectrum_df",
            velocity_vorticity_spectrum_output_name("beads_velocity_vorticity_spectrum.parquet", vector_cfg),
        )
        if vorticity_spectrum_df.empty or "k_rad_per_um" not in vorticity_spectrum_df.columns:
            continue
        grouped = vorticity_spectrum_df.copy()
        grouped["dataset_order"] = int(dataset_order)
        grouped["dataset_color"] = spec.color
        grouped["legend_label"] = spec.label
        vorticity_spectrum_parts.append(grouped)

    if plot_vorticity_spectrum_enabled and vorticity_spectrum_parts:
        vorticity_spectrum_cmp = pd.concat(vorticity_spectrum_parts, ignore_index=True)
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=150)
            x_series: list[np.ndarray] = []
            y_series: list[np.ndarray] = []
            for dataset_order, dataset_df in vorticity_spectrum_cmp.groupby("dataset_order", sort=True):
                dataset_df = dataset_df.sort_values("k_rad_per_um")
                if dataset_df.empty:
                    continue
                color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                x = dataset_df["k_rad_per_um"].to_numpy(dtype=float)
                if "enstrophy" in dataset_df.columns:
                    y_col = "enstrophy"
                elif "energy" in dataset_df.columns:
                    y_col = "energy"
                else:
                    y_col = dataset_df.columns[1]
                y = dataset_df[y_col].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if vorticity_spectrum_plot_range is not None:
                    lower, upper = vorticity_spectrum_plot_range
                    if lower is not None:
                        mask &= x >= float(lower)
                    if upper is not None:
                        mask &= x <= float(upper)
                x = x[mask]
                y = y[mask]
                if x.size == 0:
                    continue
                x_series.append(x)
                y_series.append(y)
                plot_vorticity_spectrum(ax, dataset_df.loc[mask], title=None, label=label, color=color, x_range=vorticity_spectrum_plot_range)

            ax.set_title("2D vorticity spectrum comparison")
            ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

            x_limits = _velocity_spectrum_log_axis_limits(x_series)
            y_limits = _velocity_spectrum_log_axis_limits(y_series)
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

            def _white_vorticity_spectrum(ax_white: Axes, table=vorticity_spectrum_cmp) -> None:
                x_series_white: list[np.ndarray] = []
                y_series_white: list[np.ndarray] = []
                for dataset_order, dataset_df in table.groupby("dataset_order", sort=True):
                    dataset_df = dataset_df.sort_values("k_rad_per_um")
                    if dataset_df.empty:
                        continue
                    color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                    label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                    x = dataset_df["k_rad_per_um"].to_numpy(dtype=float)
                    if "enstrophy" in dataset_df.columns:
                        y_col = "enstrophy"
                    elif "energy" in dataset_df.columns:
                        y_col = "energy"
                    else:
                        y_col = dataset_df.columns[1]
                    y = dataset_df[y_col].to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                    if vorticity_spectrum_plot_range is not None:
                        lower, upper = vorticity_spectrum_plot_range
                        if lower is not None:
                            mask &= x >= float(lower)
                        if upper is not None:
                            mask &= x <= float(upper)
                    x = x[mask]
                    y = y[mask]
                    if x.size == 0:
                        continue
                    x_series_white.append(x)
                    y_series_white.append(y)
                    plot_vorticity_spectrum(ax_white, dataset_df.loc[mask], title=None, label=label, color=color, x_range=vorticity_spectrum_plot_range)

                ax_white.set_title("2D vorticity spectrum comparison")
                ax_white.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
                x_limits = _velocity_spectrum_log_axis_limits(x_series_white)
                y_limits = _velocity_spectrum_log_axis_limits(y_series_white)
                if x_limits is not None:
                    ax_white.set_xlim(*x_limits)
                if y_limits is not None:
                    ax_white.set_ylim(*y_limits)

            stem = f"{comparison_name}_vorticity_spectrum"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_vorticity_spectrum)
            plt.close(fig)

    comparison_vector_cfg = dict(vector_cfg)
    comparison_vector_overrides = comparison_cfg.get("vector_corr", {})
    if isinstance(comparison_vector_overrides, dict):
        comparison_vector_cfg.update(comparison_vector_overrides)
    plot_vector_temporal_enabled = _config_bool(comparison_vector_cfg, "plot_temporal_enabled", _config_bool(comparison_vector_cfg, "enabled", True))
    plot_vector_spatial_enabled = _config_bool(comparison_vector_cfg, "plot_spatial_enabled", _config_bool(comparison_vector_cfg, "enabled", True))
    plot_tensor_enabled = _config_bool(comparison_vector_cfg, "plot_tensor_enabled", _config_bool(comparison_vector_cfg, "tensor_enabled", True))
    plot_tensor_parts = set(_config_str_list(comparison_vector_cfg, "plot_tensor_parts", ["full", "symmetric", "antisymmetric"]))
    plot_tensor_pair_fits_enabled = _config_bool(comparison_vector_cfg, "plot_tensor_pair_fits_enabled", _config_bool(comparison_vector_cfg, "tensor_fit_enabled", False))
    plot_tensor_time_series_enabled = _config_bool(comparison_vector_cfg, "plot_tensor_time_series_enabled", _config_bool(comparison_vector_cfg, "tensor_time_series_enabled", False))
    temporal_range = _range_from_config(comparison_vector_cfg.get("temporal_plot_lag_s_min"), comparison_vector_cfg.get("temporal_plot_lag_s_max"))

    temporal_parts: list[pd.DataFrame] = []
    for dataset_order, (spec, runner) in enumerate(sorted_records):
        temporal_df = _read_output_or_parquet(
            runner,
            "vector_corr",
            "temporal_vector_corr_df",
            _velocity_output_name("beads_vector_correlation_temporal.parquet", vector_cfg),
        )
        if temporal_df.empty or not {"lag_s", "corr"}.issubset(temporal_df.columns):
            continue
        grouped = (
            temporal_df.dropna(subset=["lag_s", "corr"])
            .groupby("lag_s", as_index=False)
            .agg(
                corr=("corr", "mean"),
                corr_std=("corr", "std"),
                n_pairs=("corr", "size"),
            )
        )
        grouped["corr_std"] = grouped["corr_std"].fillna(0.0)
        fit_popt, fit_tau, fit_tau_err = _fit_temporal_decay_with_error(temporal_df, "lag_s", "corr", fit_range=temporal_range)
        grouped["tau_str_s"] = float(fit_tau) if fit_tau is not None and np.isfinite(fit_tau) else np.nan
        grouped["tau_err_s"] = float(fit_tau_err) if fit_tau_err is not None and np.isfinite(fit_tau_err) else np.nan
        grouped["amp"] = float(fit_popt[0]) if fit_popt is not None and len(fit_popt) > 0 else np.nan
        grouped["offset"] = float(fit_popt[2]) if fit_popt is not None and len(fit_popt) > 2 else np.nan
        grouped["dataset_order"] = int(dataset_order)
        grouped["dataset_color"] = spec.color
        grouped["legend_label"] = spec.label
        temporal_parts.append(grouped)

    if plot_vector_temporal_enabled and temporal_parts:
        temporal_cmp = pd.concat(temporal_parts, ignore_index=True)
        with comparison_style_context("dark"):
            fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
            x_series: list[np.ndarray] = []
            y_series: list[np.ndarray] = []
            for dataset_order, dataset_df in temporal_cmp.groupby("dataset_order", sort=True):
                dataset_df = dataset_df.sort_values("lag_s")
                if dataset_df.empty:
                    continue
                color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                x = dataset_df["lag_s"].to_numpy(dtype=float)
                y = dataset_df["corr"].to_numpy(dtype=float)
                yerr = dataset_df["corr_std"].to_numpy(dtype=float) if "corr_std" in dataset_df.columns else None
                mask = np.isfinite(x) & np.isfinite(y)
                if temporal_range is not None:
                    lower, upper = temporal_range
                    if lower is not None:
                        mask &= x >= float(lower)
                    if upper is not None:
                        mask &= x <= float(upper)
                x = x[mask]
                y = y[mask]
                if yerr is not None:
                    yerr = yerr[mask]
                if x.size == 0:
                    continue
                x_series.append(x)
                y_series.append(y)

                tau_values = dataset_df["tau_str_s"].to_numpy(dtype=float) if "tau_str_s" in dataset_df.columns else np.array([], dtype=float)
                tau_values = tau_values[np.isfinite(tau_values)]
                tau = float(np.nanmedian(tau_values)) if tau_values.size else np.nan
                tau_err_values = dataset_df["tau_err_s"].to_numpy(dtype=float) if "tau_err_s" in dataset_df.columns else np.array([], dtype=float)
                tau_err_values = tau_err_values[np.isfinite(tau_err_values)] if tau_err_values.size else np.array([], dtype=float)
                tau_err = float(np.nanmedian(tau_err_values)) if tau_err_values.size else np.nan
                amp_values = dataset_df["amp"].to_numpy(dtype=float) if "amp" in dataset_df.columns else np.array([], dtype=float)
                amp_values = amp_values[np.isfinite(amp_values)] if amp_values.size else np.array([], dtype=float)
                offset_values = dataset_df["offset"].to_numpy(dtype=float) if "offset" in dataset_df.columns else np.array([], dtype=float)
                offset_values = offset_values[np.isfinite(offset_values)] if offset_values.size else np.array([], dtype=float)
                if np.isfinite(tau) and tau > 0 and amp_values.size and offset_values.size:
                    tau_label = _format_math_uncertainty_label(r"\tau", tau, tau_err, r"\mathrm{s}")
                    label = f"{label} {tau_label}"
                    x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 250)
                    ax.plot(x_fit, _exp_decay(x_fit, float(np.nanmedian(amp_values)), tau, float(np.nanmedian(offset_values))), ls="--", lw=1.4, color=color, alpha=0.85, label="_nolegend_")

                ax.plot(x, y, lw=2.0, color=color, label=label)
                if yerr is not None and np.any(np.isfinite(yerr)):
                    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                    ax.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=color, linewidth=0)

            ax.set_title("Temporal vector correlation comparison")
            ax.set_xlabel("lag (s)")
            ax.set_ylabel("correlation")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=True)

            x_limits = _finite_limits(x_series)
            y_limits = _finite_limits(y_series)
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

            def _white_temporal(ax_white: Axes, table=temporal_cmp) -> None:
                x_series_white: list[np.ndarray] = []
                y_series_white: list[np.ndarray] = []
                for dataset_order, dataset_df in table.groupby("dataset_order", sort=True):
                    dataset_df = dataset_df.sort_values("lag_s")
                    if dataset_df.empty:
                        continue
                    color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
                    label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
                    x = dataset_df["lag_s"].to_numpy(dtype=float)
                    y = dataset_df["corr"].to_numpy(dtype=float)
                    yerr = dataset_df["corr_std"].to_numpy(dtype=float) if "corr_std" in dataset_df.columns else None
                    mask = np.isfinite(x) & np.isfinite(y)
                    if temporal_range is not None:
                        lower, upper = temporal_range
                        if lower is not None:
                            mask &= x >= float(lower)
                        if upper is not None:
                            mask &= x <= float(upper)
                    x = x[mask]
                    y = y[mask]
                    if yerr is not None:
                        yerr = yerr[mask]
                    if x.size == 0:
                        continue
                    x_series_white.append(x)
                    y_series_white.append(y)

                    tau_values = dataset_df["tau_str_s"].to_numpy(dtype=float) if "tau_str_s" in dataset_df.columns else np.array([], dtype=float)
                    tau_values = tau_values[np.isfinite(tau_values)]
                    tau = float(np.nanmedian(tau_values)) if tau_values.size else np.nan
                    tau_err_values = dataset_df["tau_err_s"].to_numpy(dtype=float) if "tau_err_s" in dataset_df.columns else np.array([], dtype=float)
                    tau_err_values = tau_err_values[np.isfinite(tau_err_values)] if tau_err_values.size else np.array([], dtype=float)
                    tau_err = float(np.nanmedian(tau_err_values)) if tau_err_values.size else np.nan
                    amp_values = dataset_df["amp"].to_numpy(dtype=float) if "amp" in dataset_df.columns else np.array([], dtype=float)
                    amp_values = amp_values[np.isfinite(amp_values)] if amp_values.size else np.array([], dtype=float)
                    offset_values = dataset_df["offset"].to_numpy(dtype=float) if "offset" in dataset_df.columns else np.array([], dtype=float)
                    offset_values = offset_values[np.isfinite(offset_values)] if offset_values.size else np.array([], dtype=float)
                    if np.isfinite(tau) and tau > 0 and amp_values.size and offset_values.size:
                        tau_label = _format_math_uncertainty_label(r"\tau", tau, tau_err, r"\mathrm{s}")
                        label = f"{label} {tau_label}"
                        x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 250)
                        ax_white.plot(x_fit, _exp_decay(x_fit, float(np.nanmedian(amp_values)), tau, float(np.nanmedian(offset_values))), ls="--", lw=1.4, color=color, alpha=0.85, label="_nolegend_")

                    ax_white.plot(x, y, lw=2.0, color=color, label=label)
                    if yerr is not None and np.any(np.isfinite(yerr)):
                        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                        ax_white.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=color, linewidth=0)

                ax_white.set_xlabel("lag (s)")
                ax_white.set_ylabel("correlation")
                ax_white.grid(True, alpha=0.25)
                ax_white.legend(frameon=True)
                x_limits = _finite_limits(x_series_white)
                y_limits = _finite_limits(y_series_white)
                if x_limits is not None:
                    ax_white.set_xlim(*x_limits)
                if y_limits is not None:
                    ax_white.set_ylim(*y_limits)

            stem = f"{comparison_name}_temporal_vector_correlation"
            saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_temporal)
            plt.close(fig)

    tensor_cfg = comparison_vector_cfg
    tensor_fit_enabled = bool(tensor_cfg.get("tensor_fit_enabled", False))
    tensor_distance_mode = str(tensor_cfg.get("tensor_distance_mode", "xyz")).strip().lower()
    tensor_fit_range = _range_from_config(
        tensor_cfg.get("tensor_fit_distance_um_min"),
        tensor_cfg.get("tensor_fit_distance_um_max"),
    )
    tensor_fit_min_points = int(tensor_cfg.get("tensor_fit_min_points", 4))

    def _plot_tensor_component_pair(
        ax: Axes,
        *,
        tensor_basis: str,
        part: str,
        component_pair: str,
        include_fit: bool,
        title: str | None,
    ) -> None:
        x_series: list[np.ndarray] = []
        y_series: list[np.ndarray] = []
        tensor_result_key, _ = _tensor_comparison_output_keys(tensor_basis)
        for dataset_order, (spec, runner) in enumerate(sorted_records):
            tensor_df = _read_output_or_parquet(
                runner,
                "vector_corr",
                tensor_result_key,
                _velocity_output_name(
                    "beads_vector_correlation_tensor_avg.parquet" if bool(tensor_cfg.get("multi_frame_average", False)) else "beads_vector_correlation_tensor.parquet",
                    tensor_cfg,
                    distance_mode=tensor_distance_mode,
                    tensor_basis=tensor_basis,
                ),
            )
            if tensor_df.empty:
                continue

            pair_df = tensor_df.loc[
                (tensor_df["part"].astype(str) == str(part))
                & (tensor_df["component_pair"].astype(str) == str(component_pair))
            ].copy()
            if pair_df.empty:
                continue

            agg: dict[str, str] = {"corr": "mean"}
            if "corr_sem" in pair_df.columns:
                agg["corr_sem"] = "mean"
            grouped = (
                pair_df.dropna(subset=["distance_um", "corr"])
                .groupby("distance_um", as_index=False)
                .agg(agg)
                .sort_values("distance_um")
            )
            if grouped.empty:
                continue

            x = grouped["distance_um"].to_numpy(dtype=float)
            y = grouped["corr"].to_numpy(dtype=float)
            x_series.append(x)
            y_series.append(y)

            color = spec.color
            legend_label = spec.label
            signal_variance = float(np.nanvar(y, ddof=0)) if y.size else float("nan")

            if "corr_sem" in grouped.columns:
                yerr = grouped["corr_sem"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                if np.any(mask):
                    ax.fill_between(x[mask], y[mask] - yerr[mask], y[mask] + yerr[mask], alpha=0.14, color=color, linewidth=0)

            fit_popt = None
            fit_xi = None
            fit_xi_err = None
            if include_fit:
                fit_popt, fit_xi, fit_xi_err = _fit_signed_decay(
                    x,
                    y,
                    yerr=grouped["corr_sem"].to_numpy(dtype=float) if "corr_sem" in grouped.columns else None,
                    fit_range=tensor_fit_range,
                    min_points=tensor_fit_min_points,
                )
                if fit_xi is not None and np.isfinite(fit_xi):
                    xi_label = _format_math_uncertainty_label(r"\xi", fit_xi, fit_xi_err, r"\mu\mathrm{m}")
                    legend_label = f"{spec.label} {xi_label}"
            elif str(component_pair).lower() == "xy" and np.isfinite(signal_variance):
                legend_label = f"{spec.label} var={signal_variance:.3g}"

            if fit_popt is not None and np.isfinite(x).any():
                finite_x = x[np.isfinite(x)]
                x_fit = np.linspace(float(np.nanmin(finite_x)), float(np.nanmax(finite_x)), 250)
                ax.plot(x_fit, _exp_decay(x_fit, *fit_popt), ls="--", lw=1.4, color=color, alpha=0.85, label="_nolegend_")

            ax.plot(x, y, lw=1.8, color=color, label=legend_label)

        if title:
            ax.set_title(title)
        ax.set_xlabel(r"distance ($\mu\mathrm{m}$)")
        ax.set_ylabel("correlation")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

        x_limits = _finite_limits(x_series)
        y_limits = _finite_limits(y_series)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    for tensor_basis in ("cartesian", "spherical"):
        for part, component_pair, include_fit, title in _tensor_comparison_component_specs(tensor_basis, tensor_fit_enabled):
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
                _plot_tensor_component_pair(
                    ax,
                    tensor_basis=tensor_basis,
                    part=part,
                    component_pair=component_pair,
                    include_fit=include_fit,
                    title=title,
                )

                def _white_tensor(ax_white: Axes, part_name=part, pair_name=component_pair, fit_enabled=include_fit, basis_name=tensor_basis) -> None:
                    _plot_tensor_component_pair(
                        ax_white,
                        tensor_basis=basis_name,
                        part=part_name,
                        component_pair=pair_name,
                        include_fit=fit_enabled,
                        title=None,
                    )

                stem = f"{comparison_name}_tensor_{part}_{component_pair}_correlation"
                if tensor_basis == "spherical":
                    stem = f"{comparison_name}_tensor_spherical_{part}_{component_pair}_correlation"
                saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_tensor)
                plt.close(fig)

    tensor_time_series_enabled = bool(vector_cfg.get("tensor_time_series_enabled", False))
    if plot_tensor_time_series_enabled and tensor_time_series_enabled:
        tensor_time_series_specs = (
            ("cartesian", "symmetric", "xx", "Tensor xx decay length over time"),
            ("cartesian", "antisymmetric", "xy", "Tensor xy decay length over time"),
            ("spherical", "symmetric", "rr", "Spherical tensor rr decay length over time"),
            ("spherical", "antisymmetric", "rtheta", "Spherical tensor rtheta decay length over time"),
        )
        for tensor_basis, part, component_pair, title in tensor_time_series_specs:
            _, tensor_time_series_key = _tensor_comparison_output_keys(tensor_basis)
            time_parts: list[pd.DataFrame] = []
            for dataset_order, (spec, runner) in enumerate(sorted_records):
                tensor_time_df = _read_output_or_parquet(
                    runner,
                    "vector_corr",
                    tensor_time_series_key,
                    _velocity_output_name(
                        "beads_vector_correlation_tensor_time_series.parquet",
                        tensor_cfg,
                        distance_mode=tensor_distance_mode,
                        tensor_basis=tensor_basis if tensor_basis == "spherical" else None,
                    ),
                )
                if tensor_time_df.empty:
                    continue

                summary = _tensor_decay_length_time_series(
                    tensor_time_df,
                    part=part,
                    component_pair=component_pair,
                    fit_range=tensor_fit_range,
                    min_points=tensor_fit_min_points,
                )
                if summary.empty:
                    continue

                summary = summary.copy()
                summary["time_min"] = summary["time_s"].astype(float) / 60.0
                summary["dataset_id"] = spec.dataset_id
                summary["dataset_label"] = spec.label
                summary["dataset_order"] = int(dataset_order)
                summary["dataset_color"] = spec.color
                summary["legend_label"] = spec.label
                time_parts.append(summary)

            if not time_parts:
                continue

            time_cmp = pd.concat(time_parts, ignore_index=True).sort_values(["dataset_order", "time_min"]).reset_index(drop=True)
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
                _plot_autocorr_length_time_series(
                    ax,
                    time_cmp,
                    title=f"{title} comparison",
                    time_col="time_min",
                    time_unit="min",
                    use_log_x=False,
                )
                ax.set_ylabel(r"tensor decay length ($\mu\mathrm{m}$)")

                def _white_tensor_time(ax_white: Axes, table=time_cmp, plot_title=title) -> None:
                    _plot_autocorr_length_time_series(
                        ax_white,
                        table,
                        title=None,
                        time_col="time_min",
                        time_unit="min",
                        use_log_x=False,
                    )
                    ax_white.set_ylabel(r"tensor decay length ($\mu\mathrm{m}$)")

                stem = f"{comparison_name}_tensor_{part}_{component_pair}_length_over_time"
                if tensor_basis == "spherical":
                    stem = f"{comparison_name}_tensor_spherical_{part}_{component_pair}_length_over_time"
                saved[stem] = save_comparison_dual_pdf(fig, out_dir, stem, white_plot_fn=_white_tensor_time)
                plt.close(fig)

    return saved


def run_batch_notebook_like(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    dataset_ids: Sequence[str],
    base_dirs: Sequence[str] | None = None,
    variation: str | None = None,
    overrides: dict[str, Any] | None = None,
    feature_order: Iterable[str] | None = None,
    feature_commands: Sequence[str] | None = None,
    enable: Sequence[str] | None = None,
    disable: Sequence[str] | None = None,
    overwrite: Sequence[str] | None = None,
    apply_feature_commands_to_single_datasets: bool = True,
    run_comparison: bool = True,
    comparison_name: str | None = None,
    comparison_output_root: str | Path = "plots/comparisons",
) -> dict[str, Any]:
    base_config = load_analysis_config(str(config_path))
    base_config = merge_overrides(base_config, overrides)
    dataset_cfg = dict(base_config.get("dataset", {}))
    default_base_dir = str(dataset_cfg.get("base_dir", "data"))
    dataset_pairs = _normalize_batch_inputs(dataset_ids, base_dirs, default_base_dir=default_base_dir)
    if not dataset_pairs:
        raise ValueError("dataset_ids must contain at least one dataset")

    specs = _comparison_specs_for_batch(base_config, dataset_pairs, variation=variation)
    runs: list[tuple[ComparisonSpec, AnalysisNotebookRunner]] = []
    commands_for_single = list(feature_commands or []) if apply_feature_commands_to_single_datasets else []

    if not apply_feature_commands_to_single_datasets and feature_commands:
        print("Skipping NOTEBOOK feature command application on single-dataset runs (comparison config override).")

    for spec in specs:
        source_base_dir = spec.raw_base_dir if spec.raw_base_dir is not None else spec.base_dir
        path_details = [f"source_base_dir={source_base_dir}"]
        if spec.base_dir is not None and spec.base_dir != source_base_dir:
            path_details.append(f"registry_base_dir={spec.base_dir}")
        if spec.raw_base_dir is not None:
            path_details.append(f"raw_base_dir={spec.raw_base_dir}")
        print(f"Running dataset {spec.dataset_id} ({', '.join(path_details)}, label={spec.label})")
        runner = load_default_runner(
            config_path=config_path,
            dataset_id=spec.dataset_id,
            variation=variation if variation is not None else spec.variation,
            overrides=overrides,
        )
        _apply_notebook_commands(
            runner,
            dataset_id=spec.dataset_id,
            variation=variation if variation is not None else spec.variation,
            base_dir=source_base_dir,
            feature_commands=commands_for_single,
            enable=list(enable or []),
            disable=list(disable or []),
            overwrite=list(overwrite or []),
        )
        runner.run_all(feature_order=feature_order)
        runs.append((spec, runner))

    comparison_paths: dict[str, dict[str, Path]] = {}
    if run_comparison:
        comparison_cfg = base_config.get("comparison", {}) if isinstance(base_config.get("comparison", {}), dict) else {}
        resolved_name = comparison_name or str(comparison_cfg.get("name", "batch_comparison")).strip() or "batch_comparison"
        resolved_root = comparison_output_root or str(comparison_cfg.get("output_root", "plots/comparisons"))
        comparison_paths = run_batch_comparison(
            runs,
            comparison_name=resolved_name,
            output_root=resolved_root,
        )
        print(f"Saved comparison plots to {comparison_output_dir(resolved_root, resolved_name, variation=str(variation or ''))}")

    return {
        "runs": runs,
        "comparison_paths": comparison_paths,
    }


@dataclass
class AnalysisNotebookRunner:
    config: dict[str, Any]
    feature_switches: dict[str, FeatureSwitch] = field(default_factory=_feature_switches)
    state: dict[str, Any] | None = None
    outputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "AnalysisNotebookRunner":
        config = load_analysis_config(str(config_path))
        config = merge_overrides(config, overrides)
        runner = cls(config=config)
        for feature_name in FEATURE_ORDER:
            feature_cfg = runner.config.get(feature_name, {})
            if isinstance(feature_cfg, dict) and "enabled" in feature_cfg:
                enabled = bool(feature_cfg.get("enabled"))
                runner.set_feature(feature_name, compute=enabled, plot=enabled)
        return runner

    def load_state(self) -> dict[str, Any]:
        if self.state is None:
            self.state = load_dataset_state(self.config["dataset"], verbose=bool(self.config.get("runtime", {}).get("verbose", True)))
        return self.state

    def set_feature(
        self,
        name: str,
        *,
        compute: bool | None = None,
        plot: bool | None = None,
        overwrite: bool | None = None,
    ) -> "AnalysisNotebookRunner":
        current = self.feature_switches[name]
        self.feature_switches[name] = FeatureSwitch(
            compute=current.compute if compute is None else bool(compute),
            plot=current.plot if plot is None else bool(plot),
            overwrite=current.overwrite if overwrite is None else bool(overwrite),
        )
        return self

    def enable(self, name: str, *, compute: bool | None = None, plot: bool | None = None) -> "AnalysisNotebookRunner":
        return self.set_feature(name, compute=True if compute is None else compute, plot=True if plot is None else plot)

    def disable(self, name: str, *, compute: bool | None = None, plot: bool | None = None) -> "AnalysisNotebookRunner":
        return self.set_feature(name, compute=False if compute is None else compute, plot=False if plot is None else plot)

    def set_overwrite(self, name: str, overwrite: bool = True) -> "AnalysisNotebookRunner":
        return self.set_feature(name, overwrite=overwrite)

    def print_feature_diagram(self) -> None:
        print("Feature map")
        print("-----------")
        print(f"{'feature':<14} {'compute':<9} {'plot':<9} {'overwrite':<10} {'description'}")
        for name in FEATURE_ORDER:
            switch = self.feature_switches[name]
            desc = FEATURE_DESCRIPTIONS.get(name, "")
            print(
                f"{name:<14} "
                f"{'on' if switch.compute else 'off':<9} "
                f"{'on' if switch.plot else 'off':<9} "
                f"{'on' if switch.overwrite else 'off':<10} "
                f"- {desc}"
            )

    def run_all(self, feature_order: Iterable[str] | None = None) -> dict[str, Any]:
        self.print_feature_diagram()
        for name in (tuple(feature_order) if feature_order else FEATURE_ORDER):
            self.run_feature(name)
        return self.outputs

    def run_feature(self, name: str) -> Any:
        switch = self.feature_switches[name]
        if name == "vector_corr" and not bool(self.config.get("vector_corr", {}).get("enabled", True)):
            self.outputs[name] = {}
            return {}
        if not (switch.compute or switch.plot):
            self.outputs[name] = {}
            return {}
        if name == "beads":
            return self._run_beads()
        if name == "autocorr":
            return self._run_autocorr()
        if name == "image_corr":
            return self._run_image_corr()
        if name == "vector_corr":
            return self._run_vector_corr()
        if name == "velocity_spectrum":
            return self._run_velocity_spectrum()
        if name == "summary":
            return self._run_summary()
        raise KeyError(f"Unknown feature: {name}")

    def _runtime_overrides(self, feature_name: str) -> dict[str, Any]:
        switch = self.feature_switches[feature_name]
        return {"runtime": {"skip_existing": not switch.overwrite}}

    def _render_displacement_overlay_movie(
        self,
        *,
        state: dict[str, Any],
        beads_cfg: dict[str, Any],
        vector_cfg: dict[str, Any],
        vel_df: pd.DataFrame,
        plot_dir: Path,
        overwrite: bool,
    ) -> Path | None:
        if not bool(beads_cfg.get("render_displacement_overlay_movie", False)):
            return None
        if vel_df is None or vel_df.empty:
            return None

        overlay_max_vectors_cfg = beads_cfg.get("displacement_overlay_max_vectors")
        overlay_max_vectors = int(overlay_max_vectors_cfg) if overlay_max_vectors_cfg is not None else None
        movie_name = build_velocity_artifact_name(f"{state['dataset_id']}_beads_displacement_per_frame_overlay.mp4", vector_cfg)
        movie_path = plot_dir / movie_name
        if movie_path.exists() and not overwrite:
            return movie_path

        fps_value = beads_cfg.get("displacement_overlay_fps")
        if fps_value is None:
            fps_value = state["calibration"].get("fps", 1.0)

        render_bead_displacement_overlay_movie(
            state["images"],
            vel_df,
            movie_path,
            px_per_micron=float(state["calibration"]["px_per_micron"]),
            vector_cfg=vector_cfg,
            bead_channel=int(beads_cfg.get("channel_to_use", 1)),
            background_channel=beads_cfg.get("displacement_overlay_background_channel"),
            scalebar_um=beads_cfg.get("displacement_overlay_scalebar_um"),
            scalebar_loc=str(beads_cfg.get("displacement_overlay_scalebar_loc", "lower right")),
            frame_step=int(beads_cfg.get("displacement_overlay_frame_step", 1)),
            vector_scale=float(beads_cfg.get("displacement_overlay_vector_scale", 1.0)),
            max_vectors=overlay_max_vectors,
            fps=float(fps_value),
            dpi=int(beads_cfg.get("displacement_overlay_dpi", 150)),
            show_title=bool(beads_cfg.get("displacement_overlay_show_title", True)),
            verbose=bool(self.config.get("runtime", {}).get("verbose", True)),
            progress_every=int(beads_cfg.get("displacement_overlay_progress_every", 10)),
        )
        return movie_path

    def _render_bead_velocity_plot(
        self,
        *,
        state: dict[str, Any],
        beads_cfg: dict[str, Any],
        vector_cfg: dict[str, Any],
        vel_df: pd.DataFrame,
        plot_dir: Path,
        overwrite: bool,
    ) -> dict[str, str] | None:
        if not bool(beads_cfg.get("plot_velocity_over_time", False)):
            return None
        if vel_df is None or vel_df.empty:
            return None

        plot_stem = build_velocity_artifact_stem(f"{state['dataset_id']}_beads_velocity_over_time", vector_cfg)
        black_path = plot_dir / f"{plot_stem}_black.pdf"
        white_path = plot_dir / f"{plot_stem}_white.pdf"
        if black_path.exists() and white_path.exists() and not overwrite:
            return {"black": str(black_path), "white": str(white_path)}

        fps_value = state["calibration"].get("fps", 1.0)
        save_velocity_over_time_dual_pdf(
            vel_df,
            plot_dir,
            plot_stem,
            fps=float(fps_value) if fps_value is not None else None,
            title="Bead speed over time",
            dpi=int(beads_cfg.get("velocity_over_time_dpi", beads_cfg.get("displacement_overlay_dpi", 150))),
        )
        return {"black": str(black_path), "white": str(white_path)}

    def _plot_bead_outputs(
        self,
        *,
        state: dict[str, Any],
        beads_cfg: dict[str, Any],
        vector_cfg: dict[str, Any],
        vel_df: pd.DataFrame,
        plot_dir: Path,
        overwrite: bool,
        result: dict[str, Any],
    ) -> None:
        if _config_bool(beads_cfg, "plot_preview_enabled", True):
            preview_summary, preview_stats, fig_preview, _ = preview_bead_detection(state, beads_cfg, show=False)
            result.update({"preview_summary": preview_summary, "preview_stats": preview_stats})
            dark_path = plot_dir / f"{state['dataset_id']}_bead_preview_dark.pdf"
            white_path = plot_dir / f"{state['dataset_id']}_bead_preview_white.pdf"
            fig_preview.savefig(dark_path, dpi=150, bbox_inches="tight")
            with comparison_style_context("white"):
                _, _, white_fig, _ = preview_bead_detection(state, beads_cfg, show=False)
            white_fig.savefig(white_path, dpi=150, bbox_inches="tight")
            plt.close(white_fig)
            plt.close(fig_preview)

        velocity_plot_paths = self._render_bead_velocity_plot(
            state=state,
            beads_cfg=beads_cfg,
            vector_cfg=vector_cfg,
            vel_df=vel_df,
            plot_dir=plot_dir,
            overwrite=overwrite,
        )
        if velocity_plot_paths is not None:
            result["velocity_over_time_plot_paths"] = velocity_plot_paths

        movie_path = self._render_displacement_overlay_movie(
            state=state,
            beads_cfg=beads_cfg,
            vector_cfg=vector_cfg,
            vel_df=vel_df,
            plot_dir=plot_dir,
            overwrite=overwrite,
        )
        if movie_path is not None:
            result["displacement_overlay_movie_path"] = movie_path

    def _run_beads(self) -> dict[str, Any]:
        switch = self.feature_switches["beads"]
        dataset_cfg = self.config.get("dataset", {})
        dataset_id = str(dataset_cfg.get("dataset_id", ""))
        variation = str(dataset_cfg.get("variation", ""))
        beads_cfg = dict(self.config.get("beads", {}))
        vector_cfg = dict(self.config.get("vector_corr", {}))
        derived_dir = Path("data") / dataset_id / "derived"
        plot_dir = _ensure_dir(Path("plots") / dataset_id / variation / "beads")

        result: dict[str, Any] = {}
        if switch.compute:
            state = self.load_state()
            derived_dir = Path(state["paths"]["derived_dir"])
            plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "beads")
            detections_df, tracks_df = detect_and_link_beads(state, beads_cfg, skip_existing=not switch.overwrite)
            vel_df = compute_velocity_from_tracks(state, tracks_df, skip_existing=not switch.overwrite)
            result.update(
                {
                    "detections_df": detections_df,
                    "tracks_df": tracks_df,
                    "tracks_vel_df": vel_df,
                }
            )
            if bool(beads_cfg.get("compute_angular_speed", False)) and not vel_df.empty:
                result["tracks_ang_df"] = compute_angular_speed_xy(state, vel_df, beads_cfg, skip_existing=not switch.overwrite)

            if switch.plot:
                self._plot_bead_outputs(
                    state=state,
                    beads_cfg=beads_cfg,
                    vector_cfg=vector_cfg,
                    vel_df=vel_df,
                    plot_dir=plot_dir,
                    overwrite=switch.overwrite,
                    result=result,
                )
        else:
            state = self.load_state()
            derived_dir = Path(state["paths"]["derived_dir"])
            plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "beads")
            result.update(
                {
                    "detections_df": _read_parquet(derived_dir / "beads_detections.parquet"),
                    "tracks_df": _read_parquet(derived_dir / "beads_tracks.parquet"),
                    "tracks_vel_df": _read_parquet(derived_dir / "beads_tracks_with_velocity.parquet"),
                }
            )
            if bool(beads_cfg.get("compute_angular_speed", False)):
                result["tracks_ang_df"] = _read_parquet(derived_dir / "beads_tracks_with_angular_speed.parquet")
            if switch.plot:
                vel_df_obj = result.get("tracks_vel_df")
                vel_df = vel_df_obj if isinstance(vel_df_obj, pd.DataFrame) else pd.DataFrame()
                self._plot_bead_outputs(
                    state=state,
                    beads_cfg=beads_cfg,
                    vector_cfg=vector_cfg,
                    vel_df=vel_df,
                    plot_dir=plot_dir,
                    overwrite=switch.overwrite,
                    result=result,
                )

        self.outputs["beads"] = result
        return result

    def _run_autocorr(self) -> dict[str, Any]:
        state = self.load_state()
        switch = self.feature_switches["autocorr"]
        derived_dir = Path(state["paths"]["derived_dir"])
        plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "autocorr")
        autocorr_cfg = dict(self.config.get("autocorr", {}))

        plot_range = (
            autocorr_cfg.get("plot_r_um_min"),
            autocorr_cfg.get("plot_r_um_max"),
        )
        plot_range = _range_from_config(plot_range[0], plot_range[1])

        print(
            "Autocorr plotting range: "
            + (f"{plot_range[0]} .. {plot_range[1]} µm" if plot_range is not None else "auto")
        )
        print(f"Autocorr fit mode: {autocorr_cfg.get('fit_mode', 'standard')}")

        if switch.compute:
            result = run_autocorr_core(self.config, state=state, overrides=self._runtime_overrides("autocorr"))
        else:
            single_enabled = bool(autocorr_cfg.get("single_frame_3d_enabled", True))
            sampled3d_enabled = bool(autocorr_cfg.get("sampled_3d_enabled", True))
            sampled2d_enabled = bool(autocorr_cfg.get("sampled_2d_enabled", True))
            radial2d_enabled = bool(autocorr_cfg.get("radial_2d_enabled", True))
            result = {
                "single3d_df": _read_parquet(derived_dir / "autocorr_3d_single_frame.parquet") if single_enabled else pd.DataFrame(),
                "sampled3d_df": _read_parquet(derived_dir / "autocorr_3d_sampled.parquet") if sampled3d_enabled else pd.DataFrame(),
                "sampled2d_df": _read_parquet(derived_dir / "autocorr_2d_sampled.parquet") if sampled2d_enabled else pd.DataFrame(),
                "radial2d_df": _read_parquet(derived_dir / "autocorr_2d_radial_single.parquet") if radial2d_enabled else pd.DataFrame(),
            }

        if not switch.compute:
            print(
                "Autocorr loaded: "
                f"single3d={len(result.get('single3d_df', []))}, "
                f"sampled3d={len(result.get('sampled3d_df', []))}, "
                f"sampled2d={len(result.get('sampled2d_df', []))}, "
                f"radial2d={len(result.get('radial2d_df', []))}"
            )

        if switch.plot:
            autocorr_fit_mode = str(autocorr_cfg.get("fit_mode", "standard")).strip().lower()
            use_weighted_near0_profile = autocorr_fit_mode in {"weighted_near0", "weighted-near0", "near0_weighted", "near0-weighted"}
            with comparison_style_context("dark"):
                for key, stem in (
                    ("single3d_df", "autocorr_3d_single_frame"),
                    ("sampled3d_df", "autocorr_3d_sampled"),
                    ("sampled2d_df", "autocorr_2d_sampled"),
                    ("radial2d_df", "autocorr_2d_radial_single"),
                ):
                    df = result.get(key, pd.DataFrame())
                    if df is None or df.empty:
                        continue
                    x_col = "r_um" if "r_um" in df.columns else df.columns[0]
                    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                    title_map = {
                        "single3d_df": "3D mass-mass autocorrelation, single frame",
                        "sampled3d_df": "3D mass-mass autocorrelation, sampled frames",
                        "sampled2d_df": "2D mass-mass autocorrelation, sampled frames",
                        "radial2d_df": "2D radial mass-mass autocorrelation, single frame",
                    }
                    title = title_map.get(key, key.replace("_", " "))
                    if use_weighted_near0_profile and key == "sampled3d_df" and x_col == "r_um":
                        _plot_autocorr_weighted_near0_profile(ax, df, title=title, x_range=plot_range)
                    else:
                        _grouped_line_plot(ax, df, x_col, title=title, x_range=plot_range)

                    def _white(ax_white: Axes, table=df, col=x_col) -> None:
                        if use_weighted_near0_profile and key == "sampled3d_df" and col == "r_um":
                            _plot_autocorr_weighted_near0_profile(ax_white, table, title=None, x_range=plot_range)
                        else:
                            _grouped_line_plot(ax_white, table, col, title=None, x_range=plot_range)

                    save_comparison_dual_pdf(fig, plot_dir, f"{state['dataset_id']}_{stem}", white_plot_fn=_white)
                    plt.close(fig)

                for key, stem, title in (
                    ("sampled3d_df", "autocorr_3d_sampled_length_over_time_linear", "3D autocorrelation length over time"),
                    ("sampled2d_df", "autocorr_2d_sampled_length_over_time_linear", "2D autocorrelation length over time"),
                ):
                    df = result.get(key, pd.DataFrame())
                    if df is None or df.empty:
                        continue
                    summary = _sampled_autocorr_length_time_series(df)
                    if summary.empty:
                        continue
                    summary = summary.copy()
                    summary["time_min"] = summary["time_s"].astype(float) / 60.0
                    summary["dataset_label"] = "sampled autocorr length"
                    summary["legend_label"] = "sampled autocorr length"
                    summary["dataset_color"] = comparison_palette("tab10", 1)[0]
                    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                    _plot_autocorr_length_time_series(
                        ax,
                        summary,
                        title=title,
                        time_col="time_min",
                        time_unit="min",
                        use_log_x=False,
                    )

                    def _white_time(ax_white: Axes, table=summary, plot_title=title) -> None:
                        _plot_autocorr_length_time_series(
                            ax_white,
                            table,
                            title=None,
                            time_col="time_min",
                            time_unit="min",
                            use_log_x=False,
                        )

                    save_comparison_dual_pdf(
                        fig,
                        plot_dir,
                        f"{state['dataset_id']}_{stem}",
                        white_plot_fn=_white_time,
                    )
                    plt.close(fig)

        self.outputs["autocorr"] = result
        return result

    def _run_image_corr(self) -> dict[str, Any]:
        switch = self.feature_switches["image_corr"]
        dataset_cfg = self.config.get("dataset", {})
        dataset_id = str(dataset_cfg.get("dataset_id", ""))
        variation = str(dataset_cfg.get("variation", ""))
        derived_dir = Path("data") / dataset_id / "derived"
        plot_dir = _ensure_dir(Path("plots") / dataset_id / variation / "image_correlation")

        image_cfg = dict(self.config.get("image_corr", {}))
        if switch.compute:
            state = self.load_state()
            derived_dir = Path(state["paths"]["derived_dir"])
            plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "image_correlation")
            raw_df = compute_raw_time_image_correlation(state, image_cfg, skip_existing=not switch.overwrite)
            fit_df = fit_time_image_correlation(state["paths"]["derived_dir"], raw_df=raw_df, skip_existing=not switch.overwrite)
        else:
            raw_df = _read_parquet(derived_dir / "image_time_correlation_raw.parquet")
            fit_df = _read_parquet(derived_dir / "image_time_correlation_fit.parquet")

        result = {"image_corr_df": raw_df, "image_corr_fit_df": fit_df}

        if switch.plot and not raw_df.empty:
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                _plot_image_correlation(ax, raw_df, fit_df, title="Image-time correlation with stretched-exponential fit")

                def _white(ax_white: Axes, raw=raw_df, fit=fit_df) -> None:
                    _plot_image_correlation(ax_white, raw, fit, title=None)

                save_comparison_dual_pdf(fig, plot_dir, f"{dataset_id}_image_time_correlation", white_plot_fn=_white)
                plt.close(fig)

        self.outputs["image_corr"] = result
        return result

    def _run_vector_corr(self) -> dict[str, Any]:
        switch = self.feature_switches["vector_corr"]
        dataset_cfg = self.config.get("dataset", {})
        vector_cfg = dict(self.config.get("vector_corr", {}))
        dataset_id = str(dataset_cfg.get("dataset_id", ""))
        variation = str(dataset_cfg.get("variation", ""))
        derived_dir = Path("data") / dataset_id / "derived"
        plot_dir = _ensure_dir(Path("plots") / dataset_id / variation / "vector_correlations")
        tensor_distance_mode = str(vector_cfg.get("tensor_distance_mode", "xyz")).strip().lower()

        if switch.compute:
            state = self.load_state()
            derived_dir = Path(state["paths"]["derived_dir"])
            plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "vector_correlations")
            result = run_vector_correlation_core(self.config, state=state, overrides=self._runtime_overrides("vector_corr"))
        else:
            multi_frame_average = bool(vector_cfg.get("multi_frame_average", False))
            temporal_name = _velocity_output_name("beads_vector_correlation_temporal.parquet", vector_cfg)
            spatial_name = _velocity_output_name(
                "beads_vector_correlation_spatial_avg.parquet" if multi_frame_average else "beads_vector_correlation_spatial.parquet",
                vector_cfg,
            )
            tensor_name = _velocity_output_name(
                "beads_vector_correlation_tensor_avg.parquet" if multi_frame_average else "beads_vector_correlation_tensor.parquet",
                vector_cfg,
                distance_mode=tensor_distance_mode,
            )
            tensor_spherical_name = _velocity_output_name(
                "beads_vector_correlation_tensor_avg.parquet" if multi_frame_average else "beads_vector_correlation_tensor.parquet",
                vector_cfg,
                distance_mode=tensor_distance_mode,
                tensor_basis="spherical",
            )
            tensor_time_series_name = _velocity_output_name(
                "beads_vector_correlation_tensor_time_series.parquet",
                vector_cfg,
                distance_mode=tensor_distance_mode,
            )
            tensor_spherical_time_series_name = _velocity_output_name(
                "beads_vector_correlation_tensor_time_series.parquet",
                vector_cfg,
                distance_mode=tensor_distance_mode,
                tensor_basis="spherical",
            )
            result = {
                "temporal_vector_corr_df": _read_parquet(derived_dir / temporal_name),
                "spatial_vector_corr_df": _read_parquet(derived_dir / spatial_name),
                "tensor_vector_corr_df": _read_parquet(derived_dir / tensor_name),
                "tensor_spherical_vector_corr_df": _read_parquet(derived_dir / tensor_spherical_name),
                "tensor_time_series_df": _read_parquet(derived_dir / tensor_time_series_name),
                "tensor_spherical_time_series_df": _read_parquet(derived_dir / tensor_spherical_time_series_name),
            }

        temporal_df = _result_frame(result, "temporal_vector_corr_df", "temporal_df")
        spatial_df = _result_frame(result, "spatial_vector_corr_df", "spatial_df")
        tensor_df = _result_frame(result, "tensor_vector_corr_df", "tensor_df")
        if not temporal_df.empty:
            print(f"Temporal vector correlation loaded: {len(temporal_df)} pairs")
        if not spatial_df.empty:
            print(f"Spatial vector correlation loaded: {len(spatial_df)} pairs")
        if not tensor_df.empty and "part" in tensor_df.columns:
            part_counts = tensor_df["part"].astype(str).value_counts().to_dict()
            print(f"Tensor vector correlation loaded: {len(tensor_df)} rows across {part_counts}")
        tensor_spherical_df = _result_frame(result, "tensor_spherical_vector_corr_df", "tensor_spherical_df")
        if not tensor_spherical_df.empty and "part" in tensor_spherical_df.columns:
            part_counts = tensor_spherical_df["part"].astype(str).value_counts().to_dict()
            print(f"Spherical tensor vector correlation loaded: {len(tensor_spherical_df)} rows across {part_counts}")
        tensor_time_series_df = _result_frame(result, "tensor_time_series_df")
        if not tensor_time_series_df.empty and {"part", "component_pair"}.issubset(tensor_time_series_df.columns):
            pair_counts = tensor_time_series_df.groupby(["part", "component_pair"]).size().to_dict()
            print(f"Tensor time-series correlation loaded: {len(tensor_time_series_df)} rows across {pair_counts}")
        tensor_spherical_time_series_df = _result_frame(result, "tensor_spherical_time_series_df")
        if not tensor_spherical_time_series_df.empty and {"part", "component_pair"}.issubset(tensor_spherical_time_series_df.columns):
            pair_counts = tensor_spherical_time_series_df.groupby(["part", "component_pair"]).size().to_dict()
            print(f"Spherical tensor time-series correlation loaded: {len(tensor_spherical_time_series_df)} rows across {pair_counts}")

        plot_dir = _tensor_plot_sample_dir(
            [tensor_df, tensor_spherical_df, tensor_time_series_df, tensor_spherical_time_series_df],
            plot_dir,
        )
        comparison_vector_cfg = vector_cfg

        if switch.plot:
            if _config_bool(vector_cfg, "plot_temporal_enabled", True) and not temporal_df.empty:
                with comparison_style_context("dark"):
                    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                    temporal_plot_range = (
                        vector_cfg.get("temporal_plot_lag_s_min"),
                        vector_cfg.get("temporal_plot_lag_s_max"),
                    )
                    temporal_plot_range = _range_from_config(temporal_plot_range[0], temporal_plot_range[1])
                    temporal_fit_range = (
                        vector_cfg.get("temporal_fit_lag_s_min"),
                        vector_cfg.get("temporal_fit_lag_s_max"),
                    )
                    temporal_fit_range = _range_from_config(temporal_fit_range[0], temporal_fit_range[1])
                    plot_temporal_vector_correlation(
                        ax,
                        temporal_df,
                        title="Temporal normalized dot-product scatter",
                        max_points=vector_cfg.get("temporal_plot_max_points"),
                        x_range=temporal_plot_range,
                        fit_range=temporal_fit_range,
                    )

                    def _white_temporal(ax_white: Axes, df=temporal_df) -> None:
                        plot_temporal_vector_correlation(
                            ax_white,
                            df,
                            title=None,
                            max_points=vector_cfg.get("temporal_plot_max_points"),
                            x_range=temporal_plot_range,
                            fit_range=temporal_fit_range,
                        )

                    temporal_stem = build_velocity_artifact_stem(f"{dataset_id}_temporal_vector_correlation", vector_cfg)
                    save_vector_correlation_dual_pdf(fig, plot_dir, temporal_stem, white_plot_fn=_white_temporal)
                    print(f"Saved vector temporal correlation plots to {plot_dir}")
                    plt.close(fig)

            if _config_bool(vector_cfg, "plot_spatial_enabled", True) and not spatial_df.empty:
                with comparison_style_context("dark"):
                    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                    spatial_plot_range = (
                        vector_cfg.get("spatial_plot_distance_um_min"),
                        vector_cfg.get("spatial_plot_distance_um_max"),
                    )
                    spatial_plot_range = _range_from_config(spatial_plot_range[0], spatial_plot_range[1])
                    plot_spatial_vector_correlation(
                        ax,
                        spatial_df,
                        title="Spatial normalized dot-product scatter, one frame",
                        x_range=spatial_plot_range,
                        mean_bin_count=int(vector_cfg.get("spatial_nbins", 40)),
                    )

                    def _white_spatial(ax_white: Axes, df=spatial_df) -> None:
                        plot_spatial_vector_correlation(
                            ax_white,
                            df,
                            title=None,
                            x_range=spatial_plot_range,
                            mean_bin_count=int(vector_cfg.get("spatial_nbins", 40)),
                        )

                    spatial_stem = build_velocity_artifact_stem(f"{dataset_id}_spatial_vector_correlation", vector_cfg)
                    save_vector_correlation_dual_pdf(fig, plot_dir, spatial_stem, white_plot_fn=_white_spatial)
                    print(f"Saved vector spatial correlation plots to {plot_dir}")
                    plt.close(fig)

            tensor_plot_enabled = _config_bool(vector_cfg, "plot_tensor_enabled", True)
            tensor_plot_parts = _config_str_list(vector_cfg, "plot_tensor_parts", ["full", "symmetric", "antisymmetric"])
            plot_tensor_enabled = _config_bool(vector_cfg, "plot_tensor_enabled", tensor_plot_enabled)
            plot_tensor_parts = set(_config_str_list(vector_cfg, "plot_tensor_parts", list(tensor_plot_parts)))
            plot_tensor_pair_fits_enabled = _config_bool(vector_cfg, "plot_tensor_pair_fits_enabled", _config_bool(vector_cfg, "tensor_fit_enabled", False))
            plot_tensor_time_series_enabled = _config_bool(vector_cfg, "plot_tensor_time_series_enabled", _config_bool(vector_cfg, "tensor_time_series_enabled", False))
            tensor_plot_range = _range_from_config(
                vector_cfg.get("tensor_plot_distance_um_min"),
                vector_cfg.get("tensor_plot_distance_um_max"),
            )
            tensor_basis_specs = [
                ("cartesian", tensor_df, "Spatial vector tensor correlation", "vector_tensor_correlation"),
                ("spherical", tensor_spherical_df, "Spherical vector tensor correlation", "vector_tensor_correlation_spherical"),
            ]
            for tensor_basis, basis_df, basis_title, stem_prefix in tensor_basis_specs:
                if not plot_tensor_enabled or basis_df.empty:
                    continue
                for part, part_title, stem in (
                    ("full", basis_title, f"{stem_prefix}_full"),
                    ("symmetric", ("Symmetric tensor correlation" if tensor_basis == "cartesian" else "Spherical symmetric tensor correlation"), f"{stem_prefix}_symmetric"),
                    ("antisymmetric", ("Antisymmetric tensor correlation" if tensor_basis == "cartesian" else "Spherical antisymmetric tensor correlation"), f"{stem_prefix}_antisymmetric"),
                ):
                    if part not in plot_tensor_parts:
                        continue
                    with comparison_style_context("dark"):
                        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
                        plot_vector_tensor_correlation(
                            ax,
                            basis_df,
                            part=part,
                            title=part_title,
                            x_range=tensor_plot_range,
                        )

                        def _white_tensor(ax_white: Axes, df=basis_df, tensor_part=part) -> None:
                            plot_vector_tensor_correlation(
                                ax_white,
                                df,
                                part=tensor_part,
                                title=None,
                                x_range=tensor_plot_range,
                            )

                        tensor_stem = _tensor_plot_stem(dataset_id, stem, vector_cfg, basis_df, distance_mode=tensor_distance_mode)
                        save_vector_correlation_dual_pdf(fig, plot_dir, tensor_stem, white_plot_fn=_white_tensor)
                        print(f"Saved tensor vector correlation plots to {plot_dir} ({tensor_basis}, {part})")
                        plt.close(fig)

                if plot_tensor_pair_fits_enabled:
                    tensor_fit_pairs = _tensor_fit_component_pairs_for_basis(
                        [str(pair).strip() for pair in vector_cfg.get("tensor_fit_component_pairs", []) if str(pair).strip()],
                        tensor_basis,
                    )
                    tensor_fit_pairs_by_part_cfg = vector_cfg.get("tensor_fit_component_pairs_by_part", {})
                    tensor_fit_pairs_by_part = {
                        str(part).strip(): _tensor_fit_component_pairs_for_basis(
                            [str(pair).strip() for pair in pairs if str(pair).strip()],
                            tensor_basis,
                        )
                        for part, pairs in tensor_fit_pairs_by_part_cfg.items()
                        if str(part).strip() and isinstance(pairs, (list, tuple))
                    } if isinstance(tensor_fit_pairs_by_part_cfg, dict) else {}
                    tensor_fit_parts = [str(part).strip() for part in vector_cfg.get("tensor_fit_parts", ["full"]) if str(part).strip()]
                    tensor_fit_range = _range_from_config(
                        vector_cfg.get("tensor_fit_distance_um_min"),
                        vector_cfg.get("tensor_fit_distance_um_max"),
                    )
                    tensor_fit_min_points = int(vector_cfg.get("tensor_fit_min_points", 4))
                    available_pairs_by_part = {
                        part_name: set(basis_df.loc[basis_df["part"].astype(str) == part_name, "component_pair"].astype(str))
                        for part_name in tensor_fit_parts
                    }
                    if tensor_fit_pairs or tensor_fit_pairs_by_part:
                        for part in tensor_fit_parts:
                            requested_pairs = tensor_fit_pairs_by_part.get(part, tensor_fit_pairs)
                            part_pairs = [pair for pair in requested_pairs if pair in available_pairs_by_part.get(part, set())]
                            if not part_pairs:
                                continue
                            with comparison_style_context("dark"):
                                fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=150)
                                plot_vector_tensor_pair_decay(
                                    ax,
                                    basis_df,
                                    part=part,
                                    component_pairs=part_pairs,
                                    title=f"Tensor component-pair decay fits ({part})" if part != "antisymmetric" else f"Tensor component-pair decay ({part})",
                                    x_range=tensor_plot_range,
                                    fit_range=tensor_fit_range,
                                    min_points=tensor_fit_min_points,
                                    fit_enabled=part != "antisymmetric",
                                )

                                def _white_tensor_fit(ax_white: Axes, df=basis_df, tensor_part=part, pairs=part_pairs) -> None:
                                    plot_vector_tensor_pair_decay(
                                        ax_white,
                                        df,
                                        part=tensor_part,
                                        component_pairs=pairs,
                                        title=None,
                                        x_range=tensor_plot_range,
                                        fit_range=tensor_fit_range,
                                        min_points=tensor_fit_min_points,
                                        fit_enabled=tensor_part != "antisymmetric",
                                    )

                                pair_tag = "-".join(part_pairs)
                                fit_stem = _tensor_plot_stem(
                                    dataset_id,
                                    f"tensor_component_pair_fits_{part}_{pair_tag}",
                                    vector_cfg,
                                    basis_df,
                                )
                                save_vector_correlation_dual_pdf(fig, plot_dir, fit_stem, white_plot_fn=_white_tensor_fit)
                                print(f"Saved tensor pair-fit plots to {plot_dir} ({part})")
                                plt.close(fig)

            if plot_tensor_time_series_enabled:
                def _plot_tensor_time_series(ax_target: Axes, table: pd.DataFrame, *, title: str | None) -> None:
                    if table.empty or "time_s" not in table.columns or "xi_um" not in table.columns:
                        ax_target.set_title(title or "")
                        ax_target.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_target.transAxes)
                        return

                    component_pairs = sorted({str(value) for value in table["component_pair"].dropna().astype(str).tolist() if str(value).strip()})
                    x_series: list[np.ndarray] = []
                    y_series: list[np.ndarray] = []
                    cmap = plt.get_cmap("tab10")
                    colors = [cmap(index % 10) for index in range(max(1, len(component_pairs)))]

                    for index, component_pair in enumerate(component_pairs):
                        pair_df = table.loc[table["component_pair"].astype(str) == str(component_pair)].sort_values("time_s")
                        if pair_df.empty:
                            continue

                        x = pair_df["time_s"].to_numpy(dtype=float)
                        y = pair_df["xi_um"].to_numpy(dtype=float)
                        yerr = pair_df["xi_err_um"].to_numpy(dtype=float) if "xi_err_um" in pair_df.columns else None
                        color = colors[index % len(colors)]

                        ax_target.plot(x, y, lw=2.0, color=color, label=component_pair)
                        if yerr is not None and np.any(np.isfinite(yerr)):
                            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                            ax_target.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=color, linewidth=0)

                        x_series.append(x)
                        y_series.append(y)

                    if title:
                        ax_target.set_title(title)
                    ax_target.set_xlabel("time (s)")
                    ax_target.set_ylabel(r"tensor decay length ($\mu\mathrm{m}$)")
                    ax_target.grid(True, alpha=0.25)
                    ax_target.legend(frameon=True)

                    x_limits = _finite_limits(x_series)
                    y_limits = _finite_limits(y_series)
                    if x_limits is not None:
                        ax_target.set_xlim(*x_limits)
                    if y_limits is not None:
                        ax_target.set_ylim(*y_limits)

                tensor_time_series_specs = (
                    ("cartesian", tensor_time_series_df, "Tensor decay length over time"),
                    ("spherical", tensor_spherical_time_series_df, "Spherical tensor decay length over time"),
                )
                for tensor_basis, time_series_df, basis_title in tensor_time_series_specs:
                    if time_series_df.empty:
                        continue

                    for part in ("full", "symmetric", "antisymmetric"):
                        part_df = time_series_df.loc[time_series_df["part"].astype(str) == str(part)].copy()
                        if part_df.empty:
                            continue

                        with comparison_style_context("dark"):
                            fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
                            _plot_tensor_time_series(ax, part_df, title=f"{basis_title} ({part})")

                            def _white_tensor_time(ax_white: Axes, df=part_df, plot_title=basis_title, part_name=part) -> None:
                                _plot_tensor_time_series(ax_white, df, title=None)

                            time_series_stem = _tensor_plot_stem(
                                dataset_id,
                                f"tensor_{part}_length_over_time",
                                vector_cfg,
                                part_df,
                                distance_mode=tensor_distance_mode,
                            )
                            if tensor_basis == "spherical":
                                time_series_stem = _tensor_plot_stem(
                                    dataset_id,
                                    f"tensor_spherical_{part}_length_over_time",
                                    vector_cfg,
                                    part_df,
                                    distance_mode=tensor_distance_mode,
                                )
                            save_vector_correlation_dual_pdf(fig, plot_dir, time_series_stem, white_plot_fn=_white_tensor_time)
                            print(f"Saved tensor time-series plots to {plot_dir} ({tensor_basis}, {part})")
                            plt.close(fig)

        self.outputs["vector_corr"] = result
        return result

    def _run_velocity_spectrum(self) -> dict[str, Any]:
        switch = self.feature_switches["velocity_spectrum"]
        dataset_cfg = self.config.get("dataset", {})
        vector_cfg = dict(self.config.get("vector_corr", {}))
        spectrum_cfg = dict(self.config.get("velocity_spectrum", {}))
        xy_vorticity_cfg = dict(spectrum_cfg.get("xy_vorticity", {}))
        xy_vorticity_spectrum_cfg = dict(xy_vorticity_cfg.get("spectrum", {}))
        dataset_id = str(dataset_cfg.get("dataset_id", ""))
        variation = str(dataset_cfg.get("variation", ""))
        derived_dir = Path("data") / dataset_id / "derived"
        plot_dir = _ensure_dir(Path("plots") / dataset_id / variation / "velocity_spectrum")

        spectrum_name = velocity_spectrum_output_name("beads_velocity_spectrum.parquet", vector_cfg)
        spectrum_frames_name = velocity_spectrum_output_name("beads_velocity_spectrum_frames.parquet", vector_cfg)
        vorticity_name = velocity_vorticity_output_name("beads_velocity_vorticity_xy.parquet", vector_cfg)
        vorticity_spectrum_name = velocity_vorticity_spectrum_output_name("beads_velocity_vorticity_spectrum.parquet", vector_cfg)

        if switch.compute:
            state = self.load_state()
            derived_dir = Path(state["paths"]["derived_dir"])
            plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "velocity_spectrum")
            result = run_velocity_spectrum_core(self.config, state=state, overrides=self._runtime_overrides("velocity_spectrum"))
        else:
            result = {
                "velocity_spectrum_df": _read_parquet(derived_dir / spectrum_name),
                "velocity_spectrum_frames_df": _read_parquet(derived_dir / spectrum_frames_name),
                "velocity_vorticity_df": _read_parquet(derived_dir / vorticity_name),
                "velocity_vorticity_spectrum_df": _read_parquet(derived_dir / vorticity_spectrum_name),
            }

        spectrum_df = _result_frame(result, "velocity_spectrum_df")
        frame_spectrum_df = _result_frame(result, "velocity_spectrum_frames_df")
        vorticity_df = _result_frame(result, "velocity_vorticity_df")
        vorticity_spectrum_df = _result_frame(result, "velocity_vorticity_spectrum_df")
        if not spectrum_df.empty:
            print(f"Velocity spectrum loaded: {len(spectrum_df)} wavenumber bins")
        if not frame_spectrum_df.empty:
            print(f"Velocity spectrum frame table loaded: {len(frame_spectrum_df)} rows")
        if not vorticity_df.empty:
            print(f"Velocity vorticity map loaded: {len(vorticity_df)} grid cells")
        if not vorticity_spectrum_df.empty:
            print(f"Velocity vorticity spectrum loaded: {len(vorticity_spectrum_df)} wavenumber bins")

        if switch.plot and not spectrum_df.empty:
            plot_range = _range_from_config(spectrum_cfg.get("plot_k_min"), spectrum_cfg.get("plot_k_max"))
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                plot_velocity_spectrum(
                    ax,
                    spectrum_df,
                    title="3D time-averaged velocity spectrum",
                    x_range=plot_range,
                )

                def _white_velocity_spectrum(ax_white: Axes, df=spectrum_df) -> None:
                    plot_velocity_spectrum(
                        ax_white,
                        df,
                        title=None,
                        x_range=plot_range,
                    )

                stem = velocity_spectrum_artifact_stem(f"{dataset_id}_velocity_spectrum", vector_cfg)
                save_comparison_dual_pdf(fig, plot_dir, stem, white_plot_fn=_white_velocity_spectrum)
                plt.close(fig)

        if switch.plot and bool(xy_vorticity_cfg.get("enabled", False)) and bool(xy_vorticity_cfg.get("plot_enabled", True)) and not vorticity_df.empty:
            quiver_stride = int(xy_vorticity_cfg.get("quiver_stride", 3))
            quiver_scale = xy_vorticity_cfg.get("quiver_scale")
            color_map = str(xy_vorticity_cfg.get("colormap", "RdBu_r"))
            frame_value = int(vorticity_df["frame"].iloc[0]) if "frame" in vorticity_df.columns and not vorticity_df.empty else -1
            title = f"XY vorticity overlay, frame {frame_value}"
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=150)
                plot_xy_vorticity_overlay(
                    ax,
                    vorticity_df,
                    title=title,
                    quiver_stride=quiver_stride,
                    quiver_scale=float(quiver_scale) if quiver_scale is not None else None,
                    cmap=color_map,
                )

                def _white_vorticity(ax_white: Axes, df=vorticity_df) -> None:
                    plot_xy_vorticity_overlay(
                        ax_white,
                        df,
                        title=None,
                        quiver_stride=quiver_stride,
                        quiver_scale=float(quiver_scale) if quiver_scale is not None else None,
                        cmap=color_map,
                    )

                stem = velocity_vorticity_artifact_stem(f"{dataset_id}_velocity_vorticity_xy", vector_cfg)
                save_comparison_dual_pdf(fig, plot_dir, stem, white_plot_fn=_white_vorticity)
                plt.close(fig)

        if switch.plot and bool(xy_vorticity_spectrum_cfg.get("enabled", False)) and bool(xy_vorticity_spectrum_cfg.get("plot_enabled", True)) and not vorticity_spectrum_df.empty:
            plot_range = _range_from_config(xy_vorticity_spectrum_cfg.get("plot_k_min"), xy_vorticity_spectrum_cfg.get("plot_k_max"))
            with comparison_style_context("dark"):
                fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
                plot_vorticity_spectrum(
                    ax,
                    vorticity_spectrum_df,
                    title="2D vorticity spectrum",
                    x_range=plot_range,
                )

                def _white_vorticity_spectrum(ax_white: Axes, df=vorticity_spectrum_df) -> None:
                    plot_vorticity_spectrum(
                        ax_white,
                        df,
                        title=None,
                        x_range=plot_range,
                    )

                stem = velocity_vorticity_spectrum_artifact_stem(f"{dataset_id}_velocity_vorticity_spectrum", vector_cfg)
                save_comparison_dual_pdf(fig, plot_dir, stem, white_plot_fn=_white_vorticity_spectrum)
                plt.close(fig)

        self.outputs["velocity_spectrum"] = result
        return result

    def _run_summary(self) -> dict[str, Any]:
        state = self.load_state()
        dataset_cfg = self.config.get("dataset", {})
        dataset_id = str(dataset_cfg.get("dataset_id", ""))
        derived_dir = Path(state["paths"]["derived_dir"])
        rows = []
        for name in FEATURE_ORDER:
            switch = self.feature_switches[name]
            rows.append(
                {
                    "feature": name,
                    "compute": switch.compute,
                    "plot": switch.plot,
                    "overwrite": switch.overwrite,
                    "has_output": name in self.outputs,
                }
            )

        summary_df = pd.DataFrame(rows)
        summary_path = derived_dir / "analysis_unified_feature_manifest.parquet"
        summary_df.to_parquet(summary_path, index=False)

        beads_cfg = dict(self.config.get("beads", {}))
        vector_cfg = dict(self.config.get("vector_corr", {}))
        bead_outputs = self.outputs.get("beads", {}) if isinstance(self.outputs.get("beads", {}), dict) else {}
        summary_movie_path = bead_outputs.get("displacement_overlay_movie_path") if isinstance(bead_outputs.get("displacement_overlay_movie_path"), Path) else None
        if summary_movie_path is None and bool(beads_cfg.get("render_displacement_overlay_movie", False)) and "displacement_overlay_movie_path" not in bead_outputs:
            plot_dir = _ensure_dir(Path(state["paths"]["plots_dir"]) / "beads")
            vel_df_obj = bead_outputs.get("tracks_vel_df")
            vel_df = vel_df_obj if isinstance(vel_df_obj, pd.DataFrame) else pd.DataFrame()
            if vel_df.empty:
                vel_df = _read_parquet(derived_dir / "beads_tracks_with_velocity.parquet")
            summary_movie_path = self._render_displacement_overlay_movie(
                state=state,
                beads_cfg=beads_cfg,
                vector_cfg=vector_cfg,
                vel_df=vel_df,
                plot_dir=plot_dir,
                overwrite=False,
            )
            if summary_movie_path is not None:
                print(f"Saved displacement overlay movie from cached data to {summary_movie_path}")
        print(summary_df.to_string(index=False))
        result = {"summary_df": summary_df, "summary_path": summary_path}
        if summary_movie_path is not None:
            result["displacement_overlay_movie_path"] = summary_movie_path
        self.outputs["summary"] = result
        return result


def load_default_runner(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    dataset_id: str | None = None,
    variation: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> AnalysisNotebookRunner:
    runner = AnalysisNotebookRunner.from_config_path(config_path, overrides=overrides)
    if dataset_id is not None:
        runner.config["dataset"]["dataset_id"] = dataset_id
    if variation is not None:
        runner.config["dataset"]["variation"] = variation
    return runner


def run_notebook_like(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    dataset_id: str | None = None,
    variation: str | None = None,
    overrides: dict[str, Any] | None = None,
    feature_order: Iterable[str] | None = None,
) -> AnalysisNotebookRunner:
    runner = load_default_runner(config_path=config_path, dataset_id=dataset_id, variation=variation, overrides=overrides)
    runner.run_all(feature_order=feature_order)
    return runner


def _reflect_into_bounds(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    if np.any(span <= 0):
        raise ValueError("upper bounds must be greater than lower bounds")
    offset = np.mod(np.asarray(values, dtype=float) - lower, 2.0 * span)
    reflected = np.where(offset > span, 2.0 * span - offset, offset)
    return reflected + lower


def _generate_synthetic_brownian_tracks(
    *,
    frame_count: int,
    particle_count: int,
    seed: int,
    field_of_view_um: tuple[float, float, float],
    drift_um_per_frame: tuple[float, float, float],
    step_sigma_um: tuple[float, float, float],
    margin_um: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fov = np.asarray(field_of_view_um, dtype=float)
    lower = np.full(3, float(margin_um), dtype=float)
    upper = fov - float(margin_um)
    start_positions = rng.uniform(lower, upper, size=(int(particle_count), 3))

    drift = np.asarray(drift_um_per_frame, dtype=float)
    step_sigma = np.asarray(step_sigma_um, dtype=float)
    positions = np.empty((int(frame_count), int(particle_count), 3), dtype=float)
    positions[0] = start_positions

    current = start_positions
    for frame_index in range(1, int(frame_count)):
        step = rng.normal(loc=drift, scale=step_sigma, size=(int(particle_count), 3))
        current = _reflect_into_bounds(current + step, lower, upper)
        positions[frame_index] = current

    rows: list[dict[str, float | int]] = []
    for frame_index in range(int(frame_count)):
        for particle_index in range(int(particle_count)):
            x_um, y_um, z_um = positions[frame_index, particle_index]
            rows.append(
                {
                    "frame": int(frame_index),
                    "particle": int(particle_index),
                    "x_um": float(x_um),
                    "y_um": float(y_um),
                    "z_um": float(z_um),
                }
            )

    return pd.DataFrame(rows)


def _generate_synthetic_motion_tracks(
    *,
    simulation_kind: str,
    frame_count: int,
    particle_count: int,
    seed: int,
    field_of_view_um: tuple[float, float, float],
    drift_um_per_frame: tuple[float, float, float],
    step_sigma_um: tuple[float, float, float],
    margin_um: float,
    radial_flow_um_per_frame: float,
) -> pd.DataFrame:
    kind = str(simulation_kind).strip().lower()
    if kind not in {"brownian", "contractile", "extensile"}:
        raise ValueError(f"Unknown synthetic motion kind: {simulation_kind!r}")

    rng = np.random.default_rng(seed)
    fov = np.asarray(field_of_view_um, dtype=float)
    lower = np.full(3, float(margin_um), dtype=float)
    upper = fov - float(margin_um)
    center = 0.5 * (lower + upper)
    start_positions = rng.uniform(lower, upper, size=(int(particle_count), 3))

    drift = np.asarray(drift_um_per_frame, dtype=float)
    step_sigma = np.asarray(step_sigma_um, dtype=float)
    radial_flow = float(radial_flow_um_per_frame)
    positions = np.empty((int(frame_count), int(particle_count), 3), dtype=float)
    positions[0] = start_positions

    current = start_positions
    for frame_index in range(1, int(frame_count)):
        step = rng.normal(loc=0.0, scale=step_sigma, size=(int(particle_count), 3))
        if kind == "brownian":
            step += drift
        else:
            radial_vector = current - center
            radial_norm = np.linalg.norm(radial_vector, axis=1, keepdims=True)
            radial_direction = np.divide(
                radial_vector,
                np.where(radial_norm > 0.0, radial_norm, 1.0),
                out=np.zeros_like(radial_vector),
                where=radial_norm > 0.0,
            )
            flow_sign = -1.0 if kind == "contractile" else 1.0
            step += flow_sign * radial_flow * radial_direction + drift

        current = _reflect_into_bounds(current + step, lower, upper)
        positions[frame_index] = current

    rows: list[dict[str, float | int]] = []
    for frame_index in range(int(frame_count)):
        for particle_index in range(int(particle_count)):
            x_um, y_um, z_um = positions[frame_index, particle_index]
            rows.append(
                {
                    "frame": int(frame_index),
                    "particle": int(particle_index),
                    "x_um": float(x_um),
                    "y_um": float(y_um),
                    "z_um": float(z_um),
                }
            )

    return pd.DataFrame(rows)


def _average_tracks_over_frame_windows(tracks_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    window_size = int(window_size)
    if window_size <= 1 or tracks_df.empty:
        return tracks_df.copy()

    required_columns = {"frame", "particle", "x_um", "y_um", "z_um"}
    missing_columns = required_columns.difference(tracks_df.columns)
    if missing_columns:
        raise ValueError(f"tracks_df is missing required columns for averaging: {sorted(missing_columns)}")

    averaged = tracks_df.copy()
    averaged["frame_window"] = averaged["frame"].astype(int) // window_size
    averaged = (
        averaged.groupby(["frame_window", "particle"], as_index=False)[["x_um", "y_um", "z_um"]]
        .mean()
        .sort_values(["frame_window", "particle"])
        .reset_index(drop=True)
    )
    averaged = averaged.rename(columns={"frame_window": "frame"})
    averaged["frame"] = averaged["frame"].astype(int)
    return averaged


def _run_synthetic_motion_vector_corr_test(
    *,
    simulation_kind: str,
    dataset_id: str,
    variation: str = "",
    base_dir: str | Path = "data",
    frame_count: int = 120,
    particle_count: int = 48,
    seed: int = 42,
    average_window_frames: int = 1,
) -> dict[str, Any]:
    kind = str(simulation_kind).strip().lower()
    if kind not in {"brownian", "contractile", "extensile"}:
        raise ValueError(f"Unknown synthetic motion kind: {simulation_kind!r}")

    average_window_frames = max(1, int(average_window_frames))
    raw_frame_count = int(frame_count)
    effective_frame_count = max(1, (raw_frame_count + average_window_frames - 1) // average_window_frames)

    config = load_analysis_config(str(DEFAULT_CONFIG_PATH))
    vector_cfg = dict(config.get("vector_corr", {}))
    vector_cfg.update(
        {
            "enabled": True,
            "exclude_velocity_outliers": False,
            "multi_frame_average": False,
            "tensor_time_series_enabled": True,
            "plot_temporal_enabled": True,
            "plot_spatial_enabled": True,
            "plot_tensor_enabled": True,
            "plot_tensor_pair_fits_enabled": True,
            "plot_tensor_time_series_enabled": True,
            "tensor_distance_mode": "xyz",
            "spatial_frame_index": 0,
            "tensor_frame_index": 0,
            "spatial_nbins": 48,
            "tensor_nbins": 36,
            "temporal_max_lag_frames": min(60, max(1, int(effective_frame_count) - 1)),
            "temporal_fit_lag_s_min": 0.2,
            "temporal_fit_lag_s_max": 5.0,
            "temporal_plot_max_points": 5000,
            "tensor_time_series_sample_count": min(8, max(1, int(effective_frame_count))),
        }
    )
    config["vector_corr"] = vector_cfg
    spectrum_cfg = dict(config.get("velocity_spectrum", {}))
    spectrum_cfg.update(
        {
            "enabled": True,
            "grid_shape": [32, 32, 32],
            "k_bins": 48,
            "subtract_mean": True,
        }
    )
    config["velocity_spectrum"] = spectrum_cfg
    config["dataset"] = {
        "base_dir": str(base_dir),
        "dataset_id": dataset_id,
        "variation": variation,
    }
    config.setdefault("runtime", {})
    config["runtime"]["skip_existing"] = False
    config["runtime"]["verbose"] = True

    paths = prepare_output_dirs(dataset_id, variation=variation)
    if kind == "brownian":
        drift_um_per_frame = (0.015, -0.008, 0.003)
        step_sigma_um = (0.22, 0.22, 0.11)
        radial_flow_um_per_frame = 0.03
    elif kind == "contractile":
        drift_um_per_frame = (0.006, -0.003, 0.001)
        step_sigma_um = (0.11, 0.11, 0.055)
        radial_flow_um_per_frame = 0.11
    else:
        drift_um_per_frame = (0.006, -0.003, 0.001)
        step_sigma_um = (0.11, 0.11, 0.055)
        radial_flow_um_per_frame = 0.11

    raw_tracks_df = _generate_synthetic_motion_tracks(
        simulation_kind=kind,
        frame_count=raw_frame_count,
        particle_count=particle_count,
        seed=seed,
        field_of_view_um=(24.0, 24.0, 10.0),
        drift_um_per_frame=drift_um_per_frame,
        step_sigma_um=step_sigma_um,
        margin_um=1.0,
        radial_flow_um_per_frame=radial_flow_um_per_frame,
    )

    tracks_df = _average_tracks_over_frame_windows(raw_tracks_df, average_window_frames)
    effective_frame_count = int(tracks_df["frame"].nunique()) if not tracks_df.empty else 0
    effective_fps = 10.0 / float(average_window_frames)

    tracks_path = Path(paths["derived_dir"]) / "beads_tracks.parquet"
    raw_tracks_path = Path(paths["derived_dir"]) / "beads_tracks_raw.parquet"
    raw_tracks_df.to_parquet(raw_tracks_path, index=False)
    tracks_df.to_parquet(tracks_path, index=False)

    state = {
        "dataset_id": dataset_id,
        "variation": variation,
        "base_dir": str(base_dir),
        "images": None,
        "handle": None,
        "raw_mask": None,
        "dims": {"T": int(effective_frame_count), "C": 1, "Z": 40, "Y": 120, "X": 120},
        "calibration": {
            "px_per_micron": 4.0,
            "px_per_micron_z": 4.0,
            "fps": float(effective_fps),
        },
        "paths": paths,
    }

    velocity_df = compute_velocity_from_tracks(state, tracks_df, skip_existing=False)

    runner = AnalysisNotebookRunner(config=config)
    runner.state = state
    runner.set_overwrite("vector_corr", True)
    runner.set_feature("velocity_spectrum", compute=True, plot=True, overwrite=True)
    result = runner.run_feature("vector_corr")
    spectrum_result = runner.run_feature("velocity_spectrum")

    temporal_df = _result_frame(result, "temporal_vector_corr_df", "temporal_df")
    spatial_df = _result_frame(result, "spatial_vector_corr_df", "spatial_df")
    tensor_df = _result_frame(result, "tensor_vector_corr_df", "tensor_df")
    tensor_spherical_df = _result_frame(result, "tensor_spherical_vector_corr_df", "tensor_spherical_df")
    tensor_time_series_df = _result_frame(result, "tensor_time_series_df")
    tensor_spherical_time_series_df = _result_frame(result, "tensor_spherical_time_series_df")
    spectrum_df = _result_frame(spectrum_result, "velocity_spectrum_df")
    spectrum_frame_df = _result_frame(spectrum_result, "velocity_spectrum_frames_df")

    print(f"Synthetic {kind} vector-corr test complete")
    print(f"  raw tracks: {len(raw_tracks_df)} rows, {raw_tracks_df['particle'].nunique()} particles, {raw_tracks_df['frame'].nunique()} frames")
    print(f"  averaged tracks: {len(tracks_df)} rows, {tracks_df['particle'].nunique()} particles, {tracks_df['frame'].nunique()} frames (window={average_window_frames})")
    print(f"  velocity rows: {len(velocity_df)}")
    print(f"  temporal rows: {len(temporal_df)}")
    print(f"  spatial rows: {len(spatial_df)}")
    print(f"  tensor rows: {len(tensor_df)}")
    print(f"  spherical tensor rows: {len(tensor_spherical_df)}")
    print(f"  tensor time-series rows: {len(tensor_time_series_df)}")
    print(f"  spherical tensor time-series rows: {len(tensor_spherical_time_series_df)}")
    print(f"  spectrum bins: {len(spectrum_df)}")
    print(f"  spectrum frame rows: {len(spectrum_frame_df)}")
    print(f"  derived dir: {paths['derived_dir']}")
    print(f"  plots dir: {paths['plots_dir']}")

    return {
        "config": config,
        "state": state,
        "raw_tracks_df": raw_tracks_df,
        "tracks_df": tracks_df,
        "velocity_df": velocity_df,
        "result": result,
        "spectrum_result": spectrum_result,
    }


def run_synthetic_brownian_vector_corr_test(
    *,
    dataset_id: str = "synthetic_brownian_vector_corr",
    variation: str = "",
    base_dir: str | Path = "data",
    frame_count: int = 120,
    particle_count: int = 48,
    seed: int = 42,
    average_window_frames: int = 1,
) -> dict[str, Any]:
    return _run_synthetic_motion_vector_corr_test(
        simulation_kind="brownian",
        dataset_id=dataset_id,
        variation=variation,
        base_dir=base_dir,
        frame_count=frame_count,
        particle_count=particle_count,
        seed=seed,
        average_window_frames=average_window_frames,
    )


def run_synthetic_contractile_vector_corr_test(
    *,
    dataset_id: str = "synthetic_contractile_vector_corr",
    variation: str = "",
    base_dir: str | Path = "data",
    frame_count: int = 120,
    particle_count: int = 48,
    seed: int = 42,
    average_window_frames: int = 1,
) -> dict[str, Any]:
    return _run_synthetic_motion_vector_corr_test(
        simulation_kind="contractile",
        dataset_id=dataset_id,
        variation=variation,
        base_dir=base_dir,
        frame_count=frame_count,
        particle_count=particle_count,
        seed=seed,
        average_window_frames=average_window_frames,
    )


def run_synthetic_extensile_vector_corr_test(
    *,
    dataset_id: str = "synthetic_extensile_vector_corr",
    variation: str = "",
    base_dir: str | Path = "data",
    frame_count: int = 120,
    particle_count: int = 48,
    seed: int = 42,
    average_window_frames: int = 1,
) -> dict[str, Any]:
    return _run_synthetic_motion_vector_corr_test(
        simulation_kind="extensile",
        dataset_id=dataset_id,
        variation=variation,
        base_dir=base_dir,
        frame_count=frame_count,
        particle_count=particle_count,
        seed=seed,
        average_window_frames=average_window_frames,
    )


def _parse_feature_switch_overrides(values: list[str]) -> dict[str, dict[str, bool]]:
    parsed: dict[str, dict[str, bool]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid feature override: {value!r}")
        feature_name, raw_settings = value.split(":", 1)
        if feature_name not in FEATURE_ORDER:
            raise KeyError(f"Unknown feature: {feature_name}")
        parsed.setdefault(feature_name, {})
        for setting in raw_settings.split(","):
            if not setting:
                continue
            if "=" not in setting:
                raise ValueError(f"Invalid switch setting: {setting!r}")
            key, raw_value = setting.split("=", 1)
            key_name = key.strip().lower()
            if key_name not in {"compute", "plot"}:
                continue
            parsed[feature_name][key_name] = raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return parsed


def _apply_switch_overrides(runner: AnalysisNotebookRunner, overrides: dict[str, dict[str, bool]]) -> None:
    for feature_name, settings in overrides.items():
        runner.set_feature(feature_name, **settings)


def _apply_notebook_commands(
    runner: AnalysisNotebookRunner,
    *,
    dataset_id: str | None = None,
    variation: str | None = None,
    base_dir: str | None = None,
    feature_commands: list[str] | None = None,
    enable: list[str] | None = None,
    disable: list[str] | None = None,
    overwrite: list[str] | None = None,
) -> None:
    if dataset_id is not None:
        runner.config["dataset"]["dataset_id"] = dataset_id
    if variation is not None:
        runner.config["dataset"]["variation"] = variation
    if base_dir is not None:
        runner.config["dataset"]["base_dir"] = str(base_dir)

    if feature_commands:
        _apply_switch_overrides(runner, _parse_feature_switch_overrides(feature_commands))

    for feature_name in enable or []:
        runner.enable(feature_name)
    for feature_name in disable or []:
        runner.disable(feature_name)
    for feature_name in overwrite or []:
        runner.set_overwrite(feature_name, True)


def main(argv: list[str] | None = None) -> Any:
    parser = argparse.ArgumentParser(description="Notebook-like orchestration for analysis_unified")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    synthetic_group = parser.add_mutually_exclusive_group()
    synthetic_group.add_argument("--synthetic-brownian", action="store_true")
    synthetic_group.add_argument("--synthetic-contractile", action="store_true")
    synthetic_group.add_argument("--synthetic-extensile", action="store_true")
    parser.add_argument("--synthetic-dataset-id", type=str, default=None)
    parser.add_argument("--synthetic-frames", type=int, default=120)
    parser.add_argument("--synthetic-particles", type=int, default=48)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--synthetic-average-window", type=int, default=1)
    parser.add_argument("--synthetic-base-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--dataset-ids", nargs="*", default=[])
    parser.add_argument("--variation", type=str, default=None)
    parser.add_argument("--base-dirs", nargs="*", default=[])
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--enable", nargs="*", default=[])
    parser.add_argument("--disable", nargs="*", default=[])
    parser.add_argument("--overwrite", nargs="*", default=[])
    parser.add_argument("--run", nargs="*", default=None)
    parser.add_argument("--comparison-name", type=str, default=None)
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--notebook-dataset-id", type=str, default=None)
    parser.add_argument("--notebook-variation", type=str, default=None)
    parser.add_argument("--notebook-feature-commands", nargs="*", default=[])
    parser.add_argument("--notebook-enable", nargs="*", default=[])
    parser.add_argument("--notebook-disable", nargs="*", default=[])
    parser.add_argument("--notebook-overwrite", nargs="*", default=[])
    parser.add_argument("--notebook-run-order", nargs="*", default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--notebook-base-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    synthetic_kind = None
    if args.synthetic_brownian:
        synthetic_kind = "brownian"
    elif args.synthetic_contractile:
        synthetic_kind = "contractile"
    elif args.synthetic_extensile:
        synthetic_kind = "extensile"

    if synthetic_kind is not None:
        default_dataset_ids = {
            "brownian": "synthetic_brownian_vector_corr",
            "contractile": "synthetic_contractile_vector_corr",
            "extensile": "synthetic_extensile_vector_corr",
        }
        dataset_id = str(args.synthetic_dataset_id).strip() if args.synthetic_dataset_id is not None else ""
        dataset_id = dataset_id or default_dataset_ids[synthetic_kind]
        if synthetic_kind == "brownian":
            return run_synthetic_brownian_vector_corr_test(
                dataset_id=dataset_id,
                variation=str(args.variation).strip() if args.variation is not None else "",
                base_dir=args.synthetic_base_dir,
                frame_count=int(args.synthetic_frames),
                particle_count=int(args.synthetic_particles),
                seed=int(args.synthetic_seed),
                average_window_frames=int(args.synthetic_average_window),
            )
        if synthetic_kind == "contractile":
            return run_synthetic_contractile_vector_corr_test(
                dataset_id=dataset_id,
                variation=str(args.variation).strip() if args.variation is not None else "",
                base_dir=args.synthetic_base_dir,
                frame_count=int(args.synthetic_frames),
                particle_count=int(args.synthetic_particles),
                seed=int(args.synthetic_seed),
                average_window_frames=int(args.synthetic_average_window),
            )
        return run_synthetic_extensile_vector_corr_test(
            dataset_id=dataset_id,
            variation=str(args.variation).strip() if args.variation is not None else "",
            base_dir=args.synthetic_base_dir,
            frame_count=int(args.synthetic_frames),
            particle_count=int(args.synthetic_particles),
            seed=int(args.synthetic_seed),
            average_window_frames=int(args.synthetic_average_window),
        )

    config = load_analysis_config(str(args.config))
    comparison_cfg = config.get("comparison", {}) if isinstance(config.get("comparison", {}), dict) else {}
    comparison_cfg = resolve_comparison_preset(comparison_cfg)

    feature_commands = [*NOTEBOOK_FEATURE_COMMANDS, *args.notebook_feature_commands, *args.feature]
    enable = [*NOTEBOOK_ENABLE, *args.notebook_enable, *args.enable]
    disable = [*NOTEBOOK_DISABLE, *args.notebook_disable, *args.disable]
    overwrite = [*NOTEBOOK_OVERWRITE, *args.notebook_overwrite, *args.overwrite]
    feature_order = args.run if args.run is not None else args.notebook_run_order if args.notebook_run_order is not None else NOTEBOOK_RUN_ORDER
    notebook_dataset_id = args.dataset_id if args.dataset_id is not None else args.notebook_dataset_id if args.notebook_dataset_id is not None else NOTEBOOK_DATASET_ID
    notebook_dataset_id = None if notebook_dataset_id is None else str(notebook_dataset_id).strip() or None
    notebook_variation = args.variation if args.variation is not None else args.notebook_variation if args.notebook_variation is not None else NOTEBOOK_VARIATION
    comparison_enabled = bool(comparison_cfg.get("enabled", False)) and not args.no_comparison
    apply_notebook_feature_commands_to_single_datasets = bool(comparison_cfg.get("apply_notebook_feature_commands_to_single_datasets", True))

    dataset_ids = [str(value).strip() for value in args.dataset_ids if str(value).strip()]
    if not dataset_ids:
        dataset_ids = [str(value).strip() for value in NOTEBOOK_BATCH_DATASET_IDS if str(value).strip()]

    base_dirs = [str(value).strip() for value in args.base_dirs if str(value).strip()]
    if not base_dirs:
        base_dirs = [str(value).strip() for value in NOTEBOOK_BATCH_BASE_DIRS if str(value).strip()]

    if dataset_ids or (comparison_enabled and not notebook_dataset_id):
        comparison_name = args.comparison_name or NOTEBOOK_COMPARISON_NAME or str(comparison_cfg.get("name", "batch_comparison"))
        comparison_output_root = str(comparison_cfg.get("output_root", "plots/comparisons"))
        variation = notebook_variation

        if not dataset_ids:
            default_base_dir = str(config.get("dataset", {}).get("base_dir", "data"))
            batch_pairs = _comparison_dataset_pairs_from_config(config, default_base_dir=default_base_dir)
            dataset_ids = [dataset_id for dataset_id, _ in batch_pairs]
            base_dirs = [base_dir for _, base_dir in batch_pairs]
            if not dataset_ids:
                raise ValueError(
                    "comparison.registry must define at least one dataset when notebook comparison is enabled and dataset id is empty"
                )

        batch_result = run_batch_notebook_like(
            config_path=args.config,
            dataset_ids=dataset_ids,
            base_dirs=base_dirs,
            variation=variation,
            feature_order=feature_order,
            feature_commands=feature_commands,
            enable=enable,
            disable=disable,
            overwrite=overwrite,
            apply_feature_commands_to_single_datasets=apply_notebook_feature_commands_to_single_datasets,
            run_comparison=comparison_enabled,
            comparison_name=comparison_name,
            comparison_output_root=comparison_output_root,
        )
        return batch_result

    runner = load_default_runner(config_path=args.config)
    single_dataset_feature_commands = feature_commands if apply_notebook_feature_commands_to_single_datasets else []
    if not apply_notebook_feature_commands_to_single_datasets and feature_commands:
        print("Skipping NOTEBOOK feature command application on single-dataset run (comparison config override).")
    _apply_notebook_commands(
        runner,
        dataset_id=notebook_dataset_id,
        variation=notebook_variation,
        base_dir=args.base_dir if args.base_dir is not None else args.notebook_base_dir if args.notebook_base_dir is not None else NOTEBOOK_BASE_DIR,
        feature_commands=single_dataset_feature_commands,
        enable=enable,
        disable=disable,
        overwrite=overwrite,
    )
    runner.run_all(feature_order=feature_order)
    return runner


if __name__ == "__main__":
    main()