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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis_pipeline.beads_track import detect_and_link_beads, preview_bead_detection
from analysis_pipeline.beads_velocity import compute_angular_speed_xy, compute_velocity_from_tracks
from analysis_pipeline.comparison import comparison_style_context, save_comparison_dual_pdf
from analysis_pipeline.config import load_analysis_config, merge_overrides
from analysis_pipeline.correlation_plots import (
    plot_spatial_vector_correlation,
    plot_temporal_vector_correlation,
    plot_vector_tensor_correlation,
    plot_vector_tensor_pair_decay,
    save_vector_correlation_dual_pdf,
)
from analysis_pipeline.image_correlation import compute_raw_time_image_correlation, fit_time_image_correlation
from analysis_pipeline.io_dataset import load_dataset_state
from analysis_pipeline.pipeline import run_autocorr_core, run_vector_correlation_core
from analysis_pipeline.velocity_movies import build_velocity_artifact_name, build_velocity_artifact_stem, render_bead_displacement_overlay_movie
from analysis_pipeline.velocity_plots import save_velocity_over_time_dual_pdf
from analysis_pipeline.vector_correlation import _velocity_output_name


DEFAULT_CONFIG_PATH = Path("config/analysis_default.yaml")
FEATURE_ORDER = ("beads", "autocorr", "image_corr", "vector_corr", "summary")

# Edit these values directly when you want to run the file like a notebook.
NOTEBOOK_DATASET_ID = "AMF_106_002__C640_C470"
NOTEBOOK_BASE_DIR = "/Volumes/Tom_Data"  # dataserver_files/Group_Bausch/Tom_Dataserver/20260407
NOTEBOOK_VARIATION: str | None = None
NOTEBOOK_FEATURE_COMMANDS = [
    "beads:compute=1,plot=1,overwrite=1",
    "autocorr:compute=1,plot=1,overwrite=1",
    "image_corr:compute=0,plot=0,overwrite=0",
    "vector_corr:compute=1,plot=1,overwrite=1",
    "summary:compute=1,plot=1,overwrite=1",
]
NOTEBOOK_ENABLE: list[str] = []
NOTEBOOK_DISABLE: list[str] = []
NOTEBOOK_OVERWRITE: list[str] = []
NOTEBOOK_RUN_ORDER: list[str] | None = None

# Short human-readable descriptions for each feature used in the printed
# summary. These make it explicit what each feature computes and plots.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "beads": "Detects and links beads, computes velocities and optional angular speed; can also render displacement overlay movies and velocity-over-time plots.",
    "autocorr": "Computes FFT-based 2D/3D autocorrelations and radial averages; saves autocorrelation plots.",
    "image_corr": "Computes time-image correlations and fits; saves raw and fitted correlation plots.",
    "vector_corr": "Computes temporal, spatial, and tensor bead-motion vector correlations; can optionally reject Westerweel-Scarano outliers before plotting.",
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


def _finite_limits(values: Sequence[np.ndarray | pd.Series | list[float]], *, pad_fraction: float = 0.06) -> tuple[float, float] | None:
    finite_parts = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size:
            finite_parts.append(array)
    if not finite_parts:
        return None

    combined = np.concatenate(finite_parts)
    lower = float(np.nanmin(combined))
    upper = float(np.nanmax(combined))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower == upper:
        margin = 1.0 if lower == 0.0 else abs(lower) * 0.1
        return lower - margin, upper + margin
    margin = (upper - lower) * float(pad_fraction)
    return lower - margin, upper + margin


def _range_from_config(min_value: Any, max_value: Any) -> tuple[float | None, float | None] | None:
    lower = None if min_value is None else float(min_value)
    upper = None if max_value is None else float(max_value)
    if lower is None and upper is None:
        return None
    return lower, upper


def _result_frame(result: dict[str, Any], *keys: str) -> pd.DataFrame:
    for key in keys:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            return value
    return pd.DataFrame()


def _exp_decay(x: np.ndarray, amplitude: float, xi: float, offset: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xi = max(float(xi), 1e-12)
    return float(amplitude) * np.exp(-np.clip(x, 0.0, None) / xi) + float(offset)


def _fit_decay_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return None, None

    x_fit = x[mask]
    y_fit = y[mask]
    positive = x_fit > 0.0
    if int(positive.sum()) >= 3:
        x_fit = x_fit[positive]
        y_fit = y_fit[positive]
    if x_fit.size < 4:
        return None, None

    amplitude_guess = max(0.05, float(np.nanmax(y_fit) - np.nanmin(y_fit)))
    xi_guess = max(1e-3, float(np.nanmedian(x_fit)))
    offset_guess = float(np.nanmin(y_fit))

    try:
        popt, _ = curve_fit(
            _exp_decay,
            x_fit,
            y_fit,
            p0=[amplitude_guess, xi_guess, offset_guess],
            bounds=([0.0, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        return popt, float(popt[1])
    except Exception:
        return None, None


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
) -> None:
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    group_cols = [column for column in ("component", "channel", "projection", "frame") if column in df.columns]
    grouped = list(df.groupby(group_cols, sort=True)) if group_cols else [("all", df)]

    cmap = plt.get_cmap("viridis")
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
        x_series.append(x)
        y_series.append(y)
        group_label = _format_group_label(group_cols, group_key)
        ax.plot(x, y, lw=1.8, color=color, label=group_label)

        fit_column = "xi_um" if x_col == "r_um" else "tau_str_s" if x_col == "lag_s" else None
        if fit_column and fit_column in sub.columns:
            fit_values = sub[fit_column].to_numpy(dtype=float)
            fit_value = float(fit_values[np.isfinite(fit_values)][0]) if np.isfinite(fit_values).any() else np.nan
            if np.isfinite(fit_value) and fit_value > 0.0 and np.isfinite(x).any() and np.isfinite(y).any():
                fit_err_column = "xi_err_um" if fit_column == "xi_um" else "tau_err_s" if fit_column == "tau_str_s" else None
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
                    amplitude = max(0.05, float(np.nanmax(finite_y) - np.nanmin(finite_y)))
                    offset = float(np.nanmin(finite_y))
                    y_fit = _exp_decay(x_fit, amplitude, fit_value, offset)
                    if fit_column == "xi_um":
                        if fit_error is not None and np.isfinite(fit_error):
                            fit_label = rf"{group_label} fit ($\xi={fit_value:.2g}\pm{fit_error:.2g}\,\mu\mathrm{{m}}$)"
                        else:
                            fit_label = rf"{group_label} fit ($\xi={fit_value:.2g}\,\mu\mathrm{{m}}$)"
                    else:
                        if fit_error is not None and np.isfinite(fit_error):
                            fit_label = rf"{group_label} fit ($\tau={fit_value:.2g}\pm{fit_error:.2g}\,\mathrm{{s}}$)"
                        else:
                            fit_label = rf"{group_label} fit ($\tau={fit_value:.2g}\,\mathrm{{s}}$)"
                    ax.plot(x_fit, y_fit, ls="--", lw=1.4, color=color, alpha=0.85, label=fit_label)

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


def _plot_image_correlation(ax: Axes, raw_df: pd.DataFrame, fit_df: pd.DataFrame, *, title: str | None = None) -> None:
    if raw_df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    group_cols = [column for column in ("channel", "projection") if column in raw_df.columns]
    grouped = list(raw_df.groupby(group_cols, sort=True)) if group_cols else [("all", raw_df)]

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
        ax.plot(x, y, lw=1.8, color=color, label=str(group_key))

        if "corr_sem" in sub.columns:
            yerr = sub["corr_sem"].to_numpy(dtype=float)
            if np.any(np.isfinite(yerr)):
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=color)

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
                    ax.plot(dt, fit, ls="--", lw=1.4, color=color, alpha=0.8, label=f"{group_key} fit (τ={tau:.2g} s, β={beta:.2g})")

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
        if name == "beads":
            return self._run_beads()
        if name == "autocorr":
            return self._run_autocorr()
        if name == "image_corr":
            return self._run_image_corr()
        if name == "vector_corr":
            return self._run_vector_corr()
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
        preview_summary, preview_stats, fig_preview, _ = preview_bead_detection(state, beads_cfg, show=True)
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
                    _grouped_line_plot(ax, df, x_col, title=title, x_range=plot_range)

                    def _white(ax_white: Axes, table=df, col=x_col) -> None:
                        _grouped_line_plot(ax_white, table, col, title=None, x_range=plot_range)

                    save_comparison_dual_pdf(fig, plot_dir, f"{state['dataset_id']}_{stem}", white_plot_fn=_white)
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
            result = {
                "temporal_vector_corr_df": _read_parquet(derived_dir / temporal_name),
                "spatial_vector_corr_df": _read_parquet(derived_dir / spatial_name),
                "tensor_vector_corr_df": _read_parquet(derived_dir / tensor_name),
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

        if switch.plot:
            if not temporal_df.empty:
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

            if not spatial_df.empty:
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

            if not tensor_df.empty:
                tensor_plot_range = _range_from_config(
                    vector_cfg.get("tensor_plot_distance_um_min"),
                    vector_cfg.get("tensor_plot_distance_um_max"),
                )
                for part, part_title, stem in (
                    ("full", "Spatial vector tensor correlation", "vector_tensor_correlation_full"),
                    ("symmetric", "Symmetric tensor correlation", "vector_tensor_correlation_symmetric"),
                    ("antisymmetric", "Antisymmetric tensor correlation", "vector_tensor_correlation_antisymmetric"),
                ):
                    with comparison_style_context("dark"):
                        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
                        plot_vector_tensor_correlation(
                            ax,
                            tensor_df,
                            part=part,
                            title=part_title,
                            x_range=tensor_plot_range,
                        )

                        def _white_tensor(ax_white: Axes, df=tensor_df, tensor_part=part) -> None:
                            plot_vector_tensor_correlation(
                                ax_white,
                                df,
                                part=tensor_part,
                                title=None,
                                x_range=tensor_plot_range,
                            )

                        tensor_stem = build_velocity_artifact_stem(f"{dataset_id}_{stem}", vector_cfg, distance_mode=tensor_distance_mode)
                        save_vector_correlation_dual_pdf(fig, plot_dir, tensor_stem, white_plot_fn=_white_tensor)
                        print(f"Saved tensor vector correlation plots to {plot_dir} ({part})")
                        plt.close(fig)

                if bool(vector_cfg.get("tensor_fit_enabled", False)):
                    tensor_fit_pairs = [str(pair).strip() for pair in vector_cfg.get("tensor_fit_component_pairs", []) if str(pair).strip()]
                    tensor_fit_parts = [str(part).strip() for part in vector_cfg.get("tensor_fit_parts", ["full"]) if str(part).strip()]
                    tensor_fit_range = _range_from_config(
                        vector_cfg.get("tensor_fit_distance_um_min"),
                        vector_cfg.get("tensor_fit_distance_um_max"),
                    )
                    tensor_fit_min_points = int(vector_cfg.get("tensor_fit_min_points", 4))
                    if tensor_fit_pairs:
                        for part in tensor_fit_parts:
                            with comparison_style_context("dark"):
                                fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=150)
                                plot_vector_tensor_pair_decay(
                                    ax,
                                    tensor_df,
                                    part=part,
                                    component_pairs=tensor_fit_pairs,
                                    title=f"Tensor component-pair decay fits ({part})",
                                    x_range=tensor_plot_range,
                                    fit_range=tensor_fit_range,
                                    min_points=tensor_fit_min_points,
                                )

                                def _white_tensor_fit(ax_white: Axes, df=tensor_df, tensor_part=part, pairs=tensor_fit_pairs) -> None:
                                    plot_vector_tensor_pair_decay(
                                        ax_white,
                                        df,
                                        part=tensor_part,
                                        component_pairs=pairs,
                                        title=None,
                                        x_range=tensor_plot_range,
                                        fit_range=tensor_fit_range,
                                        min_points=tensor_fit_min_points,
                                    )

                                fit_stem = build_velocity_artifact_stem(f"{dataset_id}_tensor_component_pair_fits_{part}", vector_cfg)
                                save_vector_correlation_dual_pdf(fig, plot_dir, fit_stem, white_plot_fn=_white_tensor_fit)
                                print(f"Saved tensor pair-fit plots to {plot_dir} ({part})")
                                plt.close(fig)

        self.outputs["vector_corr"] = result
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
            parsed[feature_name][key.strip()] = raw_value.strip().lower() in {"1", "true", "yes", "on"}
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


def main(argv: list[str] | None = None) -> AnalysisNotebookRunner:
    parser = argparse.ArgumentParser(description="Notebook-like orchestration for analysis_unified")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--variation", type=str, default=None)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--enable", nargs="*", default=[])
    parser.add_argument("--disable", nargs="*", default=[])
    parser.add_argument("--overwrite", nargs="*", default=[])
    parser.add_argument("--run", nargs="*", default=None)
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

    runner = load_default_runner(config_path=args.config)
    _apply_notebook_commands(
        runner,
        dataset_id=args.dataset_id if args.dataset_id is not None else args.notebook_dataset_id if args.notebook_dataset_id is not None else NOTEBOOK_DATASET_ID,
        variation=args.variation if args.variation is not None else args.notebook_variation if args.notebook_variation is not None else NOTEBOOK_VARIATION,
        base_dir=args.base_dir if args.base_dir is not None else args.notebook_base_dir if args.notebook_base_dir is not None else NOTEBOOK_BASE_DIR,
        feature_commands=[*NOTEBOOK_FEATURE_COMMANDS, *args.notebook_feature_commands, *args.feature],
        enable=[*NOTEBOOK_ENABLE, *args.notebook_enable, *args.enable],
        disable=[*NOTEBOOK_DISABLE, *args.notebook_disable, *args.disable],
        overwrite=[*NOTEBOOK_OVERWRITE, *args.notebook_overwrite, *args.overwrite],
    )

    feature_order = args.run if args.run is not None else args.notebook_run_order if args.notebook_run_order is not None else NOTEBOOK_RUN_ORDER
    runner.run_all(feature_order=feature_order)
    return runner


if __name__ == "__main__":
    main()