from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from analysis_pipeline.comparison import comparison_palette, comparison_style_context, save_comparison_dual_pdf
from analysis_unified import _sampled_autocorr_length_time_series


def _derived_dir(base_dir: str | Path, dataset_id: str) -> Path:
    return Path(base_dir) / dataset_id / "derived"


def _load_sampled_length_table(base_dir: str | Path, dataset_id: str, *, key: str) -> pd.DataFrame:
    filename_map = {
        "sampled3d_df": "autocorr_3d_sampled.parquet",
        # "sampled2d_df": "autocorr_2d_sampled.parquet",
    }
    path = _derived_dir(base_dir, dataset_id) / filename_map[key]
    if not path.exists():
        return pd.DataFrame()

    raw_df = pd.read_parquet(path)
    summary = _sampled_autocorr_length_time_series(raw_df)
    if summary.empty:
        return summary

    summary = summary.copy()
    summary["time_min"] = summary["time_s"].astype(float) / 60.0
    summary["dataset_id"] = dataset_id
    summary["dataset_label"] = dataset_id
    return summary


def _finite_limits(values: list[np.ndarray], *, pad_fraction: float = 0.06) -> tuple[float, float] | None:
    finite_parts: list[np.ndarray] = []
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


def _finite_positive_limits(values: list[np.ndarray], *, pad_fraction: float = 0.06) -> tuple[float, float] | None:
    limits = _finite_limits(values, pad_fraction=pad_fraction)
    if limits is None:
        return None
    lower, upper = limits
    lower = max(lower, np.finfo(float).tiny)
    if lower >= upper:
        upper = lower * 10.0
    return lower, upper


def _second_positive_value(values: list[np.ndarray]) -> float | None:
    finite_parts: list[np.ndarray] = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        array = array[np.isfinite(array) & (array > 0)]
        if array.size:
            finite_parts.append(array)
    if not finite_parts:
        return None

    combined = np.unique(np.sort(np.concatenate(finite_parts)))
    if combined.size >= 2:
        return float(combined[1])
    return float(combined[0])


def _merge_two_series(first: pd.DataFrame, second: pd.DataFrame, *, gap_minutes: float = 30.0) -> pd.DataFrame:
    if first.empty and second.empty:
        return pd.DataFrame()
    if first.empty:
        return second.copy()
    if second.empty:
        return first.copy()

    first = first.copy().sort_values("time_min")
    second = second.copy().sort_values("time_min")

    first_end = float(np.nanmax(first["time_min"].to_numpy(dtype=float)))
    second = second.copy()
    second["time_min"] = second["time_min"].astype(float) + first_end + float(gap_minutes)
    return pd.concat([first, second], ignore_index=True).sort_values(["dataset_order", "time_min"]).reset_index(drop=True)


def _plot_merged_series(output_dir: Path, stem: str, table: pd.DataFrame, title: str) -> None:
    if table.empty:
        print(f"Skipping {stem}: no data")
        return

    def _plot(ax_target: Axes, table_df: pd.DataFrame) -> None:
        x_series: list[np.ndarray] = []
        y_series: list[np.ndarray] = []
        for dataset_order, dataset_df in table_df.groupby("dataset_order", sort=True):
            dataset_df = dataset_df.sort_values("time_min")
            if dataset_df.empty:
                continue
            color = str(dataset_df["dataset_color"].iloc[0]) if "dataset_color" in dataset_df.columns else None
            label = str(dataset_df["legend_label"].iloc[0]) if "legend_label" in dataset_df.columns else str(dataset_order)
            x = dataset_df["time_min"].to_numpy(dtype=float)
            y = dataset_df["xi_um"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if not np.any(mask):
                continue
            x = x[mask]
            y = y[mask]
            yerr = None
            if "xi_err_um" in dataset_df.columns:
                yerr_raw = dataset_df["xi_err_um"].to_numpy(dtype=float)[mask]
                if np.any(np.isfinite(yerr_raw)):
                    yerr = np.where(np.isfinite(yerr_raw), yerr_raw, 0.0)
            x = np.clip(x, np.finfo(float).tiny, None)
            if yerr is not None:
                yerr = np.maximum(yerr, 0.0)
                ax_target.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="o",
                    linestyle="none",
                    markersize=4.5,
                    capsize=3,
                    color=color,
                    label=label,
                    alpha=0.9,
                )
            else:
                ax_target.plot(x, y, "o", markersize=4.5, color=color, label=label, alpha=0.9)
            x_series.append(x)
            y_series.append(y)

        ax_target.set_title(title)
        ax_target.set_xlabel("time (min)")
        ax_target.set_ylabel(r"autocorr length ($\mu\mathrm{m}$)")
        ax_target.set_xscale("log")
        ax_target.grid(True, alpha=0.25)
        ax_target.legend(frameon=True)

        x_limits = _finite_positive_limits(x_series)
        y_limits = _finite_limits(y_series)
        if x_limits is not None:
            left_x = _second_positive_value(x_series)
            ax_target.set_xlim(left_x if left_x is not None else x_limits[0], x_limits[1])
        if y_limits is not None:
            ax_target.set_ylim(*y_limits)

    with comparison_style_context("dark"):
        fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
        _plot(ax, table)

        def _white(ax_white):
            _plot(ax_white, table)

        save_comparison_dual_pdf(fig, output_dir, stem, white_plot_fn=_white)
        plt.close(fig)
        print(f"Saved {stem}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge autocorrelation length-over-time plots for two consecutive acquisitions.")
    parser.add_argument("--base-dir", default="data", help="Base directory containing the dataset folders")
    parser.add_argument("--dataset-a", default="AMF_108_002__C640_C470", help="First dataset id")
    parser.add_argument("--dataset-b", default="AMF_108_003__C640_C470", help="Second dataset id")
    parser.add_argument("--output-root", default="plots/merged_autocorr", help="Directory for the merged output PDFs")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    color = comparison_palette("tab10", 1)[0]
    for key, title in (("sampled3d_df", "3D autocorrelation length over time"),):
        table_a = _load_sampled_length_table(base_dir, args.dataset_a, key=key)
        table_b = _load_sampled_length_table(base_dir, args.dataset_b, key=key)
        if table_a.empty and table_b.empty:
            print(f"Skipping {key}: no data in either dataset")
            continue

        table_a = table_a.copy()
        table_b = table_b.copy()
        if not table_a.empty:
            table_a["dataset_order"] = 0
            table_a["dataset_color"] = color
            table_a["legend_label"] = "sampled autocorr length"
        if not table_b.empty:
            table_b["dataset_order"] = 1
            table_b["dataset_color"] = color
            table_b["legend_label"] = "_nolegend_"

        merged = _merge_two_series(table_a, table_b)
        stem = f"{args.dataset_a}_to_{args.dataset_b}_{key}_autocorr_length_over_time_logx"
        _plot_merged_series(output_root, stem, merged, f"{title} merged")


if __name__ == "__main__":
    main()
