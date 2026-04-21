from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .comparison import comparison_style_context, save_comparison_dual_pdf


def _component_order(component: str) -> int:
    order = {"vector": 0, "x": 1, "y": 2, "z": 3, "r": 1, "theta": 2, "phi": 3}
    return order.get(str(component).lower(), 99)


def _exp_decay(x: np.ndarray, amplitude: float, xi: float, offset: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xi = max(float(xi), 1e-12)
    return float(amplitude) * np.exp(-np.clip(x, 0.0, None) / xi) + float(offset)


def _finite_limits(values: list[np.ndarray], *, pad_fraction: float = 0.06) -> tuple[float, float] | None:
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


def _fit_component_decay(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, float | None, float | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return None, None, None

    x_fit = x[mask]
    y_fit = y[mask]
    positive = x_fit > 0.0
    if int(positive.sum()) >= 3:
        x_fit = x_fit[positive]
        y_fit = y_fit[positive]
    if x_fit.size < 4:
        return None, None, None

    amplitude_guess = max(0.05, float(np.nanmax(y_fit) - np.nanmin(y_fit)))
    xi_guess = max(1e-3, float(np.nanmedian(x_fit)))
    offset_guess = float(np.nanmin(y_fit))

    try:
        popt, pcov = curve_fit(
            _exp_decay,
            x_fit,
            y_fit,
            p0=[amplitude_guess, xi_guess, offset_guess],
            bounds=([0.0, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full(popt.shape, np.nan)
        return popt, float(popt[1]), float(perr[1]) if perr.size > 1 else np.nan
    except Exception:
        return None, None, None


def _decay_symbol_and_unit(x_col: str) -> tuple[str, str]:
    if x_col == "lag_s":
        return r"\tau", r"\mathrm{s}"
    return r"\xi", r"\mu\mathrm{m}"


def _format_decay_annotation(symbol: str, value: float, uncertainty: float | None, unit: str) -> str:
    if not np.isfinite(value):
        return rf"${symbol}=\mathrm{{nan}}\,{unit}$"

    if uncertainty is not None and np.isfinite(uncertainty) and uncertainty > 0:
        decimals = max(0, int(max(0, -np.floor(np.log10(abs(float(uncertainty)))) + 1)))
    else:
        decimals = 3

    value_str = f"{float(value):.{decimals}f}"
    if uncertainty is not None and np.isfinite(uncertainty):
        uncertainty_str = f"{float(uncertainty):.{decimals}f}"
        return rf"${symbol}={value_str}\pm{uncertainty_str}\,{unit}$"
    return rf"${symbol}={value_str}\,{unit}$"


def _score_column(df: pd.DataFrame) -> str:
    return "score" if "score" in df.columns else "corr"


def _scatter_limits(values: list[np.ndarray], *, pad_fraction: float = 0.06) -> tuple[float, float] | None:
    return _finite_limits(values, pad_fraction=pad_fraction)


def _mean_curve(x: np.ndarray, y: np.ndarray, *, bin_count: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return None

    if bin_count is None or int(bin_count) <= 0 or x.size <= int(bin_count):
        grouped = pd.DataFrame({"x": x, "y": y}).groupby("x", sort=True, as_index=False).mean(numeric_only=True)
        if grouped.empty:
            return None
        return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)

    bins = max(2, int(bin_count))
    edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), bins + 1)
    centers: list[float] = []
    means: list[float] = []
    for index in range(bins):
        if index == bins - 1:
            mask_bin = (x >= edges[index]) & (x <= edges[index + 1])
        else:
            mask_bin = (x >= edges[index]) & (x < edges[index + 1])
        if not np.any(mask_bin):
            continue
        centers.append(float(np.nanmean(x[mask_bin])))
        means.append(float(np.nanmean(y[mask_bin])))
    if not centers:
        return None
    return np.asarray(centers, dtype=float), np.asarray(means, dtype=float)


def _downsample_scatter_points(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None:
        return x, y

    try:
        limit = int(max_points)
    except Exception:
        return x, y

    if limit <= 0 or x.size <= limit:
        return x, y

    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]
    sample_idx = np.linspace(0, x_sorted.size - 1, limit).astype(int)
    return x_sorted[sample_idx], y_sorted[sample_idx]


def _fit_temporal_decay(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray | None, float | None]:
    grouped = df[[x_col, y_col]].dropna().groupby(x_col, as_index=False).mean(numeric_only=True)
    if grouped.empty:
        return None, None
    popt, xi, _ = _fit_component_decay(grouped[x_col].to_numpy(dtype=float), grouped[y_col].to_numpy(dtype=float))
    return popt, xi


def _fit_temporal_decay_with_error(df: pd.DataFrame, x_col: str, y_col: str, *, fit_range: tuple[float | None, float | None] | None = None) -> tuple[np.ndarray | None, float | None, float | None]:
    grouped = df[[x_col, y_col]].dropna().groupby(x_col, as_index=False).mean(numeric_only=True)
    if grouped.empty:
        return None, None, None

    if fit_range is not None:
        lower, upper = fit_range
        mask = np.ones(len(grouped), dtype=bool)
        if lower is not None:
            mask &= grouped[x_col].to_numpy(dtype=float) >= float(lower)
        if upper is not None:
            mask &= grouped[x_col].to_numpy(dtype=float) <= float(upper)
        grouped = grouped.loc[mask]
        if grouped.empty:
            return None, None, None

    return _fit_component_decay(grouped[x_col].to_numpy(dtype=float), grouped[y_col].to_numpy(dtype=float))


def _fit_signed_decay(
    x: np.ndarray,
    y: np.ndarray,
    *,
    yerr: np.ndarray | None = None,
    fit_range: tuple[float | None, float | None] | None = None,
    min_points: int = 4,
) -> tuple[np.ndarray | None, float | None, float | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if fit_range is not None:
        lower, upper = fit_range
        if lower is not None:
            mask &= x >= float(lower)
        if upper is not None:
            mask &= x <= float(upper)

    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        mask &= np.isfinite(yerr)

    if int(mask.sum()) < int(min_points):
        return None, None, None

    x_fit = x[mask]
    y_fit = y[mask]
    sigma = None
    if yerr is not None:
        sigma = np.clip(yerr[mask], 1e-12, None)

    offset_guess = float(np.nanmean(y_fit[-max(3, y_fit.size // 5) :])) if y_fit.size else 0.0
    if not np.isfinite(offset_guess):
        offset_guess = 0.0
    amplitude_guess = float(y_fit[0] - offset_guess) if y_fit.size else 1.0
    if not np.isfinite(amplitude_guess) or amplitude_guess == 0.0:
        centered = y_fit - offset_guess
        amplitude_guess = float(centered[np.nanargmax(np.abs(centered))]) if centered.size else 1.0
    xi_guess = max(1e-3, float(np.nanmedian(x_fit)))

    try:
        popt, pcov = curve_fit(
            _exp_decay,
            x_fit,
            y_fit,
            p0=[amplitude_guess, xi_guess, offset_guess],
            bounds=([-np.inf, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
            sigma=sigma,
            absolute_sigma=sigma is not None,
            maxfev=20000,
        )
        perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full(popt.shape, np.nan)
        return popt, float(popt[1]), float(perr[1]) if perr.size > 1 else np.nan
    except Exception:
        return None, None, None


def _plot_scatter(
    ax,
    df: pd.DataFrame,
    x_col: str,
    *,
    title: str | None = None,
    fit: bool = False,
    max_points: int | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    fit_range: tuple[float | None, float | None] | None = None,
    show_mean_line: bool = False,
    mean_bin_count: int | None = None,
) -> None:
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    y_col = _score_column(df)
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if x_range is not None:
        lower, upper = x_range
        mask = np.isfinite(x) & np.isfinite(y)
        if lower is not None:
            mask &= x >= float(lower)
        if upper is not None:
            mask &= x <= float(upper)
        x = x[mask]
        y = y[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        return

    x_plot, y_plot = _downsample_scatter_points(x, y, max_points)
    scatter_label = "binned mean" if "corr_sem" in df.columns and np.isfinite(df["corr_sem"].to_numpy(dtype=float)).any() else "pairs"
    ax.scatter(x_plot, y_plot, s=16, alpha=0.55, color="#4C78A8", edgecolors="none", label=scatter_label)

    if "corr_sem" in df.columns:
        sem_df = df[[x_col, y_col, "corr_sem"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, "corr_sem"])
        if not sem_df.empty:
            sem_df = sem_df.groupby(x_col, as_index=False).mean(numeric_only=True)
            if not sem_df.empty:
                ax.fill_between(
                    sem_df[x_col].to_numpy(dtype=float),
                    sem_df[y_col].to_numpy(dtype=float) - sem_df["corr_sem"].to_numpy(dtype=float),
                    sem_df[y_col].to_numpy(dtype=float) + sem_df["corr_sem"].to_numpy(dtype=float),
                    alpha=0.15,
                    color="#4C78A8",
                    label="mean ± sem",
                )

    if show_mean_line:
        mean_curve = _mean_curve(x, y, bin_count=mean_bin_count)
        if mean_curve is not None:
            mean_x, mean_y = mean_curve
            ax.plot(mean_x, mean_y, lw=2.1, color="#54A24B", label="mean")

    fit_line = None
    xi = None
    xi_err = None
    if fit:
        popt, xi, xi_err = _fit_temporal_decay_with_error(df, x_col, y_col, fit_range=fit_range)

        if popt is not None and xi is not None:
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            if fit_range is not None:
                lower, upper = fit_range
                if lower is not None:
                    x_min = max(x_min, float(lower))
                if upper is not None:
                    x_max = min(x_max, float(upper))
            x_fit = np.linspace(x_min, x_max, 250)
            fit_line = _exp_decay(x_fit, *popt)
            symbol, unit = _decay_symbol_and_unit(x_col)
            if xi_err is not None and np.isfinite(xi_err):
                label = rf"fit (${symbol}={xi:.4g}\pm{xi_err:.4g}\,{unit}$)"
            else:
                label = rf"fit (${symbol}={xi:.4g}\,{unit}$)"
            ax.plot(x_fit, fit_line, ls="--", lw=1.7, color="#F58518", label=label)

    if title:
        ax.set_title(title)
    ax.legend(frameon=True)

    x_limits = _scatter_limits([x])
    y_limits = _scatter_limits([y])
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _plot_by_component(ax, df: pd.DataFrame, x_col: str, *, title: str | None = None) -> None:
    if df.empty:
        ax.set_title(title or "")
        return

    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []
    for component in sorted(df["component"].dropna().unique().tolist(), key=_component_order):
        sub = df.loc[df["component"] == component].sort_values(x_col)
        if sub.empty:
            continue
        x = sub[x_col].to_numpy(dtype=float)
        y = sub["corr"].to_numpy(dtype=float)
        x_series.append(x)
        y_series.append(y)
        yerr = sub["corr_sem"].to_numpy(dtype=float) if "corr_sem" in sub.columns else None
        ax.plot(x, y, lw=2.0, label=str(component))
        if yerr is not None and np.any(np.isfinite(yerr)):
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)

        fit_x = x_col in {"lag_s", "r_um"}
        if fit_x:
            popt, xi, xi_err = _fit_component_decay(x, y)
            if popt is not None and xi is not None:
                x_fit = np.linspace(float(np.nanmin(x[np.isfinite(x)])), float(np.nanmax(x[np.isfinite(x)])), 250)
                y_fit = _exp_decay(x_fit, *popt)
                symbol, unit = _decay_symbol_and_unit(x_col)
                if xi_err is not None and np.isfinite(xi_err):
                    fit_label = rf"{component} fit (${symbol}={xi:.3g}\pm{xi_err:.3g}\,{unit}$)"
                else:
                    fit_label = rf"{component} fit (${symbol}={xi:.3g}\,{unit}$)"
                ax.plot(x_fit, y_fit, ls="--", lw=1.5, label=fit_label)

    if title:
        ax.set_title(title)
    ax.legend(frameon=True)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def plot_temporal_vector_correlation(
    ax,
    df: pd.DataFrame,
    title: str | None = None,
    *,
    max_points: int | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    fit_range: tuple[float | None, float | None] | None = None,
) -> None:
    _plot_scatter(
        ax,
        df,
        "lag_s",
        title=title,
        fit=True,
        max_points=max_points,
        x_range=x_range,
        fit_range=fit_range,
        show_mean_line=True,
        mean_bin_count=None,
    )
    ax.set_xlabel(r"lag ($\mathrm{s}$)")
    ax.set_ylabel(r"score = $\hat{v}_i \cdot \hat{v}_j$")
    ax.grid(True, alpha=0.25)


def plot_spatial_vector_correlation(
    ax,
    df: pd.DataFrame,
    title: str | None = None,
    *,
    x_range: tuple[float | None, float | None] | None = None,
    mean_bin_count: int | None = None,
) -> None:
    x_col = "distance_um" if "distance_um" in df.columns else "r_um"
    _plot_scatter(
        ax,
        df,
        x_col,
        title=title,
        fit=False,
        x_range=x_range,
        show_mean_line=True,
        mean_bin_count=mean_bin_count,
    )
    ax.set_xlabel(r"distance ($\mu\mathrm{m}$)")
    ax.set_ylabel(r"score = $\hat{v}_i \cdot \hat{v}_j$")
    ax.grid(True, alpha=0.25)


def _tensor_component_sequence(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    row_components = df.get("row_component", pd.Series(dtype=object)).dropna().astype(str).tolist()
    col_components = df.get("col_component", pd.Series(dtype=object)).dropna().astype(str).tolist()
    components = sorted(set(row_components + col_components), key=_component_order)
    return components


def _tensor_pair_order(df: pd.DataFrame) -> list[str]:
    components = _tensor_component_sequence(df)
    return [f"{row}{col}" for row in components for col in components]


def _tensor_basis_from_df(df: pd.DataFrame) -> str:
    if "tensor_basis" not in df.columns:
        return "cartesian"
    basis_values = df["tensor_basis"].dropna().astype(str).str.strip().str.lower().tolist()
    if not basis_values:
        return "cartesian"
    if "spherical" in basis_values:
        return "spherical"
    return basis_values[0]


def _tensor_component_symbol(component: str, tensor_basis: str | None) -> str:
    basis = str(tensor_basis or "cartesian").strip().lower()
    component = str(component).strip()
    if basis != "spherical":
        return component
    return {
        "r": "r",
        "theta": r"\theta",
        "phi": r"\phi",
    }.get(component.lower(), component)


def _tensor_component_display(component: str, tensor_basis: str | None) -> str:
    basis = str(tensor_basis or "cartesian").strip().lower()
    if basis != "spherical":
        return str(component)
    return rf"${_tensor_component_symbol(component, basis)}$"


def _tensor_component_pair_display(row_component: str, col_component: str, tensor_basis: str | None) -> str:
    basis = str(tensor_basis or "cartesian").strip().lower()
    if basis != "spherical":
        return f"{row_component}{col_component}"
    return rf"${_tensor_component_symbol(row_component, basis)}\,{_tensor_component_symbol(col_component, basis)}$"


def _tensor_component_pair_display_map(df: pd.DataFrame) -> dict[str, str]:
    if not {"row_component", "col_component"}.issubset(df.columns):
        return {}

    basis = _tensor_basis_from_df(df)
    display_map: dict[str, str] = {}
    for row_component in _tensor_component_sequence(df):
        for col_component in _tensor_component_sequence(df):
            pair_name = f"{row_component}{col_component}"
            display_map[pair_name] = _tensor_component_pair_display(row_component, col_component, basis)
    return display_map


def plot_vector_tensor_correlation(
    ax,
    df: pd.DataFrame,
    *,
    part: str = "full",
    title: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
) -> None:
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    sub = df.copy()
    if "part" in sub.columns:
        sub = sub.loc[sub["part"].astype(str) == str(part)]
    if sub.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, f"no {part} data", ha="center", va="center", transform=ax.transAxes)
        return

    if x_range is not None:
        lower, upper = x_range
        mask = np.isfinite(sub["distance_um"].to_numpy(dtype=float)) & np.isfinite(sub["corr"].to_numpy(dtype=float))
        if lower is not None:
            mask &= sub["distance_um"].to_numpy(dtype=float) >= float(lower)
        if upper is not None:
            mask &= sub["distance_um"].to_numpy(dtype=float) <= float(upper)
        sub = sub.loc[mask]
    if sub.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        return

    if "component_pair" not in sub.columns:
        if {"row_component", "col_component"}.issubset(sub.columns):
            sub = sub.assign(component_pair=sub["row_component"].astype(str) + sub["col_component"].astype(str))
        else:
            raise ValueError("tensor dataframe must contain component_pair or row_component/col_component")

    pair_display_map = _tensor_component_pair_display_map(sub)

    pivot = sub.pivot_table(index="component_pair", columns="distance_um", values="corr", aggfunc="mean")
    tensor_pair_order = _tensor_pair_order(sub)
    pivot = pivot.reindex(tensor_pair_order)
    if pivot.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no tensor grid", ha="center", va="center", transform=ax.transAxes)
        return

    dist_vals = pivot.columns.to_numpy(dtype=float)
    valid_dist = np.isfinite(dist_vals)
    dist_vals = dist_vals[valid_dist]
    pivot = pivot.loc[:, valid_dist]
    if dist_vals.size == 0:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no distance bins", ha="center", va="center", transform=ax.transAxes)
        return

    data = pivot.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size:
        limit = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
    else:
        limit = 1.0

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        extent=[float(np.nanmin(dist_vals)), float(np.nanmax(dist_vals)), -0.5, len(tensor_pair_order) - 0.5],
    )
    ax.set_yticks(np.arange(len(tensor_pair_order)))
    ax.set_yticklabels([pair_display_map.get(pair, pair) for pair in tensor_pair_order])
    tick_count = min(6, len(dist_vals))
    tick_positions = np.linspace(float(np.nanmin(dist_vals)), float(np.nanmax(dist_vals)), tick_count)
    ax.set_xticks(tick_positions)
    ax.set_xlabel(r"distance ($\mu\mathrm{m}$)")
    ax.set_ylabel(r"component pair $R_{ij}$")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="correlation")


def plot_vector_tensor_pair_decay(
    ax,
    df: pd.DataFrame,
    *,
    part: str = "full",
    component_pairs: list[str] | None = None,
    title: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    fit_range: tuple[float | None, float | None] | None = None,
    min_points: int = 4,
    fit_enabled: bool = True,
) -> None:
    if df.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    sub = df.copy()
    if "part" in sub.columns:
        sub = sub.loc[sub["part"].astype(str) == str(part)]
    if sub.empty:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, f"no {part} data", ha="center", va="center", transform=ax.transAxes)
        return

    if "component_pair" not in sub.columns:
        if {"row_component", "col_component"}.issubset(sub.columns):
            sub = sub.assign(component_pair=sub["row_component"].astype(str) + sub["col_component"].astype(str))
        else:
            raise ValueError("tensor dataframe must contain component_pair or row_component/col_component")

    pair_display_map = _tensor_component_pair_display_map(sub)

    if component_pairs:
        requested_pairs = [str(pair).strip() for pair in component_pairs if str(pair).strip()]
        sub = sub.loc[sub["component_pair"].astype(str).isin(requested_pairs)]
        pair_order = requested_pairs
    else:
        pair_order = [pair for pair in _tensor_pair_order(sub) if pair in set(sub["component_pair"].astype(str))]

    if sub.empty or not pair_order:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no selected pairs", ha="center", va="center", transform=ax.transAxes)
        return

    cmap = plt.get_cmap("tab10")
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []

    for index, pair in enumerate(pair_order):
        pair_df = sub.loc[sub["component_pair"].astype(str) == pair].copy()
        if pair_df.empty:
            continue
        pair_df = pair_df.sort_values("distance_um")
        if x_range is not None:
            lower, upper = x_range
            mask = np.isfinite(pair_df["distance_um"].to_numpy(dtype=float)) & np.isfinite(pair_df["corr"].to_numpy(dtype=float))
            if lower is not None:
                mask &= pair_df["distance_um"].to_numpy(dtype=float) >= float(lower)
            if upper is not None:
                mask &= pair_df["distance_um"].to_numpy(dtype=float) <= float(upper)
            pair_df = pair_df.loc[mask]
        if pair_df.empty:
            continue

        grouped = pair_df.groupby("distance_um", as_index=False).mean(numeric_only=True)
        x = grouped["distance_um"].to_numpy(dtype=float)
        y = grouped["corr"].to_numpy(dtype=float)
        yerr = grouped["corr_sem"].to_numpy(dtype=float) if "corr_sem" in grouped.columns else None
        valid = np.isfinite(x) & np.isfinite(y)
        if yerr is not None:
            valid &= np.isfinite(yerr)
        x = x[valid]
        y = y[valid]
        if yerr is not None:
            yerr = yerr[valid]
        if x.size == 0:
            continue

        color = cmap(index % 10)
        x_series.append(x)
        y_series.append(y)

        display_pair = pair_display_map.get(pair, pair)
        data_label = display_pair
        symbol, unit = r"\xi", r"\mu\mathrm{m}"
        if yerr is not None and np.any(np.isfinite(yerr)):
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.12, color=color)

        if fit_enabled:
            popt, xi, xi_err = _fit_signed_decay(x, y, yerr=yerr, fit_range=fit_range or x_range, min_points=min_points)
            if popt is not None and xi is not None:
                x_min = float(np.nanmin(x))
                x_max = float(np.nanmax(x))
                if fit_range is not None:
                    lower, upper = fit_range
                    if lower is not None:
                        x_min = max(x_min, float(lower))
                    if upper is not None:
                        x_max = min(x_max, float(upper))
                elif x_range is not None:
                    lower, upper = x_range
                    if lower is not None:
                        x_min = max(x_min, float(lower))
                    if upper is not None:
                        x_max = min(x_max, float(upper))
                if x_max > x_min:
                    x_fit = np.linspace(x_min, x_max, 250)
                    y_fit = _exp_decay(x_fit, *popt)
                    data_label = f"{display_pair} ({_format_decay_annotation(symbol, xi, xi_err, unit)})"
                    ax.plot(x_fit, y_fit, ls="--", lw=1.5, color=color, alpha=0.85, label="_nolegend_")

        ax.plot(x, y, lw=1.8, color=color, label=data_label)

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"distance ($\mu\mathrm{m}$)")
    ax.set_ylabel(r"tensor correlation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    x_limits = _finite_limits(x_series)
    y_limits = _finite_limits(y_series)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def save_vector_correlation_dual_pdf(
    fig,
    output_dir: str | Path,
    stem: str,
    *,
    white_plot_fn: Callable[[Any], None] | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    return save_comparison_dual_pdf(fig, output_dir, stem, white_plot_fn=white_plot_fn, dpi=dpi)


def render_vector_correlation_summary(
    df: pd.DataFrame,
    x_col: str,
    *,
    title: str | None = None,
    max_points: int | None = None,
    figsize: tuple[float, float] = (7.2, 4.6),
):
    with comparison_style_context("dark"):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        _plot_scatter(
            ax,
            df,
            x_col,
            title=title,
            fit=x_col == "lag_s",
            max_points=max_points if x_col == "lag_s" else None,
            show_mean_line=True,
            mean_bin_count=None if x_col == "lag_s" else 30,
        )
        ax.set_xlabel(r"lag ($\mathrm{s}$)" if x_col == "lag_s" else r"distance ($\mu\mathrm{m}$)")
        ax.set_ylabel(r"score = $\hat{v}_i \cdot \hat{v}_j$")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
    return fig, ax