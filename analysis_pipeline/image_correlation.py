from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm


RAW_CORRELATION_FILENAME = "image_time_correlation_raw.parquet"
FIT_CORRELATION_FILENAME = "image_time_correlation_fit.parquet"
LEGACY_CORRELATION_FILENAME = "image_time_correlation.parquet"
FIT_RESULT_COLUMNS = (
    "tau_str_s",
    "tau_str_err_s",
    "beta",
    "beta_err",
    "fit_r2",
    "fit_points",
)


def _stretched_exponential(dt_s: np.ndarray, tau_str_s: float, beta: float) -> np.ndarray:
    dt_s = np.asarray(dt_s, dtype=float)
    tau_str_s = max(float(tau_str_s), 1e-12)
    beta = max(float(beta), 1e-12)
    return np.exp(-np.power(np.clip(dt_s, 0.0, None) / tau_str_s, beta))


def _resolve_channel(image_corr_cfg: Dict[str, Any]) -> int:
    if "channel" in image_corr_cfg:
        return int(image_corr_cfg["channel"])
    if "channel_2d" in image_corr_cfg:
        return int(image_corr_cfg["channel_2d"])
    return int(image_corr_cfg.get("channel_3d", 0))


def _resolve_z_mode(image_corr_cfg: Dict[str, Any]) -> Any:
    if "z_mode" in image_corr_cfg:
        return image_corr_cfg["z_mode"]
    return image_corr_cfg.get("middle_z_for_2d", "middle")


def _project_frame_to_2d(frame: np.ndarray, z_mode: Any) -> tuple[np.ndarray, int | None]:
    frame = np.asarray(frame, dtype=float)
    if frame.ndim == 2:
        return frame, None
    if frame.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D frame, got ndim={frame.ndim}")

    if isinstance(z_mode, str):
        mode = z_mode.strip().lower()
        if mode in {"middle", "mid", "center", "centre"}:
            z_idx = int(frame.shape[0] // 2)
            return frame[z_idx], z_idx
        if mode in {"mip", "max", "maximum"}:
            return np.nanmax(frame, axis=0), None
        if mode in {"mean", "avg", "average"}:
            return np.nanmean(frame, axis=0), None
        raise ValueError(f"Unsupported z_mode: {z_mode}")

    z_idx = int(z_mode)
    if z_idx < 0 or z_idx >= frame.shape[0]:
        raise ValueError(f"z_mode index out of range: {z_idx}")
    return frame[z_idx], z_idx


def _load_2d_frame(images, frame_idx: int, channel_idx: int, z_mode: Any) -> tuple[np.ndarray, int | None]:
    frame = images[int(frame_idx), int(channel_idx)].compute()
    frame = np.asarray(frame, dtype=float)
    return _project_frame_to_2d(frame, z_mode)


def _fit_stretched_exponential(dt_s: np.ndarray, corr: np.ndarray, n_pairs: np.ndarray) -> Dict[str, float]:
    dt_s = np.asarray(dt_s, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n_pairs = np.asarray(n_pairs, dtype=float)

    valid = np.isfinite(dt_s) & np.isfinite(corr) & (dt_s >= 0.0)
    if int(valid.sum()) < 3:
        return {
            "tau_str_s": np.nan,
            "tau_str_err_s": np.nan,
            "beta": np.nan,
            "beta_err": np.nan,
            "fit_r2": np.nan,
            "fit_points": int(valid.sum()),
        }

    fit_mask = valid & (corr > 0.0)
    if int(fit_mask.sum()) < 3:
        fit_mask = valid

    dt_fit = dt_s[fit_mask]
    corr_fit = corr[fit_mask]
    weights = np.clip(n_pairs[fit_mask], 1.0, None)
    sigma = 1.0 / np.sqrt(weights)

    positive_dt = dt_fit[dt_fit > 0.0]
    tau_guess = float(np.nanmedian(positive_dt)) if positive_dt.size else 1.0
    tau_guess = max(tau_guess, 1e-6)
    beta_guess = 1.0

    try:
        popt, pcov = curve_fit(
            _stretched_exponential,
            dt_fit,
            corr_fit,
            p0=[tau_guess, beta_guess],
            bounds=([1e-12, 0.1], [np.inf, 5.0]),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
        perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full(2, np.nan)
        fitted = _stretched_exponential(dt_fit, popt[0], popt[1])
        residual = float(np.sum((corr_fit - fitted) ** 2))
        total = float(np.sum((corr_fit - np.mean(corr_fit)) ** 2))
        r2 = 1.0 - residual / total if total > 0.0 else np.nan
        return {
            "tau_str_s": float(popt[0]),
            "tau_str_err_s": float(perr[0]) if perr.size > 0 else np.nan,
            "beta": float(popt[1]),
            "beta_err": float(perr[1]) if perr.size > 1 else np.nan,
            "fit_r2": float(r2),
            "fit_points": int(fit_mask.sum()),
        }
    except Exception:
        return {
            "tau_str_s": np.nan,
            "tau_str_err_s": np.nan,
            "beta": np.nan,
            "beta_err": np.nan,
            "fit_r2": np.nan,
            "fit_points": int(fit_mask.sum()),
        }


def _raw_correlation_path(derived_dir: str) -> str:
    return os.path.join(derived_dir, RAW_CORRELATION_FILENAME)


def _fit_correlation_path(derived_dir: str) -> str:
    return os.path.join(derived_dir, FIT_CORRELATION_FILENAME)


def _legacy_correlation_path(derived_dir: str) -> str:
    return os.path.join(derived_dir, LEGACY_CORRELATION_FILENAME)


def _drop_fit_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [column for column in FIT_RESULT_COLUMNS if column in df.columns]
    return df.drop(columns=drop_cols) if drop_cols else df.copy()


def _normalize_fit_lag_frames(fit_lag_frames: Any) -> set[int] | None:
    if fit_lag_frames is None:
        return None
    if isinstance(fit_lag_frames, (int, np.integer)):
        return {int(fit_lag_frames)}
    try:
        return {int(frame) for frame in fit_lag_frames}
    except TypeError:
        return {int(fit_lag_frames)}


def _filter_fit_rows(
    raw_df: pd.DataFrame,
    fit_dt_min_s: float | None = None,
    fit_dt_max_s: float | None = None,
    fit_lag_frames: Any = None,
    exclude_zero_lag: bool = False,
) -> pd.DataFrame:
    filtered = _drop_fit_columns(raw_df)
    if len(filtered) == 0:
        return filtered

    required_columns = {"dt_s", "corr", "n_pairs"}
    if not required_columns.issubset(filtered.columns):
        return pd.DataFrame(columns=list(filtered.columns))

    mask = np.isfinite(filtered["dt_s"]) & np.isfinite(filtered["corr"]) & np.isfinite(filtered["n_pairs"])
    if fit_dt_min_s is not None:
        mask &= filtered["dt_s"] >= float(fit_dt_min_s)
    if fit_dt_max_s is not None:
        mask &= filtered["dt_s"] <= float(fit_dt_max_s)

    lag_selection = _normalize_fit_lag_frames(fit_lag_frames)
    if lag_selection is not None:
        mask &= filtered["lag_frames"].isin(lag_selection)
    if exclude_zero_lag and "lag_frames" in filtered.columns:
        mask &= filtered["lag_frames"] > 0

    return filtered.loc[mask].copy()


def compute_raw_time_image_correlation(state: Dict[str, Any], image_corr_cfg: Dict[str, Any], skip_existing: bool = True) -> pd.DataFrame:
    """Compute and persist the raw two-time image correlation table."""

    derived_dir = state["paths"]["derived_dir"]
    out_path = _raw_correlation_path(derived_dir)
    legacy_path = _legacy_correlation_path(derived_dir)

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing raw image-time correlation from disk")
        return pd.read_parquet(out_path)

    if skip_existing and not os.path.exists(out_path) and os.path.exists(legacy_path):
        legacy_df = pd.read_parquet(legacy_path)
        raw_df = _drop_fit_columns(legacy_df)
        raw_df.to_parquet(out_path, index=False)
        print(f"Migrated legacy image-time correlation to {out_path}")
        return raw_df

    images = state["images"]
    t_len = int(state["dims"]["T"])
    if t_len <= 0:
        raise ValueError("Dataset contains no frames")

    channel = _resolve_channel(image_corr_cfg)
    z_mode = _resolve_z_mode(image_corr_cfg)

    frame_step = max(1, int(image_corr_cfg.get("frame_step", 1)))
    lag_step = max(1, int(image_corr_cfg.get("lag_step", 1)))
    max_lag_frames = image_corr_cfg.get("max_lag_frames", None)
    if max_lag_frames is None:
        max_lag_frames = image_corr_cfg.get("max_lag", None)

    frame_indices = np.arange(0, t_len, frame_step, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for image-time correlation")

    fps = state["calibration"].get("fps")
    time_step_s = (1.0 / float(fps)) if (fps and float(fps) > 0.0) else 1.0

    frame_cache: List[Dict[str, Any]] = []
    print(
        "Computing time correlation",
        f"(frames={len(frame_indices)}, channel={channel}, z_mode={z_mode}, dt={time_step_s:.4g} s)",
    )
    for frame_idx in tqdm(frame_indices, desc="load frames"):
        frame_2d, z_idx = _load_2d_frame(images, int(frame_idx), int(channel), z_mode)
        finite = np.isfinite(frame_2d)
        if not np.any(finite):
            frame_cache.append(
                {
                    "frame": int(frame_idx),
                    "z_index": z_idx,
                    "centered": np.array([], dtype=np.float32),
                    "mean": np.nan,
                    "std": np.nan,
                    "n_pixels": 0,
                }
            )
            continue

        values = np.asarray(frame_2d[finite], dtype=float)
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))
        centered = (values - mean_value).astype(np.float32, copy=False)
        frame_cache.append(
            {
                "frame": int(frame_idx),
                "z_index": z_idx,
                "centered": centered,
                "mean": mean_value,
                "std": std_value,
                "n_pixels": int(centered.size),
            }
        )

    valid_frames = [entry for entry in frame_cache if entry["n_pixels"] > 0 and np.isfinite(entry["std"]) and entry["std"] > 0.0]
    if len(valid_frames) < 2:
        print("Not enough valid frames to compute a time correlation")
        return pd.DataFrame()

    available_pairs = len(valid_frames) - 1
    if max_lag_frames is None:
        max_lag_idx = available_pairs
    else:
        max_lag_idx = max(0, min(available_pairs, int(max_lag_frames) // frame_step))

    lag_indices = np.arange(0, max_lag_idx + 1, lag_step, dtype=int)
    rows: List[Dict[str, Any]] = []
    for lag_idx in tqdm(lag_indices, desc="compute lags"):
        pair_corrs: List[float] = []
        pair_count = len(valid_frames) - int(lag_idx)
        if pair_count <= 0:
            continue

        for start_idx in range(pair_count):
            left = valid_frames[start_idx]
            right = valid_frames[start_idx + int(lag_idx)]
            if left["n_pixels"] != right["n_pixels"]:
                continue
            denom = float(left["std"] * right["std"])
            if not np.isfinite(denom) or denom <= 0.0:
                continue
            numerator = float(np.mean(left["centered"] * right["centered"]))
            pair_corrs.append(numerator / denom)

        corr_value = float(np.nanmean(pair_corrs)) if pair_corrs else np.nan
        corr_std = float(np.nanstd(pair_corrs, ddof=1)) if len(pair_corrs) > 1 else np.nan
        corr_sem = float(corr_std / np.sqrt(len(pair_corrs))) if len(pair_corrs) > 1 and np.isfinite(corr_std) else np.nan
        lag_frames = int(lag_idx) * frame_step
        rows.append(
            {
                "dataset_id": state["dataset_id"],
                "channel": int(channel),
                "z_index": int(valid_frames[0]["z_index"]) if valid_frames[0]["z_index"] is not None else np.nan,
                "projection": str(z_mode),
                "frame_step": int(frame_step),
                "lag_step": int(lag_step),
                "lag_index": int(lag_idx),
                "lag_frames": lag_frames,
                "dt_s": float(lag_frames) * float(time_step_s),
                "corr": corr_value,
                "corr_std": corr_std,
                "corr_sem": corr_sem,
                "n_pairs": int(len(pair_corrs)),
            }
        )

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        print("No lag rows were generated for the time correlation")
        return out_df

    out_df.to_parquet(out_path, index=False)
    print(f"Saved raw image-time correlation to {out_path}")
    return out_df


def fit_time_image_correlation(
    derived_dir: str,
    raw_df: pd.DataFrame | None = None,
    skip_existing: bool = True,
    fit_dt_min_s: float | None = None,
    fit_dt_max_s: float | None = None,
    fit_lag_frames: Any = None,
    exclude_zero_lag: bool = False,
) -> pd.DataFrame:
    """Fit the stretched exponential model from a saved raw correlation table."""

    raw_path = _raw_correlation_path(derived_dir)
    fit_path = _fit_correlation_path(derived_dir)
    legacy_path = _legacy_correlation_path(derived_dir)

    selection_active = (
        fit_dt_min_s is not None
        or fit_dt_max_s is not None
        or fit_lag_frames is not None
        or exclude_zero_lag
    )

    if skip_existing and os.path.exists(fit_path) and not selection_active:
        print("Loaded existing image-time correlation fit from disk")
        return pd.read_parquet(fit_path)

    if raw_df is None:
        if os.path.exists(raw_path):
            raw_df = pd.read_parquet(raw_path)
        elif os.path.exists(legacy_path):
            raw_df = _drop_fit_columns(pd.read_parquet(legacy_path))
        else:
            print("No raw image-time correlation table available for fitting")
            return pd.DataFrame()
    elif len(raw_df) == 0:
        print("No raw image-time correlation rows were selected for fitting")
        return pd.DataFrame()

    fit_input_df = _filter_fit_rows(
        raw_df,
        fit_dt_min_s=fit_dt_min_s,
        fit_dt_max_s=fit_dt_max_s,
        fit_lag_frames=fit_lag_frames,
        exclude_zero_lag=exclude_zero_lag,
    )
    required_columns = {"dt_s", "corr", "n_pairs"}
    if not required_columns.issubset(fit_input_df.columns):
        missing = sorted(required_columns.difference(raw_df.columns))
        raise ValueError(f"Raw image-time correlation table is missing required columns: {missing}")

    if len(fit_input_df) == 0:
        print("No rows matched the requested fit selection")
        return pd.DataFrame()

    fit_stats = _fit_stretched_exponential(
        fit_input_df["dt_s"].to_numpy(dtype=float),
        fit_input_df["corr"].to_numpy(dtype=float),
        fit_input_df["n_pairs"].to_numpy(dtype=float),
    )

    lag_selection = _normalize_fit_lag_frames(fit_lag_frames)
    fit_selection_label = "all"
    if selection_active:
        labels: List[str] = []
        if fit_dt_min_s is not None or fit_dt_max_s is not None:
            labels.append(
                "dt_range"
            )
        if lag_selection is not None:
            labels.append("lag_frames")
        if not exclude_zero_lag:
            labels.append("include_zero_lag")
        fit_selection_label = "+".join(labels) if labels else "custom"

    first_row = fit_input_df.iloc[0]
    fit_row = {
        "dataset_id": first_row.get("dataset_id", ""),
        "channel": int(first_row.get("channel", 0)) if np.isfinite(first_row.get("channel", np.nan)) else np.nan,
        "projection": first_row.get("projection", ""),
        "frame_step": int(first_row.get("frame_step", 1)) if np.isfinite(first_row.get("frame_step", np.nan)) else np.nan,
        "lag_step": int(first_row.get("lag_step", 1)) if np.isfinite(first_row.get("lag_step", np.nan)) else np.nan,
        "fit_selection": fit_selection_label,
        "fit_dt_min_s": float(fit_dt_min_s) if fit_dt_min_s is not None else np.nan,
        "fit_dt_max_s": float(fit_dt_max_s) if fit_dt_max_s is not None else np.nan,
        "fit_lag_frames": ",".join(str(frame) for frame in sorted(lag_selection)) if lag_selection is not None else "",
        "fit_exclude_zero_lag": bool(exclude_zero_lag),
        "raw_path": raw_path,
        "fit_path": fit_path,
        "n_lags": int(len(raw_df)),
        "n_fit_rows": int(len(fit_input_df)),
        **fit_stats,
    }
    fit_df = pd.DataFrame([fit_row])
    fit_df.to_parquet(fit_path, index=False)
    print(f"Saved image-time correlation fit to {fit_path}")
    print(
        "Fit summary:",
        f"tau_str={fit_stats.get('tau_str_s', np.nan):.4g} s",
        f"beta={fit_stats.get('beta', np.nan):.4g}",
        f"R^2={fit_stats.get('fit_r2', np.nan):.4g}",
    )
    return fit_df


def compute_time_image_correlation(state: Dict[str, Any], image_corr_cfg: Dict[str, Any], skip_existing: bool = True) -> pd.DataFrame:
    """Backward-compatible wrapper for the raw correlation computation."""

    return compute_raw_time_image_correlation(state, image_corr_cfg, skip_existing=skip_existing)