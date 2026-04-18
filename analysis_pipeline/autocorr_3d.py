from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import fft as spfft
from tqdm import tqdm


def _exp_decay(r, amplitude, xi, offset):
    return amplitude * np.exp(-r / xi) + offset


def _fit_range_from_cfg(autocorr_cfg: Dict[str, Any]) -> tuple[float | None, float | None] | None:
    lower = autocorr_cfg.get("fit_r_um_min")
    upper = autocorr_cfg.get("fit_r_um_max")
    if lower is None and upper is None:
        return None
    return (
        float(lower) if lower is not None else None,
        float(upper) if upper is not None else None,
    )


def _fit_decay(
    r_um: np.ndarray,
    corr: np.ndarray,
    fit_range_um: tuple[float | None, float | None] | None = None,
    *,
    fit_mode: str = "standard",
) -> Dict[str, float]:
    mask = (r_um > 0) & np.isfinite(r_um) & np.isfinite(corr)
    fit_mode_norm = str(fit_mode).strip().lower()
    if fit_range_um is not None and fit_mode_norm not in {"weighted_near0", "weighted-near0", "near0_weighted", "near0-weighted"}:
        lower, upper = fit_range_um
        if lower is not None:
            mask &= r_um >= float(lower)
        if upper is not None:
            mask &= r_um <= float(upper)
    if int(mask.sum()) < 5:
        return {"xi_um": np.nan, "xi_err_um": np.nan}

    r_fit = r_um[mask]
    c_fit = corr[mask]
    amp_guess = max(0.05, float(np.nanmax(c_fit) - np.nanmin(c_fit)))
    xi_guess = max(0.1, float(np.nanmax(r_fit))/10)
    offset_guess = float(np.nanmin(c_fit))

    sigma = None
    absolute_sigma = False
    if fit_mode_norm in {"weighted_near0", "weighted-near0", "near0_weighted", "near0-weighted"}:
        r_scale = max(float(np.nanmax(r_fit)), 1e-6)
        sigma = 0.2 + (2.5 / r_scale) * r_fit
    elif fit_mode_norm not in {"standard", "unweighted", "default", ""}:
        raise ValueError("fit_mode must be 'standard' or 'weighted_near0'")

    try:
        popt, pcov = curve_fit(
            _exp_decay,
            r_fit,
            c_fit,
            p0=[amp_guess, xi_guess, offset_guess],
            bounds=([0.0, 0.01, -np.inf], [np.inf, np.inf, np.inf]),
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            maxfev=20000,
        )
        perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full(popt.shape, np.nan)
        return {
            "xi_um": float(popt[1]),
            "xi_err_um": float(perr[1]) if perr.size > 1 else np.nan,
            "amp": float(popt[0]),
            "offset": float(popt[2]),
        }
    except Exception:
        return {"xi_um": np.nan, "xi_err_um": np.nan}


def _pxz(state: Dict[str, Any]) -> float:
    px_xy = state["calibration"].get("px_per_micron")
    px_z = state["calibration"].get("px_per_micron_z")
    if not px_xy:
        raise ValueError("px_per_micron missing")
    return float(px_z) if px_z and float(px_z) > 0 else float(px_xy)


def _radial_average_3d(ac: np.ndarray, px_xy: float, px_z: float) -> tuple[np.ndarray, np.ndarray]:
    z_len, y_len, x_len = ac.shape
    dz = 1.0 / float(px_z)
    dxy = 1.0 / float(px_xy)

    z_coords = (np.arange(z_len) - z_len // 2) * dz
    y_coords = (np.arange(y_len) - y_len // 2) * dxy
    x_coords = (np.arange(x_len) - x_len // 2) * dxy

    xy_radius_sq = y_coords[:, None] ** 2 + x_coords[None, :] ** 2
    center_y = y_len // 2
    center_x = x_len // 2
    center_z = z_len // 2

    bin_size = min(dz, dxy)
    r_max = float(np.sqrt(np.max(xy_radius_sq) + np.max(z_coords**2)))
    nbins = int(np.floor(r_max / bin_size)) + 1
    sums = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=float)

    for z_idx, z_val in enumerate(z_coords):
        r = np.sqrt(xy_radius_sq + float(z_val) ** 2)
        bin_index = np.floor(r / bin_size).astype(np.int32)
        values = np.asarray(ac[z_idx], dtype=float)
        mask = np.ones_like(values, dtype=bool)
        if z_idx == center_z:
            mask[center_y, center_x] = False
        valid = mask.ravel()
        bins = bin_index.ravel()[valid]
        vals = values.ravel()[valid]
        sums += np.bincount(bins, weights=vals, minlength=nbins)
        counts += np.bincount(bins, minlength=nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        valid_bins = counts > 0
        radial_mean = sums[valid_bins] / counts[valid_bins]

    r_centers = (np.arange(nbins) + 0.5) * bin_size
    r_centers = r_centers[valid_bins]
    zero_lag = float(ac[center_z, center_y, center_x])
    return np.concatenate(([0.0], r_centers)), np.concatenate(([zero_lag], radial_mean))


def _autocorr_3d_reference(
    volume: np.ndarray,
    px_xy: float,
    px_z: float,
    subtract_mean: bool = False,
    fft_mode: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(volume, dtype=np.float32)
    if bool(subtract_mean):
        image = image - float(np.mean(image))

    fft_mode_norm = str(fft_mode).strip().lower()
    if fft_mode_norm in {"linear", "padded", "zero_padded", "zero-padded"}:
        pad_shape: tuple[int, ...] = tuple(int(2 * int(size) - 1) for size in image.shape)
        spectrum = spfft.rfftn(image, s=pad_shape)
        ac = spfft.irfftn(np.abs(spectrum) ** 2, s=pad_shape)  # type: ignore[call-overload]
    elif fft_mode_norm in {"circular", "periodic"}:
        pad_shape: tuple[int, ...] = tuple(int(size) for size in image.shape)
        spectrum = spfft.rfftn(image)
        ac = spfft.irfftn(np.abs(spectrum) ** 2, s=pad_shape)  # type: ignore[call-overload]
    else:
        raise ValueError("fft_mode must be 'linear' or 'circular'")

    ac = np.fft.fftshift(np.asarray(ac, dtype=np.float32))
    ac = ac / float(image.size)

    var0 = float(np.var(image))
    if not np.isfinite(var0) or var0 <= 0:
        raise RuntimeError("Cannot normalize 3D autocorrelation of a zero-variance volume")
    ac = ac / var0

    return _radial_average_3d(ac, px_xy, px_z)


def _resolve_parallel_workers(requested_workers: Any, task_count: int) -> int:
    task_count = max(0, int(task_count))
    if task_count <= 1:
        return 1

    try:
        workers = int(requested_workers)
    except Exception:
        workers = 0

    if workers <= 0:
        return 1

    return max(1, min(workers, task_count))


def _parallel_map(worker_fn, items, max_workers: int, desc: str | None = None):
    items = list(items)
    if len(items) <= 1 or int(max_workers) <= 1:
        iterator = tqdm(items, total=len(items), desc=desc) if desc else items
        return [worker_fn(item) for item in iterator]

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        iterator = executor.map(worker_fn, items)
        if desc:
            iterator = tqdm(iterator, total=len(items), desc=desc)
        return list(iterator)


def _sampled_3d_frame_rows(args):
    (
        images,
        dataset_id,
        fr,
        channel,
        subtract_mean,
        fft_mode,
        fit_mode,
        px_xy,
        px_z,
        time_step_s,
        fit_range_um,
    ) = args

    vol = images[int(fr), channel].compute()
    vol = np.asarray(vol)
    if vol.ndim != 3:
        return []

    r, corr = _autocorr_3d_reference(vol, px_xy, px_z, subtract_mean=subtract_mean, fft_mode=fft_mode)
    fit = _fit_decay(r, corr, fit_range_um=fit_range_um, fit_mode=fit_mode)

    return [
        {
            "dataset_id": dataset_id,
            "frame": int(fr),
            "time_s": float(fr) * float(time_step_s),
            "channel": int(channel),
            "r_um": float(ri),
            "corr": float(ci),
            "xi_um": fit.get("xi_um", np.nan),
            "xi_err_um": fit.get("xi_err_um", np.nan),
            "amp": fit.get("amp", np.nan),
            "offset": fit.get("offset", np.nan),
        }
        for ri, ci in zip(r.tolist(), corr.tolist())
    ]


def compute_single_frame_3d(state: Dict[str, Any], autocorr_cfg: Dict[str, Any], skip_existing: bool = True) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    out_path = os.path.join(derived_dir, "autocorr_3d_single_frame.parquet")

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing single-frame 3D autocorr from disk")
        return pd.read_parquet(out_path)

    images = state["images"]
    t_len = int(state["dims"]["T"])
    frame = int(autocorr_cfg.get("single_frame_3d", 0))
    channel = int(autocorr_cfg.get("channel_3d", 0))
    subtract_mean = bool(autocorr_cfg.get("subtract_mean", False))
    fft_mode = str(autocorr_cfg.get("fft_mode", "linear"))
    fit_mode = str(autocorr_cfg.get("fit_mode", "standard"))
    fit_range_um = _fit_range_from_cfg(autocorr_cfg)

    if frame < 0 or frame >= t_len:
        raise ValueError(f"single_frame_3d out of range: {frame}")

    vol = images[frame, channel].compute()
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")

    px_xy = float(state["calibration"]["px_per_micron"])
    px_z = _pxz(state)

    print(f"Computing single-frame 3D autocorr for frame {frame} (channel={channel})")
    if fit_range_um is not None:
        print(f"  Fit range: {fit_range_um[0]} .. {fit_range_um[1]} µm")

    r_um, c = _autocorr_3d_reference(vol, px_xy, px_z, subtract_mean=subtract_mean, fft_mode=fft_mode)
    fit = _fit_decay(np.asarray(r_um, dtype=float), np.asarray(c, dtype=float), fit_range_um=fit_range_um, fit_mode=fit_mode)

    out_df = pd.DataFrame(
        {
            "dataset_id": [state["dataset_id"]] * len(r_um),
            "frame": [frame] * len(r_um),
            "channel": [channel] * len(r_um),
            "r_um": np.asarray(r_um, dtype=float),
            "corr": np.asarray(c, dtype=float),
            "xi_um": [fit.get("xi_um", np.nan)] * len(r_um),
            "xi_err_um": [fit.get("xi_err_um", np.nan)] * len(r_um),
            "amp": [fit.get("amp", np.nan)] * len(r_um),
            "offset": [fit.get("offset", np.nan)] * len(r_um),
        }
    )
    out_df.to_parquet(out_path, index=False)
    print(f"Saved single-frame 3D autocorr to {out_path}")
    return out_df


def compute_sampled_3d(state: Dict[str, Any], autocorr_cfg: Dict[str, Any], skip_existing: bool = True) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    out_path = os.path.join(derived_dir, "autocorr_3d_sampled.parquet")

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing sampled 3D autocorr from disk")
        return pd.read_parquet(out_path)

    images = state["images"]
    t_len = int(state["dims"]["T"])
    channel = int(autocorr_cfg.get("channel_3d", 0))
    subtract_mean = bool(autocorr_cfg.get("subtract_mean", False))
    fft_mode = str(autocorr_cfg.get("fft_mode", "linear"))
    fit_mode = str(autocorr_cfg.get("fit_mode", "standard"))
    fit_range_um = _fit_range_from_cfg(autocorr_cfg)

    sample_count = min(max(1, int(autocorr_cfg.get("sample_count_3d", 10))), t_len)
    frames = np.unique(np.linspace(0, max(t_len - 1, 0), sample_count, dtype=int))
    parallel_workers = _resolve_parallel_workers(autocorr_cfg.get("parallel_workers", 0), len(frames))

    px_xy = float(state["calibration"]["px_per_micron"])
    px_z = _pxz(state)
    fps = state["calibration"].get("fps")
    time_step_s = (1.0 / float(fps)) if (fps and float(fps) > 0) else 1.0

    print(f"Computing sampled 3D autocorr for {len(frames)} frames (workers={parallel_workers})")
    if fit_range_um is not None:
        print(f"  Fit range: {fit_range_um[0]} .. {fit_range_um[1]} µm")

    jobs = [
        (
            images,
            state["dataset_id"],
            int(fr),
            int(channel),
            bool(subtract_mean),
            fft_mode,
            fit_mode,
            px_xy,
            px_z,
            time_step_s,
            fit_range_um,
        )
        for fr in frames
    ]
    rows_nested = _parallel_map(_sampled_3d_frame_rows, jobs, parallel_workers, desc="sampled 3d autocorr")
    rows: List[Dict[str, Any]] = [row for rows_per_frame in rows_nested for row in rows_per_frame]

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved sampled 3D autocorr to {out_path}")
    return out_df
