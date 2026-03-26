from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from tomsUtilities import mass_mass_corr_per_frame


def _exp_decay(r, amplitude, xi, offset):
    return amplitude * np.exp(-r / xi) + offset


def _fit_decay(r_um: np.ndarray, corr: np.ndarray) -> Dict[str, float]:
    mask = (r_um > 0) & np.isfinite(r_um) & np.isfinite(corr)
    if int(mask.sum()) < 5:
        return {"xi_um": np.nan, "xi_err_um": np.nan}

    r_fit = r_um[mask]
    c_fit = corr[mask]
    amp_guess = max(0.05, float(np.nanmax(c_fit) - np.nanmin(c_fit)))
    xi_guess = max(0.1, float(np.nanmean(r_fit)))
    offset_guess = float(np.nanmin(c_fit))

    try:
        popt, pcov = curve_fit(
            _exp_decay,
            r_fit,
            c_fit,
            p0=[amp_guess, xi_guess, offset_guess],
            bounds=([0.0, 0.01, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full(popt.shape, np.nan)
        return {
            "xi_um": float(popt[1]),
            "xi_err_um": float(perr[1]) if perr.size > 1 else np.nan,
        }
    except Exception:
        return {"xi_um": np.nan, "xi_err_um": np.nan}


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
        return [worker_fn(item) for item in items]

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        iterator = executor.map(worker_fn, items)
        if desc:
            iterator = tqdm(iterator, total=len(items), desc=desc)
        return list(iterator)


def _sampled_2d_frame_rows(args):
    (
        images,
        dataset_id,
        fr,
        channel,
        z_idx,
        nbins,
        subtract_mean,
        normalize,
        px_xy,
        time_step_s,
    ) = args

    img = images[int(fr), channel, int(z_idx)].compute()
    img = np.asarray(img)
    if img.ndim != 2:
        return []

    r_um, corr_prof = mass_mass_corr_per_frame(
        imgs=img[np.newaxis, ...],
        px_per_micron=px_xy,
        px_per_micron_z=px_xy,
        nbins=nbins,
        subtract_mean=subtract_mean,
        normalize=normalize,
    )
    corr = np.asarray(corr_prof, dtype=float)[0]
    r = np.asarray(r_um, dtype=float)
    fit = _fit_decay(r, corr)

    return [
        {
            "dataset_id": dataset_id,
            "frame": int(fr),
            "time_s": float(fr) * float(time_step_s),
            "z_index": int(z_idx),
            "channel": int(channel),
            "r_um": float(ri),
            "corr": float(ci),
            "xi_um": fit.get("xi_um", np.nan),
            "xi_err_um": fit.get("xi_err_um", np.nan),
        }
        for ri, ci in zip(r.tolist(), corr.tolist())
    ]


def compute_sampled_2d(state: Dict[str, Any], autocorr_cfg: Dict[str, Any], skip_existing: bool = True) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    out_path = os.path.join(derived_dir, "autocorr_2d_sampled.parquet")

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing sampled 2D autocorr from disk")
        return pd.read_parquet(out_path)

    images = state["images"]
    t_len = int(state["dims"]["T"])
    z_len = int(state["dims"]["Z"])

    channel = int(autocorr_cfg.get("channel_2d", autocorr_cfg.get("channel_3d", 0)))
    nbins = int(autocorr_cfg.get("nbins", 120))
    subtract_mean = bool(autocorr_cfg.get("subtract_mean", False))
    normalize = autocorr_cfg.get("normalize", "c0")

    sample_count = min(max(1, int(autocorr_cfg.get("sample_count_2d", 20))), t_len)
    frames = np.unique(np.linspace(0, max(t_len - 1, 0), sample_count, dtype=int))
    parallel_workers = _resolve_parallel_workers(autocorr_cfg.get("parallel_workers", 0), len(frames))

    px_xy = float(state["calibration"]["px_per_micron"])
    fps = state["calibration"].get("fps")
    time_step_s = (1.0 / float(fps)) if (fps and float(fps) > 0) else 1.0

    z_mode = autocorr_cfg.get("middle_z_for_2d", "middle")
    if z_mode == "middle":
        z_idx = z_len // 2
    else:
        z_idx = int(z_mode)

    jobs = [
        (
            images,
            state["dataset_id"],
            int(fr),
            int(channel),
            int(z_idx),
            int(nbins),
            bool(subtract_mean),
            normalize,
            px_xy,
            time_step_s,
        )
        for fr in frames
    ]
    rows_nested = _parallel_map(_sampled_2d_frame_rows, jobs, parallel_workers, desc="sampled 2d autocorr")
    rows: List[Dict[str, Any]] = [row for rows_per_frame in rows_nested for row in rows_per_frame]

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved sampled 2D autocorr to {out_path}")
    return out_df


def compute_radial_2d_single(state: Dict[str, Any], autocorr_cfg: Dict[str, Any], skip_existing: bool = True) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    out_path = os.path.join(derived_dir, "autocorr_2d_radial_single.parquet")

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing radial 2D autocorr from disk")
        return pd.read_parquet(out_path)

    images = state["images"]
    t_len = int(state["dims"]["T"])
    z_len = int(state["dims"]["Z"])

    frame = int(autocorr_cfg.get("frame_2d", 0))
    channel = int(autocorr_cfg.get("channel_2d", autocorr_cfg.get("channel_3d", 0)))
    if frame < 0 or frame >= t_len:
        raise ValueError(f"frame_2d out of range: {frame}")

    vol = images[frame, channel].compute()
    vol = np.asarray(vol, dtype=float)
    if vol.ndim == 3:
        z_idx = z_len // 2
        img = vol[z_idx]
    elif vol.ndim == 2:
        z_idx = 0
        img = vol
    else:
        raise ValueError(f"Unexpected image ndim={vol.ndim}")

    px_per_um = float(state["calibration"]["px_per_micron"])
    pix_size_um = 1.0 / px_per_um

    img0 = img - float(np.nanmean(img))
    F = np.fft.fft2(img0)
    ac = np.fft.ifft2(np.abs(F) ** 2).real
    ac = np.fft.fftshift(ac)

    center = (ac.shape[0] // 2, ac.shape[1] // 2)
    c0 = ac[center]
    if not np.isfinite(c0) or c0 == 0:
        raise RuntimeError("Invalid central autocorrelation value")
    ac_norm = ac / float(c0)

    y_grid, x_grid = np.indices(ac_norm.shape)
    r_pix = np.sqrt((x_grid - center[1]) ** 2 + (y_grid - center[0]) ** 2)
    r_int = r_pix.astype(np.int32)
    max_r = int(np.min(center))

    inds = r_int.ravel() <= max_r
    r_flat = r_int.ravel()[inds]
    ac_flat = ac_norm.ravel()[inds]

    sums = np.bincount(r_flat, weights=ac_flat, minlength=max_r + 1)
    counts = np.bincount(r_flat, minlength=max_r + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = sums / counts

    radial = radial[: max_r + 1]
    radial_r_um = np.arange(len(radial), dtype=float) * pix_size_um

    fit = _fit_decay(radial_r_um.astype(float), radial.astype(float))

    out_df = pd.DataFrame(
        {
            "dataset_id": [state["dataset_id"]] * len(radial_r_um),
            "frame": [frame] * len(radial_r_um),
            "channel": [channel] * len(radial_r_um),
            "z_index": [int(z_idx)] * len(radial_r_um),
            "r_um": radial_r_um,
            "corr": radial,
            "xi_um": [fit.get("xi_um", np.nan)] * len(radial_r_um),
            "xi_err_um": [fit.get("xi_err_um", np.nan)] * len(radial_r_um),
        }
    )
    out_df.to_parquet(out_path, index=False)
    print(f"Saved radial 2D autocorr to {out_path}")
    return out_df
