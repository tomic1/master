from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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


def _fit_decay(r_um: np.ndarray, corr: np.ndarray, fit_range_um: tuple[float | None, float | None] | None = None) -> Dict[str, float]:
    mask = (r_um > 0) & np.isfinite(r_um) & np.isfinite(corr)
    if fit_range_um is not None:
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

    try:
        popt, pcov = curve_fit(
            _exp_decay,
            r_fit,
            c_fit,
            p0=[amp_guess, xi_guess, offset_guess],
            bounds=([0.0, 0.01, -np.inf], [np.inf, np.inf, np.inf]),
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


def _autocorr_3d_reference(volume: np.ndarray, px_xy: float, px_z: float, subtract_mean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(volume, dtype=float)
    image = image - float(np.mean(image))

    spectrum = np.fft.fftn(image)
    ac = np.fft.fftshift(np.real(np.fft.ifftn(np.abs(spectrum) ** 2)))
    ac = ac / image.size

    var0 = float(np.var(np.asarray(volume, dtype=float)))
    if var0 > 0:
        ac = ac / var0
    else:
        ac = ac / np.max(ac)

    dz = 1.0 / float(px_z)
    dxy = 1.0 / float(px_xy)
    z, y, x = image.shape
    zz = (np.arange(z) - z // 2) * dz
    yy = (np.arange(y) - y // 2) * dxy
    xx = (np.arange(x) - x // 2) * dxy
    zg, yg, xg = np.meshgrid(zz, yy, xx, indexing="ij")
    r = np.sqrt(xg**2 + yg**2 + zg**2)

    bin_size = min(dz, dxy)
    r_flat = r.ravel()
    v_flat = ac.ravel()
    bin_index = np.floor(r_flat / bin_size).astype(int)
    nbins = int(bin_index.max()) + 1 if bin_index.size else 0
    sums = np.bincount(bin_index, weights=v_flat, minlength=nbins)
    counts = np.bincount(bin_index, minlength=nbins)
    radial_mean = sums / np.maximum(counts, 1)
    r_centers = (np.arange(nbins) + 0.5) * bin_size
    return r_centers, radial_mean


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
        px_xy,
        px_z,
        time_step_s,
        fit_range_um,
    ) = args

    vol = images[int(fr), channel].compute()
    vol = np.asarray(vol)
    if vol.ndim != 3:
        return []

    r, corr = _autocorr_3d_reference(vol, px_xy, px_z, subtract_mean=subtract_mean)
    fit = _fit_decay(r, corr, fit_range_um=fit_range_um)

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

    r_um, c = _autocorr_3d_reference(vol, px_xy, px_z, subtract_mean=subtract_mean)
    fit = _fit_decay(np.asarray(r_um, dtype=float), np.asarray(c, dtype=float), fit_range_um=fit_range_um)

    out_df = pd.DataFrame(
        {
            "dataset_id": [state["dataset_id"]] * len(r_um),
            "frame": [frame] * len(r_um),
            "channel": [channel] * len(r_um),
            "r_um": np.asarray(r_um, dtype=float),
            "corr": np.asarray(c, dtype=float),
            "xi_um": [fit.get("xi_um", np.nan)] * len(r_um),
            "xi_err_um": [fit.get("xi_err_um", np.nan)] * len(r_um),
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
