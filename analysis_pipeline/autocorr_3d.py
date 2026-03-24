from __future__ import annotations

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
    nbins = int(autocorr_cfg.get("nbins", 120))
    subtract_mean = bool(autocorr_cfg.get("subtract_mean", False))
    normalize = autocorr_cfg.get("normalize", "c0")

    if frame < 0 or frame >= t_len:
        raise ValueError(f"single_frame_3d out of range: {frame}")

    vol = images[frame, channel].compute()
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")

    px_xy = float(state["calibration"]["px_per_micron"])
    px_z = _pxz(state)

    r_um, corr_prof = mass_mass_corr_per_frame(
        imgs=vol[np.newaxis, ...],
        px_per_micron=px_xy,
        px_per_micron_z=px_z,
        nbins=nbins,
        subtract_mean=subtract_mean,
        normalize=normalize,
    )
    c = np.asarray(corr_prof)[0]
    fit = _fit_decay(np.asarray(r_um, dtype=float), np.asarray(c, dtype=float))

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
    nbins = int(autocorr_cfg.get("nbins", 120))
    subtract_mean = bool(autocorr_cfg.get("subtract_mean", False))
    normalize = autocorr_cfg.get("normalize", "c0")

    sample_count = min(max(1, int(autocorr_cfg.get("sample_count_3d", 10))), t_len)
    frames = np.unique(np.linspace(0, max(t_len - 1, 0), sample_count, dtype=int))

    px_xy = float(state["calibration"]["px_per_micron"])
    px_z = _pxz(state)
    fps = state["calibration"].get("fps")
    time_step_s = (1.0 / float(fps)) if (fps and float(fps) > 0) else 1.0

    rows: List[Dict[str, Any]] = []

    for fr in tqdm(frames, desc="sampled 3d autocorr"):
        vol = images[int(fr), channel].compute()
        vol = np.asarray(vol)
        if vol.ndim != 3:
            continue

        r_um, corr_prof = mass_mass_corr_per_frame(
            imgs=vol[np.newaxis, ...],
            px_per_micron=px_xy,
            px_per_micron_z=px_z,
            nbins=nbins,
            subtract_mean=subtract_mean,
            normalize=normalize,
        )
        corr = np.asarray(corr_prof, dtype=float)[0]
        r = np.asarray(r_um, dtype=float)
        fit = _fit_decay(r, corr)

        for ri, ci in zip(r.tolist(), corr.tolist()):
            rows.append(
                {
                    "dataset_id": state["dataset_id"],
                    "frame": int(fr),
                    "time_s": float(fr) * float(time_step_s),
                    "channel": int(channel),
                    "r_um": float(ri),
                    "corr": float(ci),
                    "xi_um": fit.get("xi_um", np.nan),
                    "xi_err_um": fit.get("xi_err_um", np.nan),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved sampled 3D autocorr to {out_path}")
    return out_df
