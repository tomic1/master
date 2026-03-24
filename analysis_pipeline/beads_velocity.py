from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def _fps_from_handle_or_value(handle: Dict[str, Any] | None, fps: float | None) -> float | None:
    fps_use = None
    try:
        calib = handle.get("meta", {}).get("calibration", {}) if isinstance(handle, dict) else {}
        dt_s = calib.get("dt_s", None)
        if dt_s in (None, "", 0):
            dt_ms = calib.get("dt_ms", None)
            dt_s = (float(dt_ms) / 1000.0) if dt_ms not in (None, "", 0) else None
        else:
            dt_s = float(dt_s)
            if dt_s > 100:
                dt_s = dt_s / 1000.0
        fps_use = (1.0 / dt_s) if (dt_s and dt_s > 0) else None
    except Exception:
        fps_use = None

    if fps_use is None and fps is not None and float(fps) > 0:
        fps_use = float(fps)
    return fps_use


def compute_velocity_from_tracks(state: Dict[str, Any], tracks_df: pd.DataFrame, skip_existing: bool = True) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    out_path = os.path.join(derived_dir, "beads_tracks_with_velocity.parquet")

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing velocity tracks from disk")
        return pd.read_parquet(out_path)

    if tracks_df is None or len(tracks_df) == 0:
        raise ValueError("tracks_df is empty")
    if "frame" not in tracks_df.columns or "particle" not in tracks_df.columns:
        raise ValueError("tracks_df must contain frame and particle columns")

    px_per_micron = state["calibration"].get("px_per_micron")
    px_per_micron_z = state["calibration"].get("px_per_micron_z")
    fps = state["calibration"].get("fps")
    handle = state.get("handle")

    df = tracks_df.copy()

    has_um = all(c in df.columns for c in ["x_um", "y_um", "z_um"])
    has_px = all(c in df.columns for c in ["x", "y", "z"])

    if not has_um:
        if not has_px:
            raise ValueError("tracks_df must contain (x_um,y_um,z_um) or (x,y,z)")
        if not px_per_micron:
            raise ValueError("px_per_micron missing; cannot convert x/y to um")
        if not px_per_micron_z:
            px_per_micron_z = px_per_micron
        df["x_um"] = df["x"].astype(float) / float(px_per_micron)
        df["y_um"] = df["y"].astype(float) / float(px_per_micron)
        df["z_um"] = df["z"].astype(float) / float(px_per_micron_z)

    df["frame"] = df["frame"].astype(int)
    df["particle"] = df["particle"].astype(int)
    df = df.sort_values(["particle", "frame"], kind="mergesort").reset_index(drop=True)

    fps_use = _fps_from_handle_or_value(handle, fps)
    df["dframe"] = df.groupby("particle", sort=False)["frame"].diff()
    if fps_use is None:
        df["dt_s"] = df["dframe"].astype(float)
    else:
        df["dt_s"] = df["dframe"].astype(float) / fps_use

    df["dx_um"] = df.groupby("particle", sort=False)["x_um"].diff()
    df["dy_um"] = df.groupby("particle", sort=False)["y_um"].diff()
    df["dz_um"] = df.groupby("particle", sort=False)["z_um"].diff()
    df["disp_um"] = np.sqrt(df["dx_um"] ** 2 + df["dy_um"] ** 2 + df["dz_um"] ** 2)

    valid_dt = df["dt_s"].to_numpy(dtype=float) > 0
    dt_arr = df["dt_s"].to_numpy(dtype=float)
    for col_d, col_v in [("dx_um", "vx_um_s"), ("dy_um", "vy_um_s"), ("dz_um", "vz_um_s")]:
        d_arr = df[col_d].to_numpy(dtype=float)
        v = np.full_like(d_arr, np.nan, dtype=float)
        v[valid_dt] = d_arr[valid_dt] / dt_arr[valid_dt]
        df[col_v] = v
    df["speed_um_s"] = np.sqrt(df["vx_um_s"] ** 2 + df["vy_um_s"] ** 2 + df["vz_um_s"] ** 2)

    disp = df["disp_um"].to_numpy(dtype=float)
    nonzero = disp > 0
    for col_d, col_u in [("dx_um", "ux"), ("dy_um", "uy"), ("dz_um", "uz")]:
        d_arr = df[col_d].to_numpy(dtype=float)
        u = np.full_like(d_arr, np.nan, dtype=float)
        u[nonzero] = d_arr[nonzero] / disp[nonzero]
        df[col_u] = u

    df.to_parquet(out_path, index=False)
    print(f"Saved tracks+velocity to {out_path} | rows={len(df)}")
    return df


def compute_angular_speed_xy(state: Dict[str, Any], vel_df: pd.DataFrame, beads_cfg: Dict[str, Any], skip_existing: bool = True):
    derived_dir = state["paths"]["derived_dir"]
    out_path = os.path.join(derived_dir, "beads_tracks_with_angular_speed.parquet")

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing angular speed tracks from disk")
        return pd.read_parquet(out_path)

    if vel_df is None or len(vel_df) == 0:
        raise ValueError("vel_df is empty")

    px_per_micron = state["calibration"].get("px_per_micron")
    if not px_per_micron:
        raise ValueError("px_per_micron is required to infer default angular origin")

    x_len = int(state["dims"]["X"])
    y_len = int(state["dims"]["Y"])

    origin_x_um = beads_cfg.get("origin_x_um", None)
    origin_y_um = beads_cfg.get("origin_y_um", None)
    if origin_x_um is None:
        origin_x_um = (x_len - 1) / (2.0 * float(px_per_micron))
    if origin_y_um is None:
        origin_y_um = (y_len - 1) / (2.0 * float(px_per_micron))

    ang_df = vel_df.copy()
    if "dt_s" not in ang_df.columns:
        raise ValueError("vel_df must contain dt_s column")

    dx0 = ang_df["x_um"].to_numpy(dtype=float) - float(origin_x_um)
    dy0 = ang_df["y_um"].to_numpy(dtype=float) - float(origin_y_um)
    ang_df["r_xy_um"] = np.sqrt(dx0**2 + dy0**2)
    ang_df["theta_rad"] = np.arctan2(dy0, dx0)
    ang_df["theta_unwrapped_rad"] = ang_df.groupby("particle", sort=False)["theta_rad"].transform(
        lambda s: pd.Series(np.unwrap(s.to_numpy(dtype=float)), index=s.index)
    )

    ang_df["dtheta_rad"] = ang_df.groupby("particle", sort=False)["theta_unwrapped_rad"].diff()
    valid = (ang_df["dt_s"].to_numpy(dtype=float) > 0) & np.isfinite(ang_df["dtheta_rad"].to_numpy(dtype=float))
    omega = np.full(len(ang_df), np.nan, dtype=float)
    omega[valid] = ang_df["dtheta_rad"].to_numpy(dtype=float)[valid] / ang_df["dt_s"].to_numpy(dtype=float)[valid]
    ang_df["omega_rad_s"] = omega
    ang_df["omega_deg_s"] = np.rad2deg(ang_df["omega_rad_s"])

    ang_df.to_parquet(out_path, index=False)
    print(f"Saved tracks+angular speed to {out_path} | rows={len(ang_df)}")
    return ang_df
