import os

import numpy as np
import pandas as pd


def add_bead_angular_speed(
    tracks_df: pd.DataFrame,
    px_per_micron: float,
    X: int,
    Y: int,
    fps=None,
    handle=None,
    origin_x_um=None,
    origin_y_um=None,
):
    if tracks_df is None or len(tracks_df) == 0:
        raise ValueError("tracks_df is empty")

    ang_df = tracks_df.copy()
    if "frame" not in ang_df.columns or "particle" not in ang_df.columns:
        raise ValueError("Tracks dataframe must contain 'frame' and 'particle' columns")

    has_um = all(c in ang_df.columns for c in ["x_um", "y_um"])
    has_px = all(c in ang_df.columns for c in ["x", "y"])
    if not has_um:
        if not has_px:
            raise ValueError("Tracks dataframe must contain either (x_um, y_um) or (x, y)")
        if not px_per_micron:
            raise ValueError("px_per_micron missing; cannot convert x/y to microns")
        ang_df["x_um"] = ang_df["x"].astype(float) / float(px_per_micron)
        ang_df["y_um"] = ang_df["y"].astype(float) / float(px_per_micron)

    ang_df["frame"] = ang_df["frame"].astype(int)
    ang_df["particle"] = ang_df["particle"].astype(int)
    ang_df = ang_df.sort_values(["particle", "frame"], kind="mergesort").reset_index(drop=True)

    if origin_x_um is None:
        origin_x_um = (X - 1) / (2.0 * float(px_per_micron))
    if origin_y_um is None:
        origin_y_um = (Y - 1) / (2.0 * float(px_per_micron))

    dx0 = ang_df["x_um"].to_numpy(dtype=float) - float(origin_x_um)
    dy0 = ang_df["y_um"].to_numpy(dtype=float) - float(origin_y_um)
    ang_df["r_xy_um"] = np.sqrt(dx0**2 + dy0**2)
    ang_df["theta_rad"] = np.arctan2(dy0, dx0)
    ang_df["theta_unwrapped_rad"] = ang_df.groupby("particle", sort=False)["theta_rad"].transform(
        lambda s: pd.Series(np.unwrap(s.to_numpy(dtype=float)), index=s.index)
    )

    if "dt_s" not in ang_df.columns:
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

        if fps_use is None:
            fps_use = float(fps) if (fps is not None and float(fps) > 0) else None

        ang_df["dframe"] = ang_df.groupby("particle", sort=False)["frame"].diff()
        if fps_use is None:
            ang_df["dt_s"] = ang_df["dframe"].astype(float)
        else:
            ang_df["dt_s"] = ang_df["dframe"].astype(float) / fps_use

    ang_df["dtheta_rad"] = ang_df.groupby("particle", sort=False)["theta_unwrapped_rad"].diff()
    dt_arr = ang_df["dt_s"].to_numpy(dtype=float)
    dtheta_arr = ang_df["dtheta_rad"].to_numpy(dtype=float)
    valid = (dt_arr > 0) & np.isfinite(dtheta_arr)

    omega = np.full(len(ang_df), np.nan, dtype=float)
    omega[valid] = dtheta_arr[valid] / dt_arr[valid]
    ang_df["omega_rad_s"] = omega
    ang_df["omega_deg_s"] = np.rad2deg(ang_df["omega_rad_s"])

    return ang_df


if __name__ == "__main__":
    raise SystemExit(
        "Import add_bead_angular_speed from this file and pass a tracks dataframe from the notebook."
    )
