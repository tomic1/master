from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .autocorr_2d import compute_radial_2d_single, compute_sampled_2d
from .autocorr_3d import compute_sampled_3d, compute_single_frame_3d
from .beads_track import detect_and_link_beads, preview_bead_detection
from .beads_velocity import compute_angular_speed_xy, compute_velocity_from_tracks
from .io_dataset import load_dataset_state


def run_bead_core(config: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config if overrides is None else _merge_runtime(config, overrides)
    runtime = cfg.get("runtime", {})

    state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))
    beads_cfg = cfg.get("beads", {})
    skip_existing = bool(runtime.get("skip_existing", True))

    preview_summary, preview_stats, _, _ = preview_bead_detection(state, beads_cfg, show=True)
    detections_df, tracks_df = detect_and_link_beads(state, beads_cfg, skip_existing=skip_existing)
    vel_df = compute_velocity_from_tracks(state, tracks_df, skip_existing=skip_existing)

    ang_df = None
    if bool(beads_cfg.get("compute_angular_speed", True)) and len(vel_df) > 0:
        ang_df = compute_angular_speed_xy(state, vel_df, beads_cfg, skip_existing=skip_existing)

    return {
        "state": state,
        "preview_summary": preview_summary,
        "preview_stats": preview_stats,
        "detections_df": detections_df,
        "tracks_df": tracks_df,
        "tracks_vel_df": vel_df,
        "tracks_ang_df": ang_df,
    }


def run_autocorr_core(config: Dict[str, Any], state: Dict[str, Any] | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, pd.DataFrame]:
    cfg = config if overrides is None else _merge_runtime(config, overrides)
    runtime = cfg.get("runtime", {})
    skip_existing = bool(runtime.get("skip_existing", True))

    if state is None:
        state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))

    autocorr_cfg = cfg.get("autocorr", {})

    single3d_df = compute_single_frame_3d(state, autocorr_cfg, skip_existing=skip_existing)
    sampled3d_df = compute_sampled_3d(state, autocorr_cfg, skip_existing=skip_existing)
    sampled2d_df = compute_sampled_2d(state, autocorr_cfg, skip_existing=skip_existing)
    radial2d_df = compute_radial_2d_single(state, autocorr_cfg, skip_existing=skip_existing)

    return {
        "single3d_df": single3d_df,
        "sampled3d_df": sampled3d_df,
        "sampled2d_df": sampled2d_df,
        "radial2d_df": radial2d_df,
    }


def run_core_pipeline(config: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    bead_out = run_bead_core(config, overrides=overrides)
    autocorr_out = run_autocorr_core(config, state=bead_out["state"], overrides=overrides)

    merged = dict(bead_out)
    merged.update(autocorr_out)
    return merged


def _merge_runtime(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    from .config import merge_overrides

    return merge_overrides(config, overrides)
