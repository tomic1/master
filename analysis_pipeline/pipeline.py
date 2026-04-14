from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .autocorr_2d import compute_radial_2d_single, compute_sampled_2d
from .autocorr_3d import compute_sampled_3d, compute_single_frame_3d
from .beads_track import detect_and_link_beads, preview_bead_detection
from .beads_velocity import compute_angular_speed_xy, compute_velocity_from_tracks
from .vector_correlation import run_vector_correlation_core as _run_vector_correlation_core
from .image_correlation import compute_raw_time_image_correlation, fit_time_image_correlation
from .io_dataset import load_dataset_state


def run_bead_core(config: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config if overrides is None else _merge_runtime(config, overrides)
    runtime = cfg.get("runtime", {})

    state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))
    beads_cfg = cfg.get("beads", {})
    skip_existing = bool(runtime.get("skip_existing", True))
    angular_mode = bool(beads_cfg.get("angular_mode", False))

    preview_summary, preview_stats, _, _ = preview_bead_detection(state, beads_cfg, show=True)
    detections_df, tracks_df = detect_and_link_beads(state, beads_cfg, skip_existing=skip_existing)
    vel_df = compute_velocity_from_tracks(state, tracks_df, skip_existing=skip_existing)

    ang_df = None
    if angular_mode and bool(beads_cfg.get("compute_angular_speed", True)) and len(vel_df) > 0:
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

    single_enabled = bool(autocorr_cfg.get("single_frame_3d_enabled", True))
    sampled3d_enabled = bool(autocorr_cfg.get("sampled_3d_enabled", True))
    sampled2d_enabled = bool(autocorr_cfg.get("sampled_2d_enabled", True))
    radial2d_enabled = bool(autocorr_cfg.get("radial_2d_enabled", True))

    print(
        "Autocorr status: "
        f"single3d={'on' if single_enabled else 'off'}, "
        f"sampled3d={'on' if sampled3d_enabled else 'off'}, "
        f"sampled2d={'on' if sampled2d_enabled else 'off'}, "
        f"radial2d={'on' if radial2d_enabled else 'off'}"
    )

    single3d_df = compute_single_frame_3d(state, autocorr_cfg, skip_existing=skip_existing) if single_enabled and int(autocorr_cfg.get("single_frame_3d", -1)) >= 0 else pd.DataFrame()
    sampled3d_df = compute_sampled_3d(state, autocorr_cfg, skip_existing=skip_existing) if sampled3d_enabled else pd.DataFrame()
    sampled2d_df = compute_sampled_2d(state, autocorr_cfg, skip_existing=skip_existing) if sampled2d_enabled else pd.DataFrame()
    radial2d_df = compute_radial_2d_single(state, autocorr_cfg, skip_existing=skip_existing) if radial2d_enabled else pd.DataFrame()

    return {
        "single3d_df": single3d_df,
        "sampled3d_df": sampled3d_df,
        "sampled2d_df": sampled2d_df,
        "radial2d_df": radial2d_df,
    }


def run_image_correlation_core(config: Dict[str, Any], state: Dict[str, Any] | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, pd.DataFrame]:
    cfg = config if overrides is None else _merge_runtime(config, overrides)
    runtime = cfg.get("runtime", {})
    skip_existing = bool(runtime.get("skip_existing", True))
    image_corr_cfg = dict(cfg.get("autocorr", {}))
    image_corr_cfg.update(cfg.get("image_corr", {}))

    if not bool(image_corr_cfg.get("enabled", True)):
        return {"image_corr_df": pd.DataFrame()}

    if state is None:
        state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))

    image_corr_df = compute_raw_time_image_correlation(state, image_corr_cfg, skip_existing=skip_existing)

    return {"image_corr_df": image_corr_df}


def run_image_correlation_fit_core(
    config: Dict[str, Any],
    state: Dict[str, Any] | None = None,
    raw_df: pd.DataFrame | None = None,
    fit_selection: Dict[str, Any] | None = None,
    skip_existing: bool | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    cfg = config if overrides is None else _merge_runtime(config, overrides)
    runtime = cfg.get("runtime", {})
    runtime_skip_existing = bool(runtime.get("skip_existing", True))
    if skip_existing is None:
        skip_existing = runtime_skip_existing

    if state is None:
        state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))

    image_corr_fit_df = fit_time_image_correlation(
        state["paths"]["derived_dir"],
        raw_df=raw_df,
        skip_existing=skip_existing,
        **(fit_selection or {}),
    )

    return {"image_corr_fit_df": image_corr_fit_df}


def run_vector_correlation_core(
    config: Dict[str, Any],
    state: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    return _run_vector_correlation_core(config, state=state, overrides=overrides)


def run_core_pipeline(config: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    bead_out = run_bead_core(config, overrides=overrides)
    autocorr_out = run_autocorr_core(config, state=bead_out["state"], overrides=overrides)
    vector_corr_out = run_vector_correlation_core(config, state=bead_out["state"], overrides=overrides)
    image_corr_out = run_image_correlation_core(config, state=bead_out["state"], overrides=overrides)
    image_corr_fit_out = run_image_correlation_fit_core(
        config,
        state=bead_out["state"],
        raw_df=image_corr_out.get("image_corr_df"),
        overrides=overrides,
    )

    merged = dict(bead_out)
    merged.update(autocorr_out)
    merged.update(vector_corr_out)
    merged.update(image_corr_out)
    merged.update(image_corr_fit_out)
    return merged


def _merge_runtime(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    from .config import merge_overrides

    return merge_overrides(config, overrides)
