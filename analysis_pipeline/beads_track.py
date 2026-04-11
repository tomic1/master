from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm

from tomsUtilities import plot_bead_detection_preview, threshold_beads


_RSO_PARAMS = set(inspect.signature(remove_small_objects).parameters)


def _remove_small_objects_compat(ar: np.ndarray, min_size: int) -> np.ndarray:
    if "max_size" in _RSO_PARAMS:
        return remove_small_objects(ar, max_size=min_size)
    if "size" in _RSO_PARAMS:
        return remove_small_objects(ar, size=min_size)
    return remove_small_objects(ar, min_size=min_size)


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


def _greedy_subnet_linker(source_set, dest_set, search_range, **kwargs):
    source_list = list(source_set)
    dest_list = list(dest_set)

    if not source_list and not dest_list:
        return [], []
    if not source_list:
        return [None] * len(dest_list), dest_list
    if not dest_list:
        return source_list, [None] * len(source_list)

    def _point_key(point):
        pos = np.asarray(getattr(point, "pos", getattr(point, "coords", (0.0,))))
        return (len(getattr(point, "forward_cands", ())), tuple(pos.tolist()))

    source_list.sort(key=_point_key)
    available_dests = list(dest_list)
    linked_sources = []
    linked_dests = []

    for source in source_list:
        forward_cands: list[Any] = list(getattr(source, "forward_cands", []))
        valid_cands = [
            (cand, dist)
            for cand, dist in forward_cands
            if cand is not None and cand in available_dests and dist <= float(search_range)
        ]
        if valid_cands:
            best_dest, _ = min(valid_cands, key=lambda item: item[1])
            available_dests.remove(best_dest)
            linked_sources.append(source)
            linked_dests.append(best_dest)
        else:
            linked_sources.append(source)
            linked_dests.append(None)

    for dest in available_dests:
        linked_sources.append(None)
        linked_dests.append(dest)

    return linked_sources, linked_dests


def _link_tracks_with_fallback(
    detections_df: pd.DataFrame,
    search_range_um: float,
    memory: int,
    px_per_micron: float | None,
    adaptive_stop_px: float = 1.0,
    adaptive_step: float = 0.95,
):
    try:
        import trackpy as tp  # type: ignore
    except Exception as exc:
        raise ImportError("trackpy is required for bead linking") from exc

    def _call_link_df(**kwargs):
        return tp.link_df(detections_df, **kwargs)

    base_kwargs = {
        "search_range": float(search_range_um),
        "memory": int(memory),
        "pos_columns": ["x_um", "y_um", "z_um"],
        "t_column": "frame",
    }

    strategy_supported = True

    if px_per_micron and float(px_per_micron) > 0:
        pixel_stop_um = float(adaptive_stop_px) / float(px_per_micron)
    else:
        pixel_stop_um = float(search_range_um)

    final_stop_um = min(float(search_range_um), max(pixel_stop_um, 1e-6))

    candidate_stops = [
        float(search_range_um) * 0.5,
        float(search_range_um) * 0.25,
        float(search_range_um) * 0.1,
        final_stop_um,
    ]
    adaptive_stops: list[float | None] = [None]
    for candidate_stop in candidate_stops:
        if candidate_stop >= final_stop_um and candidate_stop not in adaptive_stops:
            adaptive_stops.append(candidate_stop)
    last_exc = None

    attempt_variants: list[dict[str, Any]] = []
    for stop_scale in adaptive_stops:
        variant = dict(base_kwargs)
        if stop_scale is not None:
            variant["adaptive_stop"] = max(float(stop_scale), 1e-6)
            variant["adaptive_step"] = float(adaptive_step)
        attempt_variants.append(variant)

    if strategy_supported:
        recursive_variant = dict(base_kwargs)
        recursive_variant["adaptive_stop"] = final_stop_um
        recursive_variant["adaptive_step"] = float(adaptive_step)
        recursive_variant["link_strategy"] = "recursive"
        attempt_variants.append(recursive_variant)

        greedy_variant = dict(base_kwargs)
        greedy_variant["link_strategy"] = _greedy_subnet_linker
        attempt_variants.append(greedy_variant)

    for attempt_index, link_kwargs in enumerate(attempt_variants):
        try:
            tracks_df = _call_link_df(**link_kwargs)
            if link_kwargs.get("adaptive_stop") is not None:
                strategy_label = link_kwargs.get("link_strategy", "auto")
                print(
                    "trackpy link_df succeeded "
                    f"(strategy={strategy_label}, adaptive_stop={link_kwargs['adaptive_stop']:.3g} um)"
                )
            return tracks_df
        except Exception as exc:
            if exc.__class__.__name__ != "SubnetOversizeException":
                raise
            last_exc = exc
            if attempt_index == 0:
                print("trackpy link_df hit a subnet oversize; retrying with adaptive_stop")
            elif link_kwargs.get("link_strategy") == "recursive":
                print(
                    "trackpy link_df still oversize with recursive strategy at "
                    f"adaptive_stop={link_kwargs['adaptive_stop']:.3g} um"
                )
            else:
                print(
                    "trackpy link_df still oversize at "
                    f"adaptive_stop={link_kwargs['adaptive_stop']:.3g} um; retrying"
                )

    if last_exc is None:
        raise RuntimeError("trackpy link_df failed before a subnet exception was captured")
    raise last_exc


def _threshold_for_frame(args) -> float:
    images, ti_s, channel_to_use = args
    vol_full = images[ti_s, channel_to_use].compute()
    try:
        return float(threshold_beads(vol_full))
    except Exception:
        return float(np.mean(vol_full))


def _detect_frame_detections(args):
    (
        images,
        ti,
        channel_to_use,
        thr,
        min_size_voxels,
        max_bbox_ratio,
        max_inertia_ratio,
        dataset_id,
    ) = args

    raw_vol = images[ti, channel_to_use].compute()
    binary = raw_vol.astype(np.float32) > float(thr)
    binary = _remove_small_objects_compat(binary, int(min_size_voxels))

    lab = label(binary, connectivity=1)
    props = regionprops(lab, intensity_image=raw_vol)

    rows = []
    for p in props:
        if p.area < min_size_voxels:
            continue

        min_z, min_y, min_x, max_z, max_y, max_x = p.bbox
        dz, dy, dx = (max_z - min_z), (max_y - min_y), (max_x - min_x)
        if min(dz, dy, dx) <= 0:
            continue
        bbox_ratio = max(dz, dy, dx) / min(dz, dy, dx)
        if bbox_ratio > max_bbox_ratio:
            continue

        try:
            eig = np.asarray(p.inertia_tensor_eigvals, dtype=float)
        except Exception:
            eig = np.linalg.eigvalsh(p.inertia_tensor).astype(float)
        if np.any(eig <= 0):
            continue
        inertia_ratio = float(eig.max() / eig.min())
        if inertia_ratio > max_inertia_ratio:
            continue

        cz, cy, cx = p.centroid
        rows.append(
            {
                "dataset_id": dataset_id,
                "frame": int(ti),
                "z": float(cz),
                "y": float(cy),
                "x": float(cx),
                "area_vox": int(p.area),
                "mean_intensity": float(getattr(p, "mean_intensity", np.nan)),
                "max_intensity": float(getattr(p, "max_intensity", np.nan)),
                "bbox_ratio": float(bbox_ratio),
                "inertia_ratio": float(inertia_ratio),
            }
        )

    return rows


def _interpolated_thresholds(images, channel_to_use: int, n_samples: int) -> np.ndarray:
    t_len = int(images.shape[0])
    if t_len < 1:
        return np.array([], dtype=float)

    n_samples = min(max(1, int(n_samples)), t_len)
    sample_idx = np.linspace(0, t_len - 1, n_samples, dtype=int)
    workers = _resolve_parallel_workers(None, len(sample_idx))
    measured = np.asarray(
        _parallel_map(
            _threshold_for_frame,
            [(images, int(ti_s), int(channel_to_use)) for ti_s in sample_idx],
            workers,
            desc="threshold samples",
        ),
        dtype=float,
    )

    thresholds = np.interp(np.arange(t_len), sample_idx, measured).astype(float)
    thresholds_next = np.empty_like(thresholds)
    if t_len > 1:
        thresholds_next[:-1] = thresholds[1:]
    thresholds_next[-1] = thresholds[-1]
    return thresholds_next


def preview_bead_detection(state: Dict[str, Any], beads_cfg: Dict[str, Any], show: bool = True):
    images = state["images"]
    t_len = int(state["dims"]["T"])

    ti_preview = 0 if t_len <= 1 else min(1, t_len - 1)
    channel_to_use = int(beads_cfg.get("channel_to_use", 1))
    min_size_voxels = int(beads_cfg.get("min_size_voxels", 5))
    max_bbox_ratio = float(beads_cfg.get("max_bbox_ratio", 5.0))
    max_inertia_ratio = float(beads_cfg.get("max_inertia_ratio", 3.0))

    raw_vol = images[ti_preview, channel_to_use].compute()
    thr_preview = float(threshold_beads(raw_vol))

    binary = raw_vol.astype(np.float32) > thr_preview
    binary = _remove_small_objects_compat(binary, min_size_voxels)
    lab = label(binary, connectivity=1)
    props = regionprops(lab, intensity_image=raw_vol)

    keep_labels = []
    stats = []
    reject_small = 0
    reject_bbox = 0
    reject_inertia = 0

    for p in props:
        min_z, min_y, min_x, max_z, max_y, max_x = p.bbox
        dz, dy, dx = (max_z - min_z), (max_y - min_y), (max_x - min_x)
        bbox_ratio = (max(dz, dy, dx) / min(dz, dy, dx)) if min(dz, dy, dx) > 0 else np.inf
        try:
            eig = np.asarray(p.inertia_tensor_eigvals, dtype=float)
        except Exception:
            eig = np.linalg.eigvalsh(p.inertia_tensor).astype(float)
        inertia_ratio = float(eig.max() / eig.min()) if (eig.size and np.all(eig > 0)) else np.inf
        cz, cy, cx = p.centroid

        stats.append(
            {
                "label": int(p.label),
                "area_vox": int(p.area),
                "bbox_ratio": float(bbox_ratio),
                "inertia_ratio": float(inertia_ratio),
                "cz": float(cz),
                "cy": float(cy),
                "cx": float(cx),
            }
        )

        if p.area < min_size_voxels:
            reject_small += 1
            continue
        if bbox_ratio > max_bbox_ratio:
            reject_bbox += 1
            continue
        if inertia_ratio > max_inertia_ratio:
            reject_inertia += 1
            continue
        keep_labels.append(p.label)

    bead_mask_preview = np.isin(lab, keep_labels) if keep_labels else np.zeros_like(binary, dtype=bool)
    fig, axes = plot_bead_detection_preview(raw_vol, binary, bead_mask_preview, show=show)

    stats_df = pd.DataFrame(stats).sort_values("area_vox", ascending=False) if stats else pd.DataFrame()
    summary = {
        "frame": ti_preview,
        "threshold": thr_preview,
        "raw_components": int(len(props)),
        "kept_components": int(len(keep_labels)),
        "reject_small": int(reject_small),
        "reject_bbox": int(reject_bbox),
        "reject_inertia": int(reject_inertia),
    }
    return summary, stats_df, fig, axes


def detect_and_link_beads(state: Dict[str, Any], beads_cfg: Dict[str, Any], skip_existing: bool = True):
    images = state["images"]
    dataset_id = state["dataset_id"]
    t_len = int(state["dims"]["T"])
    y_len = int(state["dims"]["Y"])
    x_len = int(state["dims"]["X"])

    px_per_micron = state["calibration"].get("px_per_micron")
    px_per_micron_z = state["calibration"].get("px_per_micron_z")
    fps = state["calibration"].get("fps")

    derived_dir = state["paths"]["derived_dir"]
    detections_out = os.path.join(derived_dir, "beads_detections.parquet")
    tracks_out = os.path.join(derived_dir, "beads_tracks.parquet")

    if skip_existing and os.path.exists(detections_out) and os.path.exists(tracks_out):
        detections_df = pd.read_parquet(detections_out)
        tracks_df = pd.read_parquet(tracks_out)
        print("Loaded existing bead detections/tracks from disk")
        return detections_df, tracks_df

    channel_to_use = int(beads_cfg.get("channel_to_use", 1))
    min_size_voxels = int(beads_cfg.get("min_size_voxels", 5))
    max_bbox_ratio = float(beads_cfg.get("max_bbox_ratio", 5.0))
    max_inertia_ratio = float(beads_cfg.get("max_inertia_ratio", 3.0))
    search_range_um = float(beads_cfg.get("search_range_um", 10.0))
    memory = int(beads_cfg.get("memory", 1))
    min_track_length = int(beads_cfg.get("min_track_length", 0))
    threshold_samples = int(beads_cfg.get("threshold_samples", 10))
    parallel_workers = _resolve_parallel_workers(beads_cfg.get("parallel_workers", 0), t_len)

    thresholds_next = _interpolated_thresholds(images, channel_to_use, threshold_samples)
    jobs = [
        (
            images,
            int(ti),
            channel_to_use,
            float(thresholds_next[ti]),
            min_size_voxels,
            max_bbox_ratio,
            max_inertia_ratio,
            dataset_id,
        )
        for ti in range(t_len)
    ]
    detections_nested = _parallel_map(_detect_frame_detections, jobs, parallel_workers, desc="detect beads")
    detections = [row for rows in detections_nested for row in rows]

    detections_df = pd.DataFrame(detections)
    if len(detections_df) == 0:
        detections_df.to_parquet(detections_out, index=False)
        tracks_df = detections_df.copy()
        tracks_df.to_parquet(tracks_out, index=False)
        print("No bead detections found after filtering")
        return detections_df, tracks_df

    if px_per_micron:
        detections_df["x_um"] = detections_df["x"] / float(px_per_micron)
        detections_df["y_um"] = detections_df["y"] / float(px_per_micron)
    else:
        detections_df["x_um"] = detections_df["x"]
        detections_df["y_um"] = detections_df["y"]

    if px_per_micron_z:
        detections_df["z_um"] = detections_df["z"] / float(px_per_micron_z)
    else:
        detections_df["z_um"] = detections_df["z"]

    if fps:
        detections_df["t_s"] = detections_df["frame"] / float(fps)

    detections_df = detections_df.sort_values(["frame"]).reset_index(drop=True)

    tracks_df = detections_df.copy()
    if len(detections_df) > 0 and t_len > 1:
        tracks_df = _link_tracks_with_fallback(
            detections_df,
            search_range_um,
            memory,
            px_per_micron=px_per_micron,
            adaptive_stop_px=float(beads_cfg.get("adaptive_stop_px", 1.0)),
            adaptive_step=float(beads_cfg.get("adaptive_step", 0.95)),
        )
        try:
            import trackpy as tp  # type: ignore
        except Exception as exc:
            raise ImportError("trackpy is required for bead linking") from exc

        tracks_df = tracks_df.reset_index(drop=True)
        tracks_df = tp.filter_stubs(tracks_df, min_track_length)
        tracks_df = tracks_df.reset_index(drop=True)
        tracks_df = tracks_df.sort_values(by=["particle", "frame"], kind="mergesort").reset_index(drop=True)

    detections_df.to_parquet(detections_out, index=False)
    tracks_df.to_parquet(tracks_out, index=False)

    print(
        f"Saved detections to {detections_out} | rows={len(detections_df)}"
        f"\nSaved tracks to {tracks_out} | rows={len(tracks_df)}"
        f"\nFOV shape yx=({y_len},{x_len})"
    )
    return detections_df, tracks_df
