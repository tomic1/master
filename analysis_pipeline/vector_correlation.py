from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm


_VECTOR_COMPONENTS = ("vector",)


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


def _velocity_source(vector_cfg: Dict[str, Any]) -> str:
    source = str(vector_cfg.get("velocity_source", "raw")).strip().lower()
    if source in {"raw", "default"}:
        return "raw"
    if source in {"drift_corrected", "drift-corrected", "corrected", "drift"}:
        return "drift_corrected"
    raise ValueError(f"Unknown velocity_source: {source!r}")


def _velocity_output_name(base_name: str, vector_cfg: Dict[str, Any], distance_mode: str | None = None) -> str:
    suffixes: list[str] = []
    source = _velocity_source(vector_cfg)
    if source != "raw":
        suffixes.append(source)
    outlier_suffix = _velocity_outlier_suffix(vector_cfg)
    if outlier_suffix is not None:
        suffixes.append(outlier_suffix)
    if distance_mode and distance_mode != "xyz":
        suffixes.append(distance_mode)
    if not suffixes:
        return base_name
    return base_name.replace(".parquet", f"_{'_'.join(suffixes)}.parquet")


def _velocity_outlier_suffix(vector_cfg: Dict[str, Any]) -> str | None:
    if not bool(vector_cfg.get("exclude_velocity_outliers", False)):
        return None

    method = str(vector_cfg.get("velocity_outlier_method", "westerweel_scarano")).strip().lower()
    if method in {"westerweel_scarano", "westerweel-scarano", "westerweelscarano", "ws"}:
        return "westerweel_scarano"
    raise ValueError(f"Unknown velocity_outlier_method: {method!r}")


def _velocity_outlier_config(vector_cfg: Dict[str, Any]) -> tuple[str, int, int, float] | None:
    if not bool(vector_cfg.get("exclude_velocity_outliers", False)):
        return None

    method = str(vector_cfg.get("velocity_outlier_method", "westerweel_scarano")).strip().lower()
    if method not in {"westerweel_scarano", "westerweel-scarano", "westerweelscarano", "ws"}:
        raise ValueError(f"Unknown velocity_outlier_method: {method!r}")

    neighbor_count = max(1, int(vector_cfg.get("velocity_outlier_neighbor_count", 7)))
    min_neighbors = max(1, min(int(vector_cfg.get("velocity_outlier_min_neighbors", 4)), neighbor_count))
    threshold = float(vector_cfg.get("velocity_outlier_threshold", 2.0))
    return "westerweel_scarano", neighbor_count, min_neighbors, threshold


def _westerweel_scarano_outlier_scores(
    frame_df: pd.DataFrame,
    velocity_source: str,
    neighbor_count: int,
    min_neighbors: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Westerweel-Scarano scores and flags for a single frame."""

    scores = np.full(len(frame_df), np.nan, dtype=float)
    flags = np.zeros(len(frame_df), dtype=bool)

    if len(frame_df) < max(2, min_neighbors + 1):
        return scores, flags

    pos_x, pos_y, pos_z = _position_column_names(frame_df)
    vx_col, vy_col, vz_col = _velocity_column_names(frame_df, velocity_source)

    pos = np.column_stack(
        [
            frame_df[pos_x].to_numpy(dtype=float),
            frame_df[pos_y].to_numpy(dtype=float),
            frame_df[pos_z].to_numpy(dtype=float),
        ]
    )
    vec = np.column_stack(
        [
            frame_df[vx_col].to_numpy(dtype=float),
            frame_df[vy_col].to_numpy(dtype=float),
            frame_df[vz_col].to_numpy(dtype=float),
        ]
    )

    valid = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vec), axis=1)
    if int(np.count_nonzero(valid)) < max(2, min_neighbors + 1):
        return scores, flags

    valid_indices = np.flatnonzero(valid)
    pos_valid = pos[valid]
    vec_valid = vec[valid]

    query_k = min(pos_valid.shape[0], max(neighbor_count + 1, min_neighbors + 1))
    if query_k < 2:
        return scores, flags

    tree = KDTree(pos_valid)
    _, neighbor_ids = tree.query(pos_valid, k=query_k)
    neighbor_ids = np.asarray(neighbor_ids)
    if neighbor_ids.ndim == 1:
        neighbor_ids = neighbor_ids[:, None]

    for local_index, neighbor_row in enumerate(neighbor_ids):
        neighbors = neighbor_row[1:]
        neighbors = neighbors[neighbors != local_index]
        if neighbors.size < min_neighbors:
            continue

        neighbor_vecs = vec_valid[neighbors]
        median_vec = np.nanmedian(neighbor_vecs, axis=0)
        residual = float(np.linalg.norm(vec_valid[local_index] - median_vec))
        neighbor_residuals = np.linalg.norm(neighbor_vecs - median_vec, axis=1)
        scale = float(np.nanmedian(neighbor_residuals))
        score = residual / scale if scale > 0.0 and np.isfinite(scale) else (0.0 if residual == 0.0 else np.inf)

        global_index = valid_indices[local_index]
        scores[global_index] = score
        flags[global_index] = bool(score > threshold)

    return scores, flags


def _prepare_velocity_dataframe(vel_df: pd.DataFrame, vector_cfg: Dict[str, Any], velocity_source: str) -> pd.DataFrame:
    """Return a copy of vel_df with optional Westerweel-Scarano filtering applied."""

    if vel_df is None or len(vel_df) == 0:
        return pd.DataFrame()

    df = vel_df.copy().reset_index(drop=True)
    if "frame" not in df.columns:
        raise ValueError("vel_df must contain a frame column")

    df["frame"] = df["frame"].astype(int)
    outlier_cfg = _velocity_outlier_config(vector_cfg)
    if outlier_cfg is None:
        return df

    _, neighbor_count, min_neighbors, threshold = outlier_cfg
    score_values = np.full(len(df), np.nan, dtype=float)
    flag_values = np.zeros(len(df), dtype=bool)

    for _, frame_df in df.groupby("frame", sort=False):
        frame_scores, frame_flags = _westerweel_scarano_outlier_scores(
            frame_df,
            velocity_source,
            neighbor_count,
            min_neighbors,
            threshold,
        )
        score_values[frame_df.index.to_numpy()] = frame_scores
        flag_values[frame_df.index.to_numpy()] = frame_flags

    df["velocity_outlier_score"] = score_values
    df["velocity_outlier_flag"] = flag_values
    return df.loc[~df["velocity_outlier_flag"]].reset_index(drop=True)


def _velocity_column_names(df: pd.DataFrame, velocity_source: str = "raw") -> tuple[str, str, str]:
    if velocity_source == "raw":
        if {"vx_um_s", "vy_um_s", "vz_um_s"}.issubset(df.columns):
            return "vx_um_s", "vy_um_s", "vz_um_s"
        if {"vx", "vy", "vz"}.issubset(df.columns):
            return "vx", "vy", "vz"
        raise ValueError("velocity dataframe must contain vx/vy/vz or vx_um_s/vy_um_s/vz_um_s")
    if velocity_source == "drift_corrected":
        if {"vx_drift_corrected_um_s", "vy_drift_corrected_um_s", "vz_drift_corrected_um_s"}.issubset(df.columns):
            return "vx_drift_corrected_um_s", "vy_drift_corrected_um_s", "vz_drift_corrected_um_s"
        raise ValueError("velocity dataframe is missing drift-corrected columns; recompute beads_tracks_with_velocity.parquet")
    raise ValueError(f"Unknown velocity_source: {velocity_source!r}")


def _position_column_names(df: pd.DataFrame) -> tuple[str, str, str]:
    if {"x_um", "y_um", "z_um"}.issubset(df.columns):
        return "x_um", "y_um", "z_um"
    if {"x", "y", "z"}.issubset(df.columns):
        return "x", "y", "z"
    raise ValueError("velocity dataframe must contain x/y/z or x_um/y_um/z_um")

def _frame_seconds(state: Dict[str, Any]) -> float:
    fps = state["calibration"].get("fps")
    if fps and float(fps) > 0:
        return 1.0 / float(fps)
    return 1.0


def _unit_vectors_from_df(df: pd.DataFrame, use_unit_vectors: bool, velocity_source: str = "raw") -> np.ndarray:
    vx_col, vy_col, vz_col = _velocity_column_names(df, velocity_source)
    vec = np.column_stack(
        [
            df[vx_col].to_numpy(dtype=float),
            df[vy_col].to_numpy(dtype=float),
            df[vz_col].to_numpy(dtype=float),
        ]
    )
    if not use_unit_vectors:
        return vec

    norms = np.linalg.norm(vec, axis=1)
    unit = np.full_like(vec, np.nan, dtype=float)
    valid = np.isfinite(norms) & (norms > 0)
    unit[valid] = vec[valid] / norms[valid, None]
    return unit


def _pair_dot_score(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return normalized dot products for matched vector pairs."""

    score = np.einsum("ij,ij->i", left, right)
    return score[np.isfinite(score)]


def _accumulate_temporal_particle(
    particle_df: pd.DataFrame,
    max_lag_frames: int,
    velocity_source: str,
    use_unit_vectors: bool,
    dataset_id: str,
) -> pd.DataFrame | None:
    if len(particle_df) < 2:
        return None

    particle_df = particle_df.sort_values("frame", kind="mergesort")
    particle_id = int(particle_df["particle"].iloc[0])
    frames = particle_df["frame"].to_numpy(dtype=int)
    vectors = _unit_vectors_from_df(particle_df, use_unit_vectors, velocity_source)
    if vectors.shape[0] < 2:
        return None

    max_lag = min(int(max_lag_frames), vectors.shape[0] - 1)
    if max_lag < 1:
        return None

    rows: list[dict[str, Any]] = []

    for lag in range(1, max_lag + 1):
        frame_diff = frames[lag:] - frames[:-lag]
        valid = frame_diff == lag
        if not np.any(valid):
            continue

        left = vectors[:-lag][valid]
        right = vectors[lag:][valid]
        pair_scores = _pair_dot_score(left, right)
        if pair_scores.size == 0:
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "particle": particle_id,
                "component": "vector",
                "lag_frames": int(lag),
                "lag_s": np.nan,
                "score": float(np.mean(pair_scores)),
                "corr": float(np.mean(pair_scores)),
                "n_pairs": int(pair_scores.size),
                "mode": "dot",
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_temporal_single_vector_correlation(
    state: Dict[str, Any],
    vel_df: pd.DataFrame,
    vector_cfg: Dict[str, Any],
    skip_existing: bool = True,
    prepared_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    velocity_source = _velocity_source(vector_cfg)
    out_path = os.path.join(derived_dir, _velocity_output_name("beads_vector_correlation_temporal.parquet", vector_cfg))

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing temporal vector correlation from disk")
        return pd.read_parquet(out_path)

    if vel_df is None or len(vel_df) == 0:
        raise ValueError("vel_df is empty")
    if "particle" not in vel_df.columns or "frame" not in vel_df.columns:
        raise ValueError("vel_df must contain particle and frame columns")

    df = prepared_df.copy() if prepared_df is not None else _prepare_velocity_dataframe(vel_df, vector_cfg, velocity_source)
    df["particle"] = df["particle"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["particle", "frame"], kind="mergesort").reset_index(drop=True)

    max_lag_frames = int(vector_cfg.get("temporal_max_lag_frames", 50))
    max_lag_frames = max(1, max_lag_frames)
    use_unit_vectors = bool(vector_cfg.get("use_unit_vectors", True))
    vx_col, vy_col, vz_col = _velocity_column_names(df, velocity_source)

    particle_ids = df["particle"].dropna().astype(int).unique().tolist()
    jobs = [
        (
            df.loc[df["particle"] == particle_id, ["particle", "frame", vx_col, vy_col, vz_col, *(_position_column_names(df))]].copy(),
            max_lag_frames,
            velocity_source,
            use_unit_vectors,
            state["dataset_id"],
        )
        for particle_id in particle_ids
    ]

    parallel_workers = _resolve_parallel_workers(vector_cfg.get("parallel_workers", 0), len(jobs))
    particle_results = _parallel_map(lambda item: _accumulate_temporal_particle(*item), jobs, parallel_workers, desc="temporal vector corr")

    if not particle_results:
        out_df = pd.DataFrame(columns=["dataset_id", "particle", "component", "lag_frames", "lag_s", "score", "corr", "n_pairs", "mode", "velocity_source"])
        out_df.to_parquet(out_path, index=False)
        return out_df

    lag_step_s = _frame_seconds(state)
    frames = [result for result in particle_results if result is not None]
    if not frames:
        out_df = pd.DataFrame(columns=["dataset_id", "particle", "component", "lag_frames", "lag_s", "score", "corr", "n_pairs", "mode", "velocity_source"])
        out_df.to_parquet(out_path, index=False)
        return out_df

    out_df = pd.concat(frames, ignore_index=True)
    out_df["lag_s"] = out_df["lag_frames"].astype(float) * float(lag_step_s)
    out_df["velocity_source"] = velocity_source

    out_df.to_parquet(out_path, index=False)
    print(f"Saved temporal vector correlation to {out_path}")
    return out_df


def _accumulate_spatial_frame(
    frame_df: pd.DataFrame,
    bin_edges: np.ndarray,
    max_radius_um: float,
    dataset_id: str,
    velocity_source: str,
) -> pd.DataFrame | None:
    if len(frame_df) < 2:
        return None

    pos_x, pos_y, pos_z = _position_column_names(frame_df)
    vx_col, vy_col, vz_col = _velocity_column_names(frame_df)

    pos = np.column_stack(
        [
            frame_df[pos_x].to_numpy(dtype=float),
            frame_df[pos_y].to_numpy(dtype=float),
            frame_df[pos_z].to_numpy(dtype=float),
        ]
    )
    vec = np.column_stack(
        [
            frame_df[vx_col].to_numpy(dtype=float),
            frame_df[vy_col].to_numpy(dtype=float),
            frame_df[vz_col].to_numpy(dtype=float),
        ]
    )

    valid = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vec), axis=1)
    pos = pos[valid]
    vec = vec[valid]
    if pos.shape[0] < 2:
        return None

    norms = np.linalg.norm(vec, axis=1)
    nonzero = norms > 0
    pos = pos[nonzero]
    vec = vec[nonzero]
    norms = norms[nonzero]
    if pos.shape[0] < 2:
        return None
    vec = vec / norms[:, None]

    tree = KDTree(pos)
    pairs = np.array(list(tree.query_pairs(r=max_radius_um)))
    if pairs.size == 0:
        return None
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)

    left = pos[pairs[:, 0]]
    right = pos[pairs[:, 1]]
    distances = np.linalg.norm(left - right, axis=1)
    valid_pairs = np.isfinite(distances)
    if not np.any(valid_pairs):
        return None

    distances = distances[valid_pairs]
    valid_vectors = (
        np.all(np.isfinite(vec[pairs[valid_pairs, 0]]), axis=1)
        & np.all(np.isfinite(vec[pairs[valid_pairs, 1]]), axis=1)
    )
    if not np.any(valid_vectors):
        return None

    distances = distances[valid_vectors]
    left_vec = vec[pairs[valid_pairs, 0]][valid_vectors]
    right_vec = vec[pairs[valid_pairs, 1]][valid_vectors]
    pair_scores = _pair_dot_score(left_vec, right_vec)
    if pair_scores.size == 0:
        return None
    rows = [
        {
            "dataset_id": dataset_id,
            "frame": int(frame_df["frame"].iloc[0]),
            "distance_um": float(distance),
            "score": float(score),
            "corr": float(score),
            "mode": "dot",
            "velocity_source": velocity_source,
        }
        for distance, score in zip(distances.tolist(), pair_scores.tolist())
    ]
    if not rows:
        return None
    return pd.DataFrame(rows)


def _accumulate_spatial_tensor_frame(
    frame_df: pd.DataFrame,
    bin_edges: np.ndarray,
    max_radius_um: float,
    dataset_id: str,
    velocity_source: str,
    use_unit_vectors: bool,
    distance_mode: str,
) -> pd.DataFrame | None:
    if len(frame_df) < 2:
        return None

    pos_x, pos_y, pos_z = _position_column_names(frame_df)
    vx_col, vy_col, vz_col = _velocity_column_names(frame_df)

    pos = np.column_stack(
        [
            frame_df[pos_x].to_numpy(dtype=float),
            frame_df[pos_y].to_numpy(dtype=float),
            frame_df[pos_z].to_numpy(dtype=float),
        ]
    )
    vec = np.column_stack(
        [
            frame_df[vx_col].to_numpy(dtype=float),
            frame_df[vy_col].to_numpy(dtype=float),
            frame_df[vz_col].to_numpy(dtype=float),
        ]
    )

    valid = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vec), axis=1)
    pos = pos[valid]
    vec = vec[valid]
    if pos.shape[0] < 2:
        return None

    if use_unit_vectors:
        norms = np.linalg.norm(vec, axis=1)
        nonzero = norms > 0
        pos = pos[nonzero]
        vec = vec[nonzero]
        norms = norms[nonzero]
        if pos.shape[0] < 2:
            return None
        vec = vec / norms[:, None]

    distance_coords = _tensor_distance_coordinates(pos, distance_mode)
    tree = KDTree(distance_coords)
    pairs = np.array(list(tree.query_pairs(r=max_radius_um)))
    if pairs.size == 0:
        return None
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)

    left = distance_coords[pairs[:, 0]]
    right = distance_coords[pairs[:, 1]]
    distances = np.linalg.norm(left - right, axis=1)
    valid_pairs = np.isfinite(distances)
    if not np.any(valid_pairs):
        return None

    distances = distances[valid_pairs]
    left_vec = vec[pairs[valid_pairs, 0]]
    right_vec = vec[pairs[valid_pairs, 1]]
    valid_vectors = np.all(np.isfinite(left_vec), axis=1) & np.all(np.isfinite(right_vec), axis=1)
    if not np.any(valid_vectors):
        return None

    distances = distances[valid_vectors]
    left_vec = left_vec[valid_vectors]
    right_vec = right_vec[valid_vectors]
    pair_tensors = left_vec[:, :, None] * right_vec[:, None, :]
    if pair_tensors.size == 0:
        return None

    bin_index = np.digitize(distances, bin_edges) - 1
    keep = (bin_index >= 0) & (bin_index < len(bin_edges) - 1)
    if not np.any(keep):
        return None

    bin_index = bin_index[keep]
    pair_tensors = pair_tensors[keep]

    component_names = ("x", "y", "z")
    rows: list[dict[str, Any]] = []
    for bin_id in range(len(bin_edges) - 1):
        bin_mask = bin_index == bin_id
        if not np.any(bin_mask):
            continue

        tensor_mean = np.nanmean(pair_tensors[bin_mask], axis=0)
        symmetric = 0.5 * (tensor_mean + tensor_mean.T)
        antisymmetric = 0.5 * (tensor_mean - tensor_mean.T)
        distance_um = float(0.5 * (bin_edges[bin_id] + bin_edges[bin_id + 1]))
        n_pairs = int(np.count_nonzero(bin_mask))

        for part_name, tensor in (("full", tensor_mean), ("symmetric", symmetric), ("antisymmetric", antisymmetric)):
            for row_index, row_name in enumerate(component_names):
                for col_index, col_name in enumerate(component_names):
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "frame": int(frame_df["frame"].iloc[0]),
                            "distance_um": distance_um,
                            "row_component": row_name,
                            "col_component": col_name,
                            "component_pair": f"{row_name}{col_name}",
                            "part": part_name,
                            "corr": float(tensor[row_index, col_index]),
                            "n_pairs": n_pairs,
                        }
                    )

    if not rows:
        return None
    return pd.DataFrame(rows)


def _frame_candidates(frame_ids: list[int], start_index: int) -> list[int]:
    if not frame_ids:
        return []
    start_index = max(0, min(int(start_index), len(frame_ids) - 1))
    return frame_ids[start_index:] + frame_ids[:start_index]


def _select_frame_sample(frame_ids: list[int], start_index: int, sample_count: int | None) -> list[int]:
    candidate_frames = _frame_candidates(frame_ids, start_index)
    if not candidate_frames:
        return []

    if sample_count is None:
        return candidate_frames

    sample_count = max(1, int(sample_count))
    if sample_count >= len(candidate_frames):
        return candidate_frames

    sample_idx = np.linspace(0, len(candidate_frames) - 1, sample_count, dtype=int)
    return [candidate_frames[index] for index in np.unique(sample_idx)]


def _tensor_distance_mode(vector_cfg: Dict[str, Any]) -> str:
    mode = str(vector_cfg.get("tensor_distance_mode", "xyz")).strip().lower()
    if mode in {"xyz", "3d"}:
        return "xyz"
    if mode in {"xy", "2d"}:
        return "xy"
    raise ValueError(f"Unknown tensor_distance_mode: {mode!r}")


def _tensor_distance_coordinates(pos: np.ndarray, distance_mode: str) -> np.ndarray:
    return pos[:, :2] if distance_mode == "xy" else pos


def _tensor_default_max_radius(state: Dict[str, Any], px_xy: float, px_z: float, distance_mode: str) -> float:
    x_um = float(state["dims"]["X"]) / px_xy
    y_um = float(state["dims"]["Y"]) / px_xy
    if distance_mode == "xy":
        return 0.5 * min(x_um, y_um)
    z_um = float(state["dims"]["Z"]) / px_z
    return 0.5 * min(x_um, y_um, z_um)


def _weighted_average_profiles(profile_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if profile_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group_key, group_df in profile_df.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        corr_values = group_df["corr"].to_numpy(dtype=float)
        pair_counts = group_df["n_pairs"].to_numpy(dtype=float)
        valid = np.isfinite(corr_values) & np.isfinite(pair_counts) & (pair_counts > 0)
        if not np.any(valid):
            continue

        corr_values = corr_values[valid]
        pair_counts = pair_counts[valid]
        corr = float(np.average(corr_values, weights=pair_counts))
        corr_sem = float(np.nanstd(corr_values, ddof=1) / np.sqrt(corr_values.size)) if corr_values.size > 1 else np.nan

        row: dict[str, Any] = {column: value for column, value in zip(group_cols, group_key)}
        row["corr"] = corr
        row["score"] = corr
        row["corr_sem"] = corr_sem
        row["n_pairs"] = int(np.nansum(pair_counts))
        row["frame_count"] = int(group_df["frame"].nunique()) if "frame" in group_df.columns else int(len(group_df))

        for column in ("dataset_id", "mode", "part", "row_component", "col_component", "component_pair", "distance_mode", "velocity_source"):
            if column in group_df.columns and column not in row:
                row[column] = group_df[column].iloc[0]
        row["frame"] = -1
        rows.append(row)

    return pd.DataFrame(rows)


def _accumulate_spatial_profile_frame(
    frame_df: pd.DataFrame,
    bin_edges: np.ndarray,
    max_radius_um: float,
    dataset_id: str,
    velocity_source: str,
) -> pd.DataFrame | None:
    if len(frame_df) < 2:
        return None

    pos_x, pos_y, pos_z = _position_column_names(frame_df)
    vx_col, vy_col, vz_col = _velocity_column_names(frame_df)

    pos = np.column_stack(
        [
            frame_df[pos_x].to_numpy(dtype=float),
            frame_df[pos_y].to_numpy(dtype=float),
            frame_df[pos_z].to_numpy(dtype=float),
        ]
    )
    vec = np.column_stack(
        [
            frame_df[vx_col].to_numpy(dtype=float),
            frame_df[vy_col].to_numpy(dtype=float),
            frame_df[vz_col].to_numpy(dtype=float),
        ]
    )

    valid = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vec), axis=1)
    pos = pos[valid]
    vec = vec[valid]
    if pos.shape[0] < 2:
        return None

    norms = np.linalg.norm(vec, axis=1)
    nonzero = norms > 0
    pos = pos[nonzero]
    vec = vec[nonzero]
    norms = norms[nonzero]
    if pos.shape[0] < 2:
        return None
    vec = vec / norms[:, None]

    tree = KDTree(pos)
    pairs = np.array(list(tree.query_pairs(r=max_radius_um)))
    if pairs.size == 0:
        return None
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)

    left = pos[pairs[:, 0]]
    right = pos[pairs[:, 1]]
    distances = np.linalg.norm(left - right, axis=1)
    valid_pairs = np.isfinite(distances)
    if not np.any(valid_pairs):
        return None

    distances = distances[valid_pairs]
    valid_vectors = (
        np.all(np.isfinite(vec[pairs[valid_pairs, 0]]), axis=1)
        & np.all(np.isfinite(vec[pairs[valid_pairs, 1]]), axis=1)
    )
    if not np.any(valid_vectors):
        return None

    distances = distances[valid_vectors]
    left_vec = vec[pairs[valid_pairs, 0]][valid_vectors]
    right_vec = vec[pairs[valid_pairs, 1]][valid_vectors]
    pair_scores = _pair_dot_score(left_vec, right_vec)
    if pair_scores.size == 0:
        return None

    bin_index = np.digitize(distances, bin_edges) - 1
    keep = (bin_index >= 0) & (bin_index < len(bin_edges) - 1)
    if not np.any(keep):
        return None

    bin_index = bin_index[keep]
    pair_scores = pair_scores[keep]

    rows: list[dict[str, Any]] = []
    for bin_id in range(len(bin_edges) - 1):
        bin_mask = bin_index == bin_id
        if not np.any(bin_mask):
            continue

        bin_scores = pair_scores[bin_mask]
        n_pairs = int(bin_scores.size)
        corr = float(np.nanmean(bin_scores))
        corr_sem = float(np.nanstd(bin_scores, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan
        rows.append(
            {
                "dataset_id": dataset_id,
                "frame": int(frame_df["frame"].iloc[0]),
                "distance_um": float(0.5 * (bin_edges[bin_id] + bin_edges[bin_id + 1])),
                "score": corr,
                "corr": corr,
                "corr_sem": corr_sem,
                "mode": "dot",
                "n_pairs": n_pairs,
                "velocity_source": velocity_source,
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


def _accumulate_spatial_tensor_profile_frame(
    frame_df: pd.DataFrame,
    bin_edges: np.ndarray,
    max_radius_um: float,
    dataset_id: str,
    velocity_source: str,
    use_unit_vectors: bool,
    distance_mode: str,
) -> pd.DataFrame | None:
    if len(frame_df) < 2:
        return None

    pos_x, pos_y, pos_z = _position_column_names(frame_df)
    vx_col, vy_col, vz_col = _velocity_column_names(frame_df)

    pos = np.column_stack(
        [
            frame_df[pos_x].to_numpy(dtype=float),
            frame_df[pos_y].to_numpy(dtype=float),
            frame_df[pos_z].to_numpy(dtype=float),
        ]
    )
    vec = np.column_stack(
        [
            frame_df[vx_col].to_numpy(dtype=float),
            frame_df[vy_col].to_numpy(dtype=float),
            frame_df[vz_col].to_numpy(dtype=float),
        ]
    )

    valid = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vec), axis=1)
    pos = pos[valid]
    vec = vec[valid]
    if pos.shape[0] < 2:
        return None

    if use_unit_vectors:
        norms = np.linalg.norm(vec, axis=1)
        nonzero = norms > 0
        pos = pos[nonzero]
        vec = vec[nonzero]
        norms = norms[nonzero]
        if pos.shape[0] < 2:
            return None
        vec = vec / norms[:, None]

    distance_coords = _tensor_distance_coordinates(pos, distance_mode)
    tree = KDTree(distance_coords)
    pairs = np.array(list(tree.query_pairs(r=max_radius_um)))
    if pairs.size == 0:
        return None
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)

    left = distance_coords[pairs[:, 0]]
    right = distance_coords[pairs[:, 1]]
    distances = np.linalg.norm(left - right, axis=1)
    valid_pairs = np.isfinite(distances)
    if not np.any(valid_pairs):
        return None

    distances = distances[valid_pairs]
    left_vec = vec[pairs[valid_pairs, 0]]
    right_vec = vec[pairs[valid_pairs, 1]]
    valid_vectors = np.all(np.isfinite(left_vec), axis=1) & np.all(np.isfinite(right_vec), axis=1)
    if not np.any(valid_vectors):
        return None

    distances = distances[valid_vectors]
    left_vec = left_vec[valid_vectors]
    right_vec = right_vec[valid_vectors]
    pair_tensors = left_vec[:, :, None] * right_vec[:, None, :]
    if pair_tensors.size == 0:
        return None

    bin_index = np.digitize(distances, bin_edges) - 1
    keep = (bin_index >= 0) & (bin_index < len(bin_edges) - 1)
    if not np.any(keep):
        return None

    bin_index = bin_index[keep]
    pair_tensors = pair_tensors[keep]
    component_names = ("x", "y", "z")
    rows: list[dict[str, Any]] = []

    for bin_id in range(len(bin_edges) - 1):
        bin_mask = bin_index == bin_id
        if not np.any(bin_mask):
            continue

        tensor_bin = pair_tensors[bin_mask]
        distance_um = float(0.5 * (bin_edges[bin_id] + bin_edges[bin_id + 1]))
        n_pairs = int(tensor_bin.shape[0])
        tensor_parts = {
            "full": tensor_bin,
            "symmetric": 0.5 * (tensor_bin + np.swapaxes(tensor_bin, 1, 2)),
            "antisymmetric": 0.5 * (tensor_bin - np.swapaxes(tensor_bin, 1, 2)),
        }

        for part_name, tensor_values in tensor_parts.items():
            tensor_mean = np.nanmean(tensor_values, axis=0)
            for row_index, row_name in enumerate(component_names):
                for col_index, col_name in enumerate(component_names):
                    component_values = tensor_values[:, row_index, col_index]
                    corr = float(tensor_mean[row_index, col_index])
                    corr_sem = float(np.nanstd(component_values, ddof=1) / np.sqrt(component_values.size)) if component_values.size > 1 else np.nan
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "frame": int(frame_df["frame"].iloc[0]),
                            "distance_um": distance_um,
                            "row_component": row_name,
                            "col_component": col_name,
                            "component_pair": f"{row_name}{col_name}",
                            "part": part_name,
                            "distance_mode": distance_mode,
                            "velocity_source": velocity_source,
                            "corr": corr,
                            "corr_sem": corr_sem,
                            "n_pairs": n_pairs,
                        }
                    )

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_spatial_vector_tensor_correlation(
    state: Dict[str, Any],
    vel_df: pd.DataFrame,
    vector_cfg: Dict[str, Any],
    skip_existing: bool = True,
    prepared_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    multi_frame_average = bool(vector_cfg.get("multi_frame_average", False))
    velocity_source = _velocity_source(vector_cfg)
    distance_mode = _tensor_distance_mode(vector_cfg)
    out_name = _velocity_output_name(
        "beads_vector_correlation_tensor_avg.parquet" if multi_frame_average else "beads_vector_correlation_tensor.parquet",
        vector_cfg,
        distance_mode=distance_mode,
    )
    out_path = os.path.join(derived_dir, out_name)

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing spatial vector tensor correlation from disk")
        return pd.read_parquet(out_path)

    if vel_df is None or len(vel_df) == 0:
        raise ValueError("vel_df is empty")
    if "frame" not in vel_df.columns:
        raise ValueError("vel_df must contain a frame column")

    df = prepared_df.copy() if prepared_df is not None else _prepare_velocity_dataframe(vel_df, vector_cfg, velocity_source)
    df["frame"] = df["frame"].astype(int)

    frame_index = int(vector_cfg.get("tensor_frame_index", vector_cfg.get("spatial_frame_index", 0)))
    use_unit_vectors = bool(vector_cfg.get("tensor_use_unit_vectors", vector_cfg.get("use_unit_vectors", True)))

    px_xy = state["calibration"].get("px_per_micron")
    px_z = state["calibration"].get("px_per_micron_z") or px_xy
    if not px_xy:
        raise ValueError("px_per_micron is required for spatial vector tensor correlation")
    px_xy = float(px_xy)
    px_z = float(px_z) if px_z else float(px_xy)

    max_radius_um = vector_cfg.get("tensor_max_radius_um", vector_cfg.get("spatial_max_radius_um"))
    if max_radius_um is None:
        max_radius_um = _tensor_default_max_radius(state, px_xy, px_z, distance_mode)
    max_radius_um = float(max_radius_um)

    nbins = int(vector_cfg.get("tensor_nbins", vector_cfg.get("spatial_nbins", 40)))
    nbins = max(1, nbins)
    bin_edges = np.linspace(0.0, max_radius_um, nbins + 1)

    frame_ids = df["frame"].dropna().astype(int).unique().tolist()
    if not frame_ids:
        out_df = pd.DataFrame(columns=["dataset_id", "frame", "distance_um", "row_component", "col_component", "component_pair", "part", "distance_mode", "velocity_source", "corr", "n_pairs"])
        out_df.to_parquet(out_path, index=False)
        return out_df

    if multi_frame_average:
        sample_count = int(vector_cfg.get("multi_frame_count", 5))
        selected_frames = _select_frame_sample(frame_ids, frame_index, sample_count)
        print(f"Averaging spatial vector tensor correlation over {len(selected_frames)} frames")
        frame_profiles = [
            _accumulate_spatial_tensor_profile_frame(
                df.loc[df["frame"] == candidate_frame].copy(),
                bin_edges,
                max_radius_um,
                state["dataset_id"],
                velocity_source,
                use_unit_vectors,
                distance_mode,
            )
            for candidate_frame in selected_frames
        ]
        frame_profiles = [profile for profile in frame_profiles if profile is not None and not profile.empty]
        if not frame_profiles:
            out_df = pd.DataFrame(columns=["dataset_id", "frame", "distance_um", "row_component", "col_component", "component_pair", "part", "distance_mode", "velocity_source", "corr", "corr_sem", "n_pairs", "frame_count"])
            out_df.to_parquet(out_path, index=False)
            return out_df

        out_df = _weighted_average_profiles(
            pd.concat(frame_profiles, ignore_index=True),
            ["distance_um", "part", "component_pair", "row_component", "col_component"],
        )
        out_df["frame"] = -1
        out_df["dataset_id"] = state["dataset_id"]
        out_df["mode"] = "dot"
        out_df["distance_mode"] = distance_mode
        out_df["velocity_source"] = velocity_source
        out_df = out_df.sort_values(["part", "component_pair", "distance_um"], kind="mergesort").reset_index(drop=True)
    else:
        candidate_frames = _frame_candidates(frame_ids, frame_index)
        out_df = None
        selected_frame = None
        for candidate_frame in candidate_frames:
            frame_df = df.loc[df["frame"] == candidate_frame].copy()
            out_df = _accumulate_spatial_tensor_frame(frame_df, bin_edges, max_radius_um, state["dataset_id"], velocity_source, use_unit_vectors, distance_mode)
            if out_df is not None and not out_df.empty:
                selected_frame = candidate_frame
                break

        if out_df is None or out_df.empty:
            out_df = pd.DataFrame(columns=["dataset_id", "frame", "distance_um", "row_component", "col_component", "component_pair", "part", "distance_mode", "velocity_source", "corr", "n_pairs"])
            out_df.to_parquet(out_path, index=False)
            return out_df

        if selected_frame is not None:
            out_df["frame"] = int(selected_frame)
            print(f"Using frame {selected_frame} for spatial vector tensor correlation")

        out_df["distance_mode"] = distance_mode
        out_df["velocity_source"] = velocity_source

    out_df.to_parquet(out_path, index=False)
    print(f"Saved spatial vector tensor correlation to {out_path}")
    return out_df


def compute_spatial_vector_correlation(
    state: Dict[str, Any],
    vel_df: pd.DataFrame,
    vector_cfg: Dict[str, Any],
    skip_existing: bool = True,
    prepared_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    derived_dir = state["paths"]["derived_dir"]
    multi_frame_average = bool(vector_cfg.get("multi_frame_average", False))
    velocity_source = _velocity_source(vector_cfg)
    out_name = _velocity_output_name(
        "beads_vector_correlation_spatial_avg.parquet" if multi_frame_average else "beads_vector_correlation_spatial.parquet",
        vector_cfg,
    )
    out_path = os.path.join(derived_dir, out_name)

    if skip_existing and os.path.exists(out_path):
        print("Loaded existing spatial vector correlation from disk")
        return pd.read_parquet(out_path)

    if vel_df is None or len(vel_df) == 0:
        raise ValueError("vel_df is empty")
    if "frame" not in vel_df.columns:
        raise ValueError("vel_df must contain a frame column")

    df = prepared_df.copy() if prepared_df is not None else _prepare_velocity_dataframe(vel_df, vector_cfg, velocity_source)
    df["frame"] = df["frame"].astype(int)
    frame_index = int(vector_cfg.get("spatial_frame_index", 0))

    px_xy = state["calibration"].get("px_per_micron")
    px_z = state["calibration"].get("px_per_micron_z") or px_xy
    if not px_xy:
        raise ValueError("px_per_micron is required for spatial vector correlation")
    px_xy = float(px_xy)
    px_z = float(px_z) if px_z else float(px_xy)

    max_radius_um = vector_cfg.get("spatial_max_radius_um")
    if max_radius_um is None:
        x_um = float(state["dims"]["X"]) / px_xy
        y_um = float(state["dims"]["Y"]) / px_xy
        z_um = float(state["dims"]["Z"]) / px_z
        max_radius_um = 0.5 * min(x_um, y_um, z_um)
    max_radius_um = float(max_radius_um)

    nbins = int(vector_cfg.get("spatial_nbins", 40))
    nbins = max(1, nbins)
    bin_edges = np.linspace(0.0, max_radius_um, nbins + 1)

    frame_ids = df["frame"].dropna().astype(int).unique().tolist()
    if not frame_ids:
        out_df = pd.DataFrame(columns=["dataset_id", "frame", "distance_um", "score", "corr", "mode", "velocity_source"])
        out_df.to_parquet(out_path, index=False)
        return out_df
    if multi_frame_average:
        sample_count = int(vector_cfg.get("multi_frame_count", 5))
        selected_frames = _select_frame_sample(frame_ids, frame_index, sample_count)
        print(f"Averaging spatial vector correlation over {len(selected_frames)} frames")
        frame_profiles = [
            _accumulate_spatial_profile_frame(
                df.loc[df["frame"] == candidate_frame].copy(),
                bin_edges,
                max_radius_um,
                state["dataset_id"],
                velocity_source,
            )
            for candidate_frame in selected_frames
        ]
        frame_profiles = [profile for profile in frame_profiles if profile is not None and not profile.empty]
        if not frame_profiles:
            out_df = pd.DataFrame(columns=["dataset_id", "frame", "distance_um", "score", "corr", "corr_sem", "n_pairs", "frame_count", "mode", "velocity_source"])
            out_df.to_parquet(out_path, index=False)
            return out_df

        out_df = _weighted_average_profiles(pd.concat(frame_profiles, ignore_index=True), ["distance_um"])
        out_df["frame"] = -1
        out_df["dataset_id"] = state["dataset_id"]
        out_df["mode"] = "dot"
        out_df["velocity_source"] = velocity_source
        out_df = out_df.sort_values(["distance_um"], kind="mergesort").reset_index(drop=True)
    else:
        candidate_frames = _frame_candidates(frame_ids, frame_index)
        out_df = None
        selected_frame = None
        for candidate_frame in candidate_frames:
            frame_df = df.loc[df["frame"] == candidate_frame].copy()
            out_df = _accumulate_spatial_frame(frame_df, bin_edges, max_radius_um, state["dataset_id"], velocity_source)
            if out_df is not None and not out_df.empty:
                selected_frame = candidate_frame
                break

        if out_df is None or out_df.empty:
            out_df = pd.DataFrame(columns=["dataset_id", "frame", "distance_um", "score", "corr", "mode", "velocity_source"])
            out_df.to_parquet(out_path, index=False)
            return out_df

        if selected_frame is not None:
            out_df["frame"] = int(selected_frame)
            print(f"Using frame {selected_frame} for spatial vector correlation")

        out_df["velocity_source"] = velocity_source

    out_df.to_parquet(out_path, index=False)
    print(f"Saved spatial vector correlation to {out_path}")
    return out_df


def run_vector_correlation_core(
    config: Dict[str, Any],
    state: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    cfg = config if overrides is None else _merge_runtime(config, overrides)
    runtime = cfg.get("runtime", {})
    skip_existing = bool(runtime.get("skip_existing", True))

    if state is None:
        from .io_dataset import load_dataset_state

        state = load_dataset_state(cfg["dataset"], verbose=bool(runtime.get("verbose", True)))

    vector_cfg = cfg.get("vector_corr", {})
    if not bool(vector_cfg.get("enabled", True)):
        return {
            "temporal_vector_corr_df": pd.DataFrame(),
            "spatial_vector_corr_df": pd.DataFrame(),
            "spatial_df": pd.DataFrame(),
            "tensor_vector_corr_df": pd.DataFrame(),
            "tensor_df": pd.DataFrame(),
        }

    from .beads_velocity import compute_velocity_from_tracks

    tracks_vel_path = os.path.join(state["paths"]["derived_dir"], "beads_tracks_with_velocity.parquet")
    if os.path.exists(tracks_vel_path):
        vel_df = pd.read_parquet(tracks_vel_path)
    else:
        tracks_path = os.path.join(state["paths"]["derived_dir"], "beads_tracks.parquet")
        if not os.path.exists(tracks_path):
            raise FileNotFoundError(f"Missing tracks dataframe: {tracks_path}")
        tracks_df = pd.read_parquet(tracks_path)
        vel_df = compute_velocity_from_tracks(state, tracks_df, skip_existing=skip_existing)

    prepared_vel_df = _prepare_velocity_dataframe(vel_df, vector_cfg, _velocity_source(vector_cfg))
    if len(prepared_vel_df) != len(vel_df):
        print(
            "Applied Westerweel-Scarano velocity outlier rejection: "
            f"removed {len(vel_df) - len(prepared_vel_df)} of {len(vel_df)} vectors"
        )

    temporal_df = compute_temporal_single_vector_correlation(
        state,
        vel_df,
        vector_cfg,
        skip_existing=skip_existing,
        prepared_df=prepared_vel_df,
    )
    spatial_df = compute_spatial_vector_correlation(
        state,
        vel_df,
        vector_cfg,
        skip_existing=skip_existing,
        prepared_df=prepared_vel_df,
    )
    tensor_df = (
        compute_spatial_vector_tensor_correlation(
            state,
            vel_df,
            vector_cfg,
            skip_existing=skip_existing,
            prepared_df=prepared_vel_df,
        )
        if bool(vector_cfg.get("tensor_enabled", True))
        else pd.DataFrame()
    )

    return {
        "temporal_vector_corr_df": temporal_df,
        "spatial_vector_corr_df": spatial_df,
        "spatial_df": spatial_df,
        "tensor_vector_corr_df": tensor_df,
        "tensor_df": tensor_df,
    }


def _merge_runtime(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    from .config import merge_overrides

    return merge_overrides(config, overrides)