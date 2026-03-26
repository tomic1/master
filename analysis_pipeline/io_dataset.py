from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import dask.array as da
import numpy as np

from tomsUtilities import open_probabilities_lazy, open_raw_lazy, read_zarr_calibration


def prepare_output_dirs(dataset_id: str, variation: str = "") -> Dict[str, str]:
    plots_dir = Path("plots") / dataset_id / variation
    cache_dir = Path("plots") / dataset_id / "cache" / variation
    derived_dir = Path("data") / dataset_id / "derived"

    plots_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    return {
        "plots_dir": str(plots_dir),
        "cache_dir": str(cache_dir),
        "derived_dir": str(derived_dir),
    }


def load_dataset_state(dataset_cfg: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    base_dir = dataset_cfg["base_dir"]
    dataset_id = dataset_cfg["dataset_id"]
    variation = dataset_cfg.get("variation", "")

    if not dataset_id:
        raise ValueError("dataset.dataset_id must be set")

    px_per_micron, _, px_per_micron_z, fps = read_zarr_calibration(dataset_id, base_dir=base_dir)

    images, handle = open_raw_lazy(dataset_id, base_dir=base_dir)

    raw_mask = None
    if dataset_cfg.get("load_mask_channel", True):
        try:
            prob, _ = open_probabilities_lazy(dataset_id, base_dir=base_dir)
            if prob.ndim >= 4:
                raw_mask = prob
                images = da.concatenate((images, raw_mask), axis=1)
        except FileNotFoundError:
            raw_mask = None

    tstop = int(dataset_cfg.get("tstop", -1))
    if tstop != -1:
        images = images[:tstop]

    z_lower = int(dataset_cfg.get("z_lower", 0))
    z_upper = dataset_cfg.get("z_upper", None)
    if z_lower or z_upper is not None:
        images = images[:, :, z_lower:z_upper]

    xy_cut = float(dataset_cfg.get("xy_cut", 0.0))
    if xy_cut > 0:
        yc, xc = int(images.shape[-2]), int(images.shape[-1])
        y0 = int(yc * xy_cut)
        x0 = int(xc * xy_cut)
        images = images[:, :, :, y0 : yc - y0, x0 : xc - x0]

    sampling = int(dataset_cfg.get("sampling", 1))
    if sampling != 1:
        raise NotImplementedError("sampling != 1 is not implemented in load_dataset_state")

    t, c, z, y, x = images.shape
    output_paths = prepare_output_dirs(dataset_id, variation=variation)

    state = {
        "dataset_id": dataset_id,
        "variation": variation,
        "base_dir": base_dir,
        "images": images,
        "handle": handle,
        "raw_mask": raw_mask,
        "dims": {"T": int(t), "C": int(c), "Z": int(z), "Y": int(y), "X": int(x)},
        "calibration": {
            "px_per_micron": px_per_micron,
            "px_per_micron_z": px_per_micron_z,
            "fps": fps,
        },
        "paths": output_paths,
    }

    if verbose:
        raw_mb = float(np.prod(images.shape) * images.dtype.itemsize / (1024**2))
        meta = handle.get("meta", {}) if isinstance(handle, dict) else {}
        source_name = meta.get("name", dataset_id)
        channels = int(images.shape[1])
        px_xy_str = f"{float(px_per_micron):.4g}" if px_per_micron else "n/a"
        px_z_str = f"{float(px_per_micron_z):.4g}" if px_per_micron_z else "n/a"
        fps_str = f"{float(fps):.4g}" if fps else "n/a"
        print(
            "Loaded dataset"
            f"\n  id: {dataset_id}"
            f"\n  name: {source_name}"
            f"\n  variation: {variation or '-'}"
            f"\n  base_dir: {base_dir}"
            f"\n  shape (T,C,Z,Y,X): {tuple(images.shape)}"
            f"\n  chunks: {images.chunksize}"
            f"\n  size_gb: {raw_mb/1000.0:.2f}"
            f"\n  channels: {channels}"
            f"\n  px_per_micron: {px_xy_str}"
            f"\n  px_per_micron_z: {px_z_str}"
            f"\n  fps: {fps_str}"
            f"\n  mask channel loaded: {bool(raw_mask is not None)}"
        )

    return state


def should_skip(output_path: str, skip_existing: bool) -> bool:
    return skip_existing and os.path.exists(output_path)
