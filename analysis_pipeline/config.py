from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_config() -> Dict[str, Any]:
    return {
        "dataset": {
            "base_dir": "/Volumes/TOM_DATA",
            "dataset_id": "",
            "variation": "",
            "sampling": 1,
            "tstop": -1,
            "xy_cut": 0.0,
            "z_lower": 0,
            "z_upper": None,
            "load_mask_channel": True,
        },
        "beads": {
            "enabled": True,
            "channel_to_use": 1,
            "min_size_voxels": 5,
            "max_bbox_ratio": 5.0,
            "max_inertia_ratio": 3.0,
            "search_range_um": 10.0,
            "memory": 1,
            "min_track_length": 0,
            "threshold_samples": 10,
            "compute_angular_speed": True,
            "origin_x_um": None,
            "origin_y_um": None,
        },
        "autocorr": {
            "enabled": True,
            "channel_3d": 0,
            "single_frame_3d": 0,
            "sample_count_3d": 10,
            "sample_count_2d": 20,
            "middle_z_for_2d": "middle",
            "frame_2d": 0,
            "channel_2d": 0,
            "nbins": 120,
            "subtract_mean": False,
            "normalize": "c0",
        },
        "runtime": {
            "skip_existing": True,
            "save_preview_masks": False,
            "verbose": True,
        },
    }


def load_analysis_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("PyYAML is required for YAML config files") from exc
        with path.open("r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
    elif suffix == ".json":
        import json

        with path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

    if not isinstance(parsed, dict):
        raise ValueError("Config root must be a mapping")

    return _deep_merge(_default_config(), parsed)


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    if not overrides:
        return deepcopy(config)
    return _deep_merge(config, overrides)
