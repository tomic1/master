from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_SERIES_RE = re.compile(r"(Series\d+(?:-\d+)?)")


def slugify_keep_case(text: str) -> str:
    """Filesystem-safe slug; preserves case, replaces non-alnum with '_' and collapses repeats."""
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "dataset"


def derive_dataset_id(source_path: str | Path, scene_name: str) -> str:
    source_path = Path(source_path)
    sample_id = slugify_keep_case(source_path.stem)
    m = _SERIES_RE.search(scene_name)
    if m:
        series_id = m.group(1)
    else:
        series_id = slugify_keep_case(scene_name)
    return f"{sample_id}__{series_id}"


def dataset_root(base_dir: str | Path, dataset_id: str) -> Path:
    return Path(base_dir) / dataset_id


def raw_zarr_path(base_dir: str | Path, dataset_id: str) -> Path:
    return dataset_root(base_dir, dataset_id) / "raw" / "image.zarr"


def dataset_json_path(base_dir: str | Path, dataset_id: str) -> Path:
    return dataset_root(base_dir, dataset_id) / "dataset.json"


def write_dataset_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_dataset_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_record(
    dataset_id: str,
    source_path: str | Path,
    scene_name: str,
    axes: str,
    shape_tczyx: tuple[int, int, int, int, int],
    dtype: str,
    chunks_tczyx: tuple[int, int, int, int, int],
    pixel_size_xy_um: Optional[float],
    pixel_size_z_um: Optional[float],
    dt_s: Optional[float],
) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source_path": str(source_path),
        "scene_name": scene_name,
        "axes": axes,
        "shape_tczyx": list(shape_tczyx),
        "dtype": dtype,
        "chunks_tczyx": list(chunks_tczyx),
        "calibration": {
            "pixel_size_xy_um": pixel_size_xy_um,
            "pixel_size_z_um": pixel_size_z_um,
            "dt_s": dt_s,
        },
    }
