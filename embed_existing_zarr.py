from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from dataset_layout import dataset_json_path, dataset_record, dataset_root, derive_dataset_id, write_dataset_json


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _series_dirs(native_zarr_path: Path) -> List[Path]:
    out: List[Path] = []
    for child in native_zarr_path.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / ".zattrs").exists():
            out.append(child)
    return sorted(out, key=lambda p: int(p.name))


def _to_um(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    u = unit.strip().lower()
    mapping = {
        "um": 1.0,
        "micrometer": 1.0,
        "micrometers": 1.0,
        "nm": 1e-3,
        "nanometer": 1e-3,
        "nanometers": 1e-3,
        "mm": 1e3,
        "millimeter": 1e3,
        "millimeters": 1e3,
        "cm": 1e4,
        "m": 1e6,
        "meter": 1e6,
        "meters": 1e6,
    }
    return mapping.get(u, 1.0)


def _to_seconds(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    u = unit.strip().lower()
    mapping = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
    return mapping.get(u, 1.0)


def _unit_to_seconds(unit: Optional[str]) -> float | None:
    if not unit:
        return None
    u = unit.strip().lower()
    mapping = {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "µs": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
        "min": 60.0,
        "mins": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
    }
    return mapping.get(u)


def _median_positive_diffs(values: List[float]) -> float | None:
    if len(values) < 2:
        return None
    uniq = sorted(set(v for v in values if isinstance(v, (int, float))))
    if len(uniq) < 2:
        return None
    diffs = [b - a for a, b in zip(uniq[:-1], uniq[1:]) if (b - a) > 0]
    return float(median(diffs)) if diffs else None


def _get_original_metadata(
    img: Any,
    cscene: Optional[str] = None,
    ns: str = "http://www.openmicroscopy.org/Schemas/OME/2016-06",
    separate_values: bool = False,
):
    ome_types = importlib.import_module("ome_types")

    imgmeta = img.metadata
    meta = ET.fromstring(ome_types.to_xml(imgmeta))
    ometa_dict: Dict[str, str] = {}
    unitsre = None
    valuere = None
    ounits: List[tuple[int, str]] = []
    ovalue: List[tuple[int, str]] = []

    if separate_values:
        if not cscene:
            return imgmeta, ometa_dict, [], []
        unitsre = re.compile(rf"^{re.escape(cscene)}\s+Units\s+#(?P<number>\d+)$")
        valuere = re.compile(rf"^{re.escape(cscene)}\s+Value\s+#(?P<number>\d+)$")

    for ometa in meta.findall(".//ns:OriginalMetadata", namespaces={"ns": ns}):
        key_elem = ometa.find("ns:Key", namespaces={"ns": ns})
        val_elem = ometa.find("ns:Value", namespaces={"ns": ns})
        if key_elem is None or key_elem.text is None or val_elem is None or val_elem.text is None:
            continue

        key = key_elem.text
        value = val_elem.text

        if cscene is None:
            ometa_dict[key] = value
            continue

        if key.startswith(cscene):
            ometa_dict[key] = value
            if separate_values and unitsre is not None and valuere is not None:
                match = unitsre.match(key)
                if match:
                    ounits.append((int(match.group("number")), value))
                match = valuere.match(key)
                if match:
                    ovalue.append((int(match.group("number")), value))

    if separate_values:
        ounits.sort(key=lambda x: x[0])
        ovalue.sort(key=lambda x: x[0])
        return imgmeta, ometa_dict, [x[1] for x in ovalue], [x[1] for x in ounits]
    return imgmeta, ometa_dict, [], []


def _compute_dt_s_from_xml(root_xml: ET.Element, idx: int) -> float | None:
    ns = {}
    if root_xml.tag.startswith("{"):
        ns["ome"] = root_xml.tag.split("}")[0].strip("{")
        pixels_elem = root_xml.find(f".//ome:Image[@ID='Image:{idx}']/ome:Pixels", namespaces=ns)
    else:
        pixels_elem = root_xml.find(f".//Image[@ID='Image:{idx}']/Pixels")
    if pixels_elem is None:
        return None

    try:
        size_t = int(pixels_elem.get("SizeT", 1))
    except Exception:
        size_t = 1

    time_increment = pixels_elem.get("TimeIncrement")
    if time_increment is not None:
        try:
            dt = float(time_increment)
        except Exception:
            dt = None
        if dt is not None:
            unit = pixels_elem.get("TimeIncrementUnit")
            factor = _unit_to_seconds(unit) or 1.0
            if dt > 0:
                return dt * factor if size_t > 1 else None

    if size_t <= 1:
        return None

    planes = pixels_elem.findall("ome:Plane", namespaces=ns) if ns else pixels_elem.findall("Plane")
    if not planes:
        return None

    unit = planes[0].get("DeltaTUnit")
    factor = _unit_to_seconds(unit) or 1.0

    per_t: Dict[int, float] = {}
    for plane in planes:
        dt_str = plane.get("DeltaT")
        if dt_str is None:
            continue
        try:
            dt_val = float(dt_str)
        except Exception:
            continue

        the_t_str = plane.get("TheT")
        the_c_str = plane.get("TheC")
        the_z_str = plane.get("TheZ")
        try:
            the_t = int(the_t_str) if the_t_str is not None else None
            the_c = int(the_c_str) if the_c_str is not None else None
            the_z = int(the_z_str) if the_z_str is not None else None
        except Exception:
            the_t = None
            the_c = None
            the_z = None

        if the_t is None:
            continue
        if the_c in (None, 0) and the_z in (None, 0):
            per_t.setdefault(the_t, dt_val)

    if len(per_t) >= 2:
        ts = sorted(per_t.keys())
        dts = [float(per_t[t]) * factor for t in ts]
        dt_from_per_t = _median_positive_diffs(dts)
        if dt_from_per_t is not None:
            return dt_from_per_t

    vals = []
    for plane in planes:
        dt_str = plane.get("DeltaT")
        if dt_str is None:
            continue
        try:
            vals.append(float(dt_str))
        except Exception:
            continue
    if len(vals) >= 2:
        vals_s = [float(v) * factor for v in vals]
        dt_from_vals = _median_positive_diffs(vals_s)
        if dt_from_vals is not None:
            return dt_from_vals
    return None


def _compute_dt_s_from_olympus_original_metadata(
    img: Any,
    scene_name: str,
    img_shape_tczyx: tuple[int, int, int, int, int],
    dimension_order: str,
) -> float | None:
    t, c, z, _, _ = (int(x) for x in img_shape_tczyx)
    if t <= 1:
        return None

    _, _, ovalue, ounits = _get_original_metadata(img, cscene=scene_name, separate_values=True)
    if not ovalue:
        return None

    unit = ounits[0] if ounits else None
    factor = _unit_to_seconds(unit) or 1.0

    n = t * c * z
    if len(ovalue) < n:
        return None

    try:
        vals = [float(v) for v in ovalue[:n]]
    except Exception:
        return None

    if dimension_order == "XYCZT":
        try:
            per_frame = []
            for t_idx in range(t):
                idx_flat = 0 * (z * t) + 0 * t + t_idx
                per_frame.append(vals[idx_flat] * factor)
        except Exception:
            return None
    else:
        try:
            per_frame = []
            for t_idx in range(t):
                idx_flat = t_idx * (c * z)
                per_frame.append(vals[idx_flat] * factor)
        except Exception:
            return None

    diffs = [b - a for a, b in zip(per_frame[:-1], per_frame[1:]) if (b - a) > 0]
    if not diffs:
        return None
    return float(median(diffs))


def _build_vsi_timing_context(source_path: Path):
    if source_path.suffix.lower() != ".vsi":
        return None

    try:
        bioio = importlib.import_module("bioio")
        bioio_bioformats = importlib.import_module("bioio_bioformats")
        ome_types = importlib.import_module("ome_types")
    except Exception as exc:
        print(f"Warning: VSI timing requires bioio/bioio_bioformats/ome_types ({exc})")
        return None

    try:
        img = bioio.BioImage(source_path, reader=bioio_bioformats.Reader, original_meta=True, memoize=0)
    except Exception as exc:
        print(f"Warning: could not open VSI source for timing ({source_path}): {exc}")
        return None

    try:
        root_xml = ET.fromstring(ome_types.to_xml(img.metadata))
    except Exception as exc:
        print(f"Warning: could not parse OME-XML for timing ({source_path}): {exc}")
        return None

    return {"img": img, "root_xml": root_xml}


def _vsi_dt_s_for_series(
    timing_ctx,
    scene_index: int,
    scene_name: str,
    shape_tczyx: tuple[int, int, int, int, int],
) -> float | None:
    if timing_ctx is None:
        return None

    img = timing_ctx["img"]
    root_xml = timing_ctx["root_xml"]

    dt_s = _compute_dt_s_from_xml(root_xml, scene_index)
    if dt_s is not None:
        return dt_s

    # Match vsi2zarr.py fallback behavior using Olympus OriginalMetadata values.
    scene_candidates = [scene_name]
    try:
        if 0 <= scene_index < len(img.scenes):
            scene_candidates.append(str(img.scenes[scene_index]))
    except Exception:
        pass

    for candidate in scene_candidates:
        dt_s = _compute_dt_s_from_olympus_original_metadata(
            img=img,
            scene_name=candidate,
            img_shape_tczyx=shape_tczyx,
            dimension_order="XYCZT",
        )
        if dt_s is not None:
            return dt_s

    return None


def _parse_series_selection(scenes: Optional[str], available: List[int]) -> List[int]:
    if scenes is None:
        return available

    selected: List[int] = []
    tokens = [tok.strip() for tok in scenes.split(",") if tok.strip()]
    for tok in tokens:
        try:
            idx = int(tok)
        except ValueError:
            print(f"Ignoring non-integer series selector: {tok!r}")
            continue
        if idx in available:
            selected.append(idx)
        else:
            print(f"Requested series index not found and will be skipped: {idx}")
    return sorted(set(selected))


def _link_or_copy(src_series: Path, dst_image: Path, mode: str) -> None:
    if mode == "symlink":
        dst_image.symlink_to(src_series)
        return
    shutil.copytree(src_series, dst_image)


def _strip_leading_date_prefix(dataset_id: str) -> str:
    if "__" not in dataset_id:
        return re.sub(r"^\d{8}_", "", dataset_id)

    sample_part, series_part = dataset_id.split("__", 1)
    sample_part = re.sub(r"^\d{8}_", "", sample_part)
    if not sample_part:
        sample_part = dataset_id.split("__", 1)[0]
    return f"{sample_part}__{series_part}"


def adapt_existing_zarr(
    zarr_path: Path,
    output_root: Path,
    source_path: Optional[Path] = None,
    scenes: Optional[str] = None,
    overwrite: bool = False,
    include_macro: bool = False,
    mode: str = "symlink",
    strip_leading_date: bool = True,
) -> int:
    if not zarr_path.exists():
        print(f"Input zarr not found: {zarr_path}")
        return 1

    if source_path is None:
        source_path = zarr_path

    timing_ctx = _build_vsi_timing_context(source_path)

    series_paths = _series_dirs(zarr_path)
    if not series_paths:
        print(f"No numeric series groups found in zarr: {zarr_path}")
        return 1

    available_idx = [int(p.name) for p in series_paths]
    selected_idx = _parse_series_selection(scenes, available_idx)
    if not selected_idx:
        print("No matching series selected for adaptation.")
        return 1

    created = 0
    for idx in selected_idx:
        src_series = zarr_path / str(idx)
        if not src_series.exists():
            print(f"Series folder not found, skipping: {src_series}")
            continue

        attrs = _load_json(src_series / ".zattrs")
        multiscales = attrs.get("multiscales", [])
        ms0 = multiscales[0] if multiscales else {}
        scene_name = ms0.get("name") or f"Series{idx:03d}"

        if (not include_macro) and ("macro" in scene_name.lower()):
            print(f"Skipping macro scene: {scene_name}")
            continue

        arr_meta = _load_json(src_series / "0" / ".zarray")
        shape = tuple(arr_meta.get("shape", []))
        chunks = tuple(arr_meta.get("chunks", []))
        dtype = str(arr_meta.get("dtype", "unknown"))

        if len(shape) != 5 or len(chunks) != 5:
            print(f"Series {idx} has unsupported shape/chunks for TCZYX: shape={shape}, chunks={chunks}")
            continue

        axes = ms0.get("axes", [])
        datasets = ms0.get("datasets", [])
        scale0 = None
        for ds in datasets:
            if str(ds.get("path")) == "0":
                for ct in ds.get("coordinateTransformations", []):
                    if ct.get("type") == "scale":
                        scale0 = ct.get("scale")
                        break
                break

        px_xy_um = None
        px_z_um = None
        dt_s = None
        if isinstance(scale0, list) and len(scale0) == len(axes):
            for ax, scale_val in zip(axes, scale0):
                ax_name = str(ax.get("name", "")).lower()
                unit = ax.get("unit")
                sval = float(scale_val)
                if ax_name in {"x", "y"}:
                    val = sval * _to_um(unit)
                    px_xy_um = val if px_xy_um is None else min(px_xy_um, val)
                elif ax_name == "z":
                    px_z_um = sval * _to_um(unit)

        # Use the original VSI metadata timing inference (same mechanism as vsi2zarr.py).
        dt_s_from_vsi = _vsi_dt_s_for_series(
            timing_ctx=timing_ctx,
            scene_index=idx,
            scene_name=scene_name,
            shape_tczyx=shape,
        )
        if dt_s_from_vsi is not None:
            dt_s = dt_s_from_vsi
        else:
            # Fallback for non-VSI usage: use time-axis scale from zarr metadata if present.
            if isinstance(scale0, list) and len(scale0) == len(axes):
                for ax, scale_val in zip(axes, scale0):
                    if str(ax.get("name", "")).lower() == "t":
                        dt_s = float(scale_val) * _to_seconds(ax.get("unit"))
                        break

        dataset_id = derive_dataset_id(source_path, scene_name)
        if strip_leading_date:
            dataset_id = _strip_leading_date_prefix(dataset_id)
        dst_root = dataset_root(output_root, dataset_id)
        dst_raw_parent = dst_root / "raw"
        dst_image = dst_raw_parent / "image.zarr"

        if dst_root.exists():
            if overwrite:
                shutil.rmtree(dst_root)
            else:
                print(f"Dataset exists, skipping (use --overwrite): {dst_root}")
                continue

        dst_raw_parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(src_series.resolve(), dst_image, mode=mode)

        meta = dataset_record(
            dataset_id=dataset_id,
            source_path=source_path,
            scene_name=scene_name,
            axes="TCZYX",
            shape_tczyx=shape,
            dtype=dtype,
            chunks_tczyx=chunks,
            pixel_size_xy_um=px_xy_um,
            pixel_size_z_um=px_z_um,
            dt_s=dt_s,
        )
        write_dataset_json(dataset_json_path(output_root, dataset_id), meta)
        print(f"Created dataset adapter: {dst_root}")
        created += 1

    if created == 0:
        print("No datasets were created.")
        return 1

    print(f"Done. Created {created} dataset adapter(s) in {output_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create notebook-compatible dataset adapter(s) from an existing bioformats-style zarr. "
            "Writes dataset.json and links/copies each selected series to raw/image.zarr."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to existing zarr root (contains numeric series folders)")
    parser.add_argument("--output-root", type=Path, default=Path("/Volumes/TOM_DATA"), help="Root where adapted datasets are written")
    parser.add_argument("--source-path", type=Path, default=None, help="Original source file path used to derive dataset_id")
    parser.add_argument("--scenes", type=str, default=None, help="Comma-separated series indices to include, e.g. 0,2")
    parser.add_argument("--include-macro", action="store_true", help="Include macro scenes (default skips them)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing adapted datasets")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Adapter mode: symlink (fast, no duplication) or copy (portable)",
    )
    parser.add_argument(
        "--embed-data",
        action="store_true",
        help="Embed data in the adapter by copying series files (equivalent to --mode copy)",
    )
    parser.add_argument(
        "--keep-leading-date",
        action="store_true",
        help="Keep leading YYYYMMDD_ token in dataset names (default strips it)",
    )
    args = parser.parse_args()

    if args.embed_data and args.mode != "symlink":
        parser.error("--embed-data cannot be combined with --mode; use only one")

    selected_mode = "copy" if args.embed_data else args.mode

    return adapt_existing_zarr(
        zarr_path=args.zarr_path.resolve(),
        output_root=args.output_root.resolve(),
        source_path=args.source_path.resolve() if args.source_path else None,
        scenes=args.scenes,
        overwrite=args.overwrite,
        include_macro=args.include_macro,
        mode=selected_mode,
        strip_leading_date=not args.keep_leading_date,
    )


if __name__ == "__main__":
    raise SystemExit(main())
