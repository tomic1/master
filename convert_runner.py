from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataset_layout import (
    dataset_json_path,
    dataset_record,
    dataset_root,
    derive_dataset_id,
    write_dataset_json,
)


def _pick_input_files_via_dialog() -> List[Path]:
    """Open a native file picker and return selected input files.

    Returns an empty list if dialogs are unavailable (e.g., headless host).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return []

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_types = [
            ("LIF files", "*.lif"),
            ("VSI files", "*.vsi"),
            ("All files", "*.*"),
        ]
        selected = filedialog.askopenfilenames(title="Select input files", filetypes=file_types)
        root.destroy()
        return [Path(p).resolve() for p in selected] if selected else []
    except Exception:
        return []


def _pick_output_dir_via_dialog() -> Optional[Path]:
    """Open a native folder picker and return selected output directory."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select output folder")
        root.destroy()
        return Path(selected).resolve() if selected else None
    except Exception:
        return None


def _hash_path(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _lock_path(output_root: Path, input_path: Path) -> Path:
    return output_root / "_locks" / f"{_hash_path(input_path)}.lock"


def _acquire_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _converter_for(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".lif":
        return "bioformats2raw"
    if ext == ".vsi":
        return "bioformats2raw"
    return None


def _custom_converter_for(path: Path) -> Optional[Path]:
    ext = path.suffix.lower()
    if ext == ".lif":
        return Path(__file__).with_name("lif2zarr.py")
    if ext == ".vsi":
        return Path(__file__).with_name("vsi2zarr.py")
    return None


def _output_zarr_path(output_root: Path, input_path: Path) -> Path:
    return output_root / f"{input_path.stem}.zarr"


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


def _parse_series_selection(scenes: Optional[str], available: List[int]) -> List[int]:
    if scenes is None:
        return available

    selected: List[int] = []
    tokens = [tok.strip() for tok in scenes.split(",") if tok.strip()]
    for tok in tokens:
        try:
            idx = int(tok)
        except ValueError:
            print(f"Ignoring non-integer series selector for bioformats adapter: {tok!r}")
            continue
        if idx in available:
            selected.append(idx)
        else:
            print(f"Requested series index not found in output and will be skipped: {idx}")
    return sorted(set(selected))


def _adapt_bioformats_output_to_custom_layout(
    native_zarr_path: Path,
    input_path: Path,
    output_root: Path,
    scenes: Optional[str],
    overwrite: bool,
    include_macro: bool,
) -> int:
    if not native_zarr_path.exists():
        print(f"Native bioformats output missing, cannot adapt: {native_zarr_path}")
        return 1

    series_paths = _series_dirs(native_zarr_path)
    if not series_paths:
        print(f"No numeric series groups found in bioformats output: {native_zarr_path}")
        return 1

    available_idx = [int(p.name) for p in series_paths]
    selected_idx = _parse_series_selection(scenes, available_idx)
    if not selected_idx:
        print("No matching series selected for custom layout adaptation.")
        return 1

    print(f"Adapting bioformats output into custom layout for series: {selected_idx}")
    converted_count = 0
    for idx in selected_idx:
        src_series = native_zarr_path / str(idx)
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
                elif ax_name == "t":
                    dt_s = sval * _to_seconds(unit)

        dataset_id = derive_dataset_id(input_path, scene_name)
        dst_root = dataset_root(output_root, dataset_id)
        dst_raw_parent = dst_root / "raw"
        dst_image = dst_raw_parent / "image.zarr"

        if dst_root.exists():
            if overwrite:
                print(f"Overwriting existing dataset: {dst_root}")
                shutil.rmtree(dst_root)
            else:
                print(f"Dataset exists, skipping (use --overwrite): {dst_root}")
                continue

        dst_raw_parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_series, dst_image)

        meta = dataset_record(
            dataset_id=dataset_id,
            source_path=input_path,
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
        print(f"Created custom dataset: {dst_root}")
        converted_count += 1

    if converted_count == 0:
        print("No datasets were adapted into custom layout.")
        return 1
    return 0


def _run_conversion(
    input_path: Path,
    output_root: Path,
    scenes: Optional[str],
    overwrite: bool,
    conda_env: str,
    backend: str,
    bioformats_layout: str,
    include_macro: bool,
) -> int:
    if _converter_for(input_path) is None:
        print(f"Skipping unsupported file: {input_path}")
        return 0

    lock_path = _lock_path(output_root, input_path)
    if not _acquire_lock(lock_path):
        print(f"Lock exists, skipping: {input_path}")
        return 0

    try:
        selected_backend = backend
        custom_converter = _custom_converter_for(input_path)
        if selected_backend == "auto":
            selected_backend = "custom" if (custom_converter and custom_converter.exists()) else "bioformats2raw"

        if selected_backend == "custom":
            if custom_converter is None or not custom_converter.exists():
                print(f"Custom converter missing, skipping: {input_path}")
                return 1
            cmd = [
                sys.executable,
                str(custom_converter),
                "--input",
                str(input_path),
                "--output-root",
                str(output_root),
            ]
            if scenes:
                cmd += ["--scenes", scenes]
            if overwrite:
                cmd += ["--overwrite"]
        else:
            out_path = _output_zarr_path(output_root, input_path)
            if out_path.exists():
                if overwrite:
                    print(f"Removing existing output: {out_path}")
                    if out_path.is_dir():
                        shutil.rmtree(out_path)
                    else:
                        out_path.unlink()
                else:
                    print(f"Output exists, skipping (use --overwrite): {out_path}")
                    return 0

            cmd = [
                "conda",
                "run",
                "-n",
                conda_env,
                "bioformats2raw",
                str(input_path),
                str(out_path),
            ]
            if scenes:
                cmd += ["--series", scenes]
        print(f"Running: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc

        if selected_backend == "bioformats2raw" and bioformats_layout == "custom":
            return _adapt_bioformats_output_to_custom_layout(
                native_zarr_path=out_path,
                input_path=input_path,
                output_root=output_root,
                scenes=scenes,
                overwrite=overwrite,
                include_macro=include_macro,
            )
        return rc
    finally:
        _release_lock(lock_path)


def _discover_inputs(root: Path, glob_pattern: str) -> List[Path]:
    return sorted({p for p in root.glob(glob_pattern) if p.is_file()})


def _iter_inputs(paths: Iterable[str], glob_pattern: Optional[str]) -> List[Path]:
    inputs: List[Path] = []
    if glob_pattern:
        inputs.extend(_discover_inputs(Path("."), glob_pattern))
    for p in paths:
        inputs.append(Path(p))
    # keep unique, existing files only
    uniq = sorted({p.resolve() for p in inputs})
    return [p for p in uniq if p.is_file()]


def main() -> None:
    # Single file, specify output folder, include scenes 0 and 2, overwrite, 4 parallel jobs
    # python /Users/tom/Uni_local/master_local/analysis/convert_runner.py /path/to/sample.lif --output-root /path/to/output --scenes 0,2 --overwrite --jobs 4

    # Use glob to convert many files under data/, 8 jobs
    # python /Users/tom/Uni_local/master_local/analysis/convert_runner.py --glob "data/**/*.vsi" --output-root /path/to/output --jobs 8

    # No args -> opens file dialog; then choose output folder interactively
    # python /Users/tom/Uni_local/master_local/analysis/convert_runner.py
    parser = argparse.ArgumentParser(description="Parallel conversion runner for .lif/.vsi datasets.")
    parser.add_argument("inputs", nargs="*", help="Input files (.lif, .vsi)")
    parser.add_argument("--glob", default=None, help="Glob pattern for input files (e.g. 'data/**/*.vsi')")
    parser.add_argument("--output-root", default=None, help="Output base folder")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene/series selectors")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset folder")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--conda-env", default="testenv", help="Conda env containing bioformats2raw (default: testenv)")
    parser.add_argument(
        "--backend",
        choices=["auto", "custom", "bioformats2raw"],
        default="auto",
        help="Converter backend: auto (prefer custom scripts if available), custom (lif2zarr/vsi2zarr), or bioformats2raw",
    )
    parser.add_argument(
        "--bioformats-layout",
        choices=["custom", "native"],
        default="custom",
        help="When backend resolves to bioformats2raw: custom writes dataset.json + raw/image.zarr, native keeps plain <name>.zarr",
    )
    parser.add_argument(
        "--include-macro",
        action="store_true",
        help="Include macro scene(s) when adapting bioformats2raw output to custom layout",
    )
    args = parser.parse_args()

    inputs = _iter_inputs(args.inputs, args.glob)
    output_root = Path('.')

    if not inputs:
        inputs = _pick_input_files_via_dialog()

    if not inputs:
        print("No input files provided.")
        return

    unsupported = [p for p in inputs if _converter_for(p) is None]
    if unsupported:
        print("Unsupported files will be skipped:")
        for p in unsupported:
            print(f"  - {p}")

    supported_inputs = [p for p in inputs if _converter_for(p) is not None]
    if not supported_inputs:
        print("No supported .lif/.vsi input files found.")
        return

    inputs = supported_inputs

    if not args.output_root:
        if input('Put output in input folder? (y/n) ').strip().lower() == 'y':
            output_root = inputs[0].parent
        else:
            picked = _pick_output_dir_via_dialog()
            if picked is not None:
                output_root = picked
    else:
        output_root = Path(args.output_root).resolve()

    jobs = max(1, int(args.jobs))
    is_windows = platform.system().lower().startswith('win')
    has_vsi = any(p.suffix.lower() == '.vsi' for p in inputs)

    # Java/Bio-Formats VSI conversion is crash-prone on Windows when many runs are concurrent.
    if is_windows and has_vsi and jobs > 1:
        print("Windows + .vsi detected: forcing --jobs 1 for stability.")
        jobs = 1

    if jobs == 1:
        for p in inputs:
            _run_conversion(
                p,
                output_root,
                args.scenes,
                args.overwrite,
                args.conda_env,
                args.backend,
                args.bioformats_layout,
                args.include_macro,
            )
        return

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [
            ex.submit(
                _run_conversion,
                p,
                output_root,
                args.scenes,
                args.overwrite,
                args.conda_env,
                args.backend,
                args.bioformats_layout,
                args.include_macro,
            )
            for p in inputs
        ]
        for fut in as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
