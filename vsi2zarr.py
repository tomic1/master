# System imports
import sys
import os
import faulthandler
#faulthandler.enable()
faulthandler.enable(file=open("faulthandler.log",'w'), all_threads=True)
print("Python exe:", sys.executable, flush=True)

# Further imports
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import shutil
import dask
from dask.diagnostics.progress import ProgressBar
import zarr
import ome_types
import re
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, add_metadata, write_plate_metadata
import traceback

# Compatibility shim: ome-zarr (FormatV01) requests `zarr_format=2`.
# With current zarr-python v3, Group.create_array expects `config={'zarr_format': 2}`
# rather than a direct `zarr_format=` kwarg. Dask forwards `zarr_format` into
# Group.create_array, which raises `TypeError` unless we translate it.
import inspect
from zarr.core.group import Group as _ZarrGroup


def _open_group_zarr_v2(store, mode: str = "w"):
    """Open a root Zarr group using *format v2* if supported.

    This codebase runs with zarr-python v3, but we want on-disk datasets to be
    compatible with readers that only understand Zarr v2.
    """
    params = inspect.signature(zarr.open_group).parameters
    if "zarr_format" in params:
        return zarr.open_group(store=store, mode=mode, zarr_format=2)
    if "config" in params:
        return zarr.open_group(store=store, mode=mode, config={"zarr_format": 2})
    return zarr.open_group(store=store, mode=mode)

_create_array_params = inspect.signature(_ZarrGroup.create_array).parameters
if "zarr_format" not in _create_array_params:
    _orig_create_array = _ZarrGroup.create_array

    def _create_array_compat(
        self,
        name,
        *args,
        zarr_format=None,
        dimension_separator=None,
        **kwargs,
    ):
        if zarr_format is not None:
            cfg = kwargs.get("config")
            if cfg is None:
                kwargs["config"] = {"zarr_format": zarr_format}
            elif isinstance(cfg, dict) and "zarr_format" not in cfg:
                kwargs["config"] = {**cfg, "zarr_format": zarr_format}

        # dask.array.to_zarr historically passes `dimension_separator` (Zarr v2).
        # zarr-python v3 uses `chunk_key_encoding` instead.
        if dimension_separator is not None and "chunk_key_encoding" not in kwargs:
            kwargs["chunk_key_encoding"] = {"name": "v2", "separator": dimension_separator}
        return _orig_create_array(self, name, *args, **kwargs)

    _ZarrGroup.create_array = _create_array_compat

from dataset_layout import (
    derive_dataset_id,
    dataset_root,
    raw_zarr_path,
    dataset_json_path,
    dataset_record,
    write_dataset_json,
)

# Java imports
import scyjava
logback_path = os.environ.get("BIOFORMATS_LOGBACK", "")
if logback_path and Path(logback_path).exists():
    scyjava.config.add_option(f"-Dlogback.configurationFile={logback_path}")

# `--enable-native-access` exists only on Java 17+.
# Some setups provide Java 11 where this option hard-fails JVM startup.
if os.environ.get("BIOFORMATS_ENABLE_NATIVE_ACCESS", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    scyjava.config.add_option(r"--enable-native-access=ALL-UNNAMED")

# Fix Bio-Formats Memoizer (Kryo) on Java 9+ / 17+ / 21+
scyjava.config.add_option(r"--add-opens=java.base/java.util.regex=ALL-UNNAMED")
scyjava.config.add_option(r"--add-opens=java.base/java.lang=ALL-UNNAMED")
scyjava.config.add_option(r"--add-opens=java.base/java.util=ALL-UNNAMED")

# Java-dependent imports
import bioio
import bioio_bioformats
from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target
print("bioio imported OK")
print(f"Bioio version: {bioio.__version__}")

# ilastik (as of today) does not support NGFF/OME-Zarr spec v0.5.
# ome-zarr-py supports writing older metadata versions via the `fmt` parameter.
try:
    from ome_zarr.format import FormatV04 as _OMEZarrFormatV04
except Exception:  # pragma: no cover
    _OMEZarrFormatV04 = None

cmapping = {
    "c555": "#FF0000",
    "BF": "#FFFFFF"
}


# --------------------------------------------------------------------------------------
# Optional script-config mode
#
# If you prefer to hardcode paths in the script (instead of passing CLI args), fill in the
# variables below. If you run without `--input`, the script will automatically fall back
# to `SCRIPT_INPUT_PATH`.
# --------------------------------------------------------------------------------------
SCRIPT_INPUT_PATH = "data/AMF_087_001.vsi"  # e.g. "/Volumes/.../file.vsi"
SCRIPT_OUTPUT_ROOT = "data"  # e.g. "/Volumes/.../20251112"
SCRIPT_SCENES = None  # e.g. "0" or "c640" or "0,1" or None for all
SCRIPT_OVERWRITE = True
SCRIPT_SKIP_MACRO = True


def create_bfmts2raw(img, zroot, oname="", opath=""):
    scenes = []
    for path, obj in zroot.groups():
        if not isinstance(obj, zarr.core.group.Group):
            continue
        if path == "OME":
            continue
        scenes.append(obj.path)
        add_metadata(obj, {"bioformats2raw.layout": 3})
    
    zroot.attrs["ome"] = {"version": "0.5",
                         "bioformats2raw.layout": 3}
    
    meta, vsi_meta, _, _ = get_originalMetadata(img)
    omegroup = zroot.create_group(name="OME")
    omegroup.attrs["scenes"] = scene_dict(scenes)
    omegroup.attrs["originalName"] = oname
    omegroup.attrs["originalLocation"] = opath
    omegroup.attrs["acquisitionSoftware"] = f"{vsi_meta['Product Name'][1:-1]} V{vsi_meta['Product Version'][1:-1]}"
    omegroup.attrs["ome"] = {"version": "0.5",
                         "series": img.scenes[:-1]}

    with open(omegroup.store.root / omegroup.path /"METADATA.ome.xml", "w", encoding="utf-8") as f:
        f.write(meta.to_xml())

def omero_dict(name, color, window):
    # Support multiple channels
    # color: list of color strings (e.g. ["#FF0000", "#00FF00"]) or single string
    # window: list of tuples (one per channel) or single tuple for all
    if isinstance(color, str):
        color = [color]
    if isinstance(window[0], (int, float)):
        window = [window] * len(color)
    channels = []
    for i, (c, w) in enumerate(zip(color, window)):
        channels.append({
            "label": f"{name}_ch{i+1}",
            "color": c[1:],
            "window": {
                "start": w[0],
                "end": w[1],
                "min": w[2],
                "max": w[3],
            },
            "active": True,
        })
    return {"name": name, "channels": channels}


def _unit_to_seconds(unit: str | None) -> float | None:
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


def _unit_to_micrometers(unit: str | None) -> float | None:
    if not unit:
        return None
    u = unit.strip().lower()
    mapping = {
        "um": 1.0,
        "µm": 1.0,
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
    return mapping.get(u)


def infer_pixel_size_z_um_from_planes(root_XML: ET.Element, idx: int) -> float | None:
    """Infer Z-step (micrometers) from Plane PositionZ samples.

    Some Olympus OME-XML exports omit Pixels@PhysicalSizeZ but populate Plane@PositionZ.
    We estimate spacing as the median positive difference between consecutive Z positions
    for a canonical stack (TheT=0, TheC=0).
    """
    ns = {}
    if root_XML.tag.startswith("{"):
        ns["ome"] = root_XML.tag.split("}")[0].strip("{")
        pixels_elem = root_XML.find(f".//ome:Image[@ID='Image:{idx}']/ome:Pixels", namespaces=ns)
    else:
        pixels_elem = root_XML.find(f".//Image[@ID='Image:{idx}']/Pixels")
    if pixels_elem is None:
        return None

    if ns:
        planes = pixels_elem.findall("ome:Plane", namespaces=ns)
    else:
        planes = pixels_elem.findall("Plane")
    if not planes:
        return None

    # Determine unit from plane if available
    unit = None
    for p in planes:
        unit = p.get("PositionZUnit")
        if unit:
            break
    factor = _unit_to_micrometers(unit) or 1.0

    # Collect canonical Z positions (T=0, C=0)
    z_pos: dict[int, float] = {}
    for p in planes:
        pos = p.get("PositionZ")
        if pos is None:
            continue
        try:
            pos_val = float(pos) * factor
        except Exception:
            continue

        the_t = p.get("TheT")
        the_c = p.get("TheC")
        the_z = p.get("TheZ")
        try:
            t = int(the_t) if the_t is not None else 0
            c = int(the_c) if the_c is not None else 0
            z = int(the_z) if the_z is not None else None
        except Exception:
            continue
        if z is None:
            continue
        if t == 0 and c == 0:
            z_pos.setdefault(z, pos_val)

    if len(z_pos) < 2:
        return None

    zs = np.array(sorted(z_pos.keys()), dtype=int)
    positions = np.array([z_pos[z] for z in zs], dtype=float)
    diffs = np.diff(positions)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs != 0]
    diffs = np.abs(diffs)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def compute_dt_s(root_XML: ET.Element, idx: int) -> float | None:
    """Infer frame spacing in seconds.

    Priority:
    1) Pixels@TimeIncrement (if present)
    2) Plane@DeltaT samples (median of diffs across T)
    """
    ns = {}
    if root_XML.tag.startswith("{"):
        ns["ome"] = root_XML.tag.split("}")[0].strip("{")
        pixels_elem = root_XML.find(f".//ome:Image[@ID='Image:{idx}']/ome:Pixels", namespaces=ns)
    else:
        pixels_elem = root_XML.find(f".//Image[@ID='Image:{idx}']/Pixels")
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

    if ns:
        planes = pixels_elem.findall("ome:Plane", namespaces=ns)
    else:
        planes = pixels_elem.findall("Plane")
    if not planes:
        return None

    unit = planes[0].get("DeltaTUnit")
    factor = _unit_to_seconds(unit) or 1.0

    per_t: dict[int, float] = {}
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
        ts = np.array(sorted(per_t.keys()), dtype=int)
        dts = np.array([per_t[t] for t in ts], dtype=float) * factor
        diffs = np.diff(dts)
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs))

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
        vals = np.array(sorted(set(vals)), dtype=float) * factor
        diffs = np.diff(vals)
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs))
    return None


def _compute_dt_s_from_olympus_original_metadata(
    img: bioio.BioImage,
    scene_name: str,
    img_shape_tczyx: tuple[int, int, int, int, int],
    dimension_order: str,
) -> float | None:
    """Fallback dt_s computation for Olympus VSI files.

    Many VSI exports store time in OriginalMetadata keys like:
    "<scene> Units #n" and "<scene> Value #n".

    Returns median positive frame-to-frame time difference in seconds.
    """
    t, c, z, _, _ = (int(x) for x in img_shape_tczyx)
    if t <= 1:
        return None

    try:
        _, _, ovalue, ounits = get_originalMetadata(img, cscene=scene_name, separate_values=True)
    except Exception:
        return None

    if not ovalue:
        return None

    # Determine unit scaling
    unit = None
    if ounits:
        unit = ounits[0]
    factor = _unit_to_seconds(unit) or 1.0

    # We only need the first T*C*Z entries.
    n = t * c * z
    if len(ovalue) < n:
        return None

    try:
        vals = np.asarray(ovalue[:n], dtype=float)
    except Exception:
        return None

    # VSI commonly reports planes in XYCZT order; rearrange into (T, C, Z).
    if dimension_order == "XYCZT":
        try:
            vals = vals.reshape((c, z, t)).transpose(2, 0, 1)
        except Exception:
            return None
    else:
        # Unknown order; best-effort: assume first dimension corresponds to T
        try:
            vals = vals.reshape((t, c, z))
        except Exception:
            return None

    per_frame = vals[:, 0, 0] * factor
    diffs = np.diff(per_frame)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def get_pixel_unit(root_XML: ET.Element, idx: int, ax: str):
    ns = {}
    if root_XML.tag.startswith("{"):
        ns["ome"] = root_XML.tag.split("}")[0].strip("{")
        img = root_XML.find(f".//ome:Image[@ID='Image:{idx}']/ome:Pixels", namespaces=ns)
    else:
        img = root_XML.find(f".//Image[@ID='Image:{idx}']/Pixels")
    if img is None:
        raise ValueError(f"Pixels not found for ID=Image:{idx!r}")

    if ax == 'X':
        unit = img.get("PhysicalSizeXUnit")
    elif ax == 'Y':
        unit = img.get("PhysicalSizeYUnit")
    elif ax == 'Z':
        unit = img.get("PhysicalSizeZUnit")
    else:
        raise ValueError(f"Axis {ax!r} not recognized. Use 'X', 'Y', or 'Z'.")
    return unit


def get_image_name(root_XML: ET.Element, idx: int) -> str:
    ns = {}
    if root_XML.tag.startswith("{"):
        ns["ome"] = root_XML.tag.split("}")[0].strip("{")
        img_elem = root_XML.find(f".//ome:Image[@ID='Image:{idx}']", namespaces=ns)
    else:
        img_elem = root_XML.find(f".//Image[@ID='Image:{idx}']")
    if img_elem is None:
        raise ValueError(f"Image element not found for ID=Image:{idx!r}")
    return img_elem.get("Name") or f"Image:{idx}"


def get_image_properties_from_xml(root_XML: ET.Element, idx: int) -> dict:
    ns = {}
    if root_XML.tag.startswith("{"):
        ns["ome"] = root_XML.tag.split("}")[0].strip("{")
        pixels_elem = root_XML.find(f".//ome:Image[@ID='Image:{idx}']/ome:Pixels", namespaces=ns)
    else:
        pixels_elem = root_XML.find(f".//Image[@ID='Image:{idx}']/Pixels")
    if pixels_elem is None:
        raise ValueError(f"Pixels not found for ID=Image:{idx!r}")

    size_t = int(pixels_elem.get("SizeT", 1))
    size_c = int(pixels_elem.get("SizeC", 1))
    size_z = int(pixels_elem.get("SizeZ", 1))
    size_y = int(pixels_elem.get("SizeY", 1))
    size_x = int(pixels_elem.get("SizeX", 1))
    dimension_order = pixels_elem.get("DimensionOrder", "XYCZT")

    pixel_type = pixels_elem.get("Type", "uint8")
    dtype_map = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "float": np.float32,
        "double": np.float64,
    }
    dtype = dtype_map.get(pixel_type, np.uint8)

    phys_size_x = float(pixels_elem.get("PhysicalSizeX", 1.0)) if pixels_elem.get("PhysicalSizeX") else None
    phys_size_y = float(pixels_elem.get("PhysicalSizeY", 1.0)) if pixels_elem.get("PhysicalSizeY") else None
    phys_size_z = float(pixels_elem.get("PhysicalSizeZ", 1.0)) if pixels_elem.get("PhysicalSizeZ") else None

    return {
        "shape": (size_t, size_c, size_z, size_y, size_x),
        "dimension_order": dimension_order,
        "dtype": dtype,
        "physical_pixel_sizes": {"X": phys_size_x, "Y": phys_size_y, "Z": phys_size_z},
    }

def axes_dict(axes_names, axes_types, axes_units):
    adict = [{"name":N, "type":T, "unit":U} for N,T,U in zip(axes_names, axes_types, axes_units)]
    # coosd = {
    #     "name" : "0",
    #     "axes" : axes_dict
    # }
    return adict

def scale_dict(axes_scale):
    sdict = [[{"type": "scale", "scale":axes_scale}]]
    return sdict

def scene_dict(scenes):
    scenesd = [{"id": S, "path": S} for S in scenes]
    return scenesd

def pos_data(img, cscene, order="TCZYX"):

    if not order=="TCZYX":
        raise NotImplementedError("Output shape must be (TCZYX)")

    idx = img.scenes.index(cscene)
    meta, _, ovalue, ounits = get_originalMetadata(img, cscene=cscene, separate_values=True)
    
    planes = meta.images[idx].pixels.planes
    if len(img.shape)==5:
        num_img = img.shape[0]*img.shape[1]*img.shape[2] # number of images in the stack
    else:
        raise NotImplementedError("Only supporting 5D data at the moment")
    
    pos_data = np.empty((5, num_img), dtype=float) # create a container to hold positional data (T, C, Z, Y, X) - same order as image
    pos_units = [ounits[0], "C", planes[0].position_z_unit.value, planes[0].position_y_unit.value, planes[0].position_x_unit.value] # Also extract units
    
    
    if meta.images[idx].pixels.dimension_order.value == "XYCZT": # This is the original order, but check if indeed true
        ounits[:num_img] = np.reshape(ounits[:num_img], (img.shape[1], img.shape[2], img.shape[0])).transpose(2,0,1).ravel() # Then do the rearrangement
        ovalue[:num_img] = np.reshape(ovalue[:num_img], (img.shape[1], img.shape[2], img.shape[0])).transpose(2,0,1).ravel() # for both units and values
    else:
        raise NotImplementedError(f"Dimension order {meta.images[idx].pixels.dimension_order.value} not supported yet. Only XYCZT is supported at the moment.")
    
    # Now, we just store them in the pos_data container
    for i in range(num_img):
        pos_data[0,i] = float(ovalue[i]) # T
        pos_data[1,i] = planes[i].the_c # C
        pos_data[2,i] = planes[i].position_z # Z
        pos_data[3,i] = planes[i].position_y # Y
        pos_data[4,i] = planes[i].position_x # X
    
    posd = [{"axis":A, "unit":U, "positioners":P.tolist()} 
            for A,U,P in zip(order, pos_units, pos_data)]
    return posd

def get_originalMetadata(img, cscene=None, ns="http://www.openmicroscopy.org/Schemas/OME/2016-06", separate_values=False):
    
    imgmeta = img.metadata
    meta = ET.fromstring(ome_types.to_xml(imgmeta))
    ometa_dict = {}
    
    if separate_values:
        unitsre = re.compile(rf"^{re.escape(cscene)}\s+Units\s+#(?P<number>\d+)$")
        valuere = re.compile(rf"^{re.escape(cscene)}\s+Value\s+#(?P<number>\d+)$")  
        ounits = []
        ovalue = []

    for ometa in meta.findall(".//ns:OriginalMetadata", namespaces={"ns": ns}):
        if cscene is None:
            key = ometa.find("ns:Key", namespaces={"ns": ns}).text
            if not key.startswith(tuple(img.scenes)) and key is not None:
                value = ometa.find("ns:Value", namespaces={"ns": ns}).text    
                ometa_dict[key] = value
        else:
            key = ometa.find("ns:Key", namespaces={"ns": ns}).text
            if key.startswith(cscene) and key is not None:
                value = ometa.find("ns:Value", namespaces={"ns": ns}).text    
                ometa_dict[key] = value
                if separate_values:
                    match = unitsre.match(key)
                    if match:
                        ounits.append((int(match.group("number")), value))
                    match = valuere.match(key)
                    if match:
                        ovalue.append((int(match.group("number")), value))
    if separate_values:
        ounits.sort(key=lambda x: x[0])
        ovalue.sort(key=lambda x: x[0])
        ounits = [x[1] for x in ounits]
        ovalue = [x[1] for x in ovalue]
        return imgmeta, ometa_dict, ovalue, ounits
    return imgmeta, ometa_dict, [], []

def convert_scene(
    img: bioio.BioImage,
    root_XML: ET.Element,
    scene_index: int,
    source_path: Path,
    output_root: Path,
    overwrite: bool = False,
    skip_macro: bool = True,
) -> None:
    try:
        image_props = get_image_properties_from_xml(root_XML, scene_index)
    except ValueError:
        print(f"Skipping scene index {scene_index}: metadata not found in XML")
        return

    img.set_scene(scene_index)
    if skip_macro and "macro" in str(img.current_scene).lower():
        print(f"Skipping scene {img.current_scene}.")
        return
    scene_name = str(img.current_scene)
    print(f"---------  {scene_name} / Nr. {img.current_scene_index}  ---------")

    img_shape = image_props["shape"]
    dimension_order = image_props["dimension_order"]
    img_dtype = image_props["dtype"]
    phys_sizes = image_props["physical_pixel_sizes"]

    if phys_sizes.get("Z") is None:
        phys_sizes["Z"] = infer_pixel_size_z_um_from_planes(root_XML, scene_index)

    data = img.get_image_dask_data("TCZYX")
    level_shapes = [data.shape]
    chunk_shapes = multiscale_chunk_size_from_memory_target(level_shapes, img_dtype, memory_target=8*1024**2) #(memory tartget in bytes! should be ~1-20MB)
    # OMERO required metadata
    if np.issubdtype(img_dtype, np.integer):
        pxrange = (np.iinfo(img_dtype).min, np.iinfo(img_dtype).max, np.iinfo(img_dtype).min, np.iinfo(img_dtype).max)
    else:
        # float fallback; OMERO windows are still required
        pxrange = (0.0, 1.0, 0.0, 1.0)
    # coordinateSystems required metadata
    axes_names = ['t','c','z','y','x']
    axes_types = ['time','channel','space','space','space']
    axes_units = [
        None,
        None,
        get_pixel_unit(root_XML, scene_index, 'Z'),
        get_pixel_unit(root_XML, scene_index, 'Y'),
        get_pixel_unit(root_XML, scene_index, 'X'),
    ]
    axes_scale = [
        1.0,
        1.0,
        phys_sizes["Z"] or 1.0,
        phys_sizes["Y"] or 1.0,
        phys_sizes["X"] or 1.0,
    ]
    dataset_id = derive_dataset_id(source_path, scene_name)
    root_path = dataset_root(output_root, dataset_id)
    zarr_path = raw_zarr_path(output_root, dataset_id)
    if root_path.exists():
        if overwrite:
            print(f"Overwriting existing dataset: {root_path}")
            try:
                shutil.rmtree(root_path)
            except PermissionError as exc:
                print(
                    f"Permission denied while removing existing dataset: {root_path}\n"
                    f"{exc}\n"
                    "Close any viewer/process using this folder and retry, or run without --overwrite.",
                    file=sys.stderr,
                    flush=True,
                )
                return
        else:
            print(f"Skipping existing dataset: {root_path}")
            return

    try:
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        print(
            f"Permission denied creating output folder: {zarr_path.parent}\n"
            f"{exc}\n"
            "Check write permissions on the share and ensure the dataset folder is not locked.",
            file=sys.stderr,
            flush=True,
        )
        return

    if root_path.exists() and not os.access(root_path, os.W_OK):
        print(
            f"Output folder is not writable: {root_path}",
            file=sys.stderr,
            flush=True,
        )
        return

    store = parse_url(zarr_path, mode='w').store
    root = _open_group_zarr_v2(store, mode="w")

    num_channels = img_shape[1]
    channel_colors = [cmapping.get(scene_name, "#FFFFFF")] * num_channels
    channel_windows = [pxrange] * num_channels
    meta_omero = omero_dict(scene_name, color=channel_colors, window=channel_windows)
    meta_axes = axes_dict(axes_names, axes_types, axes_units)
    meta_ascale = scale_dict(axes_scale)
    fmt = _OMEZarrFormatV04() if _OMEZarrFormatV04 is not None else None
    with dask.config.set(scheduler="threads"):
        with ProgressBar():
                write_image(
                    image=data,
                    group=root,
                    axes=meta_axes,
                    scaler=None,  # ideally, also include sharding for better access but easier local storage
                    storage_options=dict(chunks=chunk_shapes[0]),  # ome-zarr expects per-level chunks
                    coordinate_transformations=meta_ascale,
                    fmt=fmt,
                )
    add_metadata(root, {"omero":meta_omero})
    add_metadata(root, {"axesTrajectory":pos_data(img, img.current_scene)})

    shape_tczyx = tuple(int(x) for x in data.shape)
    dt_s = compute_dt_s(root_XML, scene_index)
    if dt_s is None:
        dt_s = _compute_dt_s_from_olympus_original_metadata(
            img=img,
            scene_name=scene_name,
            img_shape_tczyx=img_shape,
            dimension_order=dimension_order,
        )
    dataset_meta = dataset_record(
        dataset_id=dataset_id,
        source_path=source_path,
        scene_name=scene_name,
        axes="t,c,z,y,x",
        shape_tczyx=shape_tczyx,
        dtype=str(img_dtype),
        chunks_tczyx=tuple(chunk_shapes[0]),
        pixel_size_xy_um=phys_sizes["X"] or phys_sizes["Y"],
        pixel_size_z_um=phys_sizes["Z"],
        dt_s=dt_s,
    )
    write_dataset_json(dataset_json_path(output_root, dataset_id), dataset_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VSI to OME-Zarr datasets (one folder per dataset).")
    parser.add_argument("--use-script-paths", action="store_true", help="Use SCRIPT_* variables in this file")
    parser.add_argument("--input", default=None, help="Path to .vsi file")
    parser.add_argument("--output-root", default=None, help="Output base folder (default: data)")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene names or indices to include")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing dataset folder",
    )
    parser.add_argument(
        "--skip-macro",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip macro scenes",
    )
    args = parser.parse_args()

    use_script_defaults = bool(args.use_script_paths) or args.input is None
    if use_script_defaults:
        if args.input is None and SCRIPT_INPUT_PATH:
            args.input = SCRIPT_INPUT_PATH
        if args.output_root is None and SCRIPT_OUTPUT_ROOT:
            args.output_root = SCRIPT_OUTPUT_ROOT
        if args.scenes is None and SCRIPT_SCENES is not None:
            args.scenes = SCRIPT_SCENES
        if args.overwrite is None:
            args.overwrite = SCRIPT_OVERWRITE
        if args.skip_macro is None:
            args.skip_macro = SCRIPT_SKIP_MACRO

    if args.input is None:
        parser.error("--input is required (or set SCRIPT_INPUT_PATH in the file)")
    if args.output_root is None:
        args.output_root = "data"
    if args.overwrite is None:
        args.overwrite = False
    if args.skip_macro is None:
        args.skip_macro = True

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # Bio-Formats memoization can fail on some Java setups (IllegalAccessError during Kryo deserialization).
    # Disabling memoization makes VSI reading robust.
    img = bioio.BioImage(input_path, reader=bioio_bioformats.Reader, original_meta=True, memoize=0)
    print(f"Saving scenes {img.scenes} in OME-Zarr format to {output_root}")

    imgmeta = img.metadata
    root_XML = ET.fromstring(ome_types.to_xml(imgmeta))

    selected = None
    if args.scenes:
        items = [s.strip() for s in args.scenes.split(",") if s.strip()]
        selected = set(items)

    for i, scene in enumerate(img.scenes):
        if selected is not None:
            if str(i) not in selected and str(scene) not in selected:
                continue
        try:
            convert_scene(
                img=img,
                root_XML=root_XML,
                scene_index=i,
                source_path=input_path,
                output_root=output_root,
                overwrite=bool(args.overwrite),
                skip_macro=bool(args.skip_macro),
            )
        except Exception:
            print(f"ERROR while converting scene index={i} scene={scene!r}", file=sys.stderr, flush=True)
            traceback.print_exc()
            raise

    print("Finished successfully.")


if __name__ == "__main__":
    main()
