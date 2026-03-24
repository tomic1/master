import numpy as np
from numpy.fft import fftn, fftshift, fftfreq
import os, json, pickle, inspect
import shutil
import scipy.ndimage as nd
import subprocess
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from skimage import io, img_as_float, morphology, measure, filters
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2, ifftshift
from scipy.interpolate import griddata
from scipy.spatial import distance, cKDTree
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
import time
import pandas as pd
import tifffile
from typing import Dict, Any, List, Tuple
from pathlib import Path
from typing import Optional
import math
from skan import csr
import dask.array as da
import dask.dataframe as dd
import zarr
import cv2

from dataset_layout import (
    dataset_json_path,
    dataset_root,
    read_dataset_json,
)


tum_colors = [
    "#E37222",  # TUM Orange
    "#0065BD",  # TUM Blue
    "#A2AD00",  # TUM Green
    "#98C6EA",  # TUM Lighter Blue
    "#DAD7CB",  # TUM Light Gray
    "#64A0C8",  # TUM Light Blue
    "#005293",  # TUM Dark Blue
    "#000000",  # Black
    "#FFFFFF",  # White
    "#999999",  # TUM Gray
]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', tum_colors)

def create_video_from_images(image_folder, output_video_path, frame_rate=7):
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort images by filename

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = "temp_video.mp4"
    video = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write the frame to the video

    # Release the VideoWriter
    video.release()
    print(f"Temporary video saved as {temp_video_path}")

    # Use ffmpeg to add metadata
    metadata = f"original framerate={frame_rate}"
    ffmpeg_command = [
        'ffmpeg', '-i', temp_video_path, '-c', 'copy',
        '-metadata', metadata, output_video_path
    ]
    subprocess.run(ffmpeg_command)

    # Clean up temporary video file
    os.remove(temp_video_path)
    print(f"Final video saved as {output_video_path} with metadata: {metadata}")

def create_video_from_tiff(tiff_file_path, output_video_path, frame_rate=7, start=0, end=None, skip=1):
    # Open the multi-frame TIFF file
    tiff_image = Image.open(tiff_file_path)
    
    frames = []
    current_frame = 0
    
    try:
        while True:
            if start <= current_frame and (end is None or current_frame <= end):
                if (current_frame - start) % skip == 0:
                    frame = tiff_image.copy()  # Copy the current frame
                    frame = frame.convert('RGB')  # Ensure the frame is in RGB mode
                    frame = np.array(frame)  # Convert PIL image to numpy array
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame)
            tiff_image.seek(tiff_image.tell() + 1)  # Move to the next frame
            current_frame += 1
    except EOFError:
        pass  # End of file reached
    
    if not frames:
        print("No frames found in the specified range.")
        return
    
    # Get dimensions from the first frame
    height, width, layers = frames[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    for frame in frames:
        video.write(frame)  # Write the frame to the video

    # Release the VideoWriter
    video.release()
    print(f"Video saved as {output_video_path}")

def create_circular_matrix(size, diameter):
    Y, X = np.ogrid[:size, :size]
    center = size // 2
    mask = (X - center)**2 + (Y - center)**2 <= (diameter/2)**2
    return mask.astype(int)/np.sum(mask)

def smooth(images,kernel_smooth = create_circular_matrix(10,10)):
    out = np.array([nd.convolve(images[i],kernel_smooth/(kernel_smooth.sum()),mode='wrap')/(nd.convolve(images[i],kernel_smooth/(kernel_smooth.sum()),mode='wrap').max()) for i in range(images.shape[0])])
    return out

def filter(images, cutoff=50):
    rows, cols = images[0].shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    f_transform = fftshift(fft2(images))
    f_transform_shifted = f_transform * mask
    return np.array(abs(ifft2(ifftshift(f_transform_shifted))))

def filter_gaus(images, sigma=50.0):
    filtered_images = []
    m, n = [(ss-1.)/2. for ss in images[0].shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    for image in images:
        f_transform = fft2(image)
        f_transform_shifted = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        gaussian_filtered = magnitude_spectrum*h
        f_transform_shifted_filtered = gaussian_filtered * np.exp(1j * np.angle(f_transform_shifted))
        f_transform_filtered = ifftshift(f_transform_shifted_filtered)
        image_filtered = np.abs(ifft2(f_transform_filtered))
        filtered_images.append(image_filtered)
    return np.array(filtered_images)

def cut_intensity(images, lower=0, upper=1):
    return np.array([(images[i]/images[i].max() > lower) * (images[i]/images[i].max() < upper) for i in range(images.shape[0])]).astype(int)*255

def dilation(images, kernel_dilation = create_circular_matrix(4,4)):
    return np.array([nd.binary_dilation(images[i], structure=kernel_dilation) for i in range(images.shape[0])]).astype(int)*255

def edge(images,kernel_size=2):
    if kernel_size==2:
        ims_edge = np.abs(np.array([nd.convolve(images[i],np.array([[0,1],[-1,0]])) for i in range(images.shape[0])]))
        ims_edge += np.abs(np.array([nd.convolve(images[i],np.array([[1,0],[0,-1]])) for i in range(images.shape[0])]))
    if kernel_size==3:
        ims_edge = np.abs(np.array([nd.convolve(images[i],np.array([[0,1,0], [-1,0,1], [0,-1,0]])) for i in range(images.shape[0])]))
        ims_edge += np.abs(np.array([nd.convolve(images[i],np.array([[0,1,0], [1,0,-1], [0,-1,0]])) for i in range(images.shape[0])]))
    return ims_edge

def opening(images):
    kernel_opening = np.array([[1,1], [0,0]])
    kernel_opening_2 = np.array([[1,0], [1,0]])   
    ims_op = [nd.binary_opening(images[i], structure=kernel_opening) for i in range(images.shape[0])]
    ims_op = [nd.binary_opening(ims_op[i], structure=kernel_opening_2) for i in range(images.shape[0])]
    return ims_op

def nlM(images,h=1,ps=5,pd=3):

    #images=images.astype(float)
    sigma_est=np.array([])
    im=np.array([])
    for img in images:
        sigma_est = estimate_sigma(img, multichannel=False)
        denoise_img = denoise_nl_means(img, h=h * sigma_est, fast_mode=True,
                               patch_size=ps, patch_distance=pd, multichannel=False)
        im = np.append(im,denoise_img)
    im = im.reshape(images.shape)
    return im

def loadTif(fn):
    ext = str(fn).lower()
    if ext.endswith((".tif", ".tiff")):
        print('using tifffile to read tiff')
        ims = tifffile.imread(fn)
    else:
        ims = io.imread(fn)
    return ims

def _open_ome_zarr_array(zarr_path: Path):
    root = zarr.open_group(zarr_path, mode="r")
    if "0" in root:
        return root["0"]
    try:
        return zarr.open(zarr_path, mode="r")
    except Exception as exc:
        raise RuntimeError(f"Could not open zarr array at {zarr_path}") from exc


def open_dataset(dataset_id_or_path: str, base_dir: str = "data") -> Dict[str, Any]:
    candidate = Path(dataset_id_or_path)
    if candidate.exists():
        if candidate.is_dir() and (candidate / "dataset.json").exists():
            root = candidate
        elif candidate.name == "dataset.json":
            root = candidate.parent
        else:
            root = candidate
    else:
        root = dataset_root(base_dir, dataset_id_or_path)

    meta_path = root / "dataset.json"
    meta = read_dataset_json(meta_path)
    return {"root": root, "meta": meta}


def ensure_tzcyx(arr_tczyx: da.Array) -> da.Array:
    return arr_tczyx.transpose(0, 2, 1, 3, 4)


def read_zarr_calibration(dataset_id_or_path: str, base_dir: str = "data"):
    handle = open_dataset(dataset_id_or_path, base_dir=base_dir)
    calib = handle["meta"].get("calibration", {})
    pixel_size_xy_um = calib.get("pixel_size_xy_um")
    pixel_size_z_um = calib.get("pixel_size_z_um")

    # Time step can be stored inconsistently across exporters.
    # We expect seconds, but some datasets store milliseconds (e.g. 3000 for 3 s).
    dt_s = calib.get("dt_s")
    if dt_s in (None, "", 0):
        dt_s = calib.get("dt_ms")
        if dt_s not in (None, "", 0):
            dt_s = float(dt_s) / 1000.0
    else:
        dt_s = float(dt_s)
        if dt_s > 100:
            dt_s = dt_s / 1000.0

    fps = (1.0 / dt_s) if (dt_s and dt_s > 0) else None

    px_per_micron_xy = (1.0 / float(pixel_size_xy_um)) if pixel_size_xy_um else None
    px_per_micron_z = (1.0 / float(pixel_size_z_um)) if pixel_size_z_um else None
    return px_per_micron_xy, None, px_per_micron_z, fps


def open_raw_lazy(dataset_id_or_path: str, base_dir: str = "data", rechunk = True):
    handle = open_dataset(dataset_id_or_path, base_dir=base_dir)
    zarr_path = handle["root"] / "raw" / "image.zarr"
    arr = _open_ome_zarr_array(zarr_path)
    darr = da.from_zarr(arr)
    # Rechunk to pick one whole Z stack: (1, Z, Y, X) chunks
    if rechunk:
        T, C, Z, Y, X = darr.shape
        darr = darr.rechunk((1, 1, Z, Y, X))
    return darr, handle


def open_probabilities_lazy(dataset_id_or_path: str, base_dir: str = "data", h5_key: Optional[str] = None):
    handle = open_dataset(dataset_id_or_path, base_dir=base_dir)
    ilastik_dir = handle["root"] / "ilastik"
    zarr_path = ilastik_dir / "probabilities.zarr"
    if zarr_path.exists():
        arr = _open_ome_zarr_array(zarr_path)
        return da.from_zarr(arr), handle

    h5_path = ilastik_dir / "probabilities.h5"
    if h5_path.exists():
        try:
            import h5py  # type: ignore
        except Exception as exc:
            raise ImportError("h5py is required to read probabilities.h5") from exc
        f = h5py.File(h5_path, "r")
        if h5_key is None:
            # pick first dataset
            def _first_dataset(g):
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        return k
                    if isinstance(v, h5py.Group):
                        sub = _first_dataset(v)
                        if sub is not None:
                            return f"{k}/{sub}"
                return None
            h5_key = _first_dataset(f)
        if h5_key is None:
            raise ValueError("No dataset found in probabilities.h5")
        ds = f[h5_key]
        return da.from_array(ds, chunks=ds.chunks), handle

    npy_path = ilastik_dir / "probabilities.npy"
    if npy_path.exists():
        arr = np.load(npy_path, mmap_mode="r")
        chunks = (1,) + arr.shape[1:] if arr.ndim >= 2 else arr.shape
        return da.from_array(arr, chunks=chunks), handle

    raise FileNotFoundError(f"No ilastik probabilities found in {ilastik_dir}")

def choose_target_dtype(arr):
    """
    Return the smallest float dtype that preserves intensity levels.
    Uses value range for integer arrays; passes through float16, else float32.
    """
    if isinstance(arr, da.Array):
        return np.float32
    dt = arr.dtype
    if np.issubdtype(dt, np.integer):
        minv = int(arr.min())
        maxv = int(arr.max())
        levels = maxv - minv + 1
        return np.float16 if levels <= 1024 else np.float32
    if dt == np.float16:
        return np.float16
    return np.float32

def norm(img):
    if isinstance(img, da.Array):
        img = img.astype(choose_target_dtype(img), copy=False)
        minv = img.min(axis=(-2, -1), keepdims=True)
        maxv = img.max(axis=(-2, -1), keepdims=True)
        denom = maxv - minv
        denom = da.where(denom == 0, 1, denom)
        return (img - minv) / denom
    img = img.astype(choose_target_dtype(img), copy=False)
    t, z, c, y, x = img.shape
    for i in range(t):
        for j in range(z):
            for k in range(c):
                img[i, j, k] = (img[i, j, k] - img[i, j, k].min()) / (img[i, j, k].max() - img[i, j, k].min())
    return img

def radial_average(image, center=None, binsize=1):
    """
    Compute the radial average of a 2D image.

    Parameters:
    - image: 2D array (image data)
    - center: tuple (y, x) specifying the center. If None, uses image center.
    - binsize: size of radial bins in pixels

    Returns:
    - r_vals: 1D array of radial distances (bin centers)
    - radial_profile: average value at each radial distance
    """
    h, w = image.shape
    y, x = np.indices((h, w))

    if center is None:
        center = (h // 2, w // 2)

    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_bin = (r / binsize).astype(int)

    r_max = r_bin.max()
    radial_sum = np.bincount(r_bin.ravel(), weights=image.ravel())
    radial_count = np.bincount(r_bin.ravel())

    radial_profile = radial_sum / np.maximum(radial_count, 1)
    r_vals = np.arange(len(radial_profile)) * binsize

    return r_vals, radial_profile

def sample(im, sample_factor):
    """
    Resample using torch.nn.functional.interpolate.

    Supports:
      - 2D: (Y, X) → resize both axes
      - 3D: (Z, Y, X) → resize Y, X only (keep Z unchanged)
      - 4D: (T, Z, Y, X)
          * if Z > 2: trilinear resize (Z, Y, X)
          * else: resize Y, X only (keep Z unchanged)
    """

    im = np.asarray(im)
    if sample_factor == 1:
        return im
    if sample_factor <= 0:
        raise ValueError("sample_factor must be > 0")

    def _torch_dtype(np_dtype):
        if np.issubdtype(np_dtype, np.floating):
            return torch.float64 if np_dtype == np.float64 else torch.float32
        return torch.float32

    def _back_to_numpy(t, orig_dtype):
        out = t.detach().cpu().numpy()
        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            out = np.rint(out).clip(info.min, info.max).astype(orig_dtype)
        elif np.issubdtype(orig_dtype, np.floating):
            out = out.astype(orig_dtype, copy=False)
        return out

    ndim = im.ndim
    orig_dtype = im.dtype

    # 2D: (Y, X)
    if ndim == 2:
        Y, X = im.shape
        Y_new = max(1, int(round(Y * sample_factor)))
        X_new = max(1, int(round(X * sample_factor)))
        t = torch.from_numpy(im).to(_torch_dtype(orig_dtype)).unsqueeze(0).unsqueeze(0)  # (1,1,Y,X)
        out = F.interpolate(t, size=(Y_new, X_new), mode='bilinear', align_corners=False)
        return _back_to_numpy(out[0, 0], orig_dtype)

    # 3D: (Z, Y, X) -> resize Y,X only (keep Z)
    if ndim == 3:
        Z, Y, X = im.shape
        Y_new = max(1, int(round(Y * sample_factor)))
        X_new = max(1, int(round(X * sample_factor)))
        t = torch.from_numpy(im).to(_torch_dtype(orig_dtype)).unsqueeze(1)  # (Z,1,Y,X) treat Z as batch
        out = F.interpolate(t, size=(Y_new, X_new), mode='bilinear', align_corners=False)
        return _back_to_numpy(out[:, 0], orig_dtype)

    # 4D: (T, Z, Y, X)
    if ndim == 4:
        T, Z, Y, X = im.shape
        Y_new = max(1, int(round(Y * sample_factor)))
        X_new = max(1, int(round(X * sample_factor)))

        # Large Z: resize (Z,Y,X) with trilinear per timepoint
        if Z > 2:
            Z_new = max(1, int(round(Z * sample_factor)))
            t = torch.from_numpy(im).to(_torch_dtype(orig_dtype)).unsqueeze(1)  # (T,1,Z,Y,X)
            out = F.interpolate(t, size=(Z_new, Y_new, X_new), mode='trilinear', align_corners=False)
            return _back_to_numpy(out[:, 0], orig_dtype)

        # Small Z (<=2): resize Y,X only; keep Z unchanged
        t = torch.from_numpy(im).to(_torch_dtype(orig_dtype)).reshape(T * Z, 1, Y, X)  # (T*Z,1,Y,X)
        out = F.interpolate(t, size=(Y_new, X_new), mode='bilinear', align_corners=False)
        out = out.reshape(T, Z, Y_new, X_new)
        return _back_to_numpy(out, orig_dtype)

    raise ValueError("Unsupported image dimensionality")

def div_free(vfield):
    # Takes in and returns vectorfield of shape (H,W,2)
    H, W = vfield[:,:,0].shape
    decomposer = nHHD(grid=(H, W), spacings=(1, 1))
    decomposer.decompose(vfield)
    divfree = decomposer.r
    return divfree

def div_free_stack(vfield_stack):
    print('computing divergent free field')
    return np.array([div_free(vec) for vec in tqdm(vfield_stack)])

def interpolate(part_vec, img):
    """
    Interpoliert Partikel-Vektorfelder und berechnet dazu
    die Winkel- und das vollständige (u,v)-Feld.

    Parameters
    ----------
    part_vec : list of (N_i×4)-arrays
        Für jeden Frame eine Liste von [x, y, u, v]-Vektoren.
    img : ndarray, shape (T+1, H, W)
        Bilderstapel, nur um Höhe/Breite zu kennen.

    Returns
    -------
    angle_fields : ndarray, shape (T, H, W)
        Winkel ∠(u,v) pro Pixel und Frame.
    interpol_field : ndarray, shape (T, H, W, 2)
        Die interpolierten Vektor­komponenten (u,v).
    """
    # Anzahl der Zeitschritte
    T = len(part_vec)
    H, W = img.shape[1], img.shape[2]

    # Vor-allokieren
    angle_fields    = np.zeros((T, H, W),      dtype=np.float32)
    interpol_field  = np.zeros((T, H, W, 2),   dtype=np.float32)

    # Gitter für die Interpolation
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    for t, vecs in tqdm(enumerate(part_vec), total=T, desc="Interpolating"):
        if vecs.shape[0] > 1:
            method = "nearest" if vecs.shape[0] < 4 else "cubic"

            x, y, u, v = vecs[:,0], vecs[:,1], vecs[:,2], vecs[:,3]

            # Interpolieren
            u_interp = griddata((x, y), u,
                                (grid_x, grid_y),
                                method=method,
                                fill_value=0)
            v_interp = griddata((x, y), v,
                                (grid_x, grid_y),
                                method=method,
                                fill_value=0)

            # Winkel und Vektorfeld speichern
            angle_fields[t]       = np.arctan2(v_interp, u_interp)
            interpol_field[t, :, :, 0] = u_interp
            interpol_field[t, :, :, 1] = v_interp
        # Für Frames ohne Partikel bleibt interpol_field[t] = 0

    return angle_fields, interpol_field

def angles(part_vec, img):
    print('Interpolate and store angle fields from particle vectors')
    T = img.shape[0] - 1
    H, W = img.shape[1:]
    angle_fields = np.zeros((T, H, W), dtype=np.float32)
    
    if isinstance(part_vec, np.ndarray):
        for t, vecs in tqdm(enumerate(part_vec)):
            if vecs.shape[0] > 0:
                u, v = vecs[:,:,0], vecs[:,:,1]
                angle_fields[t] = np.arctan2(v, u)

    else:
        for t, vecs in tqdm(enumerate(part_vec)):
            if vecs.shape[0] > 0:
                x, y, u, v = vecs[:, 0], vecs[:, 1], vecs[:, 2], vecs[:, 3]
                angle_fields[t] = np.arctan2(v, u)
    return angle_fields

def animate_vector_fields(images, particle_vectors, interpolated_field, angle_fields, filename, infps, vidfps=10):
    mpl.rcParams.update(plt.rcParamsDefault)
    start = time.time()
    print("Creating animation...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    writer = FFMpegWriter(fps=vidfps, bitrate=20000)
    cbar = None

    with writer.saving(fig, filename, dpi=150):
        for t in tqdm(range(len(particle_vectors))):
            axs[0].cla()
            axs[1].cla()
            axs[2].cla()

            axs[0].imshow(images[t], cmap='gray')
            axs[1].imshow(images[t], cmap='gray')
            axs[2].imshow(images[t], cmap='gray')

            vecs = particle_vectors[t]
            x, y, u, v = vecs[:, 0], vecs[:, 1], vecs[:, 2], vecs[:, 3]
            axs[0].quiver(x, y, u, v, color='red', angles='xy')

            step = 50
            yq, xq = np.mgrid[0:images.shape[1]:step, 0:images.shape[2]:step]
            
            axs[1].quiver(xq, yq, interpolated_field[t][yq, xq,0], interpolated_field[t][yq, xq,1], color='blue', angles='xy')

            angle = angle_fields[t]
            im = axs[2].imshow(angle, cmap='twilight', vmin=-np.pi, vmax=np.pi)

            axs[0].set_title("Raw Particle Vectors")
            axs[1].set_title("Interpolated Div-free Vector Field")
            axs[2].set_title("Div-free Angle Field (radians)")

            for ax in axs:
                ax.axis('off')
                ax.text(10, 20, f"t = {t * 1/infps:.1f} s", color='white', fontsize=15, ha='left', va='top')

            if cbar:
                cbar.remove()
            cbar = fig.colorbar(im, ax=axs[2], orientation='vertical', shrink=0.7)

            writer.grab_frame()

    plt.close(fig)
    print(f"Animation saved to {filename}. [{time.time() - start:.2f} s]")

def animate_vector_fields_gen(images, vect_fields, headers, filename, infps, vidfps=10, bitrate=20000):
    mpl.rcParams.update(plt.rcParamsDefault)
    start = time.time()
    print("Creating animation...")

    plot_count = len(vect_fields)
    T = len(images)
    writer = FFMpegWriter(fps=vidfps, bitrate=bitrate)
    cbar = None
    if plot_count == 1:
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        with writer.saving(fig, filename, dpi=150):
            for t in tqdm(range(T-1)):
                plt.cla()
                plt.imshow(images[t], cmap='gray')
                if isinstance(vect_fields[0], np.ndarray):
                    step = 50
                    yq, xq = np.mgrid[0:images.shape[1]:step, 0:images.shape[2]:step]
                    if vect_fields[0][t].size > 0:
                        plt.quiver(xq, yq, vect_fields[0][t][yq, xq,0], vect_fields[0][t][yq, xq,1], color=tum_colors[0], angles='xy')
                else:
                    vecs = vect_fields[0][t]
                    x, y, u, v = vecs[:, 0], vecs[:, 1], vecs[:, 2], vecs[:, 3]
                    if vect_fields[0][t].size > 0:
                        plt.quiver(x, y, u, v, color='red', angles='xy')
                plt.title(headers[0])
                plt.text(10, 20, f"t = {t * 1/infps:.1f} s", color=tum_colors[0], fontsize=15, ha='left', va='top')
                writer.grab_frame()
    if plot_count > 1:
        fig, axs = plt.subplots(1, plot_count, figsize=(6*plot_count, 6), constrained_layout=True)
        with writer.saving(fig, filename, dpi=150):
            for t in tqdm(range(T-1)):
                for plot in range(plot_count):
                    axs[plot].cla()
                    axs[plot].imshow(images[t], cmap='gray')
                    if isinstance(vect_fields[plot], np.ndarray):
                        step = 50
                        yq, xq = np.mgrid[0:images.shape[1]:step, 0:images.shape[2]:step]
                        if vect_fields[plot][t].size > 0:
                            axs[plot].quiver(xq, yq, vect_fields[plot][t][yq, xq,0], vect_fields[plot][t][yq, xq,1], color=tum_colors[plot], angles='xy')
                    else:
                        vecs = vect_fields[plot][t]
                        x, y, u, v = vecs[:, 0], vecs[:, 1], vecs[:, 2], vecs[:, 3]
                        if vect_fields[plot][t].size > 0:
                            axs[0].quiver(x, y, u, v, color='red', angles='xy')
                    axs[plot].set_title(headers[plot])
    
                for ax in axs:
                    ax.axis('off')
                    ax.text(10, 20, f"t = {t * 1/infps:.1f} s", color=tum_colors[plot], fontsize=15, ha='left', va='top')

                writer.grab_frame()

    print(f"Animation saved to {filename}. [{time.time() - start:.2f} s]")
    plt.close(fig)

def compute_temporal_angle_correlation(angle_fields):
    T = angle_fields.shape[0]
    C_t = []
    for dt in tqdm(range(1, T // 2)):
        valid = np.isfinite(angle_fields[:-dt]) & np.isfinite(angle_fields[dt:])
        product = np.cos(angle_fields[:-dt] - angle_fields[dt:])
        C_t.append(np.mean(product[valid]))
    return np.array(C_t)

def compute_spatial_angle_self_correlation(angle_frame):
    f = angle_frame - np.mean(angle_frame)
    F = fftn(f)
    C = np.real(ifftn(F * np.conj(F)))
    C = fftshift(C)
    C /= C.max()
    y, x = np.indices(C.shape)
    r = np.sqrt((x - x.mean())**2 + (y - y.mean())**2).astype(int)
    r_flat = r.ravel()
    C_flat = C.ravel()
    r_vals = np.arange(0, r.max())
    C_r = np.array([C_flat[r_flat == i].mean() for i in r_vals])
    return r_vals, C_r

def compute_spatial_self_correlation_2d(ims):
    # assume all ims have the same shape
    H, W = ims[0].shape
    # center-based integer radii
    y, x = np.indices((H, W))
    r = np.sqrt((x - W//2)**2 + (y - H//2)**2).astype(int)
    r_flat = r.ravel()
    max_r = r_flat.max() + 1

    all_Cr = []
    for im in tqdm(ims):
        # compute auto‐correlation
        F = fft2(im)
        C = fftshift(np.real(ifft2(F * np.conj(F))))
        C /= C.max()
        C_flat = C.ravel()

        # sum and count per radius
        sum_r = np.bincount(r_flat, weights=C_flat, minlength=max_r)
        cnt_r = np.bincount(r_flat, minlength=max_r)
        Cr = sum_r / cnt_r

        all_Cr.append(Cr)

    r_vals = np.arange(max_r)
    return r_vals, np.array(all_Cr)

def exp_decay(x, A, tau):
    return A * np.exp(-x / tau)

def show_images(imgs, colorbar=[], size=None):
    T = len(imgs)
    if size is None:
        size = (8*T, 8)
    fig, axs = plt.subplots(1, T, figsize=size, constrained_layout=True)
    # Ensure axs is always iterable
    if T == 1:
        axs = [axs]
    for count in range(T):
        axs[count].cla()
        im = axs[count].imshow(imgs[count], cmap='gray')
        if colorbar and len(colorbar) > count and colorbar[count]:
            fig.colorbar(im, ax=axs[count], orientation='vertical', shrink=0.7)
    plt.show()

def get_darkest_color(cmap_name_or_obj, samples: int = 512):
    # Accept a colormap name or a Colormap object and return the darkest RGBA color
    cmap = get_cmap(cmap_name_or_obj) if isinstance(cmap_name_or_obj, str) else cmap_name_or_obj
    xs = np.linspace(0, 1, samples)
    rgb = cmap(xs)[:, :3]

    # Convert sRGB to linear and compute relative luminance (WCAG)
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
    luminance = lin @ np.array([0.2126, 0.7152, 0.0722])

    idx = int(np.argmin(luminance))
    return cmap(xs[idx])  # RGBA

def animate(images, headers, filename, vidfps=10, infps=2, timestamp=True, bitrate=20000,
            scalebar_um=None, px_per_micron=20, scalebar_px=None,
            scalebar_height_px=6, scalebar_color='white', scalebar_loc='lower right',
            scalebar_pad_px=10, scalebar_label=None, fsize=12, map='Greys',
            auto_contrast=True, contrast_mode='per_frame', clip_percentiles=(1, 99)):
    """
    Animate images with optional scale bar and automatic contrast.

    Parameters
    - scalebar_um: length of the scale bar in microns. If provided, requires px_per_micron.
    - px_per_micron: pixels per micron (XY). Needed if scalebar_um is given.
    - scalebar_px: length of the scale bar in pixels (alternative to scalebar_um).
    - scalebar_height_px: bar thickness in pixels.
    - scalebar_color: bar/text color.
    - scalebar_loc: one of {'lower right','lower left','upper right','upper left'}.
    - scalebar_pad_px: padding from axes border in pixels.
    - scalebar_label: override label text (default: '<scalebar_um> µm' if scalebar_um set).
    - fsize: font size for the scale bar label and timestamp.
    - map: colormap for the images.
    - auto_contrast: if True, set display vmin/vmax automatically from percentiles.
    - contrast_mode: 'per_frame' (vary each frame) or 'global' (fixed over stack).
    - clip_percentiles: (low, high) percentiles used for vmin/vmax, e.g. (1,99).
    """

    def _compute_sb_len_px(img_shape):
        if scalebar_px is not None:
            return int(max(1, round(scalebar_px)))
        if scalebar_um is not None and px_per_micron is not None:
            return int(max(1, round(float(scalebar_um) * float(px_per_micron))))
        return None

    def _add_scalebar(ax, img_shape):
        Lpx = _compute_sb_len_px(img_shape)
        if Lpx is None:
            return None, None
        H, W = int(img_shape[0]), int(img_shape[1])
        if Lpx > W//2:
            print('scalebar too long for image width')
        # convert pixel sizes to axes fraction
        w_frac = Lpx / max(W, 1)
        h_frac = float(scalebar_height_px) / max(H, 1)
        pad_x = float(scalebar_pad_px) / max(W, 1)
        pad_y = float(scalebar_pad_px) / max(H, 1)

        loc = (scalebar_loc or 'lower right').lower()
        if loc in ('lower right', 'lr'):
            x0, y0 = 1.0 - pad_x - w_frac, pad_y
            valign_text = 'bottom'
            y_text = y0 + h_frac + 0.01
        elif loc in ('lower left', 'll'):
            x0, y0 = pad_x, pad_y
            valign_text = 'bottom'
            y_text = y0 + h_frac + 0.01
        elif loc in ('upper right', 'ur'):
            x0, y0 = 1.0 - pad_x - w_frac, 1.0 - pad_y - h_frac
            valign_text = 'top'
            y_text = y0 - 0.01
        elif loc in ('upper left', 'ul'):
            x0, y0 = pad_x, 1.0 - pad_y - h_frac
            valign_text = 'top'
            y_text = y0 - 0.01
        else:
            x0, y0 = 1.0 - pad_x - w_frac, pad_y
            valign_text = 'bottom'
            y_text = y0 + h_frac + 0.01

        rect = Rectangle((x0, y0), w_frac, h_frac,
                         transform=ax.transAxes, facecolor=get_darkest_color(map)[:3],
                         edgecolor='none', linewidth=0, zorder=5)
        ax.add_patch(rect)

        label = scalebar_label
        if label is None and scalebar_um is not None:
            val = float(scalebar_um)
            label = f"{val:g} µm"
        txt_artist = None
        if label:
            txt_artist = ax.text(x0 + 0.5 * w_frac, y_text, label,
                                 transform=ax.transAxes, color=get_darkest_color(map)[:3],
                                 ha='center', va=valign_text, fontsize=fsize, zorder=6)
        return rect, txt_artist

    def _timestamp_anchor(img_shape):
        # Upper-right corner with same pixel padding logic as scalebar
        H, W = int(img_shape[0]), int(img_shape[1])
        pad_x = float(scalebar_pad_px) / max(W, 1)
        pad_y = float(scalebar_pad_px) / max(H, 1)
        x = 1.0 - pad_x
        y = 1.0 - pad_y
        return x, y, 'right', 'top'

    def _percentile_limits(arr):
        if not auto_contrast:
            return None, None
        lo, hi = clip_percentiles
        vmin, vmax = np.nanpercentile(arr, [lo, hi])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return None, None
        return float(vmin), float(vmax)

    start = time.time()
    print("Creating animation...")
    mpl.rcParams.update(plt.rcParamsDefault)
    plt.style.use('dark_background')
    writer = FFMpegWriter(fps=vidfps, bitrate=bitrate)

    if isinstance(images, list):
        images = np.array(images, ndmin=4)
        T = images.shape[0]
        L = images.shape[1]
        fig, axs = plt.subplots(1, T, figsize=(6*T, 6), constrained_layout=True)
        if T == 1:
            axs = [axs]

        # Precompute global limits per panel if requested
        global_limits = [ (None, None) ] * T
        if auto_contrast and contrast_mode == 'global':
            for count in range(T):
                global_limits[count] = _percentile_limits(images[count])

        with writer.saving(fig, filename, dpi=150):
            for l in tqdm(range(L)):
                for count in range(T):
                    axs[count].cla()
                    if auto_contrast and contrast_mode == 'per_frame':
                        vmin, vmax = _percentile_limits(images[count, l])
                    else:
                        vmin, vmax = global_limits[count]
                    axs[count].imshow(images[count, l], cmap=map, vmin=vmin, vmax=vmax)
                    if headers:
                        axs[count].set_title(headers[count])
                    _add_scalebar(axs[count], images[count, l].shape)

                if timestamp:
                    x, y, ha, va = _timestamp_anchor(images[0, l].shape)
                    axs[0].text(x, y, f"t = {l * 1/infps:.1f} s",
                                transform=axs[0].transAxes,
                                color=scalebar_color,
                                fontsize=fsize, ha=ha, va=va, zorder=6)
                writer.grab_frame()
        print(f"Animation saved to {filename}. [{time.time() - start:.2f} s]")
        plt.close(fig)
    else:
        # single panel stack (L, H, W)
        # Global limits if requested
        global_vmin = global_vmax = None
        if auto_contrast and contrast_mode == 'global':
            global_vmin, global_vmax = _percentile_limits(images)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        init_vmin, init_vmax = (global_vmin, global_vmax)
        if auto_contrast and contrast_mode == 'per_frame':
            init_vmin, init_vmax = _percentile_limits(images[0])
        im = ax.imshow(images[0], cmap=map, interpolation='nearest', origin='lower',
                       vmin=init_vmin, vmax=init_vmax)

        if headers:
            hdr = headers[0] if isinstance(headers, (list, tuple)) else headers
            ax.set_title(hdr)

        _add_scalebar(ax, images[0].shape)

        x, y, ha, va = _timestamp_anchor(images[0].shape)
        txt = ax.text(x, y, '', color=get_darkest_color(map)[:3], fontsize=fsize,
                      ha=ha, va=va, transform=ax.transAxes, zorder=6)

        with writer.saving(fig, filename, dpi=150):
            for l in tqdm(range(len(images))):
                im.set_data(images[l])
                if auto_contrast:
                    if contrast_mode == 'per_frame':
                        vmin, vmax = _percentile_limits(images[l])
                    else:
                        vmin, vmax = global_vmin, global_vmax
                    if vmin is not None and vmax is not None:
                        im.set_clim(vmin, vmax)
                if headers:
                    hdr = headers[l] if isinstance(headers, (list, tuple)) else headers
                    ax.set_title(hdr)
                if timestamp:
                    t_sec = l / infps
                    txt.set_text(f"t = {t_sec:.1f} s")
                writer.grab_frame()

        plt.close(fig)
        print(f"Animation saved to {filename}. [{time.time()-start:.2f} s]")

def crop_to_nonzero_border(angle_field):
    """
    Peel off any outer rows/columns of a (T,H,W) angle_field
    until the remaining 2D border (at every timepoint) has no zeros.

    Parameters
    ----------
    angle_field : np.ndarray, shape (T, H, W)
        Your angle field; zeros mark invalid edge pixels.

    Returns
    -------
    cropped : np.ndarray, shape (T, h, w)
        The field cropped so that its outermost rows/columns contain no zeros.
    crop_idx : tuple (y0, y1, x0, x1)
        Inclusive slice indices along Y and X.
    """
    # Build a 2D mask: True where any timepoint is zero
    mask2d = np.any(angle_field == 0, axis=0)  # shape (H, W)

    H, W = mask2d.shape
    y0, y1 = 0, H - 1
    x0, x1 = 0, W - 1

    # Iteratively shave off invalid edges
    while True:
        trimmed = False

        # Top row invalid?
        if y0 <= y1 and np.any(mask2d[y0, x0:x1+1]):
            y0 += 1
            trimmed = True

        # Bottom row invalid?
        if y0 <= y1 and np.any(mask2d[y1, x0:x1+1]):
            y1 -= 1
            trimmed = True

        # Left column invalid?
        if x0 <= x1 and np.any(mask2d[y0:y1+1, x0]):
            x0 += 1
            trimmed = True

        # Right column invalid?
        if x0 <= x1 and np.any(mask2d[y0:y1+1, x1]):
            x1 -= 1
            trimmed = True

        # If we trimmed, loop again; otherwise we’re done
        if not trimmed:
            break

        # If we’ve crossed over, there’s no valid region
        if y0 > y1 or x0 > x1:
            raise ValueError("No non‐zero border region could be found.")

    # Crop and return
    cropped = angle_field[:, y0:y1+1, x0:x1+1]
    return cropped, (y0, y1, x0, x1)

def apply_crop(arr, crop_idx):
    """
    Apply the same Y/X crop to any array whose last two dims are H,W.

    Parameters
    ----------
    arr : np.ndarray, shape (..., H, W)
    crop_idx : tuple (y0, y1, x0, x1)

    Returns
    -------
    np.ndarray, shape (..., y1-y0+1, x1-x0+1)
    """
    y0, y1, x0, x1 = crop_idx
    # Build a slicing tuple: [:, :, ..., y0:y1+1, x0:x1+1]
    slices = [slice(None)] * (arr.ndim - 2) + [slice(y0, y1+1), slice(x0, x1+1)]
    return arr[tuple(slices)]

def compute_divergence(field, px_spacing=1.0):
    """
    Compute the 2D divergence of a vector field.

    Parameters
    ----------
    field : np.ndarray
        The vector field, shape (..., H, W, 2).
        The last dimension holds (u_x, u_y).
    px_spacing : float or tuple of floats
        Physical size of one pixel: either a single number (same in x&y)
        or (dy, dx).

    Returns
    -------
    div : np.ndarray
        The divergence, shape (..., H, W).
    """
    # Ensure spacing tuple
    if np.isscalar(px_spacing):
        dy = dx = px_spacing
    else:
        dy, dx = px_spacing

    # Split components
    u_x = field[..., 0]
    u_y = field[..., 1]

    # Compute spatial derivatives
    # gradient returns list [∂/∂y, ∂/∂x] when given spacing
    dux_dy, dux_dx = np.gradient(u_x, dy, dx, axis=(-2, -1))
    duy_dy, duy_dx = np.gradient(u_y, dy, dx, axis=(-2, -1))

    # divergence = ∂u_x/∂x + ∂u_y/∂y
    div = dux_dx + duy_dy
    return div

def estimate_diams_and_mean_dist(image, min_size=2, threshold=None):
    """
    Parameters
    ----------
    image : 2D ndarray
        Grauwertbild mit punktförmigen Objekten.
    min_size : int
        Minimale Pixelzahl, um Rauschelemente zu filtern.
    threshold : float or None
        Schwellwert; falls None, wird automatisch Otsu gewählt.

    Returns
    -------
    diameters : list of float
        Äquivalent-Durchmesser aller Objekte in Pixeln:
            d = sqrt(4*A/π)
    mean_nn_dist : float
        Mittlere Abstand zur nächsten Nachbar-Zentroid  
        (0, falls <2 Objekte gefunden wurden).
    """
    # 1) Binarisieren
    if threshold is None:
        threshold = filters.threshold_otsu(image)
    bw = image > threshold

    # 2) Rauschen entfernen
    bw = morphology.remove_small_objects(bw, min_size=min_size)

    # 3) Labeln & Regionprops
    labels = measure.label(bw)
    props  = measure.regionprops(labels)

    # 4) Diameter aus Fläche
    diameters = min([prop.equivalent_diameter for prop in props])

    # 5) Zentroiden sammeln
    centroids = np.array([prop.centroid for prop in props])
    n = len(centroids)
    if n < 2:
        mean_nn_dist = 0.0
    else:
        # paarweise Abstände
        D = distance.cdist(centroids, centroids)
        # Diagonale unbrauchbar (0), setze sie auf +∞
        np.fill_diagonal(D, np.inf)
        # nächster Nachbar je Zeile
        nn = D.min(axis=1)
        mean_nn_dist = float(nn.min())

    return diameters, mean_nn_dist

def binary_to_tp_df(centroids, diameters, frame_number):
    """
    Wandelt die Ausgabe von handle_binary() in ein DataFrame um,
    das tp.link lesen kann.

    Parameters
    ----------
    centroids : (N,3) ndarray
        Zeilen-/Spalten-Koordinaten (row, col, z) der Objekte.
    diameters : array-like, Länge N
        Äquivalent-Durchmesser jedes Objekts (in µm oder px).
    frame_number : int
        Die Bildnummer (frame) dieses Datensatzes.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame mit Spalten ['frame','x','y','z','diameter'], geeignet für tp.link.
    """
    if len(centroids) == 0:
        return pd.DataFrame(columns=['frame','x','y','z','diameter'])

    # trackpy erwartet x=horizontal (col), y=vertikal (row)
    df = pd.DataFrame({
        'frame':   frame_number,
        'x':       centroids[:,2],
        'y':       centroids[:,1],
        'z':       centroids[:,0], 
        'diameter': diameters
    })
    return df

def speed_stats(vector_field, fps=1.0, px_per_micron=1.0):
    """
    Berechnet pro Frame den mittleren Geschwindigkeitsbetrag und dessen Unsicherheit
    sowie die overall Durchschnittsgeschwindigkeit und deren Unsicherheit.

    Parameters
    ----------
    vector_field : ndarray, shape (T, H, W, 2)
        Vektorfeld mit Komponenten (u, v).

    Returns
    -------
    mean_speeds : ndarray, shape (T,)
        Mittlerer Betrag √(u²+v²) pro Frame.
    sem_speeds : ndarray, shape (T,)
        Standardfehler des Mittels pro Frame:
            σ_frame/√(H·W),  wobei σ_frame = std über alle Pixel im Frame.
    overall_mean : float
        Durchschnitt aller Frame-Mittelwerte.
    overall_sem : float
        Standardfehler des overall mean:
            σ_means/√T,  wobei σ_means = std der Frame-Mittelwerte.
    """
    T, H, W, _ = vector_field.shape

    # 1) Geschwindigkeit am Pixel: √(u² + v²) → shape (T, H, W)
    speeds = np.linalg.norm(vector_field, axis=-1)*fps/px_per_micron

    # 2) per-frame mean und std (über Pixel)
    mean_speeds = speeds.mean(axis=(1, 2))
    std_frame   = speeds.std(axis=(1, 2), ddof=1)
    n_pixels    = H * W
    sem_speeds  = std_frame / np.sqrt(n_pixels)

    # 3) overall mean und sem (über Frames)
    overall_mean    = mean_speeds.mean()
    std_of_means    = mean_speeds.std(ddof=1)
    overall_sem     = std_of_means / np.sqrt(T)

    return mean_speeds, sem_speeds, overall_mean, overall_sem

def speed_stats_from_part_vec(part_vec, fps=1.0, px_per_micron=1.0):
    """
    Berechnet pro Frame und insgesamt die mittlere Geschwindigkeit mit Unsicherheit
    aus Partikel-Vektor-Daten (x, y, u, v) pro Frame.

    Parameters
    ----------
    part_vec : list of ndarray, length T
        Liste von Partikel-Vektorfeldern pro Frame. Jeder Eintrag ist ein (N_i, 4)-Array
        mit Spalten [x, y, dx, dy] in Pixel pro Frame.
    fps : float, optional
        Bildwiederholrate (Frames pro Sekunde), Standard 1.0.
    px_per_micron : float, optional
        Skalierungsfaktor von Pixel auf Mikrometer, Standard 1.0.

    Returns
    -------
    mean_speeds : ndarray, shape (T,)
        Mittlere Geschwindigkeit (µm/s) pro Frame.
    sem_speeds : ndarray, shape (T,)
        Standardfehler des Mittels pro Frame.
    overall_mean : float
        Gesamtmittlere Geschwindigkeit über alle Frames.
    overall_sem : float
        Standardfehler der gesamtmittleren Geschwindigkeit.
    """
    T = len(part_vec)
    mean_speeds = np.zeros(T, dtype=float)
    sem_speeds  = np.zeros(T, dtype=float)

    # 1) Berechne pro Frame
    for idx, vecs in enumerate(part_vec):
        # vecs: (Ni,4) mit dx,dy in Spalten 2,3
        if vecs is None or vecs.size == 0:
            mean_speeds[idx] = 0.0
            sem_speeds[idx]  = 0.0
            continue
        # Geschwindigkeitsbeträge in Pixel/Frame umrechnen zu µm/s
        dx = vecs[:, 2]
        dy = vecs[:, 3]
        speeds_px = np.sqrt(dx**2 + dy**2)
        speeds_um = speeds_px * fps / px_per_micron
        # Statistik
        mean_speeds[idx] = speeds_um.mean()
        if speeds_um.size > 1:
            std = speeds_um.std(ddof=1)
            sem_speeds[idx] = std / np.sqrt(speeds_um.size)
        else:
            sem_speeds[idx] = 0.0

    # 2) Gesamtstatistik über Frame-Mittelwerte
    overall_mean = mean_speeds.mean()
    if T > 1:
        std_means = mean_speeds.std(ddof=1)
        overall_sem = std_means / np.sqrt(T)
    else:
        overall_sem = 0.0

    return mean_speeds, sem_speeds, overall_mean, overall_sem

def speed_stats_from_part_vec_3d(part_vec, fps=1.0, px_per_micron=1.0, px_per_micron_z=1.0):
    """
    Berechnet pro Frame und insgesamt die mittlere Geschwindigkeit mit Unsicherheit
    aus Partikel-Vektor-Daten (x, y, z, u, v, w) pro Frame.

    Parameters
    ----------
    part_vec : list of ndarray, length T
        Liste von Partikel-Vektorfeldern pro Frame. Jeder Eintrag ist ein (N_i, 6)-Array
        mit Spalten [x, y, z, dx, dy, dz] in Pixel pro Frame.
    fps : float, optional
        Bildwiederholrate (Frames pro Sekunde), Standard 1.0.
    px_per_micron : float, optional
        Skalierungsfaktor von Pixel auf Mikrometer, Standard 1.0.
    px_per_micron_z : float, optional
        Skalierungsfaktor von Pixel auf Mikrometer in z Richtung, Standard 1.0.

    Returns
    -------
    mean_speeds : ndarray, shape (T,)
        Mittlere Geschwindigkeit (µm/s) pro Frame.
    sem_speeds : ndarray, shape (T,)
        Standardfehler des Mittels pro Frame.
    overall_mean : float
        Gesamtmittlere Geschwindigkeit über alle Frames.
    overall_sem : float
        Standardfehler der gesamtmittleren Geschwindigkeit.
    """
    T = len(part_vec)
    mean_speeds = np.zeros(T, dtype=float)
    sem_speeds  = np.zeros(T, dtype=float)

    # 1) Berechne pro Frame
    for idx, vecs in enumerate(part_vec):
        # vecs: (Ni,6) mit dx,dy,dz in Spalten 4,5,6
        if vecs is None or vecs.size == 0:
            mean_speeds[idx] = 0.0
            sem_speeds[idx]  = 0.0
            continue
        # Geschwindigkeitsbeträge in Pixel/Frame umrechnen zu µm/s
        dx = vecs[:, 3]/ px_per_micron
        dy = vecs[:, 4]/ px_per_micron
        dz = vecs[:, 5]/ px_per_micron_z
        speeds_px = np.sqrt(dx**2 + dy**2 + dz**2)
        speeds_um = speeds_px * fps
        # Statistik
        mean_speeds[idx] = speeds_um.mean()
        if speeds_um.size > 1:
            std = speeds_um.std(ddof=1)
            sem_speeds[idx] = std / np.sqrt(speeds_um.size)
        else:
            sem_speeds[idx] = 0.0

    # 2) Gesamtstatistik über Frame-Mittelwerte
    overall_mean = mean_speeds.mean()
    if T > 1:
        std_means = mean_speeds.std(ddof=1)
        overall_sem = std_means / np.sqrt(T)
    else:
        overall_sem = 0.0

    return mean_speeds, sem_speeds, overall_mean, overall_sem

def read_tiff_calibration(path):
    """
    Liest aus einem ImageJ-kompatiblen TIFF
    - die X/Y-Pixelgröße in µm (aus den TIFF-Tags XResolution, YResolution + ResolutionUnit)
    - die Z-Pixelgröße in µm (aus imagej_metadata['spacing'], falls vorhanden)
    - das Zeitintervall in s (aus imagej_metadata['finterval'] oder aus fps)

    Returns
    -------
    px_x, px_y, px_z, dt : float oder None
    """
    px_x = px_y = px_z = dt = None

    with tifffile.TiffFile(path) as tif:
        # Zugriff auf erste Seite und deren Tags
        page = tif.pages[0]
        tags = page.tags

        # 1) Einheit der Auflösung (ResolutionUnit): 2=inches, 3=cm
        unit_tag = tags.get('ResolutionUnit')
        if unit_tag:
            if unit_tag.value == 2:
                unit_to_um = 25_400.0     # inch → µm
            elif unit_tag.value == 3:
                unit_to_um = 10_000.0     # cm → µm
            else:
                unit_to_um = 1.0
        else:
            unit_to_um = 1.0

        # 2) X/Y-Auflösung (NumPixels / Unit) ⇒ Unit/Pixel ⇒ µm/Pixel
        for axis, var in [('XResolution', 'px_x'), ('YResolution', 'px_y')]:
            tag = tags.get(axis)
            if tag:
                num, den = tag.value
                # den/num = Einheiten pro Pixel
                val = (den / num) * unit_to_um
                if axis == 'XResolution':
                    px_x = val
                else:
                    px_y = val

        # 3) Z-Pixelgröße aus ImageJ-Spacing
        ij = tif.imagej_metadata or {}
        if 'spacing' in ij:
            px_z = float(ij['spacing'])

        # 4) Zeitintervall: erst finterval, dann fps als Fallback
        if 'finterval' in ij:
            dt = float(ij['finterval'])
        elif 'fps' in ij:
            dt = 1.0 / float(ij['fps'])

    if px_x and px_y and px_z and dt:
        return 1/px_x, 1/px_y, 1/px_z, 1/dt, 
    if px_x and px_y and  not px_z and dt:
        print('No z-scale')
        return 1/px_x, 1/px_y, np.inf, 1/dt, 
    if px_x and px_y and not px_z and not dt:
        print('No z- and t-scale')
        return 1/px_x, 1/px_y, np.inf, None, 
    if px_x and px_y and px_z and not dt:
        print('No t-scale')
        return 1/px_x, 1/px_y, 1/px_z, None, 
    else:
        print('No scale found')
        return None,

def autocorr_1d_axis(volume, axis, px_scale=1.0, max_lag=None):
    """
    Compute 1D spatial autocorrelation of a 3D volume along a specified axis.
    (unchanged)
    """
    arr = np.moveaxis(volume, axis, -1)
    M, N = int(np.prod(arr.shape[:-1])), arr.shape[-1]
    if max_lag is None or max_lag > N - 1:
        max_lag = N - 1
    arr2 = arr.reshape(M, N).astype(np.float64)
    arr2 -= arr2.mean()
    C0    = np.mean(arr2 * arr2)
    lags  = np.arange(max_lag + 1)
    corr  = np.array([np.mean(arr2[:, :N - lag] * arr2[:, lag:]) / C0
                      for lag in lags])
    distances = lags * px_scale
    return distances, corr

def fit_exp_decay(distances, corr, fit_fraction=1.0):
    """
    Fit corr(d) = A * exp(-d / tau) using only the first `fit_fraction` of distances > 0.
    
    Parameters
    ----------
    distances : 1D array
    corr      : 1D array, same length
    fit_fraction : float in (0,1]
        Only distances <= fit_fraction * max(distances) are used in the fit.
    
    Returns
    -------
    A_opt   : float
        Fitted amplitude.
    tau_opt : float
        Fitted decay constant (correlation length).
  """
    max_d = distances.max() * max(0.0, min(fit_fraction, 1.0))
    mask = (distances > distances[1:].min()) & (distances <= max_d)
    d, c = distances[mask], corr[mask]
    if d.size < 2:
        return None, None
    A0   = c[0]
    tau0 = (d.max() - d.min()) / 2.0 if d.size > 1 else 1.0
    try:
        popt, _ = curve_fit(lambda x, A, tau: A * np.exp((-x/tau)),
                            d, c, p0=[A0, tau0], bounds=([0,0],[np.inf,np.inf]))
        return popt[0], popt[1]
    except Exception:
        return None, None
    
def fit_double_exp_decay(distances, corr, fit_fraction=1.0):
    """
    Fit corr(d) = A * exp(-d / tau) using only the first `fit_fraction` of distances > 0.
    
    Parameters
    ----------
    distances : 1D array
    corr      : 1D array, same length
    fit_fraction : float in (0,1]
        Only distances <= fit_fraction * max(distances) are used in the fit.
    
    Returns
    -------
    A_opt   : float
        Fitted amplitude.
    tau_opt : float
        Fitted decay constant (correlation length).
  """
    max_d = distances.max() * max(0.0, min(fit_fraction, 1.0))
    mask = (distances > distances[1:].min()) & (distances <= max_d)
    d, c = distances[mask], corr[mask]
    if d.size < 2:
        return None, None
    A0 = c[0]
    B0 = A0*0.1
    tau1 = (d.max() - d.min()) / 2.0 if d.size > 1 else 1.0
    tau2 = tau1*10
    try:
        popt, _ = curve_fit(lambda x, A, B, t1, t2: A * np.exp((-x/t1)) + B * np.exp((-x/t2)),
                            d, c, p0=[A0, B0, tau1, tau2], bounds=([0]*4,[np.inf]*4))
        return popt[0], popt[1], popt[2], popt[3]
    except Exception:
        return None, None, None, None

def spatial_autocorr_axes_stack(images, px_xy=1.0, px_z=1.0, fit_fraction=1.0):

    """
    Compute 1D spatial autocorrelation along x, y, z for each timepoint,
    and fit both amplitude A and decay tau for exp_decay model.
    
    Parameters
    ----------
    images : ndarray, shape (T, Z, Y, X)
        4D image stack.
    px_xy : float
        Physical pixel size in x and y directions.
    px_z : float
        Physical pixel size in z direction.
    fit_fraction : float
        Fraction of distance axis to use for fitting.
    
    Returns
    -------
    distances : dict of ndarray
        {'x': dx_array, 'y': dy_array, 'z': dz_array}
    corrs     : dict of ndarray
        {'x': corr_x, 'y': corr_y, 'z': corr_z}
    amps      : dict of ndarray
        {'x': A_x, 'y': A_y, 'z': A_z}, each shape (T,)
    taus      : dict of ndarray
        {'x': tau_x, 'y': tau_y, 'z': tau_z}, each shape (T,)
    """
    T, Z, Y, X = images.shape
    
    # Compute distance axes
    dx, _ = autocorr_1d_axis(images[0], axis=2, px_scale=px_xy)
    dy, _ = autocorr_1d_axis(images[0], axis=1, px_scale=px_xy)
    dz, _ = autocorr_1d_axis(images[0], axis=0, px_scale=px_z)
    
    # Prepare outputs
    corr_x = np.zeros((T, len(dx)), dtype=float)
    corr_y = np.zeros((T, len(dy)), dtype=float)
    corr_z = np.zeros((T, len(dz)), dtype=float)
    A_x    = np.zeros(T, dtype=float)
    A_y    = np.zeros(T, dtype=float)
    A_z    = np.zeros(T, dtype=float)
    tau_x  = np.zeros(T, dtype=float)
    tau_y  = np.zeros(T, dtype=float)
    tau_z  = np.zeros(T, dtype=float)

    for t in tqdm(range(T)):
        vol = images[t]
        _, corr_x[t] = autocorr_1d_axis(vol, axis=2, px_scale=px_xy, max_lag=dx.size-1)
        _, corr_y[t] = autocorr_1d_axis(vol, axis=1, px_scale=px_xy, max_lag=dy.size-1)
        _, corr_z[t] = autocorr_1d_axis(vol, axis=0, px_scale=px_z,  max_lag=dz.size-1)
        
        A_x[t],  tau_x[t]  = fit_exp_decay(dx, corr_x[t], fit_fraction)
        A_y[t],  tau_y[t]  = fit_exp_decay(dy, corr_y[t], fit_fraction)
        A_z[t],  tau_z[t]  = fit_exp_decay(dz, corr_z[t], fit_fraction)
    
    distances = {'x': dx, 'y': dy, 'z': dz}
    corrs     = {'x': corr_x, 'y': corr_y, 'z': corr_z}
    amps      = {'x': A_x,  'y': A_y,  'z': A_z}
    taus      = {'x': tau_x,'y': tau_y,'z': tau_z}
    return distances, corrs, amps, taus

def group3d(images, df):
    T = images.shape[0] - 1
    #H, W = images.shape[1:]
    particle_vectors = [[] for _ in range(T)]

    # Group by particle and process each trajectory
    for pid, track in tqdm(df.groupby('particle')):
        track = track.sort_values('frame')
        frames = track['frame'].values
        x = track['x'].values
        y = track['y'].values
        z = track['z'].values

        for i in range(len(frames) - 1):
            f = frames[i]
            if frames[i+1] == f + 1:
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                dz = z[i+1] - z[i]
                particle_vectors[f].append([x[i], y[i], z[i], dx, dy, dz])

    return [np.asarray(vectors) if vectors else np.empty((0, 6)) for vectors in particle_vectors]

def group3d_fast(images, df):
    """
    Vectorized, much faster version of group3d:
    builds all particle‐to‐particle displacements in one pass.
    Output: (x,y,z,dx,dy,dz) per particle per frame
    """
    T = images.shape[0] - 1

    # 1) sort and compute "next" via groupby‐shift
    df2 = df.sort_values(['particle','frame'])
    df2[['frame_next','x_next','y_next','z_next']] = (
        df2.groupby('particle')[['frame','x','y','z']]
           .shift(-1)
    )

    # 2) keep only consecutive‐frame links
    mask = df2['frame_next'] == df2['frame'] + 1
    df2 = df2.loc[mask, ['frame','x','y','z','x_next','y_next','z_next']]

    # 3) compute displacements
    df2['dx'] = df2['x_next'] - df2['x']
    df2['dy'] = df2['y_next'] - df2['y']
    df2['dz'] = df2['z_next'] - df2['z']

    # 4) for each frame f, grab all rows with frame==f
    pv = [None] * T
    for f, grp in df2.groupby('frame'):
        if 0 <= f < T:
            pv[f] = grp[['x','y','z','dx','dy','dz']].to_numpy()

    # 5) fill missing frames with empty arrays
    for i in range(T):
        if pv[i] is None:
            pv[i] = np.empty((0, 6), float)

    return pv

def testimg():
    testimg = np.zeros([10, 18, 512, 512])
    for t in range(10):
        testimg[t,:,5+t*3:40+t*3,5:40]=1
        testimg[t,t+1,100:130,200:220]=1
    return testimg

def mean_particle_vectors(particle_vectors, px_per_micron=1, px_per_micron_z=1):
    """
    Compute the per‐frame mean displacement vector from a list/array of particle_vectors,
    rescaled into microns.

    Parameters
    ----------
    particle_vectors : sequence of ndarray
        Each element is an (N, D) array of particle vectors for one frame, where
        the first K columns are positions and the last M columns are displacements.
        E.g. shape (N,6) with [x,y,z, dx,dy,dz].
    px_per_micron : float
        Conversion factor for x,y displacements (pixels per micron).
    px_per_micron_z : float
        Conversion factor for z displacements (pixels per micron).

    Returns
    -------
    mean_disp : ndarray, shape (T, M)
        The mean displacement (componentwise) per frame, in microns.
    """
    pv = np.asarray(particle_vectors, dtype=object)
    mean_disp = []
    for frame in pv:
        if frame.size:
            disp = frame[:, 3:].astype(float)
            # scale dx,dy by px_per_micron; dz by px_per_micron_z
            disp[:, 0:2] /= px_per_micron
            if disp.shape[1] > 2:  # if dz exists
                disp[:, 2] /= px_per_micron_z
            mean_disp.append(disp.mean(axis=0))
        else:
            mean_disp.append(np.zeros(frame.shape[1] - 3))
    return np.vstack(mean_disp)

def group3d_by_particle(df, num_frames=None):
    # Determine total number of frames
    if num_frames is None:
        num_frames = int(df['frame'].max()) + 1
    # Unique particles and frame index
    particles = df['particle'].unique()
    frames = np.arange(num_frames)
    # Build complete index
    idx = pd.MultiIndex.from_product([particles, frames], names=['particle','frame'])
    # Pivot positions into wide format, reindex to include missing frames
    pos = (
        df.set_index(['particle','frame'])[['x','y','z']]
          .reindex(idx)
    )
    # Compute displacements by group
    disp = pos.groupby(level=0).diff()
    disp.columns = ['dx','dy','dz']
    # Combine into a single DataFrame
    traj = pd.concat([pos, disp], axis=1).reset_index()
    return traj

def temporal_autocorr_vector(orient_traj, max_lag=None):
    orient = np.asarray(orient_traj, float)
    T = orient.shape[0]
    if max_lag is None or max_lag > T - 1:
        max_lag = T - 1
    lags = np.arange(max_lag + 1)
    # compute full dot-product matrix once
    # orientation vectors may contain NaNs
    dot_mat = orient @ orient.T
    corr = np.array([
        np.nanmean(np.diag(dot_mat, k=lag))
        for lag in lags
    ])
    return lags, corr

def temporal_autocorr_vector_S(orient_traj, max_lag=None):
    orient = np.asarray(orient_traj, float)
    T = orient.shape[0]
    if max_lag is None or max_lag > T - 1:
        max_lag = T - 1
    lags = np.arange(max_lag + 1)
    # compute full dot-product matrix once
    # orientation vectors may contain NaNs
    dot_mat = orient @ orient.T
    corr = (np.array([
        np.nanmean(np.diag(dot_mat, k=lag)**2)
        for lag in lags
    ])-1/3)*3/2
    return lags, corr

def temporal_autocorr_all_fast(df, max_lag=None, compute_S=True):
    """
    Compute per-particle temporal autocorrelations of the 3D displacement unit vectors,
    but much faster by doing one big sort and slicing into the raw NumPy arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['particle', 'frame', 'dx', 'dy', 'dz'].
    max_lag : int or None
        Maximum lag to pass through to temporal_autocorr_vector.  If None, full length.

    Returns
    -------
    results : dict
        Keys are particle IDs, values are (lags, corr) from temporal_autocorr_vector.
    """
    # 1) Extract the raw values and do a single sort by particle, then frame.
    arr = df[['particle','frame','dx','dy','dz']].to_numpy()
    # sort by particle first, then frame
    idx_sort = np.lexsort((arr[:,1], arr[:,0]))
    arr = arr[idx_sort]

    # 2) Pull out unique particles, where they start, and how many rows each has
    particles, idx_start, counts = np.unique(arr[:,0], return_index=True, return_counts=True)
    T = int(arr[:,1].max()) + 1

    results = {}
    # 3) For each particle, slice its chunk out of the big array
    for pid, start, cnt in tqdm(zip(particles, idx_start, counts),
                                 total=particles.size,
                                 desc='Temporal autocorr'):
        sub = arr[start:start+cnt]
        frames = sub[:,1].astype(int)
        disps = sub[:, 2:].astype(float)

        # normalize to get unit-direction vectors, drop zero-length
        norms = np.linalg.norm(disps, axis=1)
        valid = norms > 0
        dirs = (disps[valid].T / norms[valid]).T  # shape (n_valid,3)
        frs  = frames[valid]

        # build full-length trajectory with NaNs
        orient_traj = np.full((T, 3), np.nan, dtype=float)
        orient_traj[frs] = dirs

        # call your existing autocorr
        if compute_S:
            lags, corr = temporal_autocorr_vector_S(orient_traj, max_lag)
        else:
            lags, corr = temporal_autocorr_vector(orient_traj, max_lag)
        results[pid] = (lags, corr)

    return results

def extract_branch_features(
    skeleton: np.ndarray,
    px_per_micron_xy: 1.0,
    px_per_micron_z: 1.0,
    orientation: str = "pca",  # "pca" (fast 3x3 eig) or "endpoints"
) -> pd.DataFrame:
    """
    Faster extractor for per-branch features with anisotropic voxel scaling.

    Parameters
    ----------
    skeleton : ndarray, shape (T, Z, Y, X)
        Binary skeleton per time.
    px_per_micron_xy : float
        Pixels per micron for X and Y (microns = pixels / px_per_micron_xy).
    px_per_micron_z : float
        Pixels per micron for Z (microns = pixels / px_per_micron_z).
    orientation : {"pca", "endpoints"}
        - "pca": principal-axis direction via 3x3 covariance eigenvector (robust, fast)
        - "endpoints": unit vector from first to last point

    Returns
    -------
    features_df : pd.DataFrame with columns
        frame, branch_id, x, y, z, xa, ya, za, length
        (positions in microns; orientation is a unit, headless director; length in microns)
    """
    # Precompute scale factors (pixels -> microns) in Z,Y,X order to match skan output
    sx = sy = 1.0 / float(px_per_micron_xy)
    sz = 1.0 / float(px_per_micron_z)
    scale_zyx = np.array([sz, sy, sx], dtype=np.float32)

    dfs = []
    T = int(skeleton.shape[0])

    for t in tqdm(range(T),'extracting branch features'):
        skel = skeleton[t]
        g = csr.Skeleton(skel)

        # Cache once per frame
        n_paths = int(g.n_paths)

        for branch_id in range(n_paths):
            # Coordinates come as (L, 3) in (z, y, x) voxel indices
            co_zyx = g.path_coordinates(branch_id)
            if co_zyx.size == 0:
                continue

            # Use float32 and scale to physical microns in the same (z, y, x) order
            co_zyx = np.asarray(co_zyx, dtype=np.float32)
            co_phys_zyx = co_zyx * scale_zyx  # (L, 3) in microns (z,y,x)

            # Centroid in microns (still z,y,x order here)
            cz, cy, cx = co_phys_zyx.mean(axis=0)

            # Physical length in microns (sum of Euclidean segment lengths)
            segs = np.diff(co_phys_zyx, axis=0)
            length = float(np.linalg.norm(segs, axis=1).sum())

            # Orientation (unit headless director)
            if orientation == "endpoints" or co_phys_zyx.shape[0] < 3:
                d_zyx = co_phys_zyx[-1] - co_phys_zyx[0]
            else:
                # Fast PCA via 3x3 covariance eigen-decomposition
                X = co_phys_zyx - np.array([cz, cy, cx], dtype=np.float32)
                cov = X.T @ X  # (3,3)
                # eigh is faster/stable for symmetric covariances
                w, V = np.linalg.eigh(cov)
                d_zyx = V[:, -1]  # eigenvector with largest eigenvalue

            # Normalize; skip pathological zero-length
            n = float(np.linalg.norm(d_zyx))
            if not np.isfinite(n) or n == 0.0:
                continue
            d_zyx = d_zyx / n

            # Reorder to (x, y, z) for output and fix headless sign deterministically
            d_xyz = d_zyx[[2, 1, 0]]
            # if d_xyz[0] < 0 or (d_xyz[0] == 0 and (d_xyz[1] < 0 or (d_xyz[1] == 0 and d_xyz[2] < 0))):
            #     d_xyz = -d_xyz

            um = px_per_micron_xy!=1.0 and px_per_micron_z!=1.0

            dfs.append({
                'frame':     int(t),
                'branch_id': int(branch_id),
                'x':         float(cx),  # centroid reordered to x,y,z
                'y':         float(cy),
                'z':         float(cz),
                'xa':        float(d_xyz[0]),
                'ya':        float(d_xyz[1]),
                'za':        float(d_xyz[2]),
                'length':    length,
                'um':        um,
            })

    return pd.DataFrame(dfs)

def subtract_mean_disp_from_pv(pv, mean_disp):
    """
    pv: list of length T, each (N_t, 6) as [x,y,z, dx,dy,dz]
    mean_disp: (T, 3) array with per-frame [dx,dy,dz] means
    Returns a new list like pv where displacements are mean-subtracted.
    """
    T = len(pv)
    out = []
    for t in range(T-1):
        arr = pv[t]
        if arr.size:
            disp_corr = arr[:, 3:] - mean_disp[t]        # (N_t,3) - (3,)
            out.append(np.hstack([arr[:, :3], disp_corr]))
        else:
            out.append(arr.copy())

    return out

def prepare_particle_vectors(pv_list, round_coords=True):
    """
    Given pv_list, a list of NumPy arrays each of shape (Ki,6) = [x,y,z,dx,dy,dz],
    return a single array of shape (sum Ki,6) that you can feed into
    annotate_branches_with_dot (or merge directly).
    
    If round_coords=True, the x,y,z will be rounded to the nearest integer.
    """
    # 1) stack into one big (N,6) array
    arr = np.vstack(pv_list)
    
    if round_coords:
        # round the first three columns (x,y,z) to ints
        coords = np.round(arr[:, :3]).astype(int)
        arr = np.hstack((coords, arr[:, 3:6]))
    
    return arr

def prepare_particle_vectors_agg(pv_list, round_coords=True, agg='mean'):
    """
    Stack a list of (Ki×6) arrays into one table and aggregate duplicate coords.
    
    Parameters
    ----------
    pv_list : list of np.ndarray
        Each entry is shape (Ki,6) = [x,y,z, dx,dy,dz].
    round_coords : bool
        Round x,y,z to ints before grouping.
    agg : {'mean','sum','max'}
        How to aggregate dx,dy,dz for duplicate coords.
    
    Returns
    -------
    pv_df : pd.DataFrame
        Columns ['x','y','z','dx','dy','dz'], one row per unique coord.
    """
    arr = np.vstack(pv_list)                 # (N,6)
    if round_coords:
        coords = np.round(arr[:, :3]).astype(int)
    else:
        coords = arr[:, :3]
    vecs   = arr[:, 3:6]                     # (N,3)
    
    df = pd.DataFrame(
        np.hstack((coords, vecs)),
        columns=['x','y','z','dx','dy','dz']
    )
    
    if agg == 'mean':
        pv_df = df.groupby(['x','y','z'], as_index=False).mean()
    elif agg == 'sum':
        pv_df = df.groupby(['x','y','z'], as_index=False).sum()
    elif agg == 'max':
        pv_df = df.groupby(['x','y','z'], as_index=False).max()
    else:
        raise ValueError("agg must be one of 'mean','sum','max'")
    
    return pv_df

def annotate_branches_with_dot(features_dir, pv_df):
    """
    Merge in the (dx,dy,dz) for each branch and compute |dot| with (xa,ya,za).
    """
    merged = features_dir.merge(
        pv_df,
        on=['x','y','z'],
        how='left',
        validate='many_to_one'   # now safe, since pv_df has unique x,y,z
    )
    # compute order parameter S
    vel_abs = np.sqrt(merged['dx']**2+merged['dy']**2+merged['dz']**2)
    merged['abs_dot'] = 3/2*(np.abs(merged['xa'] * merged['dx']/vel_abs + merged['ya'] * merged['dy']/vel_abs + merged['za'] * merged['dz']/vel_abs)**2-1/3)
    # pick only the columns you want
    return merged[['frame','branch_id','length','x','y','z','abs_dot']]

def plot_S_distribution(df, bins=50, safe=None):
    """
    Plot the distribution (count) of S values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing an 'abs_dot' column.
    bins : int or sequence, optional
        Number of bins or bin edges for the histogram.
    safe : str or None, optional
        Filename to save the figure if provided.
    """
    # Extract and clean values
    values = df['S'].dropna().to_numpy()
    if values.size == 0:
        raise ValueError('No S values to plot.')

    # Compute histogram
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width, align='center', alpha=0.7, color=tum_colors[1], edgecolor='black')

    # Labels and title
    ax.set_xlabel('Absolute order parameter')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of absolute order parameters ($S$) (n={values.size})')
    ax.set_xlim(-0.5,1)
    ax.set_ylim(0,counts.max()*1.1)
    fig.tight_layout()

    # Save or show
    if safe:
        fig.savefig(safe)
    plt.show()

def plot_temporal_autocorr_sample_fast(results, fraction=0.5, seed=0, infps=1, safe=None, title=f'Filament temporal autocorrelation'):
    """
    Plot the temporal autocorrelation functions for a random subset of particles, with a shaded ±1σ tube around the sample mean,
    and convert frame-based lags into seconds using the provided frame rate.

    Parameters
    ----------
    results : dict
        Mapping particle_id -> (lags, corr_profile) as returned by temporal_autocorr_all.
    fraction : float, optional
        Fraction of particles to sample and plot (0 < fraction ≤ 1).
    seed : int or None, optional
        Random seed for reproducibility.
    infps : float, optional
        Frames per second of the original movie to convert lags into seconds.
    safe : str or None, optional
        Filename to save the figure if provided.
    """
    import random, numpy as np, matplotlib.pyplot as plt
    from tqdm import tqdm
    import matplotlib as mpl
    from scipy.optimize import curve_fit
    import numpy as np

    #from dask import delayed
    import pandas as pd
    import numpy as _np
    import pandas as _pd

    mpl.rcParams['text.usetex'] = False  # disable external LaTeX

    # 1. sample particle IDs
    pids = list(results.keys())
    n_sample = max(1, int(len(pids) * fraction))
    rng = random.Random(seed)
    sample = rng.sample(pids, n_sample)

    # 2. build a list of (time_s, corr) segments for LineCollection
    segments = []
    for pid in sample:
        lags, corr = results[pid]
        times = lags / infps
        seg = np.column_stack([times, corr])
        segments.append(seg)

    # 3. create figure & add all samples as one collection
    fig, ax = plt.subplots(figsize=(8, 6))
    lc = LineCollection(segments,
                        colors='gray',
                        linewidths=0.5,
                        alpha=0.3)
    ax.add_collection(lc)
    ax.autoscale()  # adjust axis to data range

    # 4. compute mean ±1σ across sampled correlograms
    all_corrs = np.vstack([results[pid][1] for pid in sample])  # shape (n_sample, n_lags)
    lags = results[sample[0]][0]
    times = lags / infps
    mean_corr = np.nanmean(all_corrs, axis=0)
    std_corr  = np.nanstd(all_corrs, axis=0)

    # 5. plot ±1σ tube around the mean
    ax.fill_between(
        times,
        mean_corr - std_corr,
        mean_corr + std_corr,
        color=tum_colors[3],
        alpha=0.4,
        label='±1σ',
        lw=2
    )

    # 6. overplot the mean
    ax.plot(times, mean_corr, color=tum_colors[3], lw=2, label='Filament-mean')
    # 7. finalize plot
    valid = ~np.isnan(mean_corr)
    if valid.any():
        max_time = times[valid].max()
        ax.set_xlim(0, max_time+2)
    else:
        ax.set_xlim(times.min(), times.max())
    ax.set_ylim(-0.6, 1.1)
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title + f' (n={n_sample}, {infps:.1f} fps)')
    ax.legend(loc='best', fontsize=10, ncol=2)
    fig.tight_layout()

    if safe:
        fig.savefig(safe)
    plt.show()

def compute_orientations(particle_vectors, px_per_micron=1.0, px_per_micron_z=1.0):
    """
    Convert a list of per-frame particle_vectors to unit orientation vectors,
    accounting for anisotropic pixel size.

    Parameters
    ----------
    particle_vectors : list of ndarray
        Each element is shape (N,6): [x,y,z, dx,dy,dz] in pixel units.
    px_scale : tuple of floats (sx, sy, sz)
        Physical size of a pixel in x, y, z directions (e.g., µm/pixel).

    Returns
    -------
    positions : list of ndarray
        Each element is shape (N,3): [x,y,z] positions in physical units.
    orients : list of ndarray
        Each element is shape (N,3): unit orientation vectors in physical units.
    """
    sx = sy = 1/px_per_micron
    sz = 1/px_per_micron_z
    positions = []
    orients   = []
    for vecs in tqdm(particle_vectors):
        if vecs.size == 0:
            positions.append(np.empty((0,3)))
            orients.append(np.empty((0,3)))
            continue
        # scale positions
        pos = vecs[:, :3].astype(float)
        pos[:, 0] *= sx
        pos[:, 1] *= sy
        pos[:, 2] *= sz
        # scale displacement
        disp = vecs[:, 3:].astype(float)
        disp[:, 0] *= sx
        disp[:, 1] *= sy
        disp[:, 2] *= sz
        # normalize to unit vectors
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        norms[norms == 0] = 1
        u = disp / norms
        positions.append(pos)
        orients.append(u)
    return positions, orients

def orient_autocorr_3d(positions, orients, nbins=50):
    """
    Compute radial autocorrelation of 3D orientation fields with anisotropic spacing.

    Parameters
    ----------
    positions : list of ndarray
        Per-frame arrays of shape (N,3) for particle positions in pixel units.
    orients : list of ndarray
        Per-frame arrays of shape (N,3) for unit orientation vectors.
    nbins : int
        Number of distance bins.

    Returns
    -------
    bin_centers : ndarray, shape (nbins,)
    corr_profiles : ndarray, shape (T, nbins)
        Mean dot-product <u·u> in each distance bin per frame.
    """
    T = len(positions)
    corr_profiles = np.zeros((T, nbins), dtype=float)
    bin_edges = None

    for t in tqdm(range(T)):
        u   = orients[t]
        pos = positions[t]

        # exclude zero-orientation (and non-finite) vectors
        valid = np.all(np.isfinite(u), axis=1) & (np.linalg.norm(u, axis=1) > 0)
        if pos.shape[0] == u.shape[0]:
            valid &= np.all(np.isfinite(pos), axis=1)

        u = u[valid]
        pos = pos[valid]
        N = pos.shape[0]
        if N < 2:
            corr_profiles[t, :] = np.nan
            continue

        # compute all pairwise distances in physical units
        dif = pos[:, None, :] - pos[None, :, :]  # (N,N,3)
        # dif[..., 0] *= 1.0  # already scaled in positions
        # distances = sqrt((dx*sx)^2+(dy*sy)^2+(dz*sz)^2) since pos scaled
        d2 = dif[..., 0]**2 + dif[..., 1]**2 + dif[..., 2]**2
        dist = np.sqrt(d2[np.triu_indices(N, k=1)])

        # order parameter S
        dots = 3/2*(np.sum(u[:, None, :] * u[None, :, :], axis=-1)**2-1/3)
        dots = dots[np.triu_indices(N, k=1)]

        # define bins once
        if bin_edges is None:
            maxd = dist.max()
            bin_edges = np.linspace(0, maxd, nbins + 1)
        valid_idx = np.where(np.isfinite(dots))[0]
        dots_valid = dots[valid_idx]
        dist_valid = dist[valid_idx]
        sums, _ = np.histogram(dist_valid, bins=bin_edges, weights=dots_valid)
        counts, _ = np.histogram(dist_valid, bins=bin_edges)
        profile = np.zeros(nbins, float)
        nonzero = counts > 0
        profile[nonzero] = sums[nonzero] / counts[nonzero]
        corr_profiles[t] = profile

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, corr_profiles

def positions_and_orientations_per_frame(
    df: pd.DataFrame,
    px_per_micron=1.0,
    px_per_micron_z=1.0,
    normalize_orient: bool = True,
    include_velocity: bool = False,
    bead_velocity: bool = False,
    branch_parts: bool = False,  # New parameter to include parts of branches
    ):
    """
    From a dataframe with columns:
      frame, branch_id, x, y, z, xa, ya, za, length
    return two lists (length T) of numpy arrays per frame:
      - pos_list[t]: shape (N_t, 3) with positions [x,y,z] in microns
      - ori_list[t]: shape (N_t, 3) with orientations [xa,ya,za]
                     (optionally normalized to unit vectors)

    If include_velocity is True, also return:
      - vel_list[t]: shape (N_t, 3) with velocities [vx,vy,vz]
        (no unit conversion applied; assumed to be in µm/s if provided by upstream)

    If branch_parts is True, the function will also handle parts of branches.

    Parameters
    ----------
    df : DataFrame
    px_per_micron : float
        Pixels per micron for x,y.
    px_per_micron_z : float
        Pixels per micron for z.
    normalize_orient : bool, default True
        If True, normalize [xa,ya,za] to unit length per row.
    include_velocity : bool, default False
        If True, also extract and return per-frame velocities [vx,vy,vz].
    branch_parts : bool, default False
        If True, process parts of branches.

    Returns
    -------
    pos_list : list[np.ndarray]
    ori_list : list[np.ndarray]
    vel_list : list[np.ndarray] (only if include_velocity=True)
    """
    # base required cols
    needed = {'frame','x','y','z','xa','ya','za'}
    if include_velocity:
        if bead_velocity:
            needed |= {'vx_bead','vy_bead','vz_bead'}  # velocity already in µm/s
        else:
            needed |= {'vx','vy','vz'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    # sort for reproducibility
    if branch_parts and 'particle' in df.columns:
        df2 = df.sort_values(['frame', 'branch_id', 'particle'] if 'particle' in df.columns else ['frame'])
    else:
        df2 = df.sort_values(['frame', 'branch_id'] if 'branch_id' in df.columns else ['frame'])

    # total frames T inferred from max frame that appears
    T = int(df2['frame'].max()) - 1 + 1 if len(df2) else 0 

    pos_list = [np.empty((0,3), float) for _ in range(T)]
    ori_list = [np.empty((0,3), float) for _ in range(T)]
    vel_list = [np.empty((0,3), float) for _ in range(T)] if include_velocity else None

    # scale factors (pixels -> microns)
    sx = sy = 1.0 / float(px_per_micron)
    sz = 1.0 / float(px_per_micron_z)

    for f, grp in df2.groupby('frame'):
        if int(f) < 0 or int(f) >= T:
            continue
        # positions in microns
        pos = np.stack([
            grp['x'].to_numpy(dtype=float) * sx,
            grp['y'].to_numpy(dtype=float) * sy,
            grp['z'].to_numpy(dtype=float) * sz
        ], axis=1)

        # orientations
        ori = grp[['xa','ya','za']].to_numpy(dtype=float)
        if normalize_orient and ori.size:
            n = np.linalg.norm(ori, axis=1, keepdims=True)
            np.divide(ori, np.where(n == 0, 1.0, n), out=ori)

        pos_list[int(f)] = pos
        ori_list[int(f)] = ori
        if include_velocity:
            if bead_velocity:
                vel = grp[['vx_bead','vy_bead','vz_bead']].to_numpy(dtype=float)
            else:
                vel = grp[['vx','vy','vz']].to_numpy(dtype=float)
            if normalize_orient and vel.size:
                n = np.linalg.norm(vel, axis=1, keepdims=True)
                np.divide(vel, np.where(n == 0, 1.0, n), out=vel)
            vel_list[int(f)] = vel


    if include_velocity:
        return pos_list, ori_list, vel_list
    return pos_list, ori_list

def _radial_bin(q_mag: np.ndarray, S_k: np.ndarray, nbins: int = 50
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radial-average a 3D spectrum S_k(qx,qy,qz) over |q|.
    Returns (q_centers, S_q).
    """
    q_flat = q_mag.ravel()
    S_flat = S_k.ravel()

    # exclude the zero-frequency voxel from the average
    mask = np.isfinite(S_flat) & np.isfinite(q_flat) & (q_flat > 0)
    q_flat = q_flat[mask]
    S_flat = S_flat[mask]

    qmin, qmax = q_flat.min(), q_flat.max()
    edges = np.linspace(qmin, qmax, nbins + 1)
    idx = np.clip(np.digitize(q_flat, edges) - 1, 0, nbins - 1)

    sums = np.bincount(idx, weights=S_flat, minlength=nbins).astype(float)
    counts = np.bincount(idx, minlength=nbins).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        S_q = np.where(counts > 0, sums / counts, np.nan)

    q_centers = 0.5 * (edges[:-1] + edges[1:])
    return q_centers, S_q

def compute_Sq_from_positions(
    pos_list: List[np.ndarray],
    vol_shape: Tuple[int, int, int],
    px_per_micron: float,
    px_per_micron_z: float,
    nbins: int = 60,
    sigma_px: float = 1.0,
    normalize: str = "number"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic static structure factor S(q) from per-frame point positions.

    Parameters
    ----------
    pos_list : list of (N_t, 3) arrays
        Per-frame positions in microns [x,y,z]; already output of extract_branch_features.
    vol_shape : (Z, Y, X)
        Image/skeleton volume shape for a single frame.
    px_per_micron : float
        Pixels per micron in XY.
    px_per_micron_z : float
        Pixels per micron in Z.
    nbins : int
        Number of radial q bins.
    sigma_px : float
        Gaussian width in pixels for anti-aliasing of the gridded density.
    normalize : {"number","volume"}
        Normalization convention (see docstring above).

    Returns
    -------
    q : (nbins,) array, in 1/µm
    S_q_frames : (T, nbins) array
    """
    Z, Y, X = vol_shape
    # convert to µm per pixel
    sx = 1.0 / px_per_micron
    sy = 1.0 / px_per_micron
    sz = 1.0 / px_per_micron_z

    # Fourier wavevectors in rad/µm
    qz = 2.0 * np.pi * fftfreq(Z, d=sz)
    qy = 2.0 * np.pi * fftfreq(Y, d=sy)
    qx = 2.0 * np.pi * fftfreq(X, d=sx)
    QZ, QY, QX = np.meshgrid(qz, qy, qx, indexing='ij')
    q_mag = np.sqrt(QX**2 + QY**2 + QZ**2)

    T = len(pos_list)
    S_q_frames = np.full((T, nbins), np.nan, dtype=float)
    q_out = None

    for t, pts in enumerate(tqdm(pos_list, desc="Computing S(q)")):
        if pts is None or len(pts) == 0:
            continue

        # convert real-space coordinates (µm) → voxel indices
        vx = np.floor(pts[:, 0] / sx).astype(int)
        vy = np.floor(pts[:, 1] / sy).astype(int)
        vz = np.floor(pts[:, 2] / sz).astype(int)

        m = (vz >= 0) & (vz < Z) & (vy >= 0) & (vy < Y) & (vx >= 0) & (vx < X)
        vx, vy, vz = vx[m], vy[m], vz[m]
        N = len(vx)
        if N == 0:
            continue

        rho = np.zeros((Z, Y, X), dtype=np.float32)
        rho[vz, vy, vx] = 1.0

        if sigma_px and sigma_px > 0:
            rho = gaussian_filter(rho, sigma=sigma_px, mode='wrap')

        rho_k = fftn(rho)
        Pk = (rho_k * np.conj(rho_k)).real

        if normalize == "number":
            Pk = Pk / max(N, 1)
        elif normalize == "volume":
            Pk = Pk / max(N * N, 1)

        Pk.flat[0] = 0.0

        q_bins, S_q = _radial_bin(q_mag, Pk, nbins=nbins)
        if q_out is None:
            q_out = q_bins
        S_q_frames[t, :] = S_q

    return q_out, S_q_frames

def plot_Sq(q, S_q_frames, path, title="Static structure factor", qmin_exclude=0.0,
            qmax=None, ymin=None, ymax=None, show_std=False, fit_exp=False, fit_fraction=0.5):
    """
    Plot mean S(q) (± std) with nice limits and save to 'path'.
    Optionally fit and plot an exponential function to S(q).

    Parameters
    ----------
    q : (nbins,) array in 1/µm
    S_q_frames : (T, nbins) array
    path : str, output file (e.g. 'plots/.../Sq_mean.png'); dirs are created
    qmin_exclude : float, exclude very small q (e.g. 0.0–first bin)
    qmax : float or None, right limit; if None, use 99th percentile of finite q
    ymin, ymax : floats or None, y-limits; auto from percentiles if None
    show_std : bool, fill band for ±1 std across frames
    fit_exp : bool, if True, fit and plot an exponential function
    fit_fraction : float, fraction of q-range to use for fitting
    """
    mpl.rcParams.update(plt.rcParamsDefault)
    plt.style.use('dark_background')
    plt.style.use('./science.mplstyle.txt')
    # Reduce over frames
    S_mean = np.nanmean(S_q_frames, axis=0)
    S_std  = np.nanstd(S_q_frames, axis=0)

    # Finite masks and low-q exclusion
    m = np.isfinite(q) & np.isfinite(S_mean) & (q > qmin_exclude)
    q_plot = q[m]
    S_mean_plot = S_mean[m]
    S_std_plot  = S_std[m]

    if q_plot.size == 0:
        raise ValueError("No valid data to plot after masking; check inputs.")

    # Auto x/y limits
    if qmax is None:
        qmax = np.quantile(q_plot, 0.99)
    if ymin is None or ymax is None:
        ylo = np.nanquantile(S_mean_plot, 0.02)
        yhi = np.nanquantile(S_mean_plot, 0.98)
        pad = 0.08 * max(1e-9, (yhi - ylo))
        ymin = ylo - pad if ymin is None else ymin
        ymax = yhi + pad if ymax is None else ymax

    # Make sure folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Plot
    plt.figure(figsize=(8,6), dpi=150)
    plt.plot(q_plot, S_mean_plot, lw=2, c=tum_colors[0], label="Mean S(q)")
    if show_std:
        plt.fill_between(q_plot, S_mean_plot - S_std_plot, S_mean_plot + S_std_plot, alpha=0.25, label="±1σ")

    # Optionally fit and plot exponential
    if fit_exp:
        # Use only a fraction of the q-range for fitting
        max_q_fit = q_plot.min() + fit_fraction * (q_plot.max() - q_plot.min())
        fit_mask = (q_plot > q_plot.min()) & (q_plot <= max_q_fit)
        q_fit = q_plot[fit_mask]
        S_fit = S_mean_plot[fit_mask]
        if q_fit.size > 2:
            def exp_func(q, A, tau):
                return A * np.exp(-q / tau)
            try:
                popt, pcov = curve_fit(exp_func, q_fit, S_fit, p0=[S_fit[0], 1.0], bounds=([0,0],[np.inf,np.inf]))
                S_exp = exp_func(q_plot, *popt)
                plt.plot(q_plot, S_exp, 'r--', c=tum_colors[0], alpha=0.7, lw=2, label=f"Exp fit: $\\tau={popt[1]:.2f} \pm {np.sqrt(np.diag(pcov))[1]:.2f} $ rad/µm")
            except Exception as e:
                print(f"[!] Exponential fit failed: {e}")

    plt.xlim(q_plot.min(), qmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r"$q$ (rad / $\mu$m)")
    plt.ylabel(r"$S(q)$")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    #plt.tight_layout()
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.show()

def _save_one(obj, base):
    if isinstance(obj, pd.DataFrame):
        path = f"{base}.parquet"
        try:
            obj.to_parquet(path)
        except Exception as e:
            print(f"[!] Error saving to Parquet. Falling back to pickle.")
            path = f"{base}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f)
    elif isinstance(obj, np.ndarray):
        path = f"{base}.npy"
        np.save(path, obj)
    elif isinstance(obj, list):
        if all(isinstance(x, (int, float, str, bool, type(None))) for x in obj):
            path = f"{base}.json"
            with open(path, "w") as f:
                json.dump(obj, f)
        else:
            path = f"{base}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f)
    else:
        path = f"{base}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    return path

def save_vars(folder=".", *objs):
    """
    Save variables by their real variable name.
    Usage:
        save_vars("results", df_loaded, trajectories)
    """
    os.makedirs(folder, exist_ok=True)
    saved = {}

    # get caller’s locals so we can recover variable names
    caller_locals = inspect.currentframe().f_back.f_locals

    for obj in objs:
        print('saving...')
        # find the variable name(s) pointing to this object
        names = [n for n, v in caller_locals.items() if v is obj]
        if not names:
            raise ValueError("Could not infer variable name – call with plain variables only.")
        name = names[0]

        base = os.path.join(folder, name)
        saved[name] = _save_one(obj, base)

    return saved


def cache_put(cache_folder: str | Path, name: str, obj, overwrite: bool = True):
    """Cache results in formats that work well with Zarr/Dask.

    - dask.array.Array -> `<cache_folder>/<name>.zarr` via `to_zarr` (lazy-friendly)
    - numpy.ndarray    -> `<cache_folder>/<name>.zarr` via `zarr.save_array`
    - pandas.DataFrame -> `<cache_folder>/<name>.parquet`
    - simple JSON types -> `<cache_folder>/<name>.json`
    - fallback -> pickle `<cache_folder>/<name>.pkl`

    Returns the written path.
    """
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name)).strip("_")
    if not safe:
        raise ValueError("cache name must not be empty")

    if isinstance(obj, da.Array):
        path = cache_folder / f"{safe}.zarr"
        obj.to_zarr(path, overwrite=overwrite)
        return str(path)

    if isinstance(obj, np.ndarray):
        path = cache_folder / f"{safe}.zarr"
        # zarr.save_array overwrites by removing the target store
        if path.exists() and overwrite:
            shutil.rmtree(path)
        zarr.save_array(path, obj)
        return str(path)

    if isinstance(obj, pd.DataFrame):
        path = cache_folder / f"{safe}.parquet"
        if path.exists() and (not overwrite):
            return str(path)
        obj.to_parquet(path)
        return str(path)

    if isinstance(obj, (dict, list, int, float, str, bool, type(None))):
        path = cache_folder / f"{safe}.json"
        if path.exists() and (not overwrite):
            return str(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f)
        return str(path)

    path = cache_folder / f"{safe}.pkl"
    if path.exists() and (not overwrite):
        return str(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return str(path)


def cache_get(cache_folder: str | Path, name: str):
    """Load cached data written by cache_put().

    - `<name>.zarr` -> returns dask array (lazy)
    - `<name>.parquet` -> pandas DataFrame
    - `<name>.json` -> python object
    - `<name>.pkl` -> python object
    """
    cache_folder = Path(cache_folder)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name)).strip("_")
    candidates = [
        cache_folder / f"{safe}.zarr",
        cache_folder / f"{safe}.parquet",
        cache_folder / f"{safe}.json",
        cache_folder / f"{safe}.pkl",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".zarr":
                return da.from_zarr(p)
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".json":
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
            if p.suffix == ".pkl":
                with p.open("rb") as f:
                    return pickle.load(f)
    raise FileNotFoundError(f"No cached object found for {name!r} in {cache_folder}")

def load_object(path):
    """
    Load an object saved with save_object().
    If the file does not exist, prints a message and returns None.
    """
    if not os.path.exists(path):
        print(f"[!] File not found: {path}")
        return None

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".npy":
            return np.load(path, allow_pickle=True)
        elif ext == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif ext == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"[!] Unknown file extension: {ext} (path: {path})")
            return None
    except Exception as e:
        print(f"[!] Error loading {path}: {e}")
        return None

def compare_metadata(actual_meta, saved_meta_path):
    """
    Compare the actual metadata with the saved metadata.
    
    Parameters:
        actual_meta (dict): The actual metadata to compare.
        saved_meta_path (str): Path to the saved metadata JSON file.
    
    Returns:
        bool: True if the user decides to continue, False otherwise.
    """
    try:
        # Load the saved metadata
        with open(saved_meta_path+"/metadata.json", "r") as f:
            saved_meta = json.load(f)
        
        # Compare the metadata
        if actual_meta != saved_meta:
            # Convert image_shape to tuple for comparison if it's a list
            if isinstance(saved_meta.get("image_shape"), list):
                saved_meta["image_shape"] = tuple(saved_meta["image_shape"])
            if isinstance(actual_meta.get("image_shape"), list):
                actual_meta["image_shape"] = tuple(actual_meta["image_shape"])
            
            if actual_meta != saved_meta:
                print("WARNING: Metadata does not match the saved metadata.")
                print("Actual metadata:", actual_meta)
                print("Saved metadata:", saved_meta)
                proceed = input("Do you want to continue? (y/n): ").strip().lower()
                if proceed != "y":
                    print("Operation aborted.")
                    return False
                else:
                    print('Continuing.')
            else:
                print("Metadata matches the saved metadata. Continuing.")
        else:
            print("Metadata matches the saved metadata. Continuing.")
    except FileNotFoundError:
        os.makedirs(saved_meta_path, exist_ok=True)
        json.dump(actual_meta, open(saved_meta_path+"/metadata.json","w"), indent=2)
        print(f"No metadata file found at {saved_meta_path}, created a cache folder.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {saved_meta_path}. Assuming no conflict.")
    
    return True

def df_to_vectors(features_dir):
    """
    Convert features_dir DataFrame (with columns frame,x,y,z,xa,ya,za)
    into a list of numpy arrays, one per frame:
        [ array((n_points, 6)), ... ]
    where each row = [x, y, z, xa, ya, za]
    """
    vectors = []
    for t, df_t in tqdm(features_dir.groupby("frame"), desc="Building vectors"):
        arr = df_t[["x","y","z","xa","ya","za"]].to_numpy()
        vectors.append(arr)
    return vectors

def generate_random_features_dir(canvas_size, n_vectors, n_frames=1, um=False, seed=None):
    """
    Generate a DataFrame matching `features_dir` structure with uncorrelated random vectors,
    including a 'length' column set to 1.

    Parameters
    - canvas_size: tuple (Z, Y, X) giving bounds for random positions (values in pixels)
    - n_vectors: number of vectors per frame
    - n_frames: number of frames to generate (default 1)
    - um: boolean flag for the 'um' column (coordinates remain in pixels unless converted externally)
    - seed: RNG seed for reproducibility

    Returns
    - pandas.DataFrame with columns:
      ['frame','branch_id','x','y','z','xa','ya','za','um','length']
    """
    rs = np.random.RandomState(seed)
    Z, Y, X = canvas_size
    rows = []
    for frame in range(int(n_frames)):
        xs = rs.uniform(0, X, size=int(n_vectors))
        ys = rs.uniform(0, Y, size=int(n_vectors))
        zs = rs.uniform(0, Z, size=int(n_vectors))

        # random orientations: normalized Gaussian -> approximately uniform on sphere
        vecs = rs.normal(size=(int(n_vectors), 3))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        uvecs = vecs / norms

        for i in range(int(n_vectors)):
            rows.append({
                "frame": int(frame),
                "branch_id": int(i),
                "x": float(xs[i]),
                "y": float(ys[i]),
                "z": float(zs[i]),
                "xa": float(uvecs[i, 0]),
                "ya": float(uvecs[i, 1]),
                "za": float(uvecs[i, 2]),
                "um": bool(um),
                "length": 1.0
            })

    df = pd.DataFrame(rows, columns=["frame","branch_id","x","y","z","xa","ya","za","um","length"])
    return df

def make_synthetic_mask_mt(
    shape: Tuple[int, int, int, int],
    mean_velocity: float = 2.0,
    density: float = 0.5,
    length_range: Tuple[int, int] = (40, 120),
    curvature: float = 0.15,
    thickness_px: float = 1.5,
    noise_sigma: float = 0.02,
    seed: int = None,
    per_frame_normalize: bool = False
) -> np.ndarray:
    """
    Create a synthetic 4D volume (T, Z, Y, X) with curved filaments that
    move linearly over time, similar in type and shape to mask_mt.

    Parameters:
      - shape: (T, Z, Y, X) of the output volume.
      - mean_velocity: mean filament speed in pixels/frame.
      - density: expected number of filaments per 1e6 voxels (Z*Y*X).
      - length_range: min/max filament length in voxels (number of points along centerline).
      - curvature: random curvature strength; higher = more wiggly.
      - thickness_px: apparent filament thickness (controls Gaussian blur sigma).
      - noise_sigma: additive Gaussian noise (after blur), in [0,1] scale before normalization.
      - seed: RNG seed for reproducibility.
      - per_frame_normalize: if True, normalize each frame independently to [0,1].
                             if False, normalize globally.

    Returns:
      - volume: float32 array of shape (T, Z, Y, X), values in [0,1].
    """
    rng = np.random.default_rng(seed)
    T, Z, Y, X = shape

    # number of filaments from density (per 1e6 voxels)
    n_fil = max(1, int(round(density * (Z * Y * X) / 1_000_000.0)))

    vol = np.zeros(shape, dtype=np.float32)

    def rand_unit_vec3(n=1):
        v = rng.normal(size=(n, 3))
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    # draw each filament as a 3D polyline and move with constant velocity
    for _ in range(n_fil):
        L = int(rng.integers(low=length_range[0], high=length_range[1]+1))
        # start point anywhere in volume
        p = np.empty((L, 3), dtype=np.float32)
        p[0] = [rng.uniform(0, Z-1), rng.uniform(0, Y-1), rng.uniform(0, X-1)]

        # random initial direction
        d = rand_unit_vec3(1)[0].astype(np.float32)

        # build a curved polyline with small random direction changes
        for i in range(1, L):
            d = d + curvature * rng.normal(size=3).astype(np.float32)
            nrm = np.linalg.norm(d) + 1e-8
            d = d / nrm
            p[i] = p[i-1] + d

        # constant velocity for this filament (px/frame)
        speed = float(max(0.0, rng.normal(loc=mean_velocity, scale=0.3 * max(1e-6, mean_velocity))))
        v = rand_unit_vec3(1)[0].astype(np.float32) * speed  # 3D velocity

        # rasterize into volume across time
        # shift the whole polyline by t*v
        ts = np.arange(T, dtype=np.float32)[:, None, None]  # (T,1,1)
        pv = p[None, :, :] + ts * v[None, None, :]          # (T, L, 3)

        # round to nearest voxels and add impulses
        zi = np.rint(pv[..., 0]).astype(np.int32)
        yi = np.rint(pv[..., 1]).astype(np.int32)
        xi = np.rint(pv[..., 2]).astype(np.int32)

        # mask in-bounds for each time
        inb = (
            (zi >= 0) & (zi < Z) &
            (yi >= 0) & (yi < Y) &
            (xi >= 0) & (xi < X)
        )

        for t in range(T):
            m = inb[t]
            if not np.any(m):
                continue
            zt = zi[t, m]; yt = yi[t, m]; xt = xi[t, m]
            vol[t, zt, yt, xt] += 1.0

    # simulate finite thickness by blurring each 3D frame
    sigma = float(max(0.1, thickness_px * 0.5))
    for t in range(T):
        vol[t] = gaussian_filter(vol[t], sigma=sigma, mode='nearest')

    # add low-level noise
    if noise_sigma > 0:
        vol += rng.normal(0.0, noise_sigma, size=vol.shape).astype(np.float32)

    # normalize to [0,1]
    if per_frame_normalize:
        for t in range(T):
            vmin, vmax = float(vol[t].min()), float(vol[t].max())
            if vmax > vmin:
                vol[t] = (vol[t] - vmin) / (vmax - vmin)
            else:
                vol[t] = 0.0
    else:
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol[:] = 0.0

    return vol.astype(np.float32)

def make_synthetic_mask_spheres(
    shape: Tuple[int, int, int, int],
    mean_velocity: float = 2.0,
    density: float = 0.5,
    radius_range: Tuple[float, float] = (4.0, 12.0),
    thickness_px: float = 1.5,
    noise_sigma: float = 0.02,
    seed: int = None,
    same_direction: bool = False,
    per_frame_normalize: bool = False
) -> np.ndarray:
    """
    Create a synthetic 4D volume (T, Z, Y, X) with moving solid spheres.

    Parameters:
        - shape: (T, Z, Y, X) of the output volume.
        - mean_velocity: mean sphere speed in pixels/frame.
        - density: expected number of spheres per 1e6 voxels (Z*Y*X).
        - radius_range: (min, max) sphere radius in pixels.
        - thickness_px: apparent blur thickness (Gaussian sigma scaling).
        - noise_sigma: additive Gaussian noise (after blur), in [0,1] scale before normalization.
        - seed: RNG seed for reproducibility.
        - per_frame_normalize: if True, normalize each frame independently to [0,1];
                                if False, normalize globally.

    Returns:
        - volume: float32 array of shape (T, Z, Y, X), values in [0,1].
    """
    rng = np.random.default_rng(seed)
    T, Z, Y, X = shape

    # number of spheres from density (per 1e6 voxels)
    n_sph = max(1, int(round(density * (Z * Y * X) / 1_000_000.0)))

    vol = np.zeros(shape, dtype=np.float32)

    rmin = float(max(1.0, radius_range[0]))
    rmax = float(max(rmin, radius_range[1]))

    def _rand_unit_vec3(n=1):
        v = rng.normal(size=(n, 3))
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    spheres = []
    for _ in range(n_sph):
        r = float(rng.uniform(rmin, rmax))
        # allow anywhere; sphere can be partially out of bounds (clipped)
        c0 = np.array([rng.uniform(0, Z - 1),
                        rng.uniform(0, Y - 1),
                        rng.uniform(0, X - 1)], dtype=np.float32)
        speed = float(max(0.0, rng.normal(loc=mean_velocity, scale=0.3 * max(1e-6, mean_velocity))))
        if same_direction:
            v = np.ones((1, 3))[0].astype(np.float32)/np.sqrt(3) * mean_velocity
        else:
            v = _rand_unit_vec3(1)[0].astype(np.float32) * speed
        spheres.append((c0, v, r))

    # rasterize spheres over time
    for t in range(T):
        frame = vol[t]
        for c0, v, r in spheres:
            c = c0 + t * v
            cz, cy, cx = float(c[0]), float(c[1]), float(c[2])

            z0 = max(0, int(np.floor(cz - r - 1)))
            z1 = min(Z, int(np.ceil(cz + r + 1)))
            y0 = max(0, int(np.floor(cy - r - 1)))
            y1 = min(Y, int(np.ceil(cy + r + 1)))
            x0 = max(0, int(np.floor(cx - r - 1)))
            x1 = min(X, int(np.ceil(cx + r + 1)))
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue

            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            mask = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)
            frame[z0:z1, y0:y1, x0:x1] += mask.astype(np.float32)

    # blur to simulate thickness / PSF
    sigma = float(max(0.1, thickness_px * 0.5))
    for t in range(T):
        vol[t] = gaussian_filter(vol[t], sigma=sigma, mode='nearest')

    # add low-level noise
    if noise_sigma > 0:
        vol += rng.normal(0.0, noise_sigma, size=vol.shape).astype(np.float32)

    # normalize to [0,1]
    if per_frame_normalize:
        for t in range(T):
            vmin, vmax = float(vol[t].min()), float(vol[t].max())
            if vmax > vmin:
                vol[t] = (vol[t] - vmin) / (vmax - vmin)
            else:
                vol[t] = 0.0
    else:
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol[:] = 0.0

    return vol.astype(np.float32)

def plot_velocity_orientation_correlation_comparison(names, labels, variation=""):
    """
    Plot velocity orientation correlation comparison for multiple datasets.

    Parameters:
    - names: List of dataset names (paths to cache folders).
    - labels: List of labels corresponding to each dataset.
    - variation: Variation string for the output file name.
    """
    mpl.rcParams.update(plt.rcParamsDefault)
    plt.style.use('./science.mplstyle.txt')
    plt.style.use('dark_background')
    path = f'plots/combined_plots/velocity_orientation_correlation_comparison{variation}{names}.pdf'

    plt.figure(figsize=(8, 6))

    # Plot additional datasets
    for i, dataset_name in enumerate(names):
        vector_corr_add = load_object(f"plots/{dataset_name}/cache/vector_corr.npy")
        vector_corr_std_add = load_object(f"plots/{dataset_name}/cache/vector_corr_std.npy")
        bin_centers = load_object(f"plots/{dataset_name}/cache/bin_centers.npy")
        
        color = tum_colors[i % len(tum_colors)]  # Automatically cycle through tum_colors
        plt.plot(bin_centers, vector_corr_add, color=color, label=f'velocity correlation ({labels[i]})', lw=2)
        # plt.fill_between(
        #     bin_centers,
        #     vector_corr_add - vector_corr_std_add,
        #     vector_corr_add + vector_corr_std_add,
        #     color=color,
        #     alpha=0.2,
        #     label=fr'±1$\sigma$ ({labels[i]})'
        # )
        # Plot direction correlation for additional datasets
        dir_corr_add = load_object(f"plots/{dataset_name}/cache/dir_corr.npy")
        dir_corr_std_add = load_object(f"plots/{dataset_name}/cache/dir_corr_std.npy")
        
        plt.plot(bin_centers, dir_corr_add, color=color, linestyle='--', label=f'direction correlation ({labels[i]})', lw=2)
        # plt.fill_between(
        #     bin_centers,
        #     dir_corr_add - dir_corr_std_add,
        #     dir_corr_add + dir_corr_std_add,
        #     color=color,
        #     alpha=0.2,
        #     label=fr'±1$\sigma$ direction ({labels[i]})'
        # )
    plt.xlim(-20, bin_centers.max() * 1.05)
    plt.ylim(min(load_object(f"plots/{n}/cache/dir_corr_std.npy").min() for n in names) - 0.1,
            max(load_object(f"plots/{n}/cache/dir_corr_std.npy").max() for n in names) + 0.1)
    plt.xlabel('Distance [µm]')
    plt.ylabel('Autocorrelation')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.title('Vector Orientation Autocorrelation Comparison')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_static_structure_factor_comparison(names, labels, variation=""):
    """
    Plot static structure factor comparison for multiple datasets.

    Parameters:
    - names: List of dataset names (paths to cache folders).
    - labels: List of labels corresponding to each dataset.
    - variation: Variation string for the output file name.
    """
    mpl.rcParams.update(plt.rcParamsDefault)
    plt.style.use('./science.mplstyle.txt')
    plt.style.use('dark_background')
    path = f'plots/combined_plots/static_structure_factor_comparison{variation}.pdf'

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel('q [rad/µm]')
    ax.set_ylabel('S(q)')
    ax.set_title('Static Structure Factor Comparison')

    # Plot additional datasets
    for i, dataset_name in enumerate(names):
        q = load_object(f"plots/{dataset_name}/cache/q.npy")
        S_q_mean = load_object(f"plots/{dataset_name}/cache/S_q_mean.npy")
        m = np.isfinite(q) & np.isfinite(S_q_mean) & (q > 0)
        q_plot = q[m]
        S_mean_plot = S_q_mean[m]
        color = tum_colors[i % len(tum_colors)]  # Automatically cycle through tum_colors
        plt.plot(q_plot, S_mean_plot, color=color, label=f'S(q) ({labels[i]})', lw=2)

    qmax = np.quantile(q_plot, 0.99)
    ylo = np.nanquantile(S_mean_plot, 0.02)
    yhi = np.nanquantile(S_mean_plot, 0.98)
    pad = 0.08 * max(1e-9, (yhi - ylo))
    ymin = ylo - pad
    ymax = yhi + pad
    plt.xlim(0, qmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('q [rad/µm]')
    plt.ylabel('S(q)')
    plt.legend(loc='best', fontsize=10)
    plt.title('Static Structure Factor Comparison')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def extract_branch_segments(
    skeleton: np.ndarray,
    px_per_micron_xy: float,
    px_per_micron_z: float,
    part_len_um: float = 5.0,
    orientation: str = "endpoints",  # "pca" or "endpoints" per segment
    keep_remainder: bool = True
) -> pd.DataFrame:
    """
    Segment each branch path into pieces of approximately `part_len_um` length
    and return per-segment features.

    Parameters
    ----------
    skeleton : ndarray, shape (T, Z, Y, X)
        Binary skeleton per time.
    px_per_micron_xy : float
        Pixels per micron for X and Y (microns = pixels / px_per_micron_xy).
    px_per_micron_z : float
        Pixels per micron for Z (microns = pixels / px_per_micron_z).
    part_len_um : float
        Target segment length in microns. The last segment of a branch can be shorter.
    orientation : {"pca","endpoints"}
        Per-segment orientation estimation.
    keep_remainder : bool
        If True, keep the last (possibly shorter) remainder segment. If False,
        drop it and keep only full-length segments.

    Returns
    -------
    features_df : pd.DataFrame with columns
        frame, branch_id, part_index, x, y, z, xa, ya, za, length, um, coords
        (positions/orientation in microns; coords is an (N,3) np.ndarray of XYZ microns)
    """
    if part_len_um <= 0:
        raise ValueError("part_len_um must be > 0")

    # pixels -> microns scale (match skan (z,y,x) order)
    sx = sy = 1.0 / float(px_per_micron_xy)
    sz = 1.0 / float(px_per_micron_z)
    scale_zyx = np.array([sz, sy, sx], dtype=np.float32)

    def _cumlen(coords: np.ndarray) -> np.ndarray:
        # coords: (L,3) in µm
        if coords.shape[0] < 2:
            return np.array([0.0], dtype=np.float32)
        segs = np.diff(coords, axis=0)
        d = np.linalg.norm(segs, axis=1)
        return np.concatenate([[0.0], np.cumsum(d).astype(np.float32)])

    def _interp_at_s(S: np.ndarray, P: np.ndarray, s: float) -> np.ndarray:
        # Piecewise-linear interpolation along arc-length
        # S: cumulative length (L,), P: (L,3), s in [0, S[-1]]
        if s <= 0:
            return P[0]
        if s >= S[-1]:
            return P[-1]
        i = int(np.searchsorted(S, s) - 1)
        i = max(0, min(i, len(S) - 2))
        ds = S[i+1] - S[i]
        if ds <= 0:
            return P[i].copy()
        a = (s - S[i]) / ds
        return P[i] + a * (P[i+1] - P[i])

    def _segment_points(P: np.ndarray, S: np.ndarray, s0: float, s1: float) -> np.ndarray:
        # Collect points inside [s0,s1] including interpolated endpoints
        p0 = _interp_at_s(S, P, s0)
        p1 = _interp_at_s(S, P, s1)
        mask_inner = (S > s0) & (S < s1)
        pts_inner = P[mask_inner]
        if pts_inner.size:
            return np.vstack([p0, pts_inner, p1])
        else:
            return np.vstack([p0, p1])

    rows = []
    T = int(skeleton.shape[0])
    um_flag = (px_per_micron_xy != 1.0) and (px_per_micron_z != 1.0)

    for t in tqdm(range(T), desc="extracting branch segments"):
        g = csr.Skeleton(skeleton[t])
        n_paths = int(g.n_paths)
        for branch_id in range(n_paths):
            co_zyx = g.path_coordinates(branch_id)
            if co_zyx.size == 0:
                continue

            # scale into microns in (z,y,x)
            P_zyx = np.asarray(co_zyx, dtype=np.float32) / scale_zyx
            S = _cumlen(P_zyx)
            Ltot = float(S[-1])
            if Ltot <= 0:
                continue

            # number of segments
            n_full = int(np.floor(Ltot / part_len_um))
            n_parts = n_full + (1 if keep_remainder and (Ltot - n_full * part_len_um) > 1e-6 else 0)
            if n_parts == 0:
                continue

            for k in range(n_parts):
                s0 = k * part_len_um
                s1 = min((k + 1) * part_len_um, Ltot)
                if s1 <= s0:
                    continue

                seg_pts_zyx = _segment_points(P_zyx, S, s0, s1)
                seg_pts_xyz = seg_pts_zyx[:, [2, 1, 0]]  # store coords in (x,y,z) microns

                # centroid
                cz, cy, cx = seg_pts_zyx.mean(axis=0)

                # orientation
                if orientation == "pca" and seg_pts_zyx.shape[0] >= 3:
                    X = seg_pts_zyx - np.array([cz, cy, cx], dtype=np.float32)
                    cov = X.T @ X
                    w, V = np.linalg.eigh(cov)
                    d_zyx = V[:, -1]
                else:
                    d_zyx = seg_pts_zyx[-1] - seg_pts_zyx[0]

                nrm = float(np.linalg.norm(d_zyx))
                if not np.isfinite(nrm) or nrm == 0.0:
                    continue
                d_zyx = d_zyx / nrm
                d_xyz = d_zyx[[2, 1, 0]]

                rows.append({
                    "frame":     int(t),
                    "branch_id": int(branch_id),
                    "part_index": int(k),          # index within the branch
                    "x":         float(cx),        # centroid in µm (x,y,z order)
                    "y":         float(cy),
                    "z":         float(cz),
                    "xa":        float(d_xyz[0]),  # unit headless director
                    "ya":        float(d_xyz[1]),
                    "za":        float(d_xyz[2]),
                    "length":    float(s1 - s0),   # segment length in µm (last can be shorter)
                    "um":        bool(um_flag),
                    "coords":    seg_pts_xyz,      # np.ndarray (N,3) in (x,y,z) µm
                })

    return pd.DataFrame(rows, columns=[
        "frame","branch_id","part_index","x","y","z","xa","ya","za","length","um","coords"
    ])

def compute_df_ang_structure_factor(df, velocity_cols=['vx','vy','vz'], orientation_cols=['xa','ya','za']):
    """
    Annotate each row with the structure factor S between orientation and velocity vector:
    S = (3/2) * ( (n·n0)^2 - 1/3 )
    where n is the orientation (xa,ya,za), n0 is the velocity direction (vx,vy,vz).
    Adds a column 'S' to the dataframe.
    """
    df = df.copy()
    mask = df[velocity_cols + orientation_cols].notnull().all(axis=1)
    vx = df.loc[mask, velocity_cols[0]]
    vy = df.loc[mask, velocity_cols[1]]
    vz = df.loc[mask, velocity_cols[2]]
    xa = df.loc[mask, orientation_cols[0]]
    ya = df.loc[mask, orientation_cols[1]]
    za = df.loc[mask, orientation_cols[2]]
    v_norm = np.sqrt(vx**2 + vy**2 + vz**2)
    u_norm = np.sqrt(xa**2 + ya**2 + za**2)
    valid = (v_norm > 0) & (u_norm > 0)
    S = np.full(len(df), np.nan)
    dot = (vx[valid]*xa[valid] + vy[valid]*ya[valid] + vz[valid]*za[valid]) / (v_norm[valid] * u_norm[valid])
    S_vals = (3/2) * (dot**2 - 1/3)
    S[mask[mask].index[valid]] = S_vals
    df['S'] = S
    return df

def link_branches_stable(features_dir,
                         max_dist_um=5.0,
                         w_pos=1.0, w_ori=4.0, w_len=0.1,
                         ori_cols=('xa','ya','za')):
    """
    Stable per-branch linking across consecutive frames using a multi-term cost.
    Returns a trajectories DataFrame with a persistent 'particle' id.
    """
    df = (features_dir
          .sort_values(['frame','branch_id'])
          .reset_index(drop=True)
         ).copy()
    T = int(df['frame'].max()) + 1

    # Prepare per-frame tables
    frames = [df[df['frame']==t].reset_index(drop=True) for t in range(T)]
    # Track IDs per row per frame
    id_maps = [np.full(len(frames[0]), -1, dtype=int)]
    id_maps[0][:] = np.arange(len(frames[0]))  # initialize unique ids at t=0
    next_id = len(frames[0])

    for t in range(T-1):
        A = frames[t]; B = frames[t+1]
        nA, nB = len(A), len(B)
        if nA == 0 and nB == 0:
            id_maps.append(np.array([], dtype=int)); continue
        if nA == 0:
            ids_next = np.arange(next_id, next_id + nB, dtype=int)
            next_id += nB
            id_maps.append(ids_next); continue
        if nB == 0:
            id_maps.append(np.array([], dtype=int)); continue

        # positions (µm) and orientations
        A_pos = A[['x','y','z']].to_numpy(float)
        B_pos = B[['x','y','z']].to_numpy(float)

        A_u = A[list(ori_cols)].to_numpy(float)
        B_u = B[list(ori_cols)].to_numpy(float)
        # normalize orientations just in case
        A_u /= (np.linalg.norm(A_u, axis=1, keepdims=True) + 1e-12)
        B_u /= (np.linalg.norm(B_u, axis=1, keepdims=True) + 1e-12)

        A_L = A['length'].to_numpy(float)
        B_L = B['length'].to_numpy(float)

        A_pi = A['part_index'].to_numpy(int)
        B_pi = B['part_index'].to_numpy(int)

        A_bid = A['branch_id'].to_numpy(int)
        B_bid = B['branch_id'].to_numpy(int)

        # candidate neighborhoods
        treeB = cKDTree(B_pos)
        neigh = treeB.query_ball_point(A_pos, r=max_dist_um)

        INF = 1e9
        C = np.full((nA, nB), INF, dtype=float)
        for i, js in enumerate(neigh):
            if not js: continue
            # vectorized costs to those js
            dpos = np.linalg.norm(B_pos[js]-A_pos[i], axis=1)  # µm
            dot  = np.abs((B_u[js] @ A_u[i]))                     # headless
            ang  = np.arccos(np.clip(dot, -1.0, 1.0))            # rad
            dlen = np.abs(B_L[js]-A_L[i])
            dord = np.abs(B_pi[js]-A_pi[i]).astype(float)
            switch = (B_bid[js] != A_bid[i]).astype(float)

            C[i, js] = w_pos*dpos + w_ori*ang + w_len*dlen

        # Hungarian assignment
        ri, cj = linear_sum_assignment(C)
        # Accept only reasonable matches
        accept = C[ri, cj] < (w_pos*max_dist_um + w_ori*np.pi/3.0)  # tunable
        ri, cj = ri[accept], cj[accept]

        # propagate ids
        ids_prev = id_maps[t]
        ids_next = np.full(nB, -1, dtype=int)
        # matched
        for i, j in zip(ri, cj):
            ids_next[j] = ids_prev[i]
        # unmatched -> new ids
        unmatched = np.where(ids_next < 0)[0]
        if unmatched.size:
            ids_next[unmatched] = np.arange(next_id, next_id + unmatched.size, int)
            next_id += unmatched.size

        id_maps.append(ids_next)

    # Build trajectories DataFrame with persistent ids
    out = []
    for t in range(T):
        if len(frames[t]) == 0:
            continue
        fr = frames[t].copy()
        fr['particle'] = id_maps[t]
        out.append(fr[['frame','particle','branch_id','part_index','x','y','z','xa','ya','za','length']])
    traj = pd.concat(out, ignore_index=True)
    return traj

def _estimate_curvature_zyx(P: np.ndarray) -> np.ndarray:
    """
    Approximate curvature per interior point of a 3D polyline P (L,3).
    Returns array (L,) with NaN at endpoints.
    Discrete formula using triangle circumradius.
    """
    L = P.shape[0]
    if L < 3:
        return np.full(L, np.nan, dtype=np.float32)
    curv = np.full(L, np.nan, dtype=np.float32)
    A = P[:-2]
    B = P[1:-1]
    C = P[2:]
    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(A - C, axis=1)
    c = np.linalg.norm(A - B, axis=1)
    # Heron area
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    # circumradius R = abc / (4 area)  => curvature k = 1/R = 4 area / (abc)
    denom = a * b * c
    with np.errstate(invalid='ignore', divide='ignore'):
        kappa = np.where(denom > 0, 4.0 * area / denom, 0.0)
    curv[1:-1] = kappa
    return curv

def extract_branch_segments_stable(
    skeleton: np.ndarray,
    px_per_micron_xy: float,
    px_per_micron_z: float,
    part_len_um: float = 5.0,
    orientation: str = "endpoints",
    keep_remainder: bool = True,
    branch_traj: Optional[pd.DataFrame] = None,
    stable: bool = True,
    min_points_per_segment: int = 2,
    images: Optional[np.ndarray] = None  # (T, Z, Y, X) original intensity stack
) -> pd.DataFrame:
    """
    Stable segmentation of branches over time, with optional per-segment intensity
    computed from the original images.

    If stable=True:
      For each tracked branch we keep normalized centers u in [0,1] from the
      first time it appears, and map those u to the current arc-length in later
      frames. This avoids re-sampling jitter.

    If `images` is given, adds columns:
      - I_mean: mean raw intensity over voxels touching the segment centerline
      - I_sum:  sum of those intensities
    """
    if part_len_um <= 0:
        raise ValueError("part_len_um must be > 0")
    if images is not None and images.shape[0] != skeleton.shape[0]:
        raise ValueError("images and skeleton must have same T along axis 0")

    # scale factors (pixels -> µm) matching (z,y,x)
    sx = sy = 1.0 / float(px_per_micron_xy)
    sz = 1.0 / float(px_per_micron_z)
    scale_zyx = np.array([sz, sy, sx], dtype=np.float32)

    def _cumlen(P):
        if P.shape[0] < 2:
            return np.array([0.0], dtype=np.float32)
        segs = np.diff(P, axis=0)
        d = np.linalg.norm(segs, axis=1)
        return np.concatenate([[0.0], np.cumsum(d).astype(np.float32)])

    def _interp_at_s(S, P, s):
        if s <= 0: return P[0]
        if s >= S[-1]: return P[-1]
        i = int(np.searchsorted(S, s) - 1)
        i = max(0, min(i, len(S) - 2))
        ds = S[i+1] - S[i]
        if ds <= 0: return P[i]
        a = (s - S[i]) / ds
        return P[i] + a * (P[i+1] - P[i])

    def _segment_points(P: np.ndarray, S: np.ndarray, s0: float, s1: float) -> np.ndarray:
        p0 = _interp_at_s(S, P, s0)
        p1 = _interp_at_s(S, P, s1)
        mask_inner = (S > s0) & (S < s1)
        pts_inner = P[mask_inner]
        if pts_inner.size:
            return np.vstack([p0, pts_inner, p1])
        else:
            return np.vstack([p0, p1])

    rows = []
    T = int(skeleton.shape[0])
    um_flag = (px_per_micron_xy != 1.0) and (px_per_micron_z != 1.0)

    # map (frame, local_id) -> global branch id if provided
    global_id_map = {}
    if branch_traj is not None and not branch_traj.empty:
        need_cols = {'frame','branch_id','particle'}
        if not need_cols <= set(branch_traj.columns):
            raise ValueError(f"branch_traj must have columns {need_cols}")
        for r in branch_traj[['frame','branch_id','particle']].itertuples(index=False):
            global_id_map[(int(r.frame), int(r.branch_id))] = int(r.particle)

    # remember normalized centers per global branch
    center_memory = {}

    for t in tqdm(range(T), desc="stable segmenting"):
        g = csr.Skeleton(skeleton[t])
        n_paths = int(g.n_paths)
        for local_id in range(n_paths):
            co_zyx = g.path_coordinates(local_id)
            if co_zyx.size == 0:
                continue
            # convert path to µm in (z,y,x)
            P_zyx = np.asarray(co_zyx, dtype=np.float32) * scale_zyx
            S = _cumlen(P_zyx)
            Ltot = float(S[-1])
            if Ltot <= 0:
                continue

            # resolve global branch id (for stability across frames)
            if global_id_map:
                gbid = global_id_map.get((t, local_id), None)
            else:
                gbid = (t, local_id)

            new_branch = gbid not in center_memory

            if (not stable) or new_branch:
                # first time: lay centers spaced by part_len_um
                n_full = int(np.floor(Ltot / part_len_um))
                n_parts = n_full + (1 if keep_remainder and (Ltot - n_full * part_len_um) > 1e-6 else 0)
                if n_parts <= 0:
                    continue
                centers_s = np.array(
                    [min((k + 0.5) * part_len_um, Ltot) for k in range(n_parts)],
                    dtype=np.float32
                )
                u_centers = np.where(Ltot > 0, centers_s / Ltot, 0.0)
                center_memory[gbid] = u_centers
            else:
                # reuse normalized centers and map to current arc-length
                u_centers = center_memory[gbid]
                centers_s = u_centers * Ltot

            if u_centers.size == 0:
                continue
            centers_s_sorted = np.sort(centers_s)

            # build boundaries (Voronoi along arc-length)
            bounds = [0.0]
            for i in range(len(centers_s_sorted) - 1):
                bounds.append(0.5 * (centers_s_sorted[i] + centers_s_sorted[i+1]))
            bounds.append(Ltot)
            bounds = np.array(bounds, dtype=np.float32)

            # curvature along the path
            kappa = _estimate_curvature_zyx(P_zyx)

            for idx_center, c_s in enumerate(centers_s_sorted):
                s0 = bounds[idx_center]
                s1 = bounds[idx_center + 1]
                if s1 <= s0:
                    continue

                seg_pts = _segment_points(P_zyx, S, s0, s1)
                if seg_pts.shape[0] < min_points_per_segment:
                    continue

                # centroid in µm (z,y,x order)
                cz, cy, cx = seg_pts.mean(axis=0)

                # orientation
                if orientation == "pca" and seg_pts.shape[0] >= 3:
                    Xc = seg_pts - np.array([cz, cy, cx], dtype=np.float32)
                    cov = Xc.T @ Xc
                    w, V = np.linalg.eigh(cov)
                    d_zyx = V[:, -1]
                else:
                    d_zyx = seg_pts[-1] - seg_pts[0]
                nrm = np.linalg.norm(d_zyx)
                if not np.isfinite(nrm) or nrm == 0:
                    continue
                d_zyx /= nrm
                d_xyz = d_zyx[[2, 1, 0]]

                # mean curvature in [s0,s1]
                seg_mask_full = (S >= s0) & (S <= s1)
                curv_vals = kappa[seg_mask_full]
                curv_mean = float(np.nanmean(curv_vals)) if np.isfinite(curv_vals).any() else np.nan

                # optional intensity over voxels touching the segment line
                I_mean = np.nan
                I_sum = np.nan
                if images is not None:
                    Z, Y, X = images.shape[1], images.shape[2], images.shape[3]
                    # convert seg_pts (µm) back to voxel indices (z,y,x)
                    vz = np.rint(seg_pts[:, 0] / sz).astype(int)
                    vy = np.rint(seg_pts[:, 1] / sy).astype(int)
                    vx = np.rint(seg_pts[:, 2] / sx).astype(int)
                    m = (vz >= 0) & (vz < Z) & (vy >= 0) & (vy < Y) & (vx >= 0) & (vx < X)
                    if np.any(m):
                        # remove duplicates
                        idx_lin = (vz[m] * (Y * X) + vy[m] * X + vx[m])
                        uidx = np.unique(idx_lin)
                        # unravel back to z,y,x
                        vz_u = uidx // (Y * X)
                        vy_u = (uidx % (Y * X)) // X
                        vx_u = (uidx % (Y * X)) % X
                        vals = images[t, vz_u, vy_u, vx_u]
                        if vals.size:
                            I_mean = float(vals.mean())
                            I_sum = float(vals.sum())

                rows.append({
                    "frame": int(t),
                    "branch_id": int(local_id),
                    "part_index": int(idx_center),
                    "x": float(cx),
                    "y": float(cy),
                    "z": float(cz),
                    "xa": float(d_xyz[0]),
                    "ya": float(d_xyz[1]),
                    "za": float(d_xyz[2]),
                    "length": float(s1 - s0),
                    "u_center": float(c_s / Ltot if Ltot > 0 else 0.0),
                    "curv_mean": curv_mean,
                    "I_mean": I_mean,
                    "I_sum": I_sum,
                    "um": bool(um_flag),
                    "coords": seg_pts[:, [2, 1, 0]],  # (x,y,z) in µm
                })

    return pd.DataFrame(rows, columns=[
        "frame","branch_id","part_index","x","y","z","xa","ya","za",
        "length","u_center","curv_mean","I_mean","I_sum","um","coords"
    ])

def link_parts_stable_intensity(features_parts,
                                max_dist_um=5.0,
                                w_pos=1.0, w_ori=4.0, w_len=0.1, w_order=0.25, w_switch=2.0,
                                w_center=2.0, w_curv=1.0, w_int=1.0,
                                intensity_col: str = 'I_mean',
                                ori_cols=('xa','ya','za')) -> pd.DataFrame:
    """
    Like link_parts_stable, but also uses an intensity term if available.

    Cost for candidate i@t -> j@t+1:
        C = w_pos*||Δpos|| + w_ori*angle(headless)
          + w_len*|Δlength| + w_order*|Δpart_index| + w_switch*[branch changes]
          + w_center*|Δu_center| + w_curv*|Δcurv_mean|
          + w_int*|ΔI|   (if intensity_col exists)
    """
    df = (features_parts
          .sort_values(['frame','branch_id','part_index'])
          .reset_index(drop=True)
         ).copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'frame','particle','branch_id','part_index',
            'x','y','z','xa','ya','za','length',
            'u_center','curv_mean', intensity_col, 'coords',
            'vx','vy','vz','vxd','vyd','vzd',
            'vx_drift','vy_drift','vz_drift','vdrift'
        ])

    has_center = 'u_center' in df.columns
    has_curv   = 'curv_mean' in df.columns
    has_coords = 'coords' in df.columns
    has_int    = intensity_col in df.columns

    T = int(df['frame'].max()) + 1
    frames = [df[df['frame'] == t].reset_index(drop=True) for t in range(T)]

    id_maps = []
    if len(frames[0]):
        id_maps.append(np.arange(len(frames[0]), dtype=int))
        next_id = len(frames[0])
    else:
        id_maps.append(np.array([], dtype=int))
        next_id = 0

    def _hungarian(C):
        try:
            return linear_sum_assignment(C)
        except Exception:
            INF = 1e18
            Cw = C.copy()
            nA, nB = Cw.shape
            rI, cJ = [], []
            while True:
                i, j = np.unravel_index(np.argmin(Cw), Cw.shape)
                if not np.isfinite(Cw[i, j]) or Cw[i, j] >= INF:
                    break
                rI.append(i); cJ.append(j)
                Cw[i, :] = INF; Cw[:, j] = INF
            return np.array(rI, int), np.array(cJ, int)

    INF = 1e12

    for t in tqdm(range(T - 1), desc="Linking parts (intensity)"):
        A = frames[t]; B = frames[t+1]
        nA, nB = len(A), len(B)
        if nA == 0 and nB == 0:
            id_maps.append(np.array([], dtype=int)); continue
        if nA == 0:
            ids_next = np.arange(next_id, next_id + nB, dtype=int)
            next_id += nB
            id_maps.append(ids_next); continue
        if nB == 0:
            id_maps.append(np.array([], dtype=int)); continue

        A_pos = A[['x','y','z']].to_numpy(float)
        B_pos = B[['x','y','z']].to_numpy(float)

        A_u = A[list(ori_cols)].to_numpy(float)
        B_u = B[list(ori_cols)].to_numpy(float)
        A_u /= (np.linalg.norm(A_u, axis=1, keepdims=True) + 1e-12)
        B_u /= (np.linalg.norm(B_u, axis=1, keepdims=True) + 1e-12)

        A_L = A['length'].to_numpy(float)
        B_L = B['length'].to_numpy(float)

        A_pi = A['part_index'].to_numpy(int)
        B_pi = B['part_index'].to_numpy(int)

        A_bid = A['branch_id'].to_numpy(int)
        B_bid = B['branch_id'].to_numpy(int)

        A_center = A['u_center'].to_numpy(float) if has_center else np.zeros(nA)
        B_center = B['u_center'].to_numpy(float) if has_center else np.zeros(nB)
        A_curv   = A['curv_mean'].fillna(0).to_numpy(float) if has_curv else np.zeros(nA)
        B_curv   = B['curv_mean'].fillna(0).to_numpy(float) if has_curv else np.zeros(nB)
        A_int    = A[intensity_col].to_numpy(float) if has_int else np.zeros(nA)
        B_int    = B[intensity_col].to_numpy(float) if has_int else np.zeros(nB)

        if cKDTree is not None:
            treeB = cKDTree(B_pos)
            neigh = treeB.query_ball_point(A_pos, r=max_dist_um)
        else:
            d_full = np.linalg.norm(A_pos[:,None,:]-B_pos[None,:,:], axis=2)
            neigh = [list(np.where(d_full[i] <= max_dist_um)[0]) for i in range(nA)]

        C = np.full((nA, nB), INF, dtype=float)
        for i, js in enumerate(neigh):
            if not js:
                continue
            dpos = np.linalg.norm(B_pos[js] - A_pos[i], axis=1)
            dot  = np.abs(B_u[js] @ A_u[i])
            ang  = np.arccos(np.clip(dot, -1.0, 1.0))
            dlen = np.abs(B_L[js] - A_L[i])
            dord = np.abs(B_pi[js] - A_pi[i]).astype(float)
            switch = (B_bid[js] != A_bid[i]).astype(float)
            dcen = np.abs(B_center[js] - A_center[i])
            dcurv = np.abs(B_curv[js] - A_curv[i])
            if has_int:
                dI = np.abs(B_int[js] - A_int[i])
            else:
                dI = 0.0

            C[i, js] = (w_pos * dpos
                        + w_ori * ang
                        + w_len * dlen
                        + w_order * dord
                        + w_switch * switch
                        + w_center * dcen
                        + w_curv * dcurv
                        + (w_int * dI if has_int else 0.0))

        ri, cj = _hungarian(C)
        thresh = w_pos * max_dist_um + w_ori * (np.pi/3)
        acc = C[ri, cj] < thresh
        ri, cj = ri[acc], cj[acc]

        ids_prev = id_maps[t]
        ids_next = np.full(nB, -1, dtype=int)
        for i, j in zip(ri, cj):
            ids_next[j] = ids_prev[i]
        unmatched = np.where(ids_next < 0)[0]
        if unmatched.size:
            ids_next[unmatched] = np.arange(next_id, next_id + unmatched.size, dtype=int)
            next_id += unmatched.size
        id_maps.append(ids_next)

    base_cols = ['frame','particle','branch_id','part_index',
                 'x','y','z','xa','ya','za','length']
    if has_center:
        base_cols.append('u_center')
    if has_curv:
        base_cols.append('curv_mean')
    if has_int:
        base_cols.append(intensity_col)
    if has_coords:
        base_cols.append('coords')

    out = []
    for t in range(T):
        if len(frames[t]) == 0:
            continue
        fr = frames[t].copy()
        fr['particle'] = id_maps[t]
        out.append(fr[base_cols])
    traj = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=base_cols)

    if not traj.empty:
        dt_s = globals().get('dt_s', None)
        if dt_s is None:
            fps = globals().get('fps', None)
            dt_s = 1.0 / float(fps) if fps not in (None, 0) else 1.0
        dt_s = float(dt_s)
        traj = traj.sort_values(['particle','frame']).reset_index(drop=True)
        dxyz = traj.groupby('particle')[['x','y','z']].diff()
        traj['vx'] = dxyz['x'] / dt_s
        traj['vy'] = dxyz['y'] / dt_s
        traj['vz'] = dxyz['z'] / dt_s

        vx_drift = traj.groupby('frame')['vx'].mean().fillna(0)
        vy_drift = traj.groupby('frame')['vy'].mean().fillna(0)
        vz_drift = traj.groupby('frame')['vz'].mean().fillna(0)
        gx = traj['frame'].map(vx_drift)
        gy = traj['frame'].map(vy_drift)
        gz = traj['frame'].map(vz_drift)
        traj['vxd'] = traj['vx'] - gx
        traj['vyd'] = traj['vy'] - gy
        traj['vzd'] = traj['vz'] - gz
        traj['vx_drift'] = gx
        traj['vy_drift'] = gy
        traj['vz_drift'] = gz
        traj = traj.sort_values(['frame','particle']).reset_index(drop=True)

    return traj

def link_parts_stable(features_parts,
                      max_dist_um=5.0,
                      w_pos=1.0, w_ori=4.0, w_len=0.1, w_order=0.25, w_switch=2.0,
                      w_center=2.0, w_curv=1.0,
                      ori_cols=('xa','ya','za')) -> pd.DataFrame:
    """
    Stable linking of per-branch PARTS (from features_parts) across frames.
    Adds penalties for difference in normalized center position (u_center),
    mean curvature (curv_mean), and preserves per-segment coords.

    Returns a trajectories DataFrame with persistent 'particle' IDs and keeps:
      x,y,z, xa,ya,za, length, (optional) u_center, curv_mean, coords
      plus velocity and drift-corrected components.
    """
    df = (features_parts
          .sort_values(['frame','branch_id','part_index'])
          .reset_index(drop=True)
         ).copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'frame','particle','branch_id','part_index',
            'x','y','z','xa','ya','za','length',
            'u_center','curv_mean','coords',
            'vx','vy','vz','vxd','vyd','vzd',
            'vx_drift','vy_drift','vz_drift','vdrift'
        ])

    has_center = 'u_center' in df.columns
    has_curv   = 'curv_mean' in df.columns
    has_coords = 'coords' in df.columns

    T = int(df['frame'].max()) + 1
    frames = [df[df['frame'] == t].reset_index(drop=True) for t in range(T)]

    id_maps = []
    if len(frames[0]):
        id_maps.append(np.arange(len(frames[0]), dtype=int))
        next_id = len(frames[0])
    else:
        id_maps.append(np.array([], dtype=int))
        next_id = 0

    def _hungarian(C):
        try:
            return linear_sum_assignment(C)
        except Exception:
            Cw = C.copy()
            INF = 1e18
            nA, nB = Cw.shape
            rI, cJ = [], []
            while True:
                i, j = np.unravel_index(np.argmin(Cw), Cw.shape)
                if not np.isfinite(Cw[i, j]) or Cw[i, j] >= INF:
                    break
                rI.append(i); cJ.append(j)
                Cw[i, :] = INF; Cw[:, j] = INF
            return np.array(rI, int), np.array(cJ, int)

    INF = 1e12

    for t in tqdm(range(T - 1), desc="Linking parts (stable+)"):
        A = frames[t]; B = frames[t+1]
        nA, nB = len(A), len(B)
        if nA == 0 and nB == 0:
            id_maps.append(np.array([], dtype=int)); continue
        if nA == 0:
            ids_next = np.arange(next_id, next_id + nB, dtype=int)
            next_id += nB
            id_maps.append(ids_next); continue
        if nB == 0:
            id_maps.append(np.array([], dtype=int)); continue

        A_pos = A[['x','y','z']].to_numpy(float)
        B_pos = B[['x','y','z']].to_numpy(float)

        A_u = A[list(ori_cols)].to_numpy(float)
        B_u = B[list(ori_cols)].to_numpy(float)
        A_u /= (np.linalg.norm(A_u, axis=1, keepdims=True) + 1e-12)
        B_u /= (np.linalg.norm(B_u, axis=1, keepdims=True) + 1e-12)

        A_L = A['length'].to_numpy(float)
        B_L = B['length'].to_numpy(float)

        A_pi = A['part_index'].to_numpy(int)
        B_pi = B['part_index'].to_numpy(int)

        A_bid = A['branch_id'].to_numpy(int)
        B_bid = B['branch_id'].to_numpy(int)

        if has_center:
            A_center = A['u_center'].to_numpy(float)
            B_center = B['u_center'].to_numpy(float)
        else:
            A_center = np.zeros(nA)
            B_center = np.zeros(nB)

        if has_curv:
            A_curv = A['curv_mean'].fillna(0).to_numpy(float)
            B_curv = B['curv_mean'].fillna(0).to_numpy(float)
        else:
            A_curv = np.zeros(nA)
            B_curv = np.zeros(nB)

        if cKDTree is not None:
            treeB = cKDTree(B_pos)
            neigh = treeB.query_ball_point(A_pos, r=max_dist_um)
        else:
            d_full = np.linalg.norm(A_pos[:,None,:]-B_pos[None,:,:], axis=2)
            neigh = [list(np.where(d_full[i] <= max_dist_um)[0]) for i in range(nA)]

        C = np.full((nA, nB), INF, dtype=float)
        for i, js in enumerate(neigh):
            if not js: 
                continue
            dpos = np.linalg.norm(B_pos[js]-A_pos[i], axis=1)
            dot  = np.abs(B_u[js] @ A_u[i])
            ang  = np.arccos(np.clip(dot, -1.0, 1.0))
            dlen = np.abs(B_L[js]-A_L[i])
            dord = np.abs(B_pi[js]-A_pi[i]).astype(float)
            switch = (B_bid[js] != A_bid[i]).astype(float)
            dcen = np.abs(B_center[js] - A_center[i])
            dcurv = np.abs(B_curv[js] - A_curv[i])

            C[i, js] = (w_pos * dpos
                        + w_ori * ang
                        + w_len * dlen
                        + w_order * dord
                        + w_switch * switch
                        + w_center * dcen
                        + w_curv * dcurv)

        ri, cj = _hungarian(C)
        thresh = w_pos * max_dist_um + w_ori * (np.pi/3)
        acc = C[ri, cj] < thresh
        ri, cj = ri[acc], cj[acc]

        ids_prev = id_maps[t]
        ids_next = np.full(nB, -1, dtype=int)
        for i, j in zip(ri, cj):
            ids_next[j] = ids_prev[i]
        unmatched = np.where(ids_next < 0)[0]
        if unmatched.size:
            ids_next[unmatched] = np.arange(next_id, next_id + unmatched.size, dtype=int)
            next_id += unmatched.size
        id_maps.append(ids_next)

    out = []
    base_cols = ['frame','particle','branch_id','part_index',
                 'x','y','z','xa','ya','za','length']
    if has_center:
        base_cols.append('u_center')
    if has_curv:
        base_cols.append('curv_mean')
    if has_coords:
        base_cols.append('coords')

    for t in range(T):
        if len(frames[t]) == 0:
            continue
        fr = frames[t].copy()
        fr['particle'] = id_maps[t]
        out.append(fr[base_cols])
    traj = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=base_cols)

    if not traj.empty:
        dt_s = globals().get('dt_s', None)
        if dt_s is None:
            fps = globals().get('fps', None)
            dt_s = 1.0 / float(fps) if fps not in (None, 0) else 1.0
        dt_s = float(dt_s)
        traj = traj.sort_values(['particle','frame']).reset_index(drop=True)
        dxyz = traj.groupby('particle')[['x','y','z']].diff()
        traj['vx'] = dxyz['x'] / dt_s
        traj['vy'] = dxyz['y'] / dt_s
        traj['vz'] = dxyz['z'] / dt_s

        vx_drift = traj.groupby('frame')['vx'].mean().fillna(0)
        vy_drift = traj.groupby('frame')['vy'].mean().fillna(0)
        vz_drift = traj.groupby('frame')['vz'].mean().fillna(0)
        gx = traj['frame'].map(vx_drift)
        gy = traj['frame'].map(vy_drift)
        gz = traj['frame'].map(vz_drift)
        traj['vxd'] = traj['vx'] - gx
        traj['vyd'] = traj['vy'] - gy
        traj['vzd'] = traj['vz'] - gz
        traj['vx_drift'] = gx
        traj['vy_drift'] = gy
        traj['vz_drift'] = gz
        traj = traj.sort_values(['frame','particle']).reset_index(drop=True)

    return traj

def extract_branch_segments(
    skeleton: np.ndarray,
    px_per_micron_xy: float,
    px_per_micron_z: float,
    part_len_um: float = 5.0,
    orientation: str = "endpoints",  # "pca" or "endpoints" per segment
    keep_remainder: bool = True
) -> pd.DataFrame:
    """
    Segment each branch path into pieces of approximately `part_len_um` length
    and return per-segment features.

    Parameters
    ----------
    skeleton : ndarray, shape (T, Z, Y, X)
        Binary skeleton per time.
    px_per_micron_xy : float
        Pixels per micron for X and Y (microns = pixels / px_per_micron_xy).
    px_per_micron_z : float
        Pixels per micron for Z (microns = pixels / px_per_micron_z).
    part_len_um : float
        Target segment length in microns. The last segment of a branch can be shorter.
    orientation : {"pca","endpoints"}
        Per-segment orientation estimation.
    keep_remainder : bool
        If True, keep the last (possibly shorter) remainder segment. If False,
        drop it and keep only full-length segments.

    Returns
    -------
    features_df : pd.DataFrame with columns
        frame, branch_id, part_index, x, y, z, xa, ya, za, length, um, coords
        (positions/orientation in microns; coords is an (N,3) np.ndarray of XYZ microns)
    """
    if part_len_um <= 0:
        raise ValueError("part_len_um must be > 0")

    # pixels -> microns scale (match skan (z,y,x) order)
    sx = sy = 1.0 / float(px_per_micron_xy)
    sz = 1.0 / float(px_per_micron_z)
    scale_zyx = np.array([sz, sy, sx], dtype=np.float32)

    def _cumlen(coords: np.ndarray) -> np.ndarray:
        # coords: (L,3) in µm
        if coords.shape[0] < 2:
            return np.array([0.0], dtype=np.float32)
        segs = np.diff(coords, axis=0)
        d = np.linalg.norm(segs, axis=1)
        return np.concatenate([[0.0], np.cumsum(d).astype(np.float32)])

    def _interp_at_s(S: np.ndarray, P: np.ndarray, s: float) -> np.ndarray:
        # Piecewise-linear interpolation along arc-length
        # S: cumulative length (L,), P: (L,3), s in [0, S[-1]]
        if s <= 0:
            return P[0]
        if s >= S[-1]:
            return P[-1]
        i = int(np.searchsorted(S, s) - 1)
        i = max(0, min(i, len(S) - 2))
        ds = S[i+1] - S[i]
        if ds <= 0:
            return P[i].copy()
        a = (s - S[i]) / ds
        return P[i] + a * (P[i+1] - P[i])

    def _segment_points(P: np.ndarray, S: np.ndarray, s0: float, s1: float) -> np.ndarray:
        # Collect points inside [s0,s1] including interpolated endpoints
        p0 = _interp_at_s(S, P, s0)
        p1 = _interp_at_s(S, P, s1)
        mask_inner = (S > s0) & (S < s1)
        pts_inner = P[mask_inner]
        if pts_inner.size:
            return np.vstack([p0, pts_inner, p1])
        else:
            return np.vstack([p0, p1])

    rows = []
    T = int(skeleton.shape[0])
    um_flag = (px_per_micron_xy != 1.0) and (px_per_micron_z != 1.0)

    for t in tqdm(range(T), desc="extracting branch segments"):
        g = csr.Skeleton(skeleton[t])
        n_paths = int(g.n_paths)
        for branch_id in range(n_paths):
            co_zyx = g.path_coordinates(branch_id)
            if co_zyx.size == 0:
                continue

            # scale into microns in (z,y,x)
            P_zyx = np.asarray(co_zyx, dtype=np.float32) * scale_zyx
            S = _cumlen(P_zyx)
            Ltot = float(S[-1])
            if Ltot <= 0:
                continue

            # number of segments
            n_full = int(np.floor(Ltot / part_len_um))
            n_parts = n_full + (1 if keep_remainder and (Ltot - n_full * part_len_um) > 1e-6 else 0)
            if n_parts == 0:
                continue

            for k in range(n_parts):
                s0 = k * part_len_um
                s1 = min((k + 1) * part_len_um, Ltot)
                if s1 <= s0:
                    continue

                seg_pts_zyx = _segment_points(P_zyx, S, s0, s1)
                seg_pts_xyz = seg_pts_zyx[:, [2, 1, 0]]  # store coords in (x,y,z) microns

                # centroid
                cz, cy, cx = seg_pts_zyx.mean(axis=0)

                # orientation
                if orientation == "pca" and seg_pts_zyx.shape[0] >= 3:
                    X = seg_pts_zyx - np.array([cz, cy, cx], dtype=np.float32)
                    cov = X.T @ X
                    w, V = np.linalg.eigh(cov)
                    d_zyx = V[:, -1]
                else:
                    d_zyx = seg_pts_zyx[-1] - seg_pts_zyx[0]

                nrm = float(np.linalg.norm(d_zyx))
                if not np.isfinite(nrm) or nrm == 0.0:
                    continue
                d_zyx = d_zyx / nrm
                d_xyz = d_zyx[[2, 1, 0]]

                rows.append({
                    "frame":     int(t),
                    "branch_id": int(branch_id),
                    "part_index": int(k),          # index within the branch
                    "x":         float(cx),        # centroid in µm (x,y,z order)
                    "y":         float(cy),
                    "z":         float(cz),
                    "xa":        float(d_xyz[0]),  # unit headless director
                    "ya":        float(d_xyz[1]),
                    "za":        float(d_xyz[2]),
                    "length":    float(s1 - s0),   # segment length in µm (last can be shorter)
                    "um":        bool(um_flag),
                    "coords":    seg_pts_xyz,      # np.ndarray (N,3) in (x,y,z) µm
                })

    return pd.DataFrame(rows, columns=[
        "frame","branch_id","part_index","x","y","z","xa","ya","za","length","um","coords"
    ])

def temporal_autocorr_from_parts(
    trajectories_parts: pd.DataFrame,
    max_lag=None,
    compute_S: bool = True,
    velocity_cols=('vx','vy','vz'),
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-particle temporal autocorrelations directly from trajectories_parts,
    returning exactly the same dict output as temporal_autocorr_all_fast by
    internally constructing ['particle','frame','dx','dy','dz'] and delegating.

    Preference order for vector columns:
      1) vxd,vyd,vzd (drift-corrected velocities), if prefer_drift_corrected=True
      2) vx,vy,vz    (raw velocities)
      3) dx,dy,dz    (displacements), if already present
      4) compute displacements via groupby diff(x,y,z)

    Notes:
      - Using velocity vs. displacement only changes scale; directions are normalized,
        so the autocorrelation is unaffected.
      - Frames without valid vectors remain NaN and are treated like in the fast path.
    """
    cols = set(trajectories_parts.columns)
    df_in = trajectories_parts[['particle','frame']].copy()
    df_in['dx'] = trajectories_parts[velocity_cols[0]].astype(float)
    df_in['dy'] = trajectories_parts[velocity_cols[1]].astype(float)
    df_in['dz'] = trajectories_parts[velocity_cols[2]].astype(float)
    df_in = df_in[['particle','frame','dx','dy','dz']]
    

    # Delegate to the canonical fast implementation to ensure identical output
    return temporal_autocorr_all_fast(df_in, max_lag=max_lag, compute_S=compute_S)

def bead_volume_fraction(diameter_um, concentration_pM):
    """
    Compute v/v percentage of spherical beads in water.

    Parameters
    ----------
    diameter_um : float
        Bead diameter in micrometers (µm)
    concentration_pM : float
        Molar concentration in picomolar (pM)

    Returns
    -------
    v_v_percent : float
        Volume/volume percentage (%)
    """

    NA = 6.022e23  # Avogadro's number (mol^-1)
    
    # Convert units
    diameter_m = diameter_um * 1e-6
    conc_mol_per_L = concentration_pM * 1e-12
    
    # Calculate number of beads per liter
    N = conc_mol_per_L * NA
    
    # Volume of one bead in liters (1 m³ = 1000 L)
    V_bead_L = (4/3) * math.pi * (diameter_m / 2)**3 * 1000
    
    # Total bead volume per liter
    total_vol_L = N * V_bead_L
    # Volume/volume percentage
    v_v_percent = total_vol_L * 100
    
    return v_v_percent

def animate_multichannel_zsum(
    channels,
    filename,
    headers=None,
    vidfps=10,
    infps=2,
    luts=None,
    alphas=None,
    timestamp=True,
    bitrate=20000,
    # scale bar
    scalebar_um=None,
    px_per_micron=20.0,
    scalebar_px=None,
    scalebar_height_px=6,
    scalebar_color='',
    scalebar_loc='lower right',
    scalebar_pad_px=10,
    scalebar_label=None,
    # contrast
    auto_contrast=True,
    contrast_mode='per_frame',  # {'per_frame','global'}
    clip_percentiles=(1, 99)
):
    """
    Animate a multi-channel movie from 3D/4D inputs by summing along Z for each channel,
    mapping channels to different LUTs, and compositing into RGB.

    channels:
        list/tuple of arrays, each of shape:
        - (T, Z, Y, X) -> summed over Z for each frame
        - (T, Y, X)    -> used as-is
        - (Z, Y, X)    -> Z-sum to a single frame
        - (Y, X)       -> single frame
    """
    plt.style.use('dark_background')
    # force black background regardless of styles
    mpl.rcParams['figure.facecolor'] = 'black'
    mpl.rcParams['axes.facecolor'] = 'black'
    mpl.rcParams['savefig.facecolor'] = 'black'

    # defaults
    if luts is None:
        luts = ['Blues', 'Reds', 'Greens', 'magma', 'viridis', 'cividis']
    if alphas is None:
        alphas = None  # will distribute evenly later
    if not scalebar_color:
        scalebar_color = get_darkest_color(luts[0])[:3]
    

    # prepare stacks as (T, Y, X) for each channel
    chans = []
    for arr in channels:
        a = np.asarray(arr)
        if a.ndim == 4:         # (T, Z, Y, X)
            chans.append(a.sum(axis=1))
        elif a.ndim == 3:
            # heuristic: small first dim -> treat as Z
            if a.shape[0] < 16 and a.shape[1] >= 32 and a.shape[2] >= 32:
                chans.append(a.sum(axis=0, keepdims=True))
            else:
                chans.append(a)  # (T, Y, X)
        elif a.ndim == 2:
            chans.append(a[None, ...])  # (1, Y, X)
        else:
            raise ValueError("Unsupported array ndim; expected 2D/3D/4D per channel")

    # unify T by repeating single-frame channels if needed
    T = max(c.shape[0] for c in chans)
    chans2 = []
    for c in chans:
        if c.shape[0] == T:
            chans2.append(c.astype(np.float32, copy=False))
        elif c.shape[0] == 1:
            chans2.append(np.repeat(c.astype(np.float32), T, axis=0))
        else:
            raise ValueError("All channels must have same T or be single-frame")
    chans = chans2

    # basic checks and alpha setup
    H, W = chans[0].shape[1:]
    for c in chans:
        if c.shape[1:] != (H, W):
            raise ValueError("All channels must share same (Y, X) after Z-sum")
    C = len(chans)
    if alphas is None:
        alphas = [1.0 / max(C, 1) for _ in range(C)]
    if len(luts) < C:
        raise ValueError("Not enough LUTs provided for number of channels")

    # precompute global vmin/vmax per channel if requested
    def _pctl_limits(data, lo, hi):
        vmin, vmax = np.nanpercentile(data, [lo, hi])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return None, None
        return float(vmin), float(vmax)

    global_mm = [(None, None)] * C
    if auto_contrast and contrast_mode == 'global':
        for i in range(C):
            global_mm[i] = _pctl_limits(chans[i], *clip_percentiles)

    # scale bar helpers
    def _sb_len_px():
        if scalebar_px is not None:
            return int(max(1, round(float(scalebar_px))))
        if scalebar_um is not None and px_per_micron is not None:
            return int(max(1, round(float(scalebar_um) * float(px_per_micron))))
        return None

    def _add_scalebar(ax):
        Lpx = _sb_len_px()
        if Lpx is None:
            return
        hpx = int(max(1, round(scalebar_height_px)))
        pad = int(max(0, round(scalebar_pad_px)))
        loc = (scalebar_loc or 'lower right').lower()
        if loc in ('lower right', 'lr'):
            x0, y0 = W - pad - Lpx, H - pad - hpx
            ha, va = 'right', 'bottom'
            tx, ty = x0 + Lpx // 2, y0 - (hpx*1.5 + 12)
        elif loc in ('lower left', 'll'):
            x0, y0 = pad, H - pad - hpx
            ha, va = 'left', 'bottom'
            tx, ty = x0 + Lpx // 2, y0 - (hpx*1.5 + 12)
        elif loc in ('upper right', 'ur'):
            x0, y0 = W - pad - Lpx, pad
            ha, va = 'right', 'top'
            tx, ty = x0 + Lpx // 2, y0 + hpx + (hpx*1.5 + 12)
        else:  # upper left
            x0, y0 = pad, pad
            ha, va = 'left', 'top'
            tx, ty = x0 + Lpx // 2, y0 + hpx + (hpx*1.5 + 12)
        rect = Rectangle((x0, y0), Lpx, hpx, transform=ax.transData,
                            facecolor=scalebar_color, edgecolor='none', zorder=5)
        ax.add_patch(rect)
        if scalebar_label or scalebar_um is not None:
            label = scalebar_label if scalebar_label is not None else f"{float(scalebar_um):g} µm"
            ax.text(tx, ty, label, color=scalebar_color, ha='center', va=va, fontsize=12, zorder=6)

    # prepare writers and figure
    writer = FFMpegWriter(fps=vidfps, bitrate=bitrate)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_axis_off()

    # build initial composite for shape
    def _composite_frame(t, vmm):
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(C):
            im = chans[i][t]
            if auto_contrast:
                if contrast_mode == 'per_frame':
                    vmin, vmax = _pctl_limits(im, *clip_percentiles)
                else:
                    vmin, vmax = vmm[i]
                if vmin is not None and vmax is not None and vmax > vmin:
                    im = np.clip((im - vmin) / (vmax - vmin), 0, 1)
                else:
                    im = np.zeros_like(im, dtype=np.float32)
            else:
                im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im) + 1e-12)
            cmap = get_cmap(luts[i])
            col = cmap(im)[..., :3] * float(alphas[i])
            rgb += col.astype(np.float32)
        return np.clip(rgb, 0, 1)

    init_rgb = _composite_frame(0, global_mm)
    im_artist = ax.imshow(init_rgb, interpolation='nearest', origin='lower')

    # title
    if headers:
        if isinstance(headers, (list, tuple)):
            ax.set_title(" + ".join(map(str, headers)))
        else:
            ax.set_title(str(headers))

    # add scale bar once (in data coords)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    _add_scalebar(ax)

    # timestamp
    ts_text = None
    if timestamp:
        ts_text = ax.text(W - scalebar_pad_px, scalebar_pad_px, "",
                            color=scalebar_color, ha='right', va='bottom',
                            fontsize=12, zorder=6)

    # render
    with writer.saving(fig, filename, dpi=150):
        for t in tqdm(range(T), desc="Rendering"):
            frame_rgb = _composite_frame(t, global_mm)
            im_artist.set_data(frame_rgb)
            if timestamp and ts_text is not None:
                ts_text.set_text(f"t = {t / max(infps, 1e-9):.1f} s")
            writer.grab_frame()

    plt.close(fig)
    print(f"Animation saved to {filename}")

def mass_mass_corr_per_frame(
    imgs,
    px_per_micron,
    px_per_micron_z=None,
    nbins=120,
    subtract_mean=False,
    normalize=None  # None | "c0" (divide by c(0)) | "mean2" (divide by <rho>^2)
):
    """
    Compute the (spatially averaged) mass-mass correlation c(r) for each frame:
        c(r) = ∫ ρ(r') ρ(r'+r) dr'
    implemented via FFT autocorrelation and radial averaging.

    Args:
        imgs: array with shape [T, Z, Y, X] (3D) or [T, Y, X] (2D).
        px_per_micron: XY pixel density [px/µm].
        px_per_micron_z: Z pixel density [px/µm] (required for 3D).
        nbins: number of radial bins.
        subtract_mean: if True, use δρ = ρ - <ρ>.
        normalize:
          - None: raw c(r) average (as defined).
          - "c0": divide each profile by its r=0 value.
          - "mean2": divide by <ρ>^2 (uses per-frame mean).

    Returns:
        r_centers_um: (nbins,) radii in µm.
        corr_profiles: (T, nbins) radial c(r) per frame (NaN where empty bin).
    """
    imgs = np.asarray(imgs)
    # allow single 2D image (Y,X) or 2D movie (T,Y,X) or 3D movie (T,Z,Y,X)
    if imgs.ndim == 2:
        # promote single 2D image to a length-1 movie
        imgs = imgs[np.newaxis, ...]
    assert imgs.ndim in (3, 4), "imgs must be [Y,X] or [T,Y,X] or [T,Z,Y,X]"

    is_3d = imgs.ndim == 4
    T = imgs.shape[0]
    if is_3d and px_per_micron_z is None:
        px_per_micron_z = px_per_micron  # fallback if not provided

    # voxel sizes in µm
    dxy = 1.0 / float(px_per_micron)
    if is_3d:
        dz = 1.0 / float(px_per_micron_z)

    # Prepare distance grid and binning (depends only on spatial shape)
    spatial_shape = imgs.shape[1:]
    if is_3d:
        Z, Y, X = spatial_shape
        zz = (np.arange(Z) - Z // 2) * dz
        yy = (np.arange(Y) - Y // 2) * dxy
        xx = (np.arange(X) - X // 2) * dxy
        Zg, Yg, Xg = np.meshgrid(zz, yy, xx, indexing='ij')
        r_um = np.sqrt(Zg**2 + Yg**2 + Xg**2)
        box_min_half = 0.5 * min(Z * dz, Y * dxy, X * dxy)
    else:
        Y, X = spatial_shape
        yy = (np.arange(Y) - Y // 2) * dxy
        xx = (np.arange(X) - X // 2) * dxy
        Yg, Xg = np.meshgrid(yy, xx, indexing='ij')
        r_um = np.sqrt(Yg**2 + Xg**2)
        box_min_half = 0.5 * min(Y * dxy, X * dxy)

    # radial bins up to the smallest half-box (avoid wrap-around artifacts)
    edges = np.linspace(0.0, box_min_half, nbins + 1)
    r_centers_um = 0.5 * (edges[:-1] + edges[1:])

    # Precompute bin indices and counts once
    r_flat = r_um.ravel()
    bin_idx = np.digitize(r_flat, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[valid]
    ones = np.ones_like(bin_idx, dtype=np.float64)
    bin_counts = np.bincount(bin_idx, weights=ones, minlength=nbins)
    bin_counts[bin_counts == 0] = np.nan  # avoid divide-by-zero later

    corr_profiles = np.empty((T, nbins), dtype=np.float64)
    for t in range(T):
        rho = imgs[t].astype(np.float32, copy=False)
        if subtract_mean:
            rho = rho - float(rho.mean())

        # c = FFT autocorrelation; bring zero-lag to center for radial averaging
        F = fftn(rho)
        c = np.fft.ifftn(np.abs(F)**2).real
        c = fftshift(c)

        w = c.ravel()[valid].astype(np.float64, copy=False)
        bin_sums = np.bincount(bin_idx, weights=w, minlength=nbins)
        prof = bin_sums / bin_counts  # spatial average over shell

        if normalize == "c0":
            zlag = prof[0]
            if np.isfinite(zlag) and zlag != 0:
                prof = prof / zlag
        elif normalize == "mean2":
            mu = float(imgs[t].mean())
            denom = mu * mu
            if denom != 0:
                prof = prof / denom

        corr_profiles[t] = prof

    return r_centers_um, corr_profiles

def to_tzcyx(images, z_lower=0, z_upper=None, order='tzcyx'):
    """
    Convert images to shape (t, z, c, y, x).

    Supports:
      - (t, z, c, y, x)          # already canonical
      - (t, z, y, x)             # 3D movie, single channel
      - (t, c, y, x)             # 2D movie, multi-channel
      - (t, y, x)                # 2D movie, single channel

    z_lower / z_upper are only applied if there is a real z-axis.
    """
    im = np.asarray(images)
    ndim = im.ndim

    if ndim == 5:
        # assume (t, z, c, y, x)
        t, z, c, y, x = im.shape
        # apply z cropping
        if z_upper is not None:
            im = im[:, z_lower:z_upper, ...]
        else:
            im = im[:, z_lower:, ...]
        t, z, c, y, x = im.shape

    elif ndim == 4:
        t, a, y, x = im.shape
        if order == 'tcyx':
            # (t, c, y, x) → add z=1
            c = a
            z = 1
            im = im[:, np.newaxis, ...]           # (t, 1, c, y, x)
        else:
            # (t, z, y, x) → add c=1
            z = a
            # apply z cropping before adding channel axis
            if z_upper is not None:
                im = im[:, z_lower:z_upper, :, :] # (t, z', y, x)
            else:
                im = im[:, z_lower:, :, :]
            t, z, y, x = im.shape
            c = 1
            im = im[:, :, np.newaxis, :, :]       # (t, z, 1, y, x)

    elif ndim == 3:
        # (t, y, x) → add z=1, c=1
        t, y, x = im.shape
        z = 1
        c = 1
        im = im[:, np.newaxis, np.newaxis, :, :]  # (t, 1, 1, y, x)

    else:
        raise ValueError(f"Unsupported ndim={ndim} for images")

    return im  # now (t, z, c, y, x)

def imshow_contrast(image, meanfaktor = 2, width=10, *args, **kwargs):
    """
    Show image with contrast adjusted around mean.

    Parameters
    ----------
    image : ndarray
        Input image.
    meanfaktor : float
        Factor to multiply the mean for contrast limits.
    *args, **kwargs :
        Additional arguments passed to plt.imshow().
    """
    img = np.asarray(image)
    mu = float(np.mean(img))
    vmin = mu / float(meanfaktor)
    vmax = mu * float(meanfaktor)
    fig, ax = plt.subplots(figsize=(width, image.shape[-1] / image.shape[-2] * width), dpi=300)
    ax.imshow(img, vmin=vmin, vmax=vmax, *args, **kwargs)
    ax.axis('off')
    fig.tight_layout()
    plt.show()

def _cumlen(P):
        if P.shape[0] < 2:
            return np.array([0.0], dtype=np.float32)
        segs = np.diff(P, axis=0)
        d = np.linalg.norm(segs, axis=1)
        return np.concatenate([[0.0], np.cumsum(d).astype(np.float32)])

def branch_geometry(P_zyx: np.ndarray, orientation="pca"):
    """Compute length, centroid, and direction of a branch given its coordinates."""
    if P_zyx.shape[0] < 2 or np.any(np.isnan(P_zyx)):
        return np.nan, np.array([np.nan, np.nan, np.nan], dtype=np.float32), np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    S = _cumlen(P_zyx)
    Ltot = float(S[-1])
    idx = np.where(S > Ltot/2)[0]
    centroid = ((P_zyx[idx]-P_zyx[idx-1])*((S[idx]-Ltot/2)/(S[idx]-S[idx-1]))[0] + P_zyx[idx-1])[0]

    if orientation == "pca" and P_zyx.shape[0] >= 3:
        Xc = P_zyx - centroid
        cov = Xc.T @ Xc
        w, V = np.linalg.eigh(cov)
        d_zyx = V[:, -1]
    else:
        d_zyx = P_zyx[-1] - P_zyx[0]
    nrm = np.linalg.norm(d_zyx)
    if nrm > 0 and np.isfinite(nrm):
        d_zyx = d_zyx / nrm
    else:
        d_zyx = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    return Ltot, centroid, d_zyx

def extract_branches_simple(
    skeleton,
    px_per_micron_xy: float,
    px_per_micron_z: float,
    orientation: str = "pca",
    images=None,  # (T, Z, Y, X) intensity stack
):
    """
    Simpler branch-level extraction:
    one row per branch per frame (no subdivision into parts).
    """

    if images is not None and images.shape[0] != skeleton.shape[0]:
        raise ValueError("images and skeleton must have same T along axis 0")

    # scale factors (pixels -> µm) matching (z,y,x)
    sx = sy = 1.0 / float(px_per_micron_xy)
    sz = 1.0 / float(px_per_micron_z)
    scale_zyx = np.array([sz, sy, sx], dtype=np.float32)

    um_flag = (px_per_micron_xy != 1.0) and (px_per_micron_z != 1.0)
    rows = []
    T = int(skeleton.shape[0])

    for t in tqdm(range(T), desc="branch-level segmentation"):
        skel_t = skeleton[t]
        if isinstance(skeleton, da.Array):
            skel_t = skel_t.compute()

        img_t = None
        if images is not None:
            img_t = images[t]
            if isinstance(images, da.Array):
                img_t = img_t.compute()

        g = csr.Skeleton(skel_t)
        n_paths = int(g.n_paths)

        for local_id in range(n_paths):
            co_zyx = g.path_coordinates(local_id)
            if co_zyx.size == 0:
                continue

            # path in µm
            P_zyx = np.asarray(co_zyx, dtype=np.float32) * scale_zyx

            # total length + centroid + orientation
            Ltot, centroid_zyx, d_zyx = branch_geometry(P_zyx, orientation=orientation)
            if Ltot <= 0 or not np.isfinite(Ltot):
                continue

            cz, cy, cx = centroid_zyx  # still (z,y,x) order
            d_xyz = d_zyx[[2, 1, 0]]

            # curvature over full path
            kappa = _estimate_curvature_zyx(P_zyx)
            curv_mean = float(np.nanmean(kappa)) if np.isfinite(kappa).any() else np.nan

            # optional intensity along full path
            I_mean = np.nan
            I_sum = np.nan
            if img_t is not None:
                Z, Y, X = img_t.shape[0], img_t.shape[1], img_t.shape[2]
                # convert µm back to voxel indices
                vz = np.rint(P_zyx[:, 0] / sz).astype(int)
                vy = np.rint(P_zyx[:, 1] / sy).astype(int)
                vx = np.rint(P_zyx[:, 2] / sx).astype(int)
                m = (vz >= 0) & (vz < Z) & (vy >= 0) & (vy < Y) & (vx >= 0) & (vx < X)
                if np.any(m):
                    idx_lin = vz[m] * (Y * X) + vy[m] * X + vx[m]
                    uidx = np.unique(idx_lin)
                    vz_u = uidx // (Y * X)
                    vy_u = (uidx % (Y * X)) // X
                    vx_u = (uidx % (Y * X)) % X
                    vals = img_t[vz_u, vy_u, vx_u]
                    if vals.size:
                        I_mean = float(vals.mean())
                        I_sum = float(vals.sum())

            rows.append({
                "frame": int(t),
                "branch_id": int(local_id),
                "part_index": 0,  # kept for compatibility, but always 0
                "x": float(cx),
                "y": float(cy),
                "z": float(cz),
                "xa": float(d_xyz[0]),
                "ya": float(d_xyz[1]),
                "za": float(d_xyz[2]),
                "length": float(Ltot),
                "u_center": 0.5,  # whole branch, so center at 0.5 by definition
                "curv_mean": curv_mean,
                "I_mean": I_mean,
                "I_sum": I_sum,
                "um": bool(um_flag),
                "coords": P_zyx[:, [2, 1, 0]],  # store (x,y,z) in µm
            })

    return pd.DataFrame(rows, columns=[
        "frame","branch_id","part_index","x","y","z","xa","ya","za",
        "length","u_center","curv_mean","I_mean","I_sum","um","coords"
    ])

def extract_branches_simple(
    skeleton,
    px_per_micron_xy: float,
    px_per_micron_z: float,
    orientation: str = "pca",
    images=None,  # (T, Z, Y, X) intensity stack
):
    """
    Daskified version: returns a dask.dataframe.DataFrame where each row is one branch.
    """


    if images is not None and images.shape[0] != skeleton.shape[0]:
        raise ValueError("images and skeleton must have same T along axis 0")

    # scale factors (pixels -> µm) matching (z,y,x)
    sx = sy = 1.0 / float(px_per_micron_xy)
    sz = 1.0 / float(px_per_micron_z)
    scale_zyx = np.array([sz, sy, sx], dtype=np.float32)

    um_flag = (px_per_micron_xy != 1.0) and (px_per_micron_z != 1.0)
    T = int(skeleton.shape[0])

    cols = [
        "frame","branch_id","part_index","x","y","z","xa","ya","za",
        "length","u_center","curv_mean","I_mean","I_sum","um","coords"
    ]

    def _process_frame(t):

        rows = []
        skel_t = skeleton[t]
        if isinstance(skeleton, da.Array):
            skel_t = skel_t.compute()

        img_t = None
        if images is not None:
            img_t = images[t]
            if isinstance(images, da.Array):
                img_t = img_t.compute()

        g = csr.Skeleton(skel_t)
        n_paths = int(g.n_paths)

        for local_id in range(n_paths):
            co_zyx = g.path_coordinates(local_id)
            if co_zyx.size == 0:
                continue

            P_zyx = _np.asarray(co_zyx, dtype=_np.float32) * scale_zyx

            Ltot, centroid_zyx, d_zyx = branch_geometry(P_zyx, orientation=orientation)
            if Ltot <= 0 or not _np.isfinite(Ltot):
                continue

            cz, cy, cx = centroid_zyx  # (z,y,x)
            d_xyz = d_zyx[[2, 1, 0]]

            kappa = _estimate_curvature_zyx(P_zyx)
            curv_mean = float(_np.nanmean(kappa)) if _np.isfinite(kappa).any() else _np.nan

            I_mean = _np.nan
            I_sum = _np.nan
            if img_t is not None:
                Z, Y, X = img_t.shape[0], img_t.shape[1], img_t.shape[2]
                vz = _np.rint(P_zyx[:, 0] / sz).astype(int)
                vy = _np.rint(P_zyx[:, 1] / sy).astype(int)
                vx = _np.rint(P_zyx[:, 2] / sx).astype(int)
                m = (vz >= 0) & (vz < Z) & (vy >= 0) & (vy < Y) & (vx >= 0) & (vx < X)
                if _np.any(m):
                    idx_lin = vz[m] * (Y * X) + vy[m] * X + vx[m]
                    uidx = _np.unique(idx_lin)
                    vz_u = uidx // (Y * X)
                    vy_u = (uidx % (Y * X)) // X
                    vx_u = (uidx % (Y * X)) % X
                    vals = img_t[vz_u, vy_u, vx_u]
                    if vals.size:
                        I_mean = float(vals.mean())
                        I_sum = float(vals.sum())

            rows.append({
                "frame": int(t),
                "branch_id": int(local_id),
                "part_index": 0,
                "x": float(cx),
                "y": float(cy),
                "z": float(cz),
                "xa": float(d_xyz[0]),
                "ya": float(d_xyz[1]),
                "za": float(d_xyz[2]),
                "length": float(Ltot),
                "u_center": 0.5,
                "curv_mean": curv_mean,
                "I_mean": I_mean,
                "I_sum": I_sum,
                "um": bool(um_flag),
                "coords": P_zyx[:, [2, 1, 0]],
            })

        return _pd.DataFrame(rows, columns=cols)

    # build delayed per-frame DataFrames
    delayed_parts = [delayed(_process_frame)(t) for t in range(T)]

    # meta (empty frame with correct dtypes) for from_delayed
    meta = pd.DataFrame({
        "frame": pd.Series(dtype="int64"),
        "branch_id": pd.Series(dtype="int64"),
        "part_index": pd.Series(dtype="int64"),
        "x": pd.Series(dtype="float64"),
        "y": pd.Series(dtype="float64"),
        "z": pd.Series(dtype="float64"),
        "xa": pd.Series(dtype="float64"),
        "ya": pd.Series(dtype="float64"),
        "za": pd.Series(dtype="float64"),
        "length": pd.Series(dtype="float64"),
        "u_center": pd.Series(dtype="float64"),
        "curv_mean": pd.Series(dtype="float64"),
        "I_mean": pd.Series(dtype="float64"),
        "I_sum": pd.Series(dtype="float64"),
        "um": pd.Series(dtype="bool"),
        "coords": pd.Series(dtype="object"),
    })

    ddf = dd.from_delayed(delayed_parts, meta=meta)
    return ddf

def add_velocities_and_drift(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Add vx, vy, vz, v and drift-corrected vxd, vyd, vzd, vd
    to a trackpy trajectories DataFrame.

    Parameters
    ----------
    df : DataFrame with columns ['particle', 'frame', 'x', 'y', 'z'].
    fps : frames per second.
    """
    df = df.sort_values(['particle', 'frame']).copy()

    # raw velocity components (µm/s)
    for coord in tqdm(['x', 'y', 'z'], desc="velocity components"):
        dpos = df.groupby('particle')[coord].diff()  # Δx per frame
        df[f'v{coord}'] = dpos * float(fps)          # Δx * fps = velocity

    # speed magnitude (µm/s)
    df['v'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

    # drift per frame = mean velocity over all particles in that frame
    drift = (
        df.groupby('frame')[['vx', 'vy', 'vz']]
          .mean()
          .rename(columns={
              'vx': 'vx_drift',
              'vy': 'vy_drift',
              'vz': 'vz_drift'
          })
    )

    # attach drift back to rows
    df = df.merge(drift, on='frame', how='left')

    # drift-corrected components
    df['vxd'] = df['vx'] - df['vx_drift']
    df['vyd'] = df['vy'] - df['vy_drift']
    df['vzd'] = df['vz'] - df['vz_drift']

    # drift-corrected speed magnitude
    df['vd'] = np.sqrt(df['vxd']**2 + df['vyd']**2 + df['vzd']**2)

    return df

def scalebar(
    img_shape,
    scalebar_px=None,
    scalebar_um=None,
    px_per_micron=None,
    scalebar_label=None,
    ax=None,
    fsize=12,
    scalebar_height_px=None,
    scalebar_pad_px=10,
    scalebar_loc='lower right',
    map='Greys',
):
    """
    Draw a simple scale bar onto `ax` for an image of shape `img_shape` (H, W, ...).
    """

    def _compute_sb_len_px():
        if scalebar_px is not None:
            return int(max(1, round(scalebar_px)))
        if scalebar_um is not None and px_per_micron is not None:
            return int(max(1, round(float(scalebar_um) * float(px_per_micron))))
        return None

    def _add_scalebar(ax_in):
        Lpx = _compute_sb_len_px()
        if Lpx is None:
            return None, None

        H, W = int(img_shape[0]), int(img_shape[1])
        if Lpx > W // 2:
            print('scalebar too long for image width')

        # convert pixel sizes to axes fraction
        w_frac = Lpx / max(W, 1)

        # use outer scalebar_height_px parameter if provided
        height_px = scalebar_height_px
        if height_px is None:
            height_px = max(1, int(H / 50))
        h_frac = float(height_px) / max(H, 1)

        pad_x = float(scalebar_pad_px) / max(W, 1)
        pad_y = float(scalebar_pad_px) / max(H, 1)

        loc = (scalebar_loc or 'lower right').lower()
        if loc in ('lower right', 'lr'):
            x0, y0 = 1.0 - pad_x - w_frac, pad_y
            valign_text = 'bottom'
            y_text = y0 + h_frac + 0.01
        elif loc in ('lower left', 'll'):
            x0, y0 = pad_x, pad_y
            valign_text = 'bottom'
            y_text = y0 + h_frac + 0.01
        elif loc in ('upper right', 'ur'):
            x0, y0 = 1.0 - pad_x - w_frac, 1.0 - pad_y - h_frac
            valign_text = 'top'
            y_text = y0 - 0.01
        elif loc in ('upper left', 'ul'):
            x0, y0 = pad_x, 1.0 - pad_y - h_frac
            valign_text = 'top'
            y_text = y0 - 0.01
        else:
            x0, y0 = 1.0 - pad_x - w_frac, pad_y
            valign_text = 'bottom'
            y_text = y0 + h_frac + 0.01

        rect = Rectangle(
            (x0, y0),
            w_frac,
            h_frac,
            transform=ax_in.transAxes,
            facecolor=get_darkest_color(map)[:3],
            edgecolor='none',
            linewidth=0,
            zorder=5,
        )
        ax_in.add_patch(rect)

        label = scalebar_label
        if label is None and scalebar_um is not None:
            val = float(scalebar_um)
            label = f"{val:g} µm"

        txt_artist = None
        if label:
            txt_artist = ax_in.text(
                x0 + 0.5 * w_frac,
                y_text,
                label,
                transform=ax_in.transAxes,
                color=get_darkest_color(map)[:3],
                ha='center',
                va=valign_text,
                fontsize=fsize,
                zorder=6,
            )
        return rect, txt_artist

    if ax is None:
        ax = plt.gca()

    return _add_scalebar(ax)

def load_cached_objects(cache_folder, namespace=None):
    if namespace is None:
        namespace = globals()

    loaded = {}

    for fname in os.listdir(cache_folder):
        path = os.path.join(cache_folder, fname)
        if not os.path.isfile(path):
            continue
        base, _ = os.path.splitext(fname)
        var_name = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in base)
        if not var_name.isidentifier():
            var_name = f'obj_{var_name}'

        try:
            obj = load_object(path)
        except Exception as e:
            print(f"Could not load {fname}: {e}")
            continue

        namespace[var_name] = obj
    return

def threshold_beads(img):
    if isinstance(img, da.Array):
        return img.mean().compute()*10
    return img.mean()*10


def plot_bead_detection_preview(
    raw_vol: np.ndarray,
    binary_mask: np.ndarray,
    kept_mask: np.ndarray,
    *,
    figsize: tuple[float, float] = (15, 5),
    cmap: str = 'gray',
    show: bool = True,
):
    """Plot raw/binary/kept z-max panels for bead detection parameter tuning."""
    raw_zmax = np.asarray(raw_vol).max(axis=0)
    bin_zmax = np.asarray(binary_mask).max(axis=0).astype(np.uint8)
    keep_zmax = np.asarray(kept_mask).max(axis=0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(raw_zmax, cmap=cmap)
    axes[0].set_title('raw z-max')
    axes[1].imshow(bin_zmax, cmap=cmap)
    axes[1].set_title('binary z-max')
    axes[2].imshow(keep_zmax, cmap=cmap)
    axes[2].set_title('kept beads z-max')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _fps_from_calibration_handle(handle: dict | None) -> float | None:
    """Return fps from dataset handle calibration, tolerant to ms-vs-s storage."""
    if not isinstance(handle, dict):
        return None
    calib = handle.get('meta', {}).get('calibration', {})
    dt_s = calib.get('dt_s', None)
    if dt_s in (None, '', 0):
        dt_ms = calib.get('dt_ms', None)
        dt_s = (float(dt_ms) / 1000.0) if dt_ms not in (None, '', 0) else None
    else:
        dt_s = float(dt_s)
        if dt_s > 100:
            dt_s = dt_s / 1000.0
    return (1.0 / dt_s) if (dt_s and dt_s > 0) else None


def plot_mean_bead_speed_over_time(
    tracks_vel_df: pd.DataFrame,
    *,
    fps: float | None = None,
    handle: dict | None = None,
    show_std: bool = True,
    ax=None,
    figsize: tuple[float, float] = (7, 4),
    dpi: int = 150,
    show: bool = True,
):
    """Plot mean bead speed per frame against time (s) when fps is available."""
    if tracks_vel_df is None or len(tracks_vel_df) == 0:
        raise ValueError('tracks_vel_df is empty')
    if 'frame' not in tracks_vel_df.columns or 'speed_um_s' not in tracks_vel_df.columns:
        raise ValueError("tracks_vel_df must contain 'frame' and 'speed_um_s'")

    speed_stats = tracks_vel_df.groupby('frame', sort=True)['speed_um_s'].agg(['mean', 'std', 'count'])
    frames = speed_stats.index.to_numpy(dtype=int)
    mean_speed = speed_stats['mean'].to_numpy(dtype=float)
    std_speed = speed_stats['std'].to_numpy(dtype=float)

    fps_use = _fps_from_calibration_handle(handle)
    if fps_use is None and fps is not None and float(fps) > 0:
        fps_use = float(fps)

    if fps_use is not None:
        x = frames / fps_use
        xlabel = 'time (s)'
    else:
        x = frames
        xlabel = 'frame'

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x, mean_speed, lw=1.5, label='mean')
    if show_std and x.size:
        y1 = mean_speed - std_speed
        y2 = mean_speed + std_speed
        valid_std = np.isfinite(y1) & np.isfinite(y2)
        if np.any(valid_std):
            ax.fill_between(
                x[valid_std],
                y1[valid_std],
                y2[valid_std],
                alpha=0.25,
                linewidth=0,
                label='mean ± std',
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('mean speed (µm/s)')
    ax.set_title('Mean bead speed over time')
    if x.size:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.grid(True, alpha=0.3)
    if show_std:
        ax.legend(loc='best', frameon=False)

    if show and created_fig:
        plt.show()
    return fig, ax, speed_stats


def velocity_velocity_correlation_3d(
    tracks_vel_df: pd.DataFrame,
    *,
    max_lag: int | None = None,
    fps: float | None = None,
    min_pairs: int = 5,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute 3D velocity-velocity time correlation Cvv(tau).

    Cvv(lag) = < v(t) . v(t+lag) > over all particles and valid frame pairs.
    """
    need = {'particle', 'frame', 'vx_um_s', 'vy_um_s', 'vz_um_s'}
    missing = need.difference(tracks_vel_df.columns)
    if missing:
        raise ValueError(f'tracks_vel_df missing columns: {sorted(missing)}')

    df = tracks_vel_df[list(need)].copy()
    df = df.dropna(subset=['particle', 'frame', 'vx_um_s', 'vy_um_s', 'vz_um_s'])
    if len(df) == 0:
        raise ValueError('No valid velocity rows after NaN filtering')

    df['particle'] = df['particle'].astype(int)
    df['frame'] = df['frame'].astype(int)

    grouped = []
    n_obs_max = 0
    for _, g in df.groupby('particle', sort=False):
        g = g.sort_values('frame', kind='mergesort')
        arr = g[['frame', 'vx_um_s', 'vy_um_s', 'vz_um_s']].to_numpy(dtype=float)
        if arr.shape[0] >= 2:
            grouped.append(arr)
            n_obs_max = max(n_obs_max, int(arr.shape[0]))

    if not grouped:
        raise ValueError('Need at least one particle track with >=2 velocity samples')

    if max_lag is None:
        max_lag = max(1, n_obs_max - 1)
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError('max_lag must be >= 0')

    sum_dot = np.zeros(max_lag + 1, dtype=float)
    n_pairs = np.zeros(max_lag + 1, dtype=int)

    for arr in grouped:
        frames = arr[:, 0].astype(int)
        vel = arr[:, 1:4]
        for lag in range(max_lag + 1):
            target = frames + lag
            j = np.searchsorted(frames, target)
            in_bounds = (j < len(frames))
            ok = np.zeros_like(in_bounds, dtype=bool)
            if np.any(in_bounds):
                ii = np.where(in_bounds)[0]
                ok[ii] = (frames[j[ii]] == target[ii])
            if not np.any(ok):
                continue
            v0 = vel[ok]
            v1 = vel[j[ok]]
            dots = np.einsum('ij,ij->i', v0, v1)
            sum_dot[lag] += float(np.sum(dots))
            n_pairs[lag] += int(dots.size)

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = sum_dot / n_pairs
    corr[n_pairs < int(min_pairs)] = np.nan

    if normalize:
        c0 = corr[0] if len(corr) else np.nan
        if np.isfinite(c0) and c0 != 0:
            corr_norm = corr / c0
        else:
            corr_norm = np.full_like(corr, np.nan, dtype=float)
    else:
        corr_norm = np.full_like(corr, np.nan, dtype=float)

    lag = np.arange(max_lag + 1, dtype=int)
    if fps is not None and float(fps) > 0:
        tau_s = lag / float(fps)
    else:
        tau_s = lag.astype(float)

    out = pd.DataFrame({
        'lag': lag,
        'tau_s': tau_s,
        'Cvv_um2_s2': corr,
        'Cvv_norm': corr_norm,
        'n_pairs': n_pairs,
    })
    return out


def vector_correlation_3d(
    tracks_vel_df: pd.DataFrame,
    *,
    max_lag: int | None = None,
    fps: float | None = None,
    min_pairs: int = 5,
    normalize: bool = False,
) -> pd.DataFrame:
    """Compute 3D vector orientation correlation using
    S(lag) = 3/2 * ( <(u(t)·u(t+lag))^2> - 1/3 ),
    where u is the unit velocity direction.
    """
    need = {'particle', 'frame', 'vx_um_s', 'vy_um_s', 'vz_um_s'}
    missing = need.difference(tracks_vel_df.columns)
    if missing:
        raise ValueError(f'tracks_vel_df missing columns: {sorted(missing)}')

    df = tracks_vel_df[list(need)].copy()
    df = df.dropna(subset=['particle', 'frame', 'vx_um_s', 'vy_um_s', 'vz_um_s'])
    if len(df) == 0:
        raise ValueError('No valid velocity rows after NaN filtering')

    df['particle'] = df['particle'].astype(int)
    df['frame'] = df['frame'].astype(int)

    grouped = []
    n_obs_max = 0
    for _, g in df.groupby('particle', sort=False):
        g = g.sort_values('frame', kind='mergesort')
        arr = g[['frame', 'vx_um_s', 'vy_um_s', 'vz_um_s']].to_numpy(dtype=float)
        if arr.shape[0] >= 2:
            grouped.append(arr)
            n_obs_max = max(n_obs_max, int(arr.shape[0]))

    if not grouped:
        raise ValueError('Need at least one particle track with >=2 velocity samples')

    if max_lag is None:
        max_lag = max(1, n_obs_max - 1)
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError('max_lag must be >= 0')

    sum_S = np.zeros(max_lag + 1, dtype=float)
    n_pairs = np.zeros(max_lag + 1, dtype=int)

    for arr in grouped:
        frames = arr[:, 0].astype(int)
        vel = arr[:, 1:4]
        norms = np.linalg.norm(vel, axis=1)
        for lag in range(max_lag + 1):
            target = frames + lag
            j = np.searchsorted(frames, target)
            in_bounds = (j < len(frames))
            ok = np.zeros_like(in_bounds, dtype=bool)
            if np.any(in_bounds):
                ii = np.where(in_bounds)[0]
                ok[ii] = (frames[j[ii]] == target[ii])
            if not np.any(ok):
                continue

            i_idx = np.where(ok)[0]
            j_idx = j[ok]

            n0 = norms[i_idx]
            n1 = norms[j_idx]
            good = (n0 > 0) & (n1 > 0) & np.isfinite(n0) & np.isfinite(n1)
            if not np.any(good):
                continue

            v0 = vel[i_idx[good]]
            v1 = vel[j_idx[good]]
            dots = np.einsum('ij,ij->i', v0, v1) / (n0[good] * n1[good])
            S_vals = 1.5 * (dots**2 - (1.0 / 3.0))

            sum_S[lag] += float(np.sum(S_vals))
            n_pairs[lag] += int(S_vals.size)

    with np.errstate(invalid='ignore', divide='ignore'):
        S_corr = sum_S / n_pairs
    S_corr[n_pairs < int(min_pairs)] = np.nan

    if normalize:
        s0 = S_corr[0] if len(S_corr) else np.nan
        if np.isfinite(s0) and s0 != 0:
            S_norm = S_corr / s0
        else:
            S_norm = np.full_like(S_corr, np.nan, dtype=float)
    else:
        S_norm = np.full_like(S_corr, np.nan, dtype=float)

    lag = np.arange(max_lag + 1, dtype=int)
    if fps is not None and float(fps) > 0:
        tau_s = lag / float(fps)
    else:
        tau_s = lag.astype(float)

    return pd.DataFrame({
        'lag': lag,
        'tau_s': tau_s,
        'S_vec': S_corr,
        'S_vec_norm': S_norm,
        'n_pairs': n_pairs,
    })


def _fit_exp_decay_positive_x(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int = 4,
    max_x: float | None = None,
) -> dict:
    """Fit y(x) = A * exp(-x/tau) on finite points with x > 0.

    Returns a dictionary with keys:
    success, A, tau, A_se, tau_se, y_fit, x_fit, n_fit, message.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if max_x is not None:
        valid &= (x <= float(max_x))

    xv = x[valid]
    yv = y[valid]
    if xv.size < int(min_points):
        return {
            'success': False,
            'A': np.nan,
            'tau': np.nan,
            'A_se': np.nan,
            'tau_se': np.nan,
            'x_fit': np.array([], dtype=float),
            'y_fit': np.array([], dtype=float),
            'n_fit': int(xv.size),
            'message': f'Not enough points for fit ({xv.size} < {int(min_points)})',
        }

    # Initial guesses: A from earliest positive-lag sample, tau from x-range.
    A0 = float(yv[0]) if np.isfinite(yv[0]) else float(np.nanmedian(yv))
    if not np.isfinite(A0):
        A0 = 1.0
    tau0 = float(max(np.nanmedian(xv), (np.nanmax(xv) - np.nanmin(xv)) / 3.0, 1e-12))

    try:
        popt, pcov = curve_fit(
            exp_decay,
            xv,
            yv,
            p0=[A0, tau0],
            bounds=([-np.inf, 1e-12], [np.inf, np.inf]),
            maxfev=20000,
        )
        A, tau = float(popt[0]), float(popt[1])
        if pcov is not None and np.ndim(pcov) == 2 and pcov.shape[0] >= 2 and pcov.shape[1] >= 2:
            A_var = float(pcov[0, 0])
            tau_var = float(pcov[1, 1])
            A_se = float(np.sqrt(A_var)) if np.isfinite(A_var) and A_var >= 0 else np.nan
            tau_se = float(np.sqrt(tau_var)) if np.isfinite(tau_var) and tau_var >= 0 else np.nan
        else:
            A_se = np.nan
            tau_se = np.nan
        x_fit = np.linspace(float(np.nanmin(xv)), float(np.nanmax(xv)), 200)
        y_fit = exp_decay(x_fit, A, tau)
        return {
            'success': True,
            'A': A,
            'tau': tau,
            'A_se': A_se,
            'tau_se': tau_se,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'n_fit': int(xv.size),
            'message': 'ok',
        }
    except Exception as exc:
        return {
            'success': False,
            'A': np.nan,
            'tau': np.nan,
            'A_se': np.nan,
            'tau_se': np.nan,
            'x_fit': np.array([], dtype=float),
            'y_fit': np.array([], dtype=float),
            'n_fit': int(xv.size),
            'message': f'Fit failed: {exc}',
        }


def plot_vector_correlation_3d(
    corr_df: pd.DataFrame,
    *,
    normalized: bool = False,
    fit_exp: bool = True,
    fit_max_tau_s: float | None = None,
    fit_min_points: int = 4,
    ax=None,
    figsize: tuple[float, float] = (7, 4),
    dpi: int = 150,
    show: bool = True,
    save_path: str | None = None,
    save_kwargs: dict | None = None,
):
    """Plot 3D vector orientation correlation S(lag) from vector_correlation_3d output."""
    y_col = 'S_vec_norm' if normalized else 'S_vec'
    if y_col not in corr_df.columns:
        raise ValueError(f'corr_df missing {y_col}')

    x = corr_df['tau_s'].to_numpy(dtype=float)
    y = corr_df[y_col].to_numpy(dtype=float)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x, y, marker='o', lw=1.5, label='data')
    ax.set_xlabel('lag time (s)')
    if normalized:
        ax.set_ylabel('S / S(0)')
    else:
        ax.set_ylabel(r'S = $\frac{3}{2}(\langle(\hat{u}\cdot\hat{u}_{\tau})^2\rangle - \frac{1}{3})$')
    ax.set_title('3D vector correlation')
    ax.grid(True, alpha=0.3)

    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        x_valid = x[valid]
        ax.set_xlim(float(x_valid[0]), float(x_valid[-1]))
    elif x.size:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))

    if normalized:
        ax.axhline(0.0, color='0.6', lw=1, ls='--')

    # Always auto-limit y-axis to finite plotted values with small padding.
    y_valid = y[np.isfinite(y)]
    if y_valid.size:
        ylo = float(np.nanmin(y_valid))
        yhi = float(np.nanmax(y_valid))
        if np.isfinite(ylo) and np.isfinite(yhi):
            if yhi > ylo:
                pad = 0.07 * (yhi - ylo)
                ax.set_ylim(ylo - pad, yhi + pad)
            else:
                pad = 0.1 * max(1.0, abs(yhi))
                ax.set_ylim(ylo - pad, yhi + pad)

    fit_result = {
        'success': False,
        'A': np.nan,
        'tau': np.nan,
        'A_se': np.nan,
        'tau_se': np.nan,
        'n_fit': 0,
        'message': 'fit disabled',
    }
    if fit_exp:
        fit_result = _fit_exp_decay_positive_x(
            x,
            y,
            min_points=int(fit_min_points),
            max_x=fit_max_tau_s,
        )
        if fit_result.get('success', False):
            tau = float(fit_result['tau'])
            tau_se = float(fit_result.get('tau_se', np.nan))
            if np.isfinite(tau_se):
                fit_label = f"exp fit (tau={tau:.3g}±{tau_se:.2g} s)"
            else:
                fit_label = f"exp fit (tau={tau:.3g} s)"
            ax.plot(
                fit_result['x_fit'],
                fit_result['y_fit'],
                ls='--',
                lw=1.5,
                label=fit_label,
            )
        else:
            print(f"[plot_vector_correlation_3d] {fit_result.get('message', 'fit failed')}")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best', frameon=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        kwargs = {'bbox_inches': 'tight', 'dpi': dpi}
        if save_kwargs:
            kwargs.update(save_kwargs)
        fig.savefig(save_path, **kwargs)

    if show and created_fig:
        plt.show()
    return fig, ax, fit_result


def spatial_vector_correlation_per_frame(
    tracks_vel_df: pd.DataFrame,
    *,
    r_max_um: float | None = None,
    nbins: int = 30,
    min_pairs: int = 5,
) -> pd.DataFrame:
    """Compute spatial vector orientation correlation S(r) for each frame.

    Uses
    S(r) = 3/2 * ( <(u_i · u_j)^2>_r - 1/3 )
    where u_i are unit velocity directions and pairs (i,j) are binned by
    spatial separation r within each frame.

    Returns a tidy dataframe with columns:
    frame, r_um, S_vec_spatial, n_pairs
    """
    need = {'frame', 'x_um', 'y_um', 'z_um', 'vx_um_s', 'vy_um_s', 'vz_um_s'}
    missing = need.difference(tracks_vel_df.columns)
    if missing:
        raise ValueError(f'tracks_vel_df missing columns: {sorted(missing)}')
    if int(nbins) <= 0:
        raise ValueError('nbins must be > 0')

    df = tracks_vel_df[list(need)].copy()
    df = df.dropna(subset=['frame', 'x_um', 'y_um', 'z_um', 'vx_um_s', 'vy_um_s', 'vz_um_s'])
    if len(df) == 0:
        raise ValueError('No valid rows after NaN filtering')

    df['frame'] = df['frame'].astype(int)

    # Use half global bbox diagonal as a conservative default radius.
    if r_max_um is None:
        dx = float(df['x_um'].max() - df['x_um'].min())
        dy = float(df['y_um'].max() - df['y_um'].min())
        dz = float(df['z_um'].max() - df['z_um'].min())
        diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        r_max_um = 0.5 * diag if diag > 0 else 1.0
    r_max_um = float(r_max_um)
    if r_max_um <= 0:
        raise ValueError('r_max_um must be > 0')

    edges = np.linspace(0.0, r_max_um, int(nbins) + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    for frame, g in df.groupby('frame', sort=True):
        pos = g[['x_um', 'y_um', 'z_um']].to_numpy(dtype=float)
        vel = g[['vx_um_s', 'vy_um_s', 'vz_um_s']].to_numpy(dtype=float)
        norms = np.linalg.norm(vel, axis=1)
        valid = np.isfinite(norms) & (norms > 0)
        if np.sum(valid) < 2:
            for rc in r_centers:
                rows.append({'frame': int(frame), 'r_um': float(rc), 'S_vec_spatial': np.nan, 'n_pairs': 0})
            continue

        pos = pos[valid]
        u = vel[valid] / norms[valid, None]

        tree = cKDTree(pos)
        pairs = np.array(list(tree.query_pairs(r_max_um)), dtype=int)

        sum_S = np.zeros(int(nbins), dtype=float)
        n_pairs = np.zeros(int(nbins), dtype=int)

        if pairs.size:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            d = np.linalg.norm(pos[i_idx] - pos[j_idx], axis=1)
            dots = np.einsum('ij,ij->i', u[i_idx], u[j_idx])
            S_vals = 1.5 * (dots**2 - (1.0 / 3.0))

            b = np.digitize(d, edges) - 1
            keep = (b >= 0) & (b < int(nbins))
            if np.any(keep):
                b = b[keep]
                S_vals = S_vals[keep]
                np.add.at(sum_S, b, S_vals)
                np.add.at(n_pairs, b, 1)

        with np.errstate(invalid='ignore', divide='ignore'):
            S_prof = sum_S / n_pairs
        S_prof[n_pairs < int(min_pairs)] = np.nan

        for rc, s, n in zip(r_centers, S_prof, n_pairs):
            rows.append({
                'frame': int(frame),
                'r_um': float(rc),
                'S_vec_spatial': float(s) if np.isfinite(s) else np.nan,
                'n_pairs': int(n),
            })

    return pd.DataFrame(rows)


def plot_spatial_vector_correlation(
    spatial_corr_df: pd.DataFrame,
    *,
    fit_exp: bool = True,
    fit_max_r_um: float | None = None,
    fit_min_points: int = 4,
    show_std: bool = True,
    show_heatmap: bool = True,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 150,
    show: bool = True,
    save_path: str | None = None,
    save_kwargs: dict | None = None,
):
    """Plot spatial vector-correlation mean profile + frame-vs-r heatmap.

    Expects columns: frame, r_um, S_vec_spatial.
    """
    need = {'frame', 'r_um', 'S_vec_spatial'}
    missing = need.difference(spatial_corr_df.columns)
    if missing:
        raise ValueError(f'spatial_corr_df missing columns: {sorted(missing)}')

    pivot_S = (
        spatial_corr_df
        .pivot(index='frame', columns='r_um', values='S_vec_spatial')
        .sort_index()
    )
    if pivot_S.shape[0] == 0 or pivot_S.shape[1] == 0:
        raise ValueError('No data to plot in spatial_corr_df')

    r_vals = pivot_S.columns.to_numpy(dtype=float)
    frame_vals = pivot_S.index.to_numpy(dtype=int)
    arr_S = pivot_S.to_numpy(dtype=float)
    mean_S = np.nanmean(arr_S, axis=0)
    std_S = np.nanstd(arr_S, axis=0)

    if show_heatmap:
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
        ax0, ax1 = axes[0], axes[1]
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(figsize[0], max(3.5, 0.55 * figsize[1])), dpi=dpi, constrained_layout=True)
        ax1 = None
        axes = np.array([ax0], dtype=object)

    ax0.plot(r_vals, mean_S, marker='o', lw=1.2, label='mean S(r)')
    if show_std:
        y1 = mean_S - std_S
        y2 = mean_S + std_S
        valid_std = np.isfinite(r_vals) & np.isfinite(y1) & np.isfinite(y2)
        if np.any(valid_std):
            ax0.fill_between(
                r_vals[valid_std],
                y1[valid_std],
                y2[valid_std],
                alpha=0.2,
                linewidth=0,
                label='mean ± std',
            )
    ax0.set_xlabel('r (µm)')
    ax0.set_ylabel('mean S(r)')
    ax0.set_title('Spatial vector correlation (mean over frames)')
    ax0.grid(True, alpha=0.3)

    valid0 = np.isfinite(r_vals) & np.isfinite(mean_S)
    if np.any(valid0):
        rv = r_vals[valid0]
        sv = mean_S[valid0]
        ax0.set_xlim(float(np.nanmin(rv)), float(np.nanmax(rv)))
        ylo = float(np.nanmin(sv))
        yhi = float(np.nanmax(sv))
        if yhi > ylo:
            pad = 0.07 * (yhi - ylo)
            ax0.set_ylim(ylo - pad, yhi + pad)
        else:
            pad = 0.1 * max(1.0, abs(yhi))
            ax0.set_ylim(ylo - pad, yhi + pad)

    fit_result = {
        'success': False,
        'A': np.nan,
        'tau': np.nan,
        'A_se': np.nan,
        'tau_se': np.nan,
        'n_fit': 0,
        'message': 'fit disabled',
    }
    if fit_exp:
        fit_result = _fit_exp_decay_positive_x(
            r_vals,
            mean_S,
            min_points=int(fit_min_points),
            max_x=fit_max_r_um,
        )
        if fit_result.get('success', False):
            xi = float(fit_result['tau'])
            xi_se = float(fit_result.get('tau_se', np.nan))
            if np.isfinite(xi_se):
                fit_label = f"exp fit (xi={xi:.3g}±{xi_se:.2g} µm)"
            else:
                fit_label = f"exp fit (xi={xi:.3g} µm)"
            ax0.plot(
                fit_result['x_fit'],
                fit_result['y_fit'],
                ls='--',
                lw=1.5,
                label=fit_label,
            )
        else:
            print(f"[plot_spatial_vector_correlation] {fit_result.get('message', 'fit failed')}")

    if ax0.get_legend_handles_labels()[0]:
        ax0.legend(loc='best', frameon=False)

    arr = arr_S
    vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else -0.5
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if not np.isfinite(vmin):
        vmin = -0.5
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    if show_heatmap and ax1 is not None:
        im = ax1.imshow(
            arr,
            aspect='auto',
            origin='lower',
            extent=[float(np.nanmin(r_vals)), float(np.nanmax(r_vals)), float(np.nanmin(frame_vals)), float(np.nanmax(frame_vals))],
            vmin=vmin,
            vmax=vmax,
            cmap='coolwarm',
        )
        ax1.set_xlabel('r (µm)')
        ax1.set_ylabel('frame')
        ax1.set_title('S(r) per frame')
        fig.colorbar(im, ax=ax1, label='S(r)')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        kwargs = {'bbox_inches': 'tight', 'dpi': dpi}
        if save_kwargs:
            kwargs.update(save_kwargs)
        fig.savefig(save_path, **kwargs)

    if show:
        plt.show()

    return fig, axes, fit_result


# Backward-compatible aliases
vector_correlation_nematic_3d = vector_correlation_3d
plot_vector_correlation_nematic_3d = plot_vector_correlation_3d
spatial_vector_correlation_nematic_per_frame = spatial_vector_correlation_per_frame
plot_spatial_vector_correlation_nematic = plot_spatial_vector_correlation


def plot_velocity_velocity_correlation_3d(
    corr_df: pd.DataFrame,
    *,
    normalized: bool = True,
    fit_exp: bool = True,
    fit_max_tau_s: float | None = None,
    fit_min_points: int = 4,
    ax=None,
    figsize: tuple[float, float] = (7, 4),
    dpi: int = 150,
    show: bool = True,
    save_path: str | None = None,
    save_kwargs: dict | None = None,
):
    """Plot 3D velocity-velocity correlation from velocity_velocity_correlation_3d output.

    Parameters
    ----------
    save_path
        Optional path to save the figure (e.g. .png/.pdf). If provided, parent
        folders are created automatically.
    save_kwargs
        Optional keyword arguments passed to ``fig.savefig``.
    """
    y_col = 'Cvv_norm' if normalized else 'Cvv_um2_s2'
    if y_col not in corr_df.columns:
        raise ValueError(f'corr_df missing {y_col}')

    x = corr_df['tau_s'].to_numpy(dtype=float)
    y = corr_df[y_col].to_numpy(dtype=float)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x, y, marker='o', lw=1.5, label='data')
    ax.set_xlabel('lag time (s)')
    if normalized:
        ylabel = 'Cvv / Cvv(0)'
    else:
        # TeX mode cannot parse plain unicode + caret exponents in normal text.
        ylabel = r'Cvv ($\mu$m$^2$/s$^2$)' if mpl.rcParams.get('text.usetex', False) else 'Cvv (µm^2/s^2)'
    ax.set_ylabel(ylabel)
    ax.set_title('3D velocity-velocity correlation')
    ax.grid(True, alpha=0.3)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        x_valid = x[valid]
        ax.set_xlim(float(x_valid[0]), float(x_valid[-1]))
    elif x.size:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    if normalized:
        ax.axhline(0.0, color='0.6', lw=1, ls='--')

    # Always auto-limit y-axis to finite plotted values with small padding.
    y_valid = y[np.isfinite(y)]
    if y_valid.size:
        ylo = float(np.nanmin(y_valid))
        yhi = float(np.nanmax(y_valid))
        if np.isfinite(ylo) and np.isfinite(yhi):
            if yhi > ylo:
                pad = 0.07 * (yhi - ylo)
                ax.set_ylim(ylo - pad, yhi + pad)
            else:
                pad = 0.1 * max(1.0, abs(yhi))
                ax.set_ylim(ylo - pad, yhi + pad)

    fit_result = {
        'success': False,
        'A': np.nan,
        'tau': np.nan,
        'A_se': np.nan,
        'tau_se': np.nan,
        'n_fit': 0,
        'message': 'fit disabled',
    }
    if fit_exp:
        fit_result = _fit_exp_decay_positive_x(
            x,
            y,
            min_points=int(fit_min_points),
            max_x=fit_max_tau_s,
        )
        if fit_result.get('success', False):
            tau = float(fit_result['tau'])
            tau_se = float(fit_result.get('tau_se', np.nan))
            if np.isfinite(tau_se):
                fit_label = f"exp fit (tau={tau:.3g}±{tau_se:.2g} s)"
            else:
                fit_label = f"exp fit (tau={tau:.3g} s)"
            ax.plot(
                fit_result['x_fit'],
                fit_result['y_fit'],
                ls='--',
                lw=1.5,
                label=fit_label,
            )
        else:
            print(f"[plot_velocity_velocity_correlation_3d] {fit_result.get('message', 'fit failed')}")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best', frameon=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        kwargs = {'bbox_inches': 'tight', 'dpi': dpi}
        if save_kwargs:
            kwargs.update(save_kwargs)
        fig.savefig(save_path, **kwargs)

    if show and created_fig:
        plt.show()
    return fig, ax, fit_result


def animate_bead_displacement_per_frame_overlay(
    img,
    tracks_vel_df: pd.DataFrame,
    out_path: str,
    *,
    px_per_micron: float,
    bead_channel: int = 1,
    scalebar_um: float | None = None,
    scalebar_loc: str = 'lower right',
    scalebar_height_px: int = 6,
    scalebar_pad_px: int = 10,
    frame_step: int = 1,
    vector_scale: float = 1.0,
    max_vectors: int | None = 3000,
    fps: float = 10,
    dpi: int = 150,
    show_title: bool = True,
    verbose: bool = True,
    progress_every: int = 10,
):
    """Animate bead displacement-per-frame arrows on z-max image background.

    Supported image shapes:
    - (T, Y, X)
    - (T, Z, Y, X)
    - (T, C, Z, Y, X)
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    if tracks_vel_df is None or len(tracks_vel_df) == 0:
        raise ValueError('tracks_vel_df is empty')
    if not px_per_micron or float(px_per_micron) <= 0:
        raise ValueError('px_per_micron must be a positive number')

    arr_shape = getattr(img, 'shape', None)
    if arr_shape is None or len(arr_shape) not in (3, 4, 5):
        raise ValueError(f'img must be shaped (T,Y,X), (T,Z,Y,X), or (T,C,Z,Y,X), got {arr_shape}')
    T = int(arr_shape[0])

    need_cols = {'frame', 'particle', 'x_um', 'y_um', 'dx_um', 'dy_um', 'dt_s'}
    missing = need_cols.difference(tracks_vel_df.columns)
    if missing:
        raise ValueError(f'tracks_vel_df missing columns: {sorted(missing)}')

    dfv = tracks_vel_df.copy()
    dfv = dfv[dfv['dt_s'].to_numpy(dtype=float) > 0].copy()
    if len(dfv) == 0:
        raise ValueError('No valid rows with dt_s > 0')
    dfv['frame'] = dfv['frame'].astype(int)
    dfv['particle'] = dfv['particle'].astype(int)
    dfv['x_px'] = dfv['x_um'].astype(float) * float(px_per_micron)
    dfv['y_px'] = dfv['y_um'].astype(float) * float(px_per_micron)

    if 'dframe' in dfv.columns:
        dframe = dfv['dframe'].astype(float).replace(0, np.nan)
    else:
        dframe = 1.0
    dfv['u_px_pf'] = (dfv['dx_um'].astype(float) / dframe) * float(px_per_micron)
    dfv['v_px_pf'] = (dfv['dy_um'].astype(float) / dframe) * float(px_per_micron)

    frames = sorted(set(int(f) for f in dfv['frame'].unique()))
    frames = [f for f in frames if 0 <= f < T]
    frames = frames[::max(1, int(frame_step))]
    if not frames:
        raise ValueError('No frame indices available after filtering')
    progress_every = max(1, int(progress_every))

    if verbose:
        print(
            f"Rendering displacement overlay animation: {len(frames)} frames "
            f"(step={max(1, int(frame_step))})",
            flush=True,
        )

    mean_speed = (
        tracks_vel_df.groupby('frame', sort=True)['speed_um_s']
        .mean()
        .reindex(range(T))
        .to_numpy()
    )

    def _bg_zmax(ti: int) -> np.ndarray:
        fr = img[int(ti)]
        if len(arr_shape) == 5:
            fr = fr[int(bead_channel)]
        if hasattr(fr, 'compute'):
            fr = fr.compute()
        fr = np.asarray(fr)
        if fr.ndim == 3:
            return fr.max(axis=0)
        if fr.ndim == 2:
            return fr
        raise ValueError(f'Unsupported frame ndim {fr.ndim} for index {ti}')

    bg0 = _bg_zmax(frames[0])
    H, W = bg0.shape
    fig, ax = plt.subplots(figsize=(7, 7 * H / W), dpi=dpi)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    im = ax.imshow(bg0, cmap='gray')

    artists = {'Q': None}
    S = ax.scatter([], [], s=6, c='cyan', alpha=0.6, edgecolors='none')

    def _nice_scalebar_um(target_um: float) -> float:
        if not np.isfinite(target_um) or target_um <= 0:
            return 1.0
        exp10 = 10 ** np.floor(np.log10(target_um))
        mant = target_um / exp10
        if mant < 1.5:
            nice = 1.0
        elif mant < 3.5:
            nice = 2.0
        elif mant < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * exp10)

    # If not provided, choose ~20% of field width as a readable default.
    sb_um = scalebar_um
    if sb_um is None and px_per_micron and float(px_per_micron) > 0:
        fov_um = float(W) / float(px_per_micron)
        sb_um = _nice_scalebar_um(0.2 * fov_um)

    sb_len_px = None
    if sb_um is not None and px_per_micron and float(px_per_micron) > 0:
        sb_len_px = int(max(1, round(float(sb_um) * float(px_per_micron))))
        sb_len_px = min(sb_len_px, max(1, W - 2 * int(scalebar_pad_px)))

    if sb_len_px is not None:
        w_frac = float(sb_len_px) / max(float(W), 1.0)
        h_frac = float(scalebar_height_px) / max(float(H), 1.0)
        pad_x = float(scalebar_pad_px) / max(float(W), 1.0)
        pad_y = float(scalebar_pad_px) / max(float(H), 1.0)
        loc = (scalebar_loc or 'lower right').lower()

        if loc in ('lower left', 'll'):
            x0, y0 = pad_x, pad_y
            y_text = y0 + h_frac + 0.01
            va = 'bottom'
        elif loc in ('upper right', 'ur'):
            x0, y0 = 1.0 - pad_x - w_frac, 1.0 - pad_y - h_frac
            y_text = y0 - 0.01
            va = 'top'
        elif loc in ('upper left', 'ul'):
            x0, y0 = pad_x, 1.0 - pad_y - h_frac
            y_text = y0 - 0.01
            va = 'top'
        else:
            x0, y0 = 1.0 - pad_x - w_frac, pad_y
            y_text = y0 + h_frac + 0.01
            va = 'bottom'

        rect = Rectangle(
            (x0, y0),
            w_frac,
            h_frac,
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='none',
            zorder=6,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.5 * w_frac,
            y_text,
            f"{float(sb_um):g} µm",
            transform=ax.transAxes,
            color='white',
            ha='center',
            va=va,
            fontsize=10,
            zorder=7,
        )

    def update(i):
        ii = int(i)
        ti = frames[ii]
        if verbose and ((ii % progress_every) == 0 or ii == (len(frames) - 1)):
            print(f"  rendering frame {ii + 1}/{len(frames)} (t={ti})", flush=True)
        im.set_data(_bg_zmax(ti))

        g = dfv[dfv['frame'] == int(ti)]
        if max_vectors is not None and len(g) > int(max_vectors):
            g = g.sample(int(max_vectors), random_state=0)

        xs = g['x_px'].to_numpy(dtype=float)
        ys = g['y_px'].to_numpy(dtype=float)
        us = g['u_px_pf'].to_numpy(dtype=float) * float(vector_scale)
        vs = g['v_px_pf'].to_numpy(dtype=float) * float(vector_scale)

        if artists['Q'] is not None:
            try:
                artists['Q'].remove()
            except Exception:
                pass
            artists['Q'] = None
        if xs.size:
            artists['Q'] = ax.quiver(
                xs,
                ys,
                us,
                vs,
                color='yellow',
                angles='xy',
                scale_units='xy',
                scale=1.0,
                width=0.002,
            )

        S.set_offsets(np.c_[xs, ys] if xs.size else np.empty((0, 2)))

        if show_title:
            ms = mean_speed[int(ti)] if int(ti) < len(mean_speed) else np.nan
            if np.isfinite(ms):
                ax.set_title(f'frame {ti}/{T-1} | mean speed={ms:.3g} µm/s', fontsize=12)
            else:
                ax.set_title(f'frame {ti}/{T-1}', fontsize=12)
        return [im, artists['Q'], S] if (artists['Q'] is not None) else [im, S]

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=200)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=12000)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    if verbose:
        print('Saved displacement-per-frame overlay animation to', out_path)
    return out_path

from matplotlib.collections import LineCollection

def animate_bead_tracks_overlay(
    img,
    tracks_df: pd.DataFrame,
    out_path: str,
    *,
    fps: float = 10,
    tail: int = 15,
    cmap_img: str = 'gray',
    alpha_img: float = 1.0,
    track_cmap: str = 'tab20',
    marker_size: float = 18,
    line_width: float = 1.5,
    x_col: str | None = None,
    y_col: str | None = None,
    coords_in_um: bool | None = None,
    px_per_micron: float | None = None,
    dpi: int = 150,
    percentile_clip: tuple[float, float] = (1.0, 99.5),
    show_title: bool = True,
    invert_y: bool = True,
    verbose: bool = True,
    ):
    """
    Animate tracking data overlaid on bead images.

    Parameters
    ----------
    img
        Background image time series. Supported shapes:
        - (T, Y, X)
        - (T, Z, Y, X)  (will z-max project)
    tracks_df
        Trackpy-style dataframe with at least columns: 'frame', 'particle' and position columns.
    out_path
        Output .mp4 path.
    tail
        Number of previous frames to draw as a trail per particle (0 disables trails).
    x_col, y_col
        Position columns. If None, prefers ('x','y') else ('x_um','y_um').
    coords_in_um
        If True, converts x/y from µm to px using px_per_micron. If None, inferred from column name ending '_um'.
    """
    if img.ndim not in (3, 4):
        raise ValueError(f"img must be (T,Y,X) or (T,Z,Y,X), got shape={getattr(img,'shape',None)}")
    T = int(img.shape[0])
    if 'frame' not in tracks_df.columns:
        raise ValueError("tracks_df must contain a 'frame' column")
    if 'particle' not in tracks_df.columns:
        raise ValueError("tracks_df must contain a 'particle' column")

    if x_col is None:
        x_col = 'x' if 'x' in tracks_df.columns else 'x_um'
    if y_col is None:
        y_col = 'y' if 'y' in tracks_df.columns else 'y_um'
    if x_col not in tracks_df.columns or y_col not in tracks_df.columns:
        raise ValueError(f"Missing position columns: {x_col=}, {y_col=}")

    if coords_in_um is None:
        coords_in_um = (str(x_col).endswith('_um') or str(y_col).endswith('_um'))
    if coords_in_um and not px_per_micron:
        raise ValueError("coords_in_um=True requires px_per_micron")

    # copy minimal columns; compute pixel coords
    df = tracks_df[['frame', 'particle', x_col, y_col]].copy()
    df = df.dropna(subset=[x_col, y_col])
    df['frame'] = df['frame'].astype(int)
    df['particle'] = df['particle'].astype(int)
    if coords_in_um:
        df['x_px'] = df[x_col].astype(float) * float(px_per_micron)
        df['y_px'] = df[y_col].astype(float) * float(px_per_micron)
    else:
        df['x_px'] = df[x_col].astype(float)
        df['y_px'] = df[y_col].astype(float)

    # keep only frames in range
    df = df[(df['frame'] >= 0) & (df['frame'] < T)].copy()
    if len(df) == 0:
        raise ValueError("No track rows within valid frame range")

    # groupings for fast access
    by_frame = {int(f): g for f, g in df.groupby('frame', sort=True)}
    by_particle = {int(pid): g.sort_values('frame')[['frame', 'x_px', 'y_px']].to_numpy()
                   for pid, g in df.groupby('particle', sort=False)}
    pids_sorted = sorted(by_particle.keys())
    cmap_obj = get_cmap(track_cmap)
    pid_to_color = {pid: cmap_obj(pid % cmap_obj.N) for pid in pids_sorted}

    # background frame helper
    def _bg_frame(ti: int) -> np.ndarray:
        fr = img[ti]
        if fr.ndim == 3:
            fr = fr.max(axis=0)
        return np.asarray(fr)

    # figure setup
    fr0 = _bg_frame(0)
    H, W = int(fr0.shape[-2]), int(fr0.shape[-1])
    fig, ax = plt.subplots(figsize=(7, 7 * H / W), dpi=dpi)
    ax.set_axis_off()
    if invert_y:
        ax.set_ylim(H, 0)
    else:
        ax.set_ylim(0, H)
    ax.set_xlim(0, W)

    vmin, vmax = np.percentile(fr0, percentile_clip)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(fr0)), float(np.nanmax(fr0))
    im = ax.imshow(fr0, cmap=cmap_img, vmin=vmin, vmax=vmax, alpha=alpha_img)

    sc = ax.scatter([], [], s=marker_size, c=[], edgecolors='none')
    lc = LineCollection([], linewidths=line_width, alpha=0.8)
    ax.add_collection(lc)

    def update(ti: int):
        fr = _bg_frame(ti)
        vmin, vmax = np.percentile(fr, percentile_clip)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(fr)), float(np.nanmax(fr))
        im.set_data(fr)
        im.set_clim(vmin=vmin, vmax=vmax)

        g = by_frame.get(int(ti), None)
        if g is None or len(g) == 0:
            sc.set_offsets(np.zeros((0, 2)))
            sc.set_color([])
            lc.set_segments([])
            lc.set_color([])
        else:
            xy = g[['x_px', 'y_px']].to_numpy(dtype=float)
            colors = [pid_to_color[int(pid)] for pid in g['particle'].to_numpy()]
            sc.set_offsets(xy)
            sc.set_color(colors)

            if tail and tail > 0:
                segments = []
                seg_colors = []
                t0 = int(ti) - int(tail)
                for pid in g['particle'].to_numpy():
                    arr = by_particle.get(int(pid), None)
                    if arr is None:
                        continue
                    m = (arr[:, 0] >= t0) & (arr[:, 0] <= int(ti))
                    coords = arr[m][:, 1:3]
                    if coords.shape[0] >= 2:
                        segments.append(coords)
                        seg_colors.append(pid_to_color[int(pid)])
                lc.set_segments(segments)
                lc.set_color(seg_colors)
            else:
                lc.set_segments([])
                lc.set_color([])

        if show_title:
            ax.set_title(f'frame {ti+1}/{T}', fontsize=12)
        return [im, sc, lc]

    ani = FuncAnimation(fig, update, frames=range(T), interval=1000/max(1e-6, fps), blit=False)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=12000)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    if verbose:
        print('Saved bead track overlay to', out_path)

# Example (uncomment when you have tracks):
# tracks_df = pd.read_parquet('data/<dataset_id>/derived/beads_tracks.parquet')
# bead_img = channels[1]  # expected shape (T, Z, Y, X) in this notebook
# animate_bead_tracks_overlay(
#     bead_img, tracks_df,
#     out_path=f'plots/{name}/{variation}/{name}_beads_tracks_overlay.mp4',
#     fps=10, tail=20,
#     x_col='x', y_col='y'
# )