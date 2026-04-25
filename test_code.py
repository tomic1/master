import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tomsUtilities import open_raw_lazy

import dask.array as da
HAS_DASK = True

# calibration
PX_PER_MICRON = 3.846
PX_PER_MICRON_Z = 0.6667


def load_one_stack(name, base_dir=None):
    raw = open_raw_lazy(name, base_dir=base_dir)
    arr0 = raw[0][0, 0] if isinstance(raw, (list, tuple)) else raw
    # Return a dask array lazily; wrap non-dask arrays
    if isinstance(arr0, da.Array):
        return arr0
    if hasattr(arr0, 'shape'):
        try:
            return da.from_array(arr0, chunks='auto')
        except Exception:
            pass
    try:
        arr = arr0.compute()
    except Exception:
        arr = np.array(arr0)
    return da.from_array(arr, chunks='auto')


def autocorr_2d(image, subtract_mean=False):
    img = np.asarray(image, dtype=float)
    if subtract_mean:
        img = img - np.mean(img)
    F = np.fft.fft2(img)
    power = np.abs(F) ** 2
    ac = np.fft.fftshift(np.real(np.fft.ifft2(power)))
    var0 = np.var(np.asarray(image, dtype=float))
    # normalize by variance squared (user requested var**2)
    denom = var0 ** 2
    if denom > 0:
        ac = ac / denom
    else:
        ac = ac / np.max(ac)
    return ac

def mean_and_fluctuations(img):
    # img is expected to be a dask array; keep operations lazy
    mu = img.mean()
    delta = img - mu
    return mu, delta


def variance_from_fluctuations(delta):
    # delta expected to be dask array; returns lazy mean
    sigma2 = (delta ** 2).mean()
    return sigma2


def autocovariance_fft(img: np.ndarray):
    _, delta = mean_and_fluctuations(img)
    # use dask FFTs lazily
    f = da.fft.fftn(delta)
    acov = da.fft.ifftn((da.abs(f) ** 2)).real
    acov = acov / delta.size
    return acov


def autocorrelation_fft(img: np.ndarray, eps: float = 1e-12):
    _, delta = mean_and_fluctuations(img)
    sigma2 = variance_from_fluctuations(delta)
    acov = autocovariance_fft(img)
    denom = da.maximum(sigma2, eps)
    return acov / denom


def fftshift_autocorrelation(img: np.ndarray, eps: float = 1e-12):
    acorr = autocorrelation_fft(img, eps=eps)
    # implement fftshift with dask.roll
    shifts = tuple(s // 2 for s in acorr.shape)
    return da.roll(acorr, shift=shifts, axis=(0, 1, 2))


def physical_distance_grid(shape: tuple[int, int, int], spacing: tuple[float, float, float]):
    nz, ny, nx = shape
    dz, dy, dx = spacing

    z = (np.arange(nz) - nz // 2) * dz
    y = (np.arange(ny) - ny // 2) * dy
    x = (np.arange(nx) - nx // 2) * dx

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    return r


def radial_average_3d(volume: np.ndarray, spacing: tuple[float, float, float], bin_size: float | None = None):
    # volume is a dask array; compute flattened values for binning
    r = physical_distance_grid(volume.shape, spacing)

    if bin_size is None:
        bin_size = min(spacing)

    r_flat = r.ravel()
    v_flat = volume.ravel().compute()

    bin_index = np.floor(r_flat / bin_size).astype(int)
    nbins = bin_index.max() + 1

    sums = np.bincount(bin_index, weights=v_flat, minlength=nbins)
    counts = np.bincount(bin_index, minlength=nbins)
    radial_mean = sums / np.maximum(counts, 1)

    r_centers = (np.arange(nbins) + 0.5) * bin_size
    return r_centers, radial_mean, counts


def full_autocorrelation_pipeline(
    img: np.ndarray,
    spacing: tuple[float, float, float],
    bin_size: float | None = None,
    eps: float = 1e-12,
):
    mu, delta = mean_and_fluctuations(img)
    sigma2 = variance_from_fluctuations(delta)
    acov = autocovariance_fft(img)
    denom = da.maximum(sigma2, eps)
    acorr = acov / denom
    shifts = tuple(s // 2 for s in acorr.shape)
    acorr_shifted = da.roll(acorr, shift=shifts, axis=(0, 1, 2))
    r_grid = physical_distance_grid(acorr_shifted.shape, spacing)
    r, c_r, counts = radial_average_3d(acorr_shifted, spacing, bin_size=bin_size)

    return {
        "mu": mu,
        "delta": delta,
        "sigma2": sigma2,
        "acov": acov,
        "acorr": acorr,
        "acorr_shifted": acorr_shifted,
        "r_grid": r_grid,
        "r": r,
        "c_r": c_r,
        "counts": counts,
    }





def axis_autocorr_1d(arr, axis, subtract_mean=False):
    # support numpy and dask arrays; keep lazy until compute()
    if HAS_DASK and isinstance(arr, da.Array):
        a = da.moveaxis(arr, axis, -1)
        L = a.shape[-1]
        flat = a.reshape((-1, L))
        if subtract_mean:
            flat = flat - flat.mean(axis=1, keepdims=True)
        F = da.fft.fft(flat, axis=1)
        power = da.absolute(F) ** 2
        ac = da.fft.ifft(power, axis=1).real
        # normalize so zero-lag = variance
        ac = ac / L
        vars = (flat ** 2).mean(axis=1)
        # avoid divide-by-zero
        vars = da.where(vars == 0, 1.0, vars)
        ac_norm = ac / vars[:, None]
        ac_mean = ac_norm.mean(axis=0)
        maxlag = int(L // 2)
        lags = np.arange(maxlag + 1)
        return lags, ac_mean[: maxlag + 1]
    else:
        a = np.asarray(arr, dtype=float)
        a = np.moveaxis(a, axis, -1)
        L = a.shape[-1]
        flat = a.reshape(-1, L)
        if subtract_mean:
            flat = flat - flat.mean(axis=1, keepdims=True)
        F = np.fft.fft(flat, axis=1)
        power = np.abs(F) ** 2
        ac = np.fft.ifft(power, axis=1).real
        ac = ac / L
        vars = (flat ** 2).mean(axis=1)
        vars[vars == 0] = 1.0
        ac_norm = ac / vars[:, None]
        ac_mean = ac_norm.mean(axis=0)
        maxlag = L // 2
        lags = np.arange(maxlag + 1)
        return lags, ac_mean[:maxlag + 1]


def exp_decay(r, A, L, C):
    return A * np.exp(-r / L) + C


def fit_1d(lags_um, ac):
    lags = np.asarray(lags_um)
    ac = np.asarray(ac)
    mask = np.isfinite(ac) & (lags > 0)
    if np.sum(mask) < 4:
        return None
    x = lags[mask]
    y = ac[mask]
    p0 = [y.max() - y.min(), max(0.1, x.max() / 5.0), y.min()]
    try:
        popt, _ = curve_fit(exp_decay, x, y, p0=p0, maxfev=20000)
        return popt
    except Exception:
        return None


def fit_radial(radial):
    radial = np.asarray(radial)
    r = np.arange(len(radial))
    mask = np.isfinite(radial) & (r > 0)
    if np.sum(mask) < 4:
        return None
    r_fit = r[mask]
    y_fit = radial[mask]
    p0 = [y_fit.max() - y_fit.min(), max(0.1, len(radial) / 10.0), y_fit.min()]
    try:
        popt, _ = curve_fit(exp_decay, r_fit, y_fit, p0=p0, maxfev=20000)
        return popt
    except Exception:
        return None


if __name__ == '__main__':
    name = "AMF_107_002__C640_C470"
    base_dir = "/Volumes/Tom_Data"
    arr = load_one_stack(name, base_dir=base_dir)
    print("Loaded array shape:", arr.shape)

    # require a 3D volume; drop all 2D-only processing
    vol3d = None
    if arr.ndim == 3:
        vol3d = arr
    elif arr.ndim >= 4:
        # try to find a 3D block in the leading dimension
        for i in range(arr.shape[0]):
            if arr[i].ndim == 3:
                vol3d = arr[i]
                break
        if vol3d is None:
            try:
                vol3d = arr.reshape(-1, arr.shape[-3], arr.shape[-2], arr.shape[-1])[0]
            except Exception:
                vol3d = None

    if vol3d is None:
        print('No 3D volume found in the input. Provide a 3D stack to compute volume autocorrelations.')
        raise SystemExit(0)

    # ensure vol3d is a dask array
    if not isinstance(vol3d, da.Array):
        vol3d_raw = da.from_array(np.asarray(vol3d, dtype=float), chunks='auto')
    else:
        vol3d_raw = vol3d

    # 3D autocorrelation pipeline (uses anisotropic spacing)
    spacing = (1.0 / PX_PER_MICRON_Z, 1.0 / PX_PER_MICRON, 1.0 / PX_PER_MICRON)  # dz, dy, dx in microns
    results = full_autocorrelation_pipeline(vol3d_raw, spacing, bin_size=None)
    acorr_shifted = results["acorr_shifted"]
    r_centers = results["r"]
    radial_mean = results["c_r"]

    # axis autocorrelations (per-line) raw and mean-subtracted
    lags_x3_px, ac_x3_raw = axis_autocorr_1d(vol3d_raw, axis=2, subtract_mean=False)
    lags_y3_px, ac_y3_raw = axis_autocorr_1d(vol3d_raw, axis=1, subtract_mean=False)
    lags_z3_px, ac_z3_raw = axis_autocorr_1d(vol3d_raw, axis=0, subtract_mean=False)
    lags_x3_px_ms, ac_x3_ms = axis_autocorr_1d(vol3d_raw, axis=2, subtract_mean=True)
    lags_y3_px_ms, ac_y3_ms = axis_autocorr_1d(vol3d_raw, axis=1, subtract_mean=True)
    lags_z3_px_ms, ac_z3_ms = axis_autocorr_1d(vol3d_raw, axis=0, subtract_mean=True)

    lags_x3_um = lags_x3_px / PX_PER_MICRON
    lags_y3_um = lags_y3_px / PX_PER_MICRON
    lags_z3_um = lags_z3_px / PX_PER_MICRON_Z

    # compute deferred dask arrays just before fitting/plotting
    if isinstance(ac_x3_raw, da.Array):
        ac_x3_raw = ac_x3_raw.compute()
    if isinstance(ac_y3_raw, da.Array):
        ac_y3_raw = ac_y3_raw.compute()
    if isinstance(ac_z3_raw, da.Array):
        ac_z3_raw = ac_z3_raw.compute()
    if isinstance(ac_x3_ms, da.Array):
        ac_x3_ms = ac_x3_ms.compute()
    if isinstance(ac_y3_ms, da.Array):
        ac_y3_ms = ac_y3_ms.compute()
    if isinstance(ac_z3_ms, da.Array):
        ac_z3_ms = ac_z3_ms.compute()
    # compute radial_mean if pipeline returned dask-backed values
    if isinstance(radial_mean, da.Array):
        radial_mean = radial_mean.compute()
    # acorr_shifted may be dask; compute if needed
    if isinstance(acorr_shifted, da.Array):
        acorr_shifted = acorr_shifted.compute()

    # fit decay lengths (in microns) along axes
    popt_x3_raw = fit_1d(lags_x3_um, ac_x3_raw)
    popt_y3_raw = fit_1d(lags_y3_um, ac_y3_raw)
    popt_z3_raw = fit_1d(lags_z3_um, ac_z3_raw)
    popt_x3_ms = fit_1d(lags_x3_um, ac_x3_ms)
    popt_y3_ms = fit_1d(lags_y3_um, ac_y3_ms)
    popt_z3_ms = fit_1d(lags_z3_um, ac_z3_ms)

    # fit radial autocorrelation (from pipeline)
    popt_r = fit_1d(r_centers, radial_mean)

    # plotting: axis autocorrs and radial averages with fits only
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Axis autocorrelations with fitted exponentials
    axes[0].plot(lags_x3_um, ac_x3_raw, label='x raw')
    axes[0].plot(lags_y3_um, ac_y3_raw, label='y raw')
    axes[0].plot(lags_z3_um, ac_z3_raw, label='z raw')
    if popt_x3_raw is not None:
        axes[0].plot(lags_x3_um, exp_decay(lags_x3_um, *popt_x3_raw), '--', label=f'x fit L={popt_x3_raw[1]:.2f}um')
    if popt_y3_raw is not None:
        axes[0].plot(lags_y3_um, exp_decay(lags_y3_um, *popt_y3_raw), '--', label=f'y fit L={popt_y3_raw[1]:.2f}um')
    if popt_z3_raw is not None:
        axes[0].plot(lags_z3_um, exp_decay(lags_z3_um, *popt_z3_raw), '--', label=f'z fit L={popt_z3_raw[1]:.2f}um')
    axes[0].set_xlabel('Lag (um)')
    axes[0].set_ylabel('Autocorr')
    axes[0].set_title('1D Axis Autocorr (raw)')
    axes[0].legend()

    # Radial autocorrelation with fits (physical distances)
    axes[1].plot(r_centers, radial_mean, label='3D radial (normalized)')
    if popt_r is not None:
        axes[1].plot(r_centers, exp_decay(r_centers, *popt_r), '--', label=f'rad L={popt_r[1]:.2f}um')
    axes[1].set_xlabel('Radius (um)')
    axes[1].set_ylabel('Average intensity')
    axes[1].set_title('3D Autocorr Radial Averages')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
