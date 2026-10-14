# fmri_analysis_core.py

from functools import lru_cache
from pathlib import Path
import numpy as np
import h5py
import cortex
from cortex.database import Database
from collections import defaultdict
import nibabel as nib
from scipy.signal import welch, butter, sosfiltfilt, resample_poly
from itertools import combinations
from fractions import Fraction


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MRI_DIR = DATA_DIR / "mri"
ROI_DIR = DATA_DIR / "rois"
FSMAT_DIR = DATA_DIR / "fsmat"
SURFACE_DIR = DATA_DIR / "surfaces"

# check 
required_dirs = [MRI_DIR, ROI_DIR, FSMAT_DIR, SURFACE_DIR]
missing = [path for path in required_dirs if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Required data folders are missing:\n"
        + "\n".join(str(path) for path in missing)
        + "\n\nDownload the dataset linked in the README and place its contents "
          "inside the repository's data/ directory."
    )

stories = [
    "wheretheressmoke",
    "fromboyhoodtofatherhood",
    "onapproachtopluto",
]

DB = Database(str(SURFACE_DIR))
cortex.db = DB
cortex.database.db = DB
cortex.utils.db = DB
cortex.dataset.braindata.db = DB
cortex.quickflat.utils.db = DB
cortex.quickflat.composite.db = DB

# endregion

# region Low-level helpers 
zs = lambda v: (v-v.mean(0))/(v.std(0) + 1e-8)

def hf5_path(subject, story):
    return MRI_DIR / subject / f"{subject}_{story}.hf5"

def get_xfm(subject):
    return f"{subject}_auto"

# endregion

# region Cache
# cache expansive computations
@lru_cache(maxsize=None)
def load_response_trials(subject: str, story: str, drop_first_trs: int = 50) -> np.ndarray:
    path = hf5_path(subject, story)
    with h5py.File(path, "r") as f:
        dsname = next(iter(f.keys()))
        arr = f[dsname][:]
    if drop_first_trs:
        arr = arr[:, drop_first_trs:, :]
    return arr

@lru_cache(maxsize=None)
def load_mask(subject: str) -> np.ndarray:
    xfm = get_xfm(subject)
    return DB.get_mask(subject, xfm)

@lru_cache(maxsize=None)
def get_mapper(subject: str):
    xfm = get_xfm(subject)
    return cortex.get_mapper(subject, xfm, "line_nearest", recache=False)

@lru_cache(maxsize=None)
def get_fsmats(subject):
    l_mat = np.load(FSMAT_DIR / f"{subject}_lh.npy", allow_pickle=True).item()
    r_mat = np.load(FSMAT_DIR / f"{subject}_rh.npy", allow_pickle=True).item()
    return l_mat, r_mat

# endregion

# region Vertex/voxel conversion utilities
# Map subject brain surfaces to template (fsaverage)
def voxel_to_vertex(subject: str, data_1d: np.ndarray):
    xfm = get_xfm(subject)
    mask3d = load_mask(subject)

    data_1d = np.asarray(data_1d)
    if data_1d.ndim != 1:
        raise ValueError(f"data must be 1D, got {data_1d.shape}")

    nmask = int(mask3d.sum())
    if data_1d.shape[0] != nmask:
        raise ValueError(f"data length {data_1d.shape[0]} != mask.sum() {nmask} for {subject}")

    mapper = get_mapper(subject)

    vol_data = cortex.unmask(mask3d, data_1d).copy()
    vmin, vmax = np.nanpercentile(data_1d, [1, 99])
    vol = cortex.Volume(vol_data, subject, xfm, vmin=vmin, vmax=vmax)

    return mapper(vol)  # vert with .left .right

def align_subj_to_fsaverage(subject: str, vert):
    l_mat, r_mat = get_fsmats(subject)
    fs_lh = l_mat @ vert.left
    fs_rh = r_mat @ vert.right
    return np.hstack([fs_lh, fs_rh])  # numpy array in fsaverage vertex space

# endregion


# region ROI builders
def build_roi_voxs(
    subject: str,
    f_rois=None, # default is AC and sPMv
    min_voxels: int = 1,
) -> dict:
    """
    Build a dict roi_voxs[roi_name] -> voxel indices (1D indices into masked voxels)
    combining functional and anatomical ROIs.
    """
    if f_rois is None:
        f_rois = ['AC', 'sPMv']

    # ---------- functional ROIs ----------
    xfm = get_xfm(subject)
    mask3d = load_mask(subject)

    f_roi_voxs = {}

    for roi in f_rois:
        try:
            one = cortex.utils.get_roi_masks(
                subject,
                xfm,
                roi_list=roi,          
                gm_sampler="cortical",
                split_lr=False,
                threshold=None,
                return_dict=True,
            )
        except KeyError:
            # ROI truly not present for this subject
            continue

        roi_mask3d = one[roi]
        idx = np.flatnonzero(roi_mask3d[mask3d])

        if idx.size >= min_voxels:
            f_roi_voxs[roi] = idx

    # ---------- anatomical ROIs ----------
    a_roi = [
        'parsopercularis', 'parstriangularis', 'superiorfrontal',
        'rostralmiddlefrontal','caudalmiddlefrontal','frontalpole',
        # 'precuneus'
        'inferiorparietal'
    ]

    roi_path = ROI_DIR / f"{subject}_roi.npy"
    roi_data = np.load(roi_path, allow_pickle=True).item()

    a_roi_voxs = {roi: roi_data[roi] for roi in a_roi}

    # combine PFC rois
    rois_to_combine = [
        'parsopercularis',
        'parstriangularis',
        'superiorfrontal',
        'rostralmiddlefrontal',
        'caudalmiddlefrontal',
        'frontalpole'
    ]

    pfc_voxs = []
    for roi in rois_to_combine:
        pfc_voxs.extend(a_roi_voxs[roi])
    pfc_voxs = list(set(pfc_voxs))  # remove potential duplicates

    a_roi_voxs['prefrontal'] = pfc_voxs
    for roi in rois_to_combine:
        del a_roi_voxs[roi]

    # ---------- merge functional + anatomical ----------
    roi_voxs = {**f_roi_voxs, **a_roi_voxs}

    return roi_voxs

def equal_voxs(roi_voxs, seed=0):
    rng = np.random.default_rng(seed)
    num_vox = min(len(vox) for vox in roi_voxs.values())
    roi_voxs_equal = {}
    
    for roi, vox in roi_voxs.items():
        vox = np.asarray(vox)
        vox_idx = rng.choice(len(vox), size=num_vox, replace=False)
        roi_voxs_equal[roi] = vox[vox_idx]

    return roi_voxs_equal 

# endregion

# region Simple metrics

# Get SNR for a given ROI (input should be ROI-averaged data)
def estimate_snr(response_trials):
    # response_trials should be ROI averaged data, with the shape (nRuns x nTimepoints)
    signal = np.mean(response_trials, axis=0)
    signal_var = np.var(signal, ddof=1)
    noise = response_trials - signal
    noise_var = np.var(noise, ddof=1) 
    snr = signal_var/(signal_var + noise_var) 
    return snr

def avg_pop_curvature(X, time_axis=1, eps=1e-12):
    """
    Compute the mean turning angle of a population trajectory.

    Parameters
    ----------
    X : np.ndarray
        Population activity.
    time_axis : int
        Axis corresponding to time.
        Use 1 for arrays shaped (features, time).
        Use 0 for arrays shaped (time, features).
    eps : float
        Threshold for excluding zero-length steps.

    Returns
    -------
    float
        Mean trajectory curvature in degrees.
    """

    # Standardize to shape (features, time)
    if time_axis == 0:
        X = X.T

    V = np.diff(X, axis=1)
    norms = np.linalg.norm(V, axis=0)

    dots = np.sum(V[:, 1:] * V[:, :-1], axis=0)
    denom = norms[1:] * norms[:-1]

    valid = denom > eps

    if not np.any(valid):
        return np.nan

    cos_angles = dots[valid] / denom[valid]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    angles = np.degrees(np.arccos(cos_angles))

    return float(np.mean(angles))


def compute_coherence_spectrum(data, fs=0.5, nfft=64):
    """
    Compute signal/noise PSDs and two "coherence-like" reliability measures from repeated time series.

    Parameters
    ----------
    data : ndarray, shape (R, T, V)
        Repeated measurements.
        R = repeats (trials), T = time points, V = voxels (or units).
    fs : float
        Sampling rate in Hz (e.g., fs = 1/TR for fMRI).
    nfft : int, default=64
        Used to set Welch segment length: nperseg = min(nfft, T).
        (Note: this argument is historically named nfft here, but is used as nperseg.)

    Returns
    -------
    freqs : ndarray, shape (F,)
        Frequency bins for Welch estimates.
    spower : ndarray, shape (F, V)
        PSD of the across-repeat mean ("signal power").
    npower : ndarray, shape (F, V)
        Mean PSD of trial residuals around the grand mean ("noise power").
    jnpower : ndarray, shape (F, V)
        Mean PSD of jackknife residuals (leave-one-out residuals). Computed via a cheap scaling identity.
    coherence_upper : ndarray, shape (F, V)
        Upper-bound reliability estimate: spower / (spower + npower).
        (Uses residuals around grand mean.)
    coherence_lower : ndarray, shape (F, V)
        Lower-bound reliability estimate: spower / (spower + jnpower).
        (Uses jackknife residuals; conservative.)

    Notes
    -----
    - This is not the classical cross-spectral coherence between two signals. It's a repeatability
      metric derived from splitting variance into "signal" (repeat mean) vs "noise" (repeat residuals),
      frequency-by-frequency.
    - The original implementation explicitly constructed jackknife residuals and ran Welch on them:
        mean_loo = (sum_all - data) / (R - 1)
        jnoises  = data - mean_loo
        jnpower  = mean PSD of jnoises
      This is expensive (extra Welch over an (R,T,V) array).
    - We keep that code below, commented out, for documentation / future variants.

    Speed improvement (exact identity)
    ----------------------------------
    Let signal = mean_r data_r and noise_r = data_r - signal.
    The jackknife residual is:
        jnoise_r = data_r - mean_{-r}
                 = (R/(R-1)) * (data_r - signal)
                 = (R/(R-1)) * noise_r
    Therefore:
        PSD(jnoise_r) = (R/(R-1))^2 * PSD(noise_r)
    Averaging over r preserves the scale, so:
        jnpower = (R/(R-1))^2 * npower
    This avoids the extra Welch call.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data must have shape (repeats, time, voxels). Got {data.shape}")

    R, T, V = data.shape
    if R < 2:
        raise ValueError("Need at least 2 repeats to compute jackknife-based coherence_lower.")

    nperseg = min(int(nfft), int(T))     # segment length for Welch
    noverlap = nperseg // 2
    params = dict(nperseg=nperseg, fs=fs, noverlap=noverlap)

    # -------------------------
    # signal: mean across repeats
    # -------------------------
    signal = data.mean(axis=0)                         # (T, V)
    freqs, spower = welch(signal, axis=0, **params)    # (F, V)

    # -------------------------
    # noise: trial residuals around the grand mean
    # -------------------------
    noise = data - signal[None, :, :]                  # (R, T, V)
    _, noise_psd = welch(noise, axis=1, **params)      # (R, F, V)
    npower = noise_psd.mean(axis=0)                    # (F, V)

    # -------------------------
    # jackknife noise: CHEAP version (recommended)
    # -------------------------
    scale = (R / (R - 1)) ** 2
    jnpower = scale * npower                           # (F, V)

    # -------------------------
    # jackknife noise: EXPLICIT version (expensive; keep for reference)
    # -------------------------
    # sum_all = data.sum(axis=0, keepdims=True)         # (1, T, V)
    # mean_loo = (sum_all - data) / (R - 1)             # (R, T, V)
    # jnoises = data - mean_loo                         # (R, T, V)
    # _, jn_psd = welch(jnoises, axis=1, **params)      # (R, F, V)
    # jnpower = jn_psd.mean(axis=0)                     # (F, V)

    # -------------------------
    # coherence-like reliability estimates
    # -------------------------
    coherence_upper = spower / (npower + spower)
    coherence_lower = spower / (jnpower + spower)

    coherence_upper = np.clip(coherence_upper, 0, 1)
    coherence_lower = np.clip(coherence_lower, 0, 1)

    return freqs, spower, npower, jnpower, coherence_upper, coherence_lower


def lowpass_and_resample(X, fs_in, fs_out, low_pass=True, cutoff=None, order=4, max_den=100000):
    X = np.asarray(X, dtype=np.float64)

    if low_pass:
        if cutoff is None:
            cutoff = 0.8 * (fs_out / 2.0)   # 80% of target Nyquist

        sos = butter(order, cutoff, btype='low', fs=fs_in, output='sos')
        X_filt = sosfiltfilt(sos, X, axis=0)
    else:
        X_filt = X

    if np.isclose(fs_in, fs_out):
        return X_filt 

    # downsampling 
    ratio = Fraction(fs_out / fs_in).limit_denominator(max_den)
    up, down = ratio.numerator, ratio.denominator

    if up < 1 or down < 1:
        raise ValueError(
            f"Bad resampling ratio: fs_out/fs_in={fs_out/fs_in}, "
            f"approximated as {up}/{down}. Increase max_den."
        )
    
    Y = resample_poly(X_filt, up=up, down=down, axis=0)
    return Y

# endregion
