"""Signal processing for AccelHR — resample, filter, and FFT peak detection.

Pipeline (from notebook):
  raw 32 Hz → linear resample to 26 Hz → bandpass / gravity removal →
  rFFT with linear temporal emphasis → argmax over [0.6, 3.6] Hz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SOURCE_FS = 32.0
TARGET_FS = 26.0

# Bandpass filters mirror firmware/get_hr.py:
#   low-movement / sleep  -> [0.6, 3.6] Hz
#   medium / high motion  -> [0.9, 4.0] Hz
SOS_BP_06_36 = butter(4, [0.6, 3.6], btype="bandpass", fs=TARGET_FS, output="sos")
SOS_BP_09_40 = butter(4, [0.9, 4.0], btype="bandpass", fs=TARGET_FS, output="sos")
SOS_BP = SOS_BP_06_36  # default for filter_ppg / filter_accel below
SOS_LP_GRAVITY = butter(2, 0.3, btype="lowpass", fs=TARGET_FS, output="sos")
SOS_HP_ACCEL = butter(2, 0.5, btype="highpass", fs=TARGET_FS, output="sos")

# Movement-state threshold from firmware/movement.h
MAX_ACCEL_FOR_LOW_MOVEMENT = 350.0

HR_MIN_BPM = 36.0
HR_MAX_BPM = 220.0
FREQ_MIN_HZ = HR_MIN_BPM / 60.0
FREQ_MAX_HZ = HR_MAX_BPM / 60.0
FFT_SIZE = 1024


def detect_low_movement(accel_mag: np.ndarray) -> bool:
    """True when accel-magnitude std is below the firmware low-movement threshold."""
    if len(accel_mag) == 0:
        return False
    return float(np.std(accel_mag)) < MAX_ACCEL_FOR_LOW_MOVEMENT


def select_bandpass(filter_choice: str, accel_mag: np.ndarray) -> tuple[np.ndarray, str]:
    """Pick an SOS bandpass.

    filter_choice: "auto" | "low" | "high"
      auto -> firmware-style: low when std(accel_mag) < MAX_ACCEL_FOR_LOW_MOVEMENT.
      low  -> [0.6, 3.6] Hz
      high -> [0.9, 4.0] Hz
    Returns (sos, label).
    """
    if filter_choice == "low":
        return SOS_BP_06_36, "low movement [0.6, 3.6] Hz"
    if filter_choice == "high":
        return SOS_BP_09_40, "high movement [0.9, 4.0] Hz"
    is_low = detect_low_movement(accel_mag)
    sos = SOS_BP_06_36 if is_low else SOS_BP_09_40
    label = ("auto → low movement [0.6, 3.6] Hz" if is_low
             else "auto → high movement [0.9, 4.0] Hz")
    return sos, label


def resample_linear(x: np.ndarray, source_fs: float = SOURCE_FS, target_fs: float = TARGET_FS) -> np.ndarray:
    """Linear interpolation, matches resample_signal() in cadence_lock_mitigation.c."""
    n_in = len(x)
    if n_in < 2:
        return x.astype(np.float64, copy=True)
    duration = (n_in - 1) / source_fs
    n_out = int(duration * target_fs) + 1
    ratio = source_fs / target_fs
    t = np.arange(n_out, dtype=np.float64) * ratio
    idx0 = np.floor(t).astype(int)
    frac = t - idx0
    idx0 = np.clip(idx0, 0, n_in - 1)
    idx1 = np.clip(idx0 + 1, 0, n_in - 1)
    return x[idx0] + frac * (x[idx1] - x[idx0])


def filter_ppg(green: np.ndarray) -> np.ndarray:
    return sosfiltfilt(SOS_BP, green)


def filter_accel(x: np.ndarray, remove_gravity: bool = True) -> np.ndarray:
    """Gravity removal + high-pass + bandpass for accel x/y. Pass remove_gravity=False for z."""
    if remove_gravity:
        x = x - sosfiltfilt(SOS_LP_GRAVITY, x)
        x = sosfiltfilt(SOS_HP_ACCEL, x)
    return sosfiltfilt(SOS_BP, x)


def rfft_emphasis(x: np.ndarray) -> tuple[np.ndarray, float]:
    """rFFT magnitude with linear 0.5→1.0 temporal emphasis on the input."""
    fft_in = np.zeros(FFT_SIZE, dtype=np.float64)
    n = min(len(x), FFT_SIZE)
    fft_in[:n] = x[:n]
    fft_in[:n] -= fft_in[:n].mean()
    if n >= 2:
        fft_in[:n] *= 0.5 + 0.5 * np.arange(n) / (n - 1)
    mags = np.abs(np.fft.rfft(fft_in))
    freq_res = TARGET_FS / FFT_SIZE
    return mags, freq_res


def find_dominant_peak(mags: np.ndarray, freq_res: float,
                       f_lo: float = FREQ_MIN_HZ, f_hi: float = FREQ_MAX_HZ) -> tuple[float, float]:
    fft_half = FFT_SIZE // 2 + 1
    k_lo = max(0, int(f_lo / freq_res))
    k_hi = min(fft_half - 1, int(f_hi / freq_res))
    if k_hi < k_lo:
        return 0.0, 0.0
    sub = mags[k_lo:k_hi + 1]
    k = int(np.argmax(sub)) + k_lo
    return k * freq_res, float(mags[k])


@dataclass
class ProcessedWindow:
    t_ms: np.ndarray  # resampled timestamps (ms since epoch)
    green_raw: np.ndarray
    accel_x_raw: np.ndarray
    accel_y_raw: np.ndarray
    accel_z_raw: np.ndarray
    accel_xy_raw: np.ndarray
    accel_xyz_raw: np.ndarray
    green: np.ndarray
    accel_x: np.ndarray
    accel_y: np.ndarray
    accel_z: np.ndarray
    accel_xy: np.ndarray
    accel_xyz: np.ndarray
    spectra: dict[str, np.ndarray]
    peaks: dict[str, float]  # bpm
    freq_res: float


def process_window(df: pd.DataFrame, source_fs: float = SOURCE_FS) -> ProcessedWindow:
    """Resample green + accel to TARGET_FS, filter, and compute FFT peaks.

    Expects df with columns timestamp (ms), green, accel_x, accel_y, accel_z.
    Rows with missing values in any of those are dropped before resampling.
    """
    cols = ["timestamp", "green", "accel_x", "accel_y", "accel_z"]
    sub = df[cols].dropna().sort_values("timestamp").reset_index(drop=True)
    if sub.empty:
        empty = np.array([], dtype=np.float64)
        return ProcessedWindow(
            t_ms=empty,
            green_raw=empty, accel_x_raw=empty, accel_y_raw=empty, accel_z_raw=empty,
            accel_xy_raw=empty, accel_xyz_raw=empty,
            green=empty, accel_x=empty, accel_y=empty, accel_z=empty,
            accel_xy=empty, accel_xyz=empty,
            spectra={}, peaks={}, freq_res=TARGET_FS / FFT_SIZE,
        )

    g = resample_linear(sub["green"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)
    ax = resample_linear(sub["accel_x"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)
    ay = resample_linear(sub["accel_y"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)
    az = resample_linear(sub["accel_z"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)

    n_out = len(g)
    t0 = float(sub["timestamp"].iloc[0])
    t_ms = t0 + np.arange(n_out) * (1000.0 / TARGET_FS)

    axy_raw = np.sqrt(ax ** 2 + ay ** 2)
    axyz_raw = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    g_f = filter_ppg(g)
    ax_f = filter_accel(ax, remove_gravity=True)
    ay_f = filter_accel(ay, remove_gravity=True)
    az_f = filter_accel(az, remove_gravity=False)
    axy_f = np.sqrt(ax_f ** 2 + ay_f ** 2)
    axyz_f = np.sqrt(ax_f ** 2 + ay_f ** 2 + az_f ** 2)

    spectra: dict[str, np.ndarray] = {}
    peaks: dict[str, float] = {}
    freq_res = TARGET_FS / FFT_SIZE
    for name, sig in (("green", g_f), ("accel_xy", axy_f), ("accel_z", az_f)):
        mags, freq_res = rfft_emphasis(sig)
        spectra[name] = mags
        f_hz, _ = find_dominant_peak(mags, freq_res)
        peaks[name] = f_hz * 60.0

    return ProcessedWindow(
        t_ms=t_ms,
        green_raw=g, accel_x_raw=ax, accel_y_raw=ay, accel_z_raw=az,
        accel_xy_raw=axy_raw, accel_xyz_raw=axyz_raw,
        green=g_f, accel_x=ax_f, accel_y=ay_f, accel_z=az_f,
        accel_xy=axy_f, accel_xyz=axyz_f,
        spectra=spectra, peaks=peaks, freq_res=freq_res,
    )


def spectrum_freqs_bpm(freq_res: float) -> np.ndarray:
    fft_half = FFT_SIZE // 2 + 1
    return np.arange(fft_half) * freq_res * 60.0


def compute_spectrum(signal: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run emphasis-FFT on a (filtered) signal and return (mags, freq_res, peak_bpm)."""
    mags, freq_res = rfft_emphasis(signal)
    f_hz, _ = find_dominant_peak(mags, freq_res)
    return mags, freq_res, f_hz * 60.0


@dataclass
class BandpassedWindow:
    t_ms: np.ndarray
    green: np.ndarray
    accel_x: np.ndarray
    accel_y: np.ndarray
    accel_z: np.ndarray
    accel_xy: np.ndarray
    accel_xyz: np.ndarray
    filter_label: str


def bandpass_window(df: pd.DataFrame, filter_choice: str = "auto",
                    source_fs: float = SOURCE_FS) -> BandpassedWindow:
    """Resample to TARGET_FS and apply ONLY the selected bandpass — no gravity
    removal or extra high-pass — to mirror the firmware get_hr() filter stage.

    filter_choice: "auto" | "low" | "high"
      auto -> low when std(accel magnitude) < MAX_ACCEL_FOR_LOW_MOVEMENT
      low  -> [0.6, 3.6] Hz
      high -> [0.9, 4.0] Hz
    """
    cols = ["timestamp", "green", "accel_x", "accel_y", "accel_z"]
    sub = df[cols].dropna().sort_values("timestamp").reset_index(drop=True)
    empty = np.array([], dtype=np.float64)
    if sub.empty:
        return BandpassedWindow(
            t_ms=empty, green=empty,
            accel_x=empty, accel_y=empty, accel_z=empty,
            accel_xy=empty, accel_xyz=empty,
            filter_label="(empty window)",
        )

    g = resample_linear(sub["green"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)
    ax = resample_linear(sub["accel_x"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)
    ay = resample_linear(sub["accel_y"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)
    az = resample_linear(sub["accel_z"].to_numpy(dtype=np.float64), source_fs, TARGET_FS)

    accel_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    sos, label = select_bandpass(filter_choice, accel_mag)

    # sosfiltfilt's default padlen is 3 * (2*n_sections + 1). For a 4th-order
    # Butterworth bandpass that's 27 samples — short windows blow up here.
    min_len = 3 * (2 * sos.shape[0] + 1) + 1
    if len(g) < min_len:
        return BandpassedWindow(
            t_ms=empty, green=empty,
            accel_x=empty, accel_y=empty, accel_z=empty,
            accel_xy=empty, accel_xyz=empty,
            filter_label=f"{label} (window too short: {len(g)} < {min_len} samples)",
        )

    g_f = sosfiltfilt(sos, g)
    ax_f = sosfiltfilt(sos, ax)
    ay_f = sosfiltfilt(sos, ay)
    az_f = sosfiltfilt(sos, az)
    axy_f = np.sqrt(ax_f ** 2 + ay_f ** 2)
    axyz_f = np.sqrt(ax_f ** 2 + ay_f ** 2 + az_f ** 2)

    n_out = len(g)
    t0 = float(sub["timestamp"].iloc[0])
    t_ms = t0 + np.arange(n_out) * (1000.0 / TARGET_FS)

    return BandpassedWindow(
        t_ms=t_ms,
        green=g_f, accel_x=ax_f, accel_y=ay_f, accel_z=az_f,
        accel_xy=axy_f, accel_xyz=axyz_f,
        filter_label=label,
    )
