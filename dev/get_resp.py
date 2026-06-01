"""
Single-window respiration-rate estimator from green PPG.

Mirrors firmware/respiration.c: extract cardiac instantaneous phase via
a narrowband complex resonator, take its derivative (Savitzky-Golay) to
get instantaneous heart rate, bandpass that signal in the resp band to
isolate RSA, then take the resp-band instantaneous phase and count
breath cycles over the window.

Pipeline (30 s @ 32 Hz):
  1. cardiac bandpass [0.5, 3.5] Hz on raw PPG (filtfilt)
  2. decimate 32 -> 16 Hz
  3. reflect-pad 480 -> 1024 samples
  4. cardiac inst. phase via narrowband resonator (1.5 Hz center, 3.0 BW)
  5. dphi/dt via Savitzky-Golay derivative (window=17, polyorder=2)
  6. scale to BPM
  7. respiratory bandpass [0.1, 0.6] Hz on the BPM signal -> RSA
  8. resp inst. phase via narrowband resonator (0.25 Hz center, 0.5 BW)
  9. rate = (phase[end] - phase[start]) / (2*pi) * 60 / window_s

Stateless (matches firmware compute path; not the EMA wrapper).
Accel inputs are accepted for signature compatibility and ignored.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import (
    butter,
    decimate,
    hilbert,
    savgol_filter,
    sosfiltfilt,
)

# ---------------------------------------------------------------------------
# Constants (match firmware/respiration.h)
# ---------------------------------------------------------------------------
SOURCE_FS = 32.0
DS_FS = 16.0
WINDOW_S = 30.0
DOWNSAMPLED_N = int(WINDOW_S * DS_FS)   # 480
PADDED_N = 1024
PAD_LEN = (PADDED_N - DOWNSAMPLED_N) // 2  # 272

CARD_BAND_HZ = (0.5, 3.5)
RESP_BAND_HZ = (0.1, 0.6)

CARDIAC_CENTER_HZ = 1.5
CARDIAC_BW_HZ = 3.0
RESP_CENTER_HZ = 0.25
RESP_BW_HZ = 0.5

SAVGOL_WINDOW = 17
SAVGOL_POLYORDER = 2

SOS_CARD = butter(4, list(CARD_BAND_HZ), btype="bandpass", fs=SOURCE_FS, output="sos")
SOS_RESP = butter(4, list(RESP_BAND_HZ), btype="bandpass", fs=DS_FS, output="sos")

# Plausible-rate gate.
RESP_MIN_BRPM = 6.0
RESP_MAX_BRPM = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resonator_phase(x: np.ndarray, fs: float, f0: float, bw: float) -> np.ndarray:
    """Narrowband complex-resonator instantaneous phase, mirroring
    firmware's inst_phase_resonator_block. Implemented as a 4th-order
    Butterworth bandpass [f0 - bw/2, f0 + bw/2] (filtfilt) followed by
    the Hilbert analytic signal -> unwrapped phase. Equivalent in the
    passband; phase wrap continuity matters more than the exact filter
    family.
    """
    lo = max(0.01, f0 - bw / 2.0)
    hi = min(fs / 2.0 - 0.01, f0 + bw / 2.0)
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    filt = sosfiltfilt(sos, x)
    return np.unwrap(np.angle(hilbert(filt)))


def _reflect_pad(x: np.ndarray, pad_len: int) -> np.ndarray:
    """Reflect-pad a 1-D signal by `pad_len` on each side, excluding the
    edge sample (numpy 'reflect' / firmware's reflect_pad)."""
    return np.pad(x, pad_len, mode="reflect")


def _resample_linear(x: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    n_in = len(x)
    if n_in == 0 or target_fs <= 0:
        return np.zeros(0, dtype=np.float64)
    duration = (n_in - 1) / source_fs
    n_out = int(duration * target_fs) + 1
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)
    ratio = source_fs / target_fs
    t = np.arange(n_out, dtype=np.float64) * ratio
    idx0 = np.floor(t).astype(int)
    frac = t - idx0
    idx0 = np.clip(idx0, 0, n_in - 1)
    idx1 = np.clip(idx0 + 1, 0, n_in - 1)
    return x[idx0] + frac * (x[idx1] - x[idx0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_resp(ppg_green, accel_x=None, accel_y=None, fs: float = SOURCE_FS) -> float:
    """Compute one respiration rate (brpm) for a window of green PPG.

    ppg_green:   PPG green-channel samples at `fs`.
    accel_x/y:   accepted for API compatibility, ignored.
    fs:          input sample rate. Default 32 Hz; resampled internally.

    Returns 0.0 on early-exit (insufficient data, implausible rate).
    Window must span ≥ WINDOW_S = 30 s.
    """
    del accel_x, accel_y  # unused (firmware uses PPG-only RSA)
    ppg = np.asarray(ppg_green, dtype=np.float64).ravel()
    needed = int(WINDOW_S * fs)
    if len(ppg) < needed:
        return 0.0

    # Trim to last 30 s (firmware operates on a fixed 30 s ring buffer).
    ppg = ppg[-needed:]

    # Step 1: cardiac bandpass at source fs.
    if abs(fs - SOURCE_FS) < 1e-6:
        sos_card = SOS_CARD
    else:
        sos_card = butter(4, list(CARD_BAND_HZ), btype="bandpass", fs=fs, output="sos")
    card_filt = sosfiltfilt(sos_card, ppg)

    # Step 2: decimate to DS_FS.
    if abs(fs - SOURCE_FS) < 1e-6:
        ds = decimate(card_filt, int(SOURCE_FS / DS_FS), ftype="iir", zero_phase=True)
    else:
        ds = _resample_linear(card_filt, fs, DS_FS)
    if len(ds) < int(WINDOW_S * DS_FS) - 2:
        return 0.0
    ds = ds[: int(WINDOW_S * DS_FS)]
    n_ds = len(ds)

    # Step 3: reflect-pad to 1024.
    pad_len = (PADDED_N - n_ds) // 2
    padded = _reflect_pad(ds, pad_len)
    if len(padded) != PADDED_N:
        # Trim/extend defensively.
        if len(padded) > PADDED_N:
            padded = padded[:PADDED_N]
        else:
            padded = np.pad(padded, (0, PADDED_N - len(padded)), mode="edge")

    # Step 4: cardiac instantaneous phase.
    card_phase = _resonator_phase(padded, DS_FS, CARDIAC_CENTER_HZ, CARDIAC_BW_HZ)

    # Step 5: Savitzky-Golay derivative of phase -> rad/sample.
    dphi = savgol_filter(
        card_phase,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        deriv=1,
        delta=1.0,
        mode="interp",
    )

    # Step 6: rad/sample -> BPM (instantaneous heart rate).
    rad_sample_to_bpm = DS_FS * 60.0 / (2.0 * np.pi)
    inst_hr_bpm = dphi * rad_sample_to_bpm

    # Step 7: respiratory bandpass on instantaneous HR -> RSA signal.
    rsa = sosfiltfilt(SOS_RESP, inst_hr_bpm)

    # Step 8: respiratory instantaneous phase.
    resp_phase = _resonator_phase(rsa, DS_FS, RESP_CENTER_HZ, RESP_BW_HZ)

    # Step 9: count cycles over the trimmed (un-padded) span.
    start = pad_len
    end = pad_len + n_ds - 1
    num_breaths = (resp_phase[end] - resp_phase[start]) / (2.0 * np.pi)
    duration_s = n_ds / DS_FS
    rate = float(abs(num_breaths) * 60.0 / duration_s)

    if rate < RESP_MIN_BRPM or rate > RESP_MAX_BRPM:
        return 0.0
    return rate


def get_resp_geomean(ppg_green, accel_x=None, accel_y=None,
                     fs: float = SOURCE_FS,
                     window_s: float = WINDOW_S, stride_s: float = 5.0) -> float:
    """Slide a `window_s` window over the input with `stride_s` step, run
    get_resp on each, and return the geometric mean of valid (>0) outputs.
    """
    del accel_x, accel_y  # unused
    ppg = np.asarray(ppg_green, dtype=np.float64).ravel()
    win = int(round(window_s * fs))
    step = int(round(stride_s * fs))
    if step < 1 or len(ppg) < win:
        return 0.0

    rrs: list[float] = []
    for start in range(0, len(ppg) - win + 1, step):
        rr = get_resp(ppg[start:start + win], fs=fs)
        if rr > 0.0:
            rrs.append(rr)

    if not rrs:
        return 0.0
    return float(np.exp(np.mean(np.log(rrs))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 32
    n = 60 * fs
    t = np.arange(n) / fs
    # 15 brpm respiration modulating heart rate ~72 bpm via RSA.
    rsa_hz = 1.2 + 0.05 * np.sin(2 * np.pi * 0.25 * t)        # 15 brpm
    pulse_phase = 2 * np.pi * np.cumsum(rsa_hz) / fs
    am_env = 1.0 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    ppg = 1000 * am_env * np.sin(pulse_phase) + 30 * rng.standard_normal(n)
    print(f"resp = {get_resp(ppg):.2f} brpm (expected ~15)")
    print(f"resp_geomean = {get_resp_geomean(ppg):.2f} brpm")
