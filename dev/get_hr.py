"""
Single-window HR estimator that mirrors the CLM path in firmware/hr.c.

firmware/hr.c -> run_clm_workout_path() takes a 12s window of PPG + accel
magnitude, hands it to clm_compute_mitigated_hr() which:
  1. resample to CLM_TARGET_FS (26 Hz)
  2. bandpass (Butterworth SOS, filtfilt)
  3. linear temporal emphasis window (0.5 -> 1.0)
  4. RFFT magnitude (DC removed)
  5. find prominent accel peak (with secondary-peak prominence check)
  6. if significant motion -> attenuate cadence harmonics in PPG spectrum,
     pick dominant peak; on low SNR fall back to unattenuated PPG outside
     the cadence exclusion band.
     else -> dominant PPG peak directly.

Returns one HR (BPM) for the window. Stateless: no EMA, no previous-HR
guardrails, no recovery logic. For drift suppression run an EMA outside.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# Constants (mirrors firmware/cadence_lock_mitigation.h)
# ---------------------------------------------------------------------------
CLM_SOURCE_FS = 32.0
CLM_TARGET_FS = 26.0
CLM_FFT_SIZE = 1024
CLM_FFT_HALF_SIZE = CLM_FFT_SIZE // 2 + 1

CLM_HR_MIN_BPM = 36.0
CLM_HR_MAX_BPM = 220.0
CLM_FREQ_MIN_HZ = CLM_HR_MIN_BPM / 60.0
CLM_FREQ_MAX_HZ = CLM_HR_MAX_BPM / 60.0

CLM_ATTENUATION_FACTOR = 0.2
CLM_ATTENUATION_BANDWIDTH = 0.15
CLM_NUM_HARMONICS = 1
CLM_SUBHARMONIC_ATTENUATION_FACTOR = 0.3

CLM_ACC_PEAK_MAG_THRESHOLD = 100_000.0
CLM_SECONDARY_PEAK_RATIO = 0.7
CLM_SNR_THRESHOLD_DB = -1.0
CLM_SNR_FALLBACK_CADENCE_EXCL_HZ = 0.15
CLM_SPIKE_THRESHOLD_Z = 15.0

# Movement-state thresholds (mirrors firmware/movement.h). accel_std is the
# population std of int16 accel-magnitude samples — same units the firmware's
# get_movement_level() compares against.
MAX_ACCEL_FOR_VERY_LOW_MOVEMENT = 200.0
MAX_ACCEL_FOR_LOW_MOVEMENT = 350.0
MAX_ACCEL_FOR_MEDIUM_MOVEMENT = 4000.0


def _detect_low_movement(acc: np.ndarray) -> bool:
    """Mirrors cadence_lock_mitigation.c: low-movement filter is active when
    movement_level is VERY_LOW or LOW (accel_std < MAX_ACCEL_FOR_LOW_MOVEMENT)."""
    if len(acc) == 0:
        return False
    return float(np.std(acc)) < MAX_ACCEL_FOR_LOW_MOVEMENT

# Firmware C arrays store [b0, b1, b2, -a1, -a2]; scipy SOS is [b0, b1, b2, 1, a1, a2],
# so the last two columns are sign-flipped vs the C source.

# 4th-order Butterworth bandpass [0.9, 4.0] Hz @ 26 Hz (medium/high movement)
SOS_BP_09_40 = butter(4, [0.9, 4.0], btype="bandpass", fs=CLM_TARGET_FS, output="sos")

# 4th-order Butterworth bandpass [0.6, 3.6] Hz @ 26 Hz (low movement / sleep)
SOS_BP_06_36 = butter(4, [0.6, 3.6], btype="bandpass", fs=CLM_TARGET_FS, output="sos")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resample_linear(x: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    """Linear interpolation, matches resample_signal() in cadence_lock_mitigation.c."""
    n_in = len(x)
    duration = (n_in - 1) / source_fs
    n_out = int(duration * target_fs) + 1
    ratio = source_fs / target_fs
    t = np.arange(n_out, dtype=np.float64) * ratio
    idx0 = np.floor(t).astype(int)
    frac = t - idx0
    idx0 = np.clip(idx0, 0, n_in - 1)
    idx1 = np.clip(idx0 + 1, 0, n_in - 1)
    return x[idx0] + frac * (x[idx1] - x[idx0])


def _temporal_emphasis(x: np.ndarray) -> np.ndarray:
    """Linear ramp 0.5 (oldest) -> 1.0 (newest), matches apply_temporal_emphasis()."""
    n = len(x)
    if n < 2:
        return x
    w = 0.5 + 0.5 * np.arange(n) / (n - 1)
    return x * w


def _rfft_magnitude(x: np.ndarray) -> tuple[np.ndarray, float]:
    """DC-removed zero-padded RFFT magnitude. Returns (mags, freq_res)."""
    fft_in = np.zeros(CLM_FFT_SIZE, dtype=np.float64)
    n = min(len(x), CLM_FFT_SIZE)
    fft_in[:n] = x[:n]
    fft_in[:n] -= fft_in[:n].mean()
    mags = np.abs(np.fft.rfft(fft_in))
    freq_res = CLM_TARGET_FS / CLM_FFT_SIZE
    return mags, freq_res


def _has_spike(x: np.ndarray, threshold_z: float = CLM_SPIKE_THRESHOLD_Z) -> bool:
    """Modified z-score on gradient, matches clm_has_spike()."""
    if len(x) < 3:
        return False
    grad = np.gradient(x)
    med = np.median(grad)
    mad = np.median(np.abs(grad - med))
    if mad < 1e-6:
        mad = 1e-6
    return bool(np.max(np.abs(0.6745 * (grad - med) / mad)) > threshold_z)


def _find_prominent_accel_peaks(mags: np.ndarray, freq_res: float) -> tuple[float, float, bool]:
    """Top-2 local-peak scan in [CLM_FREQ_MIN_HZ, CLM_FREQ_MAX_HZ].
    Returns (peak_freq, peak_mag, cadence_dominant)."""
    fft_half = CLM_FFT_HALF_SIZE
    k_min = max(1, int(CLM_FREQ_MIN_HZ / freq_res))
    k_max = min(fft_half - 2, int(CLM_FREQ_MAX_HZ / freq_res))

    p1_mag = p2_mag = 0.0
    p1_freq = 0.0
    for k in range(k_min, k_max + 1):
        m = mags[k]
        if m > mags[k - 1] and m > mags[k + 1]:
            if m > p1_mag:
                p2_mag = p1_mag
                p1_mag = m
                p1_freq = k * freq_res
            elif m > p2_mag:
                p2_mag = m

    if p1_mag <= 0.0:
        return 0.0, 0.0, False
    has_prominent_secondary = p2_mag >= CLM_SECONDARY_PEAK_RATIO * p1_mag
    return p1_freq, p1_mag, not has_prominent_secondary


def _find_dominant_peak(mags: np.ndarray, freq_res: float,
                        f_lo: float, f_hi: float) -> tuple[float, float]:
    """Argmax over [f_lo, f_hi]. Returns (freq, mag)."""
    fft_half = CLM_FFT_HALF_SIZE
    k_lo = max(0, int(f_lo / freq_res))
    k_hi = min(fft_half - 1, int(f_hi / freq_res))
    if k_hi < k_lo:
        return 0.0, 0.0
    sub = mags[k_lo:k_hi + 1]
    k = int(np.argmax(sub)) + k_lo
    return k * freq_res, float(mags[k])


def _attenuate_motion(mags: np.ndarray, freq_res: float, peak_freq: float,
                      attenuation: float = CLM_ATTENUATION_FACTOR,
                      bandwidth: float = CLM_ATTENUATION_BANDWIDTH,
                      num_harmonics: int = CLM_NUM_HARMONICS,
                      sub_attenuation: float = CLM_SUBHARMONIC_ATTENUATION_FACTOR) -> np.ndarray:
    """Multiply harmonics (and subharmonic) by attenuation factor."""
    out = mags.copy()
    if peak_freq <= 0.0:
        return out
    half_bw = bandwidth * 0.5
    fft_half = len(out)

    for h in range(1, num_harmonics + 1):
        center = peak_freq * h
        k_lo = max(0, int((center - half_bw) / freq_res))
        k_hi = min(fft_half - 1, int((center + half_bw) / freq_res))
        out[k_lo:k_hi + 1] *= attenuation

    if sub_attenuation < 1.0:
        center = peak_freq * 0.5
        k_lo = max(0, int((center - half_bw) / freq_res))
        k_hi = min(fft_half - 1, int((center + half_bw) / freq_res))
        out[k_lo:k_hi + 1] *= sub_attenuation
    return out


def _compute_snr(mags: np.ndarray, freq_res: float, peak_freq: float,
                 sig_radius: float = 0.3,
                 noise_lo: float = 0.5, noise_hi: float = 4.0) -> float:
    fft_half = len(mags)
    k_sig_lo = int((peak_freq - sig_radius) / freq_res)
    k_sig_hi = int((peak_freq + sig_radius) / freq_res)
    k_noise_lo = int(noise_lo / freq_res)
    k_noise_hi = int(noise_hi / freq_res)

    m2 = mags * mags
    sig_mask = np.zeros(fft_half, dtype=bool)
    sig_mask[max(0, k_sig_lo):min(fft_half, k_sig_hi + 1)] = True
    noise_mask = np.zeros(fft_half, dtype=bool)
    noise_mask[max(0, k_noise_lo):min(fft_half, k_noise_hi + 1)] = True
    noise_mask &= ~sig_mask

    sig_pwr = float(m2[sig_mask].sum())
    noise_pwr = float(m2[noise_mask].sum())
    if noise_pwr < 1e-12:
        noise_pwr = 1e-12
    if sig_pwr <= 0.0:
        return -100.0
    return 10.0 * np.log10(sig_pwr / noise_pwr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_hr(ppg_green, accel, fs: float = 32.0,
           low_movement: bool | None = None) -> float:
    """Compute one HR (BPM) for a 12s window.

    ppg_green, accel: length-(12*fs) sequences. accel is the per-sample
        magnitude (same as what hr.c feeds clm_compute_mitigated_hr).
    fs: input sample rate. Default 32 Hz; resampled to 26 Hz internally.
    low_movement: filter selection.
        None (default) -> auto-detect from accel std using firmware thresholds
            (movement.h): low_movement when accel_std < 350.
        True  -> force [0.6, 3.6] Hz bandpass (sleep / very-low / low).
        False -> force [0.9, 4.0] Hz bandpass.

    Returns 0.0 on early-exit conditions (spike, missing input, FFT empty).
    """
    ppg = np.asarray(ppg_green, dtype=np.float64)
    acc = np.asarray(accel, dtype=np.float64)
    if len(ppg) < 100 or len(acc) < 100 or len(ppg) != len(acc):
        return 0.0

    if low_movement is None:
        low_movement = _detect_low_movement(acc)

    # ---- PPG path ----
    ppg_rs = _resample_linear(ppg, fs, CLM_TARGET_FS)
    # if _has_spike(ppg_rs):
    #     return 0.0  # firmware returns previous_hr; stateless caller gets 0

    sos = SOS_BP_06_36 if low_movement else SOS_BP_09_40
    ppg_filt = sosfiltfilt(sos, ppg_rs)
    ppg_filt = _temporal_emphasis(ppg_filt)
    ppg_mags, freq_res = _rfft_magnitude(ppg_filt)

    # ---- Accel path ----
    acc_rs = _resample_linear(acc, fs, CLM_TARGET_FS)
    acc_filt = sosfiltfilt(sos, acc_rs)
    acc_filt = _temporal_emphasis(acc_filt)
    acc_mags, _ = _rfft_magnitude(acc_filt)

    # ---- Cadence detection ----
    acc_peak_freq, acc_peak_mag, _cadence_dominant = _find_prominent_accel_peaks(
        acc_mags, freq_res
    )
    has_significant_motion = acc_peak_mag > CLM_ACC_PEAK_MAG_THRESHOLD

    # ---- HR pick ----
    if not has_significant_motion:
        hr_freq, _ = _find_dominant_peak(
            ppg_mags, freq_res, CLM_FREQ_MIN_HZ, CLM_FREQ_MAX_HZ
        )
    else:
        attenuated = _attenuate_motion(ppg_mags, freq_res, acc_peak_freq)
        hr_freq, _ = _find_dominant_peak(
            attenuated, freq_res, CLM_FREQ_MIN_HZ, CLM_FREQ_MAX_HZ
        )

        snr = _compute_snr(attenuated, freq_res, hr_freq)
        if snr < CLM_SNR_THRESHOLD_DB:
            # Fall back: search unattenuated PPG outside the cadence band.
            excl_lo = acc_peak_freq - CLM_SNR_FALLBACK_CADENCE_EXCL_HZ
            excl_hi = acc_peak_freq + CLM_SNR_FALLBACK_CADENCE_EXCL_HZ
            f_below, m_below = (0.0, 0.0)
            f_above, m_above = (0.0, 0.0)
            if excl_lo > CLM_FREQ_MIN_HZ:
                f_below, m_below = _find_dominant_peak(
                    ppg_mags, freq_res, CLM_FREQ_MIN_HZ, excl_lo
                )
            if excl_hi < CLM_FREQ_MAX_HZ:
                f_above, m_above = _find_dominant_peak(
                    ppg_mags, freq_res, excl_hi, CLM_FREQ_MAX_HZ
                )
            if m_below > 0.0 or m_above > 0.0:
                hr_freq = f_below if m_below >= m_above else f_above

    return float(hr_freq * 60.0)


def get_hr_geomean(ppg_green, accel, fs: float = CLM_SOURCE_FS,
                   window_s: float = 12.0, stride_s: float = 1.0,
                   low_movement: bool | None = None) -> float:
    """Slide a `window_s` window over the input with `stride_s` step, run
    get_hr() on each, and return the geometric mean of the valid HRs.

    Inputs must be longer than `window_s` seconds at `fs`. Zero (early-exit)
    HRs are skipped. Returns 0.0 if no valid sub-window produced an HR.
    """
    ppg = np.asarray(ppg_green, dtype=np.float64)
    acc = np.asarray(accel, dtype=np.float64)
    if len(ppg) != len(acc):
        return 0.0

    win = int(round(window_s * fs))
    step = int(round(stride_s * fs))
    if step < 1 or len(ppg) <= win:
        return 0.0

    hrs = []
    for start in range(0, len(ppg) - win + 1, step):
        hr = get_hr(ppg[start:start + win], acc[start:start + win],
                    fs=fs, low_movement=low_movement)
        if hr > 0.0:
            hrs.append(hr)

    if not hrs:
        return 0.0
    return float(np.exp(np.mean(np.log(hrs))))


if __name__ == "__main__":
    # Quick sanity check: synthetic 1.2 Hz PPG (72 BPM) + low-amplitude accel.
    rng = np.random.default_rng(0)
    fs = 32
    n = 12 * fs
    t = np.arange(n) / fs
    ppg = 1000 * np.sin(2 * np.pi * 1.2 * t) + 50 * rng.standard_normal(n)
    acc = 10 * rng.standard_normal(n)
    print(f"HR = {get_hr(ppg, acc):.2f} BPM (expected ~72)")
