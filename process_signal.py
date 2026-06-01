import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


class SignalProcessing:
    TARGET_FS = 26.0
    FFT_SIZE = 1024

    GRAVITY_CUTOFF_HZ_ACC = 0.3
    SOS_GRAVITY_ACC = butter(2, GRAVITY_CUTOFF_HZ_ACC, btype="lowpass", fs=TARGET_FS, output="sos")

    RESP_HP_CUTOFF_HZ = 0.5
    SOS_RESP_HP = butter(2, RESP_HP_CUTOFF_HZ, btype="highpass", fs=TARGET_FS, output="sos")

    HR_BAND_HZ = (0.7, 3.5)
    SOS_HR_BP = butter(4, list(HR_BAND_HZ), btype="bandpass", fs=TARGET_FS, output="sos")

    HR_BAND_PPG_HZ = (0.6, 3.6)
    SOS_HR_PPG = butter(4, list(HR_BAND_PPG_HZ), btype="bandpass", fs=TARGET_FS, output="sos")

    @staticmethod
    def filter_accel(sig):
        sig = sig - np.mean(sig)
        sig = sig - sosfiltfilt(SignalProcessing.SOS_GRAVITY_ACC, sig)
        sig = sosfiltfilt(SignalProcessing.SOS_RESP_HP, sig)
        sig = sosfiltfilt(SignalProcessing.SOS_HR_BP, sig)
        return sig
    
    @staticmethod
    def filter_ppg(sig):
        sig = sosfiltfilt(SignalProcessing.SOS_HR_PPG, sig)
        return sig


    @staticmethod
    def padded(x: np.ndarray) -> tuple[np.ndarray, int]:
        """Zero-pad x into a length-FFT_SIZE buffer. Returns (buf, n_used)."""
        buf = np.zeros(SignalProcessing.FFT_SIZE, dtype=np.float64)
        n = min(len(x), SignalProcessing.FFT_SIZE)
        buf[:n] = x[:n]
        return buf, n

    @staticmethod
    def rfft_magnitude(x: np.ndarray) -> tuple[np.ndarray, float]:
        """rFFT magnitude with linear ramp window. Returns (mags, freq_res_hz)."""
        buf, n = SignalProcessing.padded(x)
        if n >= 2:
            buf[:n] *= 0.5 + 0.5 * np.arange(n) / (n - 1)
        mags = np.abs(np.fft.rfft(buf))
        freq_res = SignalProcessing.TARGET_FS / SignalProcessing.FFT_SIZE
        return mags, freq_res
    
    @staticmethod
    def autocorrelation(x: np.ndarray) -> tuple[np.ndarray, float]:
        """Linear ACF via zero-padded FFT. Returns (acf[:n], lag_step_s)."""
        buf, n = SignalProcessing.padded(x)
        lag_step = 1.0 / SignalProcessing.TARGET_FS
        if n == 0:
            return np.zeros(0, dtype=np.float64), lag_step
        spec = np.fft.rfft(buf)
        acf = np.fft.irfft(spec * np.conj(spec), n=SignalProcessing.FFT_SIZE)[:n]
        if acf[0] > 0:
            acf = acf / acf[0]
        return acf, lag_step
    

class PeakPicking:
    HR_MIN_BPM = 40.0
    HR_MAX_BPM = 180.0
    F_LO = HR_MIN_BPM / 60.0
    F_HI = HR_MAX_BPM / 60.0

    @staticmethod
    def find_dominant_peak_fft(mags: np.ndarray, freq_res: float) -> float:
        """Argmax of mags over [F_LO, F_HI]. Returns freq in Hz."""
        k_lo = max(0, int(PeakPicking.F_LO / freq_res))
        k_hi = min(len(mags) - 1, int(PeakPicking.F_HI / freq_res))
        if k_hi < k_lo:
            return 0.0
        k = int(np.argmax(mags[k_lo:k_hi + 1])) + k_lo
        return k * freq_res * 60
    
    def find_all_peaks_acf(acf: np.ndarray, lag_step: float) -> list[float]:
        """All ACF local maxima within [F_LO, F_HI], sorted by amplitude desc. Returns bpm list."""
        n = len(acf)
        if n < 2:
            return []
        k_lo = max(1, int((1.0 / PeakPicking.F_HI) / lag_step))
        k_hi = min(n - 1, int((1.0 / PeakPicking.F_LO) / lag_step))
        if k_hi <= k_lo:
            return []
        seg = acf[k_lo:k_hi + 1]
        peak_rel, _ = find_peaks(seg)
        if peak_rel.size == 0:
            return []
        order = np.argsort(seg[peak_rel])[::-1]
        bpms: list[float] = []
        for rel in peak_rel[order]:
            period_s = (k_lo + int(rel)) * lag_step
            if period_s > 0:
                bpms.append(60.0 / period_s)
        return bpms



def predict_hr_accel(accel_x, accel_y):    
    # Bandpass
    accel_x = SignalProcessing.filter_accel(accel_x)
    accel_y = SignalProcessing.filter_accel(accel_y)

    # Fusion 
    fusion_mag = np.sqrt(accel_x**2 + accel_y**2)
    
    M = np.column_stack([accel_x, accel_y])
    cov = (M.T @ M) / max(len(M) - 1, 1)
    w, V = np.linalg.eigh(cov)
    fusion_pca = V[:, np.argmax(w)]

    # Autocorrelation
    acf_mag, lag_step = SignalProcessing.autocorrelation(fusion_mag)
    acf_pca, _ = SignalProcessing.autocorrelation(fusion_pca)

    # Peaks
    peaks_mag = PeakPicking.find_all_peaks_acf(acf_mag, lag_step)
    peaks_pca = PeakPicking.find_all_peaks_acf(acf_pca, lag_step)
    peaks = np.array(list(peaks_mag) + list(peaks_pca))
    return peaks
    if peaks.size == 0:
        return float(last_hr)
    # return float(peaks[np.argmin(np.abs(peaks - last_hr))])


def predict_hr_ppg(green, t_ppg_ms, t_accel_ms):
    green = np.interp(t_accel_ms, t_ppg_ms, green)

    # Bandpass
    green = SignalProcessing.filter_ppg(green)

    # FFT
    fft_mag, freq_res = SignalProcessing.rfft_magnitude(green)

    # Peak
    return PeakPicking.find_dominant_peak_fft(fft_mag, freq_res)