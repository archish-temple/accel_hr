"""HR picking pipeline.

Preprocess each green/accel signal, fuse accel x/y, and pick peaks via
FFT (raw channels) or autocorrelation (fused channels).

- pick_window(merged_window): run the full pipeline on one window slice.
- pick_all_dense_windows(): iterate dense_windows.csv, load each user's
  sleep CSV, slice each window, and call pick_window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from pull_temple_data import fetch_all_signals, get_firmware_hr, get_user_id_by_email
from get_hr import get_hr_geomean

TARGET_FS = 26.0
GRAVITY_CUTOFF_HZ_ACC = 0.3
RESP_HP_CUTOFF_HZ = 0.5
HR_BAND_HZ = (0.7, 3.5)
HR_BAND_PPG_HZ = (0.6, 3.6)

SOS_GRAVITY_ACC = butter(2, GRAVITY_CUTOFF_HZ_ACC, btype="lowpass", fs=TARGET_FS, output="sos")
SOS_RESP_HP = butter(2, RESP_HP_CUTOFF_HZ, btype="highpass", fs=TARGET_FS, output="sos")
SOS_HR_BP = butter(4, list(HR_BAND_HZ), btype="bandpass", fs=TARGET_FS, output="sos")
SOS_HR_PPG = butter(4, list(HR_BAND_PPG_HZ), btype="bandpass", fs=TARGET_FS, output="sos")

FFT_SIZE = 1024
HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0
F_LO = HR_MIN_BPM / 60.0
F_HI = HR_MAX_BPM / 60.0

DENSE_WINDOWS_CSV = Path("sleep/dense_windows_15.csv")
SLEEP_DATA_DIR = Path("sleep/data")


def preprocess_accel(sig: np.ndarray) -> np.ndarray:
    sig = sig - np.mean(sig)
    sig = sig - sosfiltfilt(SOS_GRAVITY_ACC, sig)
    sig = sosfiltfilt(SOS_RESP_HP, sig)
    return sosfiltfilt(SOS_HR_BP, sig)


def preprocess_ppg(sig: np.ndarray) -> np.ndarray:
    return sosfiltfilt(SOS_HR_PPG, sig)


def fusion_magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(x ** 2 + y ** 2)


def fusion_pca(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """First principal component of [x, y] — axis with most variance."""
    M = np.column_stack([x, y])
    cov = (M.T @ M) / max(len(M) - 1, 1)
    w, V = np.linalg.eigh(cov)
    pc1 = V[:, np.argmax(w)]
    return M @ pc1


def _padded(x: np.ndarray) -> tuple[np.ndarray, int]:
    buf = np.zeros(FFT_SIZE, dtype=np.float64)
    n = min(len(x), FFT_SIZE)
    buf[:n] = x[:n]
    return buf, n


def rfft_magnitude(x: np.ndarray) -> tuple[np.ndarray, float]:
    buf, n = _padded(x)
    if n >= 2:
        buf[:n] *= 0.5 + 0.5 * np.arange(n) / (n - 1)
    mags = np.abs(np.fft.rfft(buf))
    freq_res = TARGET_FS / FFT_SIZE
    return mags, freq_res


def find_dominant_peak(mags: np.ndarray, freq_res: float) -> float:
    """Argmax of mags over [F_LO, F_HI]. Returns freq in Hz."""
    k_lo = max(0, int(F_LO / freq_res))
    k_hi = min(len(mags) - 1, int(F_HI / freq_res))
    if k_hi < k_lo:
        return 0.0
    k = int(np.argmax(mags[k_lo:k_hi + 1])) + k_lo
    return k * freq_res


def autocorrelation(x: np.ndarray) -> tuple[np.ndarray, float]:
    buf, n = _padded(x)
    lag_step = 1.0 / TARGET_FS
    if n == 0:
        return np.zeros(0, dtype=np.float64), lag_step
    spec = np.fft.rfft(buf)
    acf = np.fft.irfft(spec * np.conj(spec), n=FFT_SIZE)[:n]
    if acf[0] > 0:
        acf = acf / acf[0]
    return acf, lag_step


def find_all_peaks_acf(acf: np.ndarray, lag_step: float) -> list[float]:
    """All ACF local maxima within [F_LO, F_HI], sorted by amplitude desc. Returns bpm list."""
    n = len(acf)
    if n < 2:
        return []
    k_lo = max(1, int((1.0 / F_HI) / lag_step))
    k_hi = min(n - 1, int((1.0 / F_LO) / lag_step))
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


def pick_window(merged_window: pd.DataFrame) -> dict:
    """Run preprocess → fuse → fft/acf → peak pick on one window slice.

    `merged_window` columns: timestamp (ms), green, accel_x, accel_y, accel_z
    (NaN where the channel didn't sample). Returns dict with bpm dominant peaks
    for green/accel_x/y/z and full peak lists for fused_mag/fused_pca.
    """
    ppg = merged_window.loc[merged_window["green"].notna(), ["timestamp", "green"]]
    acc = merged_window.loc[
        merged_window["accel_x"].notna()
        & merged_window["accel_y"].notna()
        & merged_window["accel_z"].notna(),
        ["timestamp", "accel_x", "accel_y", "accel_z"],
    ]
    if ppg.empty or acc.empty:
        return {}

    t_ppg = ppg["timestamp"].to_numpy()
    t_acc = acc["timestamp"].to_numpy()
    green = np.interp(t_acc, t_ppg, ppg["green"].to_numpy())
    ax = acc["accel_x"].to_numpy()
    ay = acc["accel_y"].to_numpy()
    az = acc["accel_z"].to_numpy()

    green_f = preprocess_ppg(green)
    ax_f = preprocess_accel(ax)
    ay_f = preprocess_accel(ay)
    az_f = preprocess_accel(az)

    fmag = fusion_magnitude(ax_f, ay_f)
    fpca = fusion_pca(ax_f, ay_f)

    out: dict = {}
    for name, sig in (("green", green_f), ("accel_x", ax_f), ("accel_y", ay_f), ("accel_z", az_f)):
        mags, fr = rfft_magnitude(sig)
        out[name] = find_dominant_peak(mags, fr) * 60.0

    for name, sig in (("fused_mag", fmag), ("fused_pca", fpca)):
        acf, ls = autocorrelation(sig)
        out[f"{name}_peaks"] = find_all_peaks_acf(acf, ls)

    # accel_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    # out["hr_geomean"] = get_hr_geomean(green, accel_mag, fs=TARGET_FS)

    return out


def _sleep_csv_path(email: str, sleep_start, sleep_end, data_dir: Path) -> Path:
    s = pd.Timestamp(sleep_start)
    e = pd.Timestamp(sleep_end)
    name = f"{email}_{s.strftime('%Y-%m-%dT%H:%M:%S')}_{e.strftime('%Y-%m-%dT%H:%M:%S')}"
    return data_dir / name


def pick_all_dense_windows(
    dense_windows_csv: Path = DENSE_WINDOWS_CSV,
    data_dir: Path = SLEEP_DATA_DIR,
) -> pd.DataFrame:
    """For each (user, window) in dense_windows_csv, load the matching sleep CSV,
    slice [window_start, window_end), and run pick_window. Returns one row per
    window with columns: email, sleep_date, window_start, window_end, green,
    accel_x, accel_y, accel_z, fused_mag_peaks, fused_pca_peaks.
    """
    windows = pd.read_csv(dense_windows_csv, skipinitialspace=True)
    windows.columns = windows.columns.str.strip()
    for c in ("email", "sleep_date"):
        if c in windows.columns and windows[c].dtype == object:
            windows[c] = windows[c].str.strip()
    for c in ("sleep_start", "sleep_end", "window_start", "window_end"):
        windows[c] = pd.to_datetime(windows[c])
    rows: list[dict] = []

    grouped = windows.groupby(["email", "sleep_start", "sleep_end"], sort=False)
    for (email, sleep_start, sleep_end), grp in grouped:
        path = _sleep_csv_path(email, sleep_start, sleep_end, Path(data_dir))
        if not path.exists():
            data = fetch_all_signals(
                        get_user_id_by_email(email),
                        sleep_start,
                        sleep_end,
                    )
            data.to_csv(path, index=False)

        data = pd.read_csv(path)
        ts_ms = pd.to_numeric(data["timestamp"], errors="coerce").astype("int64").to_numpy()

        user_id = get_user_id_by_email(email)

        for _, w in grp.iterrows():
            ws = pd.Timestamp(w["window_start"]).floor("s")
            we = pd.Timestamp(w["window_end"]).floor("s")
            ws_ms = int(ws.value // 1_000_000)
            we_ms = int(we.value // 1_000_000)
            mask = (ts_ms >= ws_ms) & (ts_ms < we_ms)
            slc = data.loc[mask]
            peaks = pick_window(slc)
            if not peaks:
                continue

            firmware_hr = float("nan")
            if "firmware_hr" in slc.columns:
                fw_series = pd.to_numeric(slc["firmware_hr"], errors="coerce").dropna()
                if not fw_series.empty:
                    firmware_hr = float(fw_series.mean())

            rows.append({
                "email": email,
                "sleep_date": w.get("sleep_date"),
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                **peaks,
                "firmware_hr": firmware_hr,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = pick_all_dense_windows()
    print(f"Picked HR for {len(out)} windows")
    if not out.empty:
        print(out.head())
    out.to_csv('sleep/hr_accel_15.csv', index=False)
