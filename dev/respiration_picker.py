"""Respiration picking pipeline.

Same structure as hr_picker but tuned to the respiration band (6–30 brpm).

- pick_window(merged_window): run the full pipeline on one window slice.
- pick_all_dense_windows(): iterate dense_windows.csv, load each user's
  sleep CSV, slice each window, and call pick_window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from get_resp import get_resp
from pull_temple_data import fetch_all_signals, get_user_id_by_email

TARGET_FS = 26.0
RESP_BAND_HZ = (0.1, 0.5)

SOS_RESP_BP = butter(4, list(RESP_BAND_HZ), btype="bandpass", fs=TARGET_FS, output="sos")
SOS_RESP_PPG = butter(4, list(RESP_BAND_HZ), btype="bandpass", fs=TARGET_FS, output="sos")

FFT_SIZE = 1024
RESP_MIN_BRPM = 6.0
RESP_MAX_BRPM = 30.0
F_LO = RESP_MIN_BRPM / 60.0
F_HI = RESP_MAX_BRPM / 60.0

DENSE_WINDOWS_CSV = Path("sleep/dense_windows_60.csv")
SLEEP_DATA_DIR = Path("sleep/data")
OUTPUT_CSV = Path("sleep/resp_accel_60.csv")


def preprocess_accel(sig: np.ndarray) -> np.ndarray:
    sig = sig - np.mean(sig)
    return sosfiltfilt(SOS_RESP_BP, sig)


def preprocess_ppg(sig: np.ndarray) -> np.ndarray:
    return sosfiltfilt(SOS_RESP_PPG, sig)


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
    """All ACF local maxima within [F_LO, F_HI], sorted by amplitude desc. Returns brpm list."""
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
    brpms: list[float] = []
    for rel in peak_rel[order]:
        period_s = (k_lo + int(rel)) * lag_step
        if period_s > 0:
            brpms.append(60.0 / period_s)
    return brpms


def pick_window(merged_window: pd.DataFrame) -> dict:
    """Run preprocess → fuse → fft/acf → peak pick on one window slice.

    Returns dict with brpm dominant peaks for green/accel_x/y/z and full peak
    lists for fused_mag/fused_pca.
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

    out["resp_smartfusion"] = get_resp(green, ax, ay, fs=TARGET_FS)

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
            rows.append({
                "email": email,
                "sleep_date": w.get("sleep_date"),
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                **peaks,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = pick_all_dense_windows()
    print(f"Picked respiration for {len(out)} windows")
    if not out.empty:
        print(out.head())
    out.to_csv(OUTPUT_CSV, index=False)
