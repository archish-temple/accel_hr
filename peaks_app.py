"""Streamlit app: given a CSV path and a start time, run the notebook's
filter+FFT pipeline over a fixed 15s window and show the dominant peak per
signal alongside the firmware HR (resolved via the email encoded in the
CSV filename).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt

from pull_temple_data import get_firmware_hr, get_user_id_by_email

WINDOW_SECONDS_DEFAULT = 15
WINDOW_SECONDS_MIN = 5
WINDOW_SECONDS_MAX = 60
SOURCE_FS = 32.0
TARGET_FS = 26.0
FFT_SIZE = 1024
HR_MIN_BPM = 36.0
HR_MAX_BPM = 220.0
FREQ_MIN_HZ = HR_MIN_BPM / 60.0
FREQ_MAX_HZ = HR_MAX_BPM / 60.0

SOS_BP_09_40 = butter(4, [0.9, 4.0], btype="bandpass", fs=TARGET_FS, output="sos")
SOS_BP_06_36 = butter(4, [0.6, 3.6], btype="bandpass", fs=TARGET_FS, output="sos")
SOS_LP_GRAVITY = butter(2, 0.3, btype="lowpass", fs=TARGET_FS, output="sos")
SOS_HP_ACCEL = butter(2, 0.5, btype="highpass", fs=TARGET_FS, output="sos")


def resample_linear(x: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
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


def rfft_magnitude_emphasis(x: np.ndarray) -> tuple[np.ndarray, float]:
    fft_in = np.zeros(FFT_SIZE, dtype=np.float64)
    n = min(len(x), FFT_SIZE)
    fft_in[:n] = x[:n]
    fft_in[:n] -= fft_in[:n].mean()
    if n >= 2:
        fft_in[:n] *= 0.5 + 0.5 * np.arange(n) / (n - 1)
    mags = np.abs(np.fft.rfft(fft_in))
    return mags, TARGET_FS / FFT_SIZE


def find_dominant_peak(mags: np.ndarray, freq_res: float, f_lo: float, f_hi: float) -> float:
    fft_half = FFT_SIZE // 2 + 1
    k_lo = max(0, int(f_lo / freq_res))
    k_hi = min(fft_half - 1, int(f_hi / freq_res))
    if k_hi < k_lo:
        return 0.0
    k = int(np.argmax(mags[k_lo:k_hi + 1])) + k_lo
    return k * freq_res


def email_from_filename(csv_path: str) -> str | None:
    stem = Path(csv_path).stem.rsplit("_", 1)[0]
    if "_at_" not in stem:
        return None
    local, domain = stem.split("_at_", 1)
    return f"{local}@{domain.replace('_', '.')}"


@st.cache_data(show_spinner="Loading CSV…")
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_ist"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    ).dt.tz_convert("Asia/Kolkata")
    return df


@st.cache_data(show_spinner="Fetching firmware HR…")
def fetch_firmware_hr_mean(email: str, start_iso: str, end_iso: str) -> float:
    user_id = get_user_id_by_email(email)
    if not user_id:
        return 0.0
    hr = get_firmware_hr(user_id, start_iso, end_iso)
    if hr is None or hr.empty:
        return 0.0
    return float(hr["firmware_hr"].mean())


def compute_peaks(window: pd.DataFrame) -> dict:
    green = resample_linear(window["green"].to_numpy(), SOURCE_FS, TARGET_FS)
    ax = resample_linear(window["accel_x"].to_numpy(), SOURCE_FS, TARGET_FS)
    ay = resample_linear(window["accel_y"].to_numpy(), SOURCE_FS, TARGET_FS)
    az = resample_linear(window["accel_z"].to_numpy(), SOURCE_FS, TARGET_FS)

    t_start_ms = float(window["timestamp"].iloc[0])
    n_out = len(green)
    t_ms = t_start_ms + np.arange(n_out) * (1000.0 / TARGET_FS)

    sos = SOS_BP_06_36 if float(np.std(az)) < 350.0 else SOS_BP_09_40
    sos_label = "[0.6, 3.6] Hz" if sos is SOS_BP_06_36 else "[0.9, 4.0] Hz"

    green_f = sosfiltfilt(sos, green)
    ax_f = sosfiltfilt(sos, sosfiltfilt(SOS_HP_ACCEL, ax - sosfiltfilt(SOS_LP_GRAVITY, ax)))
    ay_f = sosfiltfilt(sos, sosfiltfilt(SOS_HP_ACCEL, ay - sosfiltfilt(SOS_LP_GRAVITY, ay)))
    az_f = sosfiltfilt(sos, az)
    axy_f = np.sqrt(ax_f ** 2 + ay_f ** 2)

    sigs = {"green": green_f, "accel_x": ax_f, "accel_y": ay_f, "accel_z": az_f, "accel_xy": axy_f}
    peaks = {}
    spectra = {}
    freq_res = TARGET_FS / FFT_SIZE
    for name, sig in sigs.items():
        mags, freq_res = rfft_magnitude_emphasis(sig)
        peak_hz = find_dominant_peak(mags, freq_res, FREQ_MIN_HZ, FREQ_MAX_HZ)
        peaks[name] = peak_hz * 60.0
        spectra[name] = mags

    resampled = {"green": green, "accel_x": ax, "accel_y": ay, "accel_z": az}
    return {
        "peaks": peaks,
        "spectra": spectra,
        "freq_res": freq_res,
        "filter": sos_label,
        "resampled": resampled,
        "t_ms": t_ms,
    }


SIGNAL_COLORS = {
    "green": "#2ca02c",
    "accel_x": "#1f77b4",
    "accel_y": "#ff7f0e",
    "accel_z": "#9467bd",
    "accel_xy": "#17becf",
}


def resampled_figure(resampled: dict, t_ms: np.ndarray) -> go.Figure:
    t_ist = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert("Asia/Kolkata")
    names = ["green", "accel_x", "accel_y", "accel_z"]
    fig = make_subplots(
        rows=len(names), cols=1, shared_xaxes=True,
        subplot_titles=names, vertical_spacing=0.04,
    )
    for i, name in enumerate(names, start=1):
        fig.add_trace(
            go.Scatter(x=t_ist, y=resampled[name], name=name,
                       line=dict(color=SIGNAL_COLORS[name], width=1)),
            row=i, col=1,
        )
    fig.update_xaxes(title_text="time (IST)", row=len(names), col=1)
    fig.update_layout(
        height=160 * len(names) + 60,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def spectra_figure(spectra: dict, peaks: dict, freq_res: float, firmware_hr: float) -> go.Figure:
    fft_half = FFT_SIZE // 2 + 1
    freqs_bpm = np.arange(fft_half) * freq_res * 60.0
    band = (freqs_bpm >= HR_MIN_BPM) & (freqs_bpm <= HR_MAX_BPM)

    names = ["green", "accel_x", "accel_y", "accel_z", "accel_xy"]
    fig = make_subplots(
        rows=len(names), cols=1, shared_xaxes=True,
        subplot_titles=[f"{n} spectrum" for n in names],
        vertical_spacing=0.04,
    )
    for i, name in enumerate(names, start=1):
        mags = spectra[name]
        peak_bpm = peaks[name]
        color = SIGNAL_COLORS[name]
        y_max = float(mags[band].max()) if mags[band].size else 0.0
        fig.add_trace(
            go.Scatter(x=freqs_bpm[band], y=mags[band], name=name,
                       line=dict(color=color, width=1)),
            row=i, col=1,
        )
        fig.add_vline(x=peak_bpm, line_dash="dash", line_color=color, row=i, col=1)
        fig.add_annotation(
            x=peak_bpm, y=y_max, text=f"{peak_bpm:.1f}",
            font=dict(color=color), showarrow=True, arrowhead=2,
            ax=60, ay=-25, row=i, col=1,
        )
        if firmware_hr > 0:
            fig.add_vline(x=firmware_hr, line_dash="solid", line_color="#ff1493", row=i, col=1)

    fig.update_xaxes(title_text="bpm", row=len(names), col=1)
    fig.update_layout(
        height=180 * len(names) + 60,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        title_text=f"Firmware HR: {firmware_hr:.1f} bpm" if firmware_hr > 0 else "Firmware HR: n/a",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="AccelHR Peaks", layout="wide")
    st.title("AccelHR Peak Detector")

    with st.sidebar:
        st.header("Inputs")
        csv_path = st.text_input(
            "CSV path",
            value="sleep/data/aadhiraj_at_temple_com_2026-05-05.csv",
            help="Absolute or relative to the project root.",
        )
        start_date = st.date_input("Start date", value=dt.date(2026, 5, 5))
        start_time_text = st.text_input("Start time (HH:MM:SS)", value="22:31:00")
        window_seconds = st.slider(
            "Window length (s)",
            min_value=WINDOW_SECONDS_MIN,
            max_value=WINDOW_SECONDS_MAX,
            value=WINDOW_SECONDS_DEFAULT,
            step=1,
        )
        fetch_fw = st.checkbox("Fetch firmware HR", value=True)

    if not csv_path or not os.path.exists(csv_path):
        st.error(f"CSV not found: {csv_path}")
        return

    try:
        start_time = dt.datetime.strptime(start_time_text.strip(), "%H:%M:%S").time()
    except ValueError:
        st.error("Start time must be HH:MM:SS")
        return

    tz = "Asia/Kolkata"
    is_sleep = "sleep" in Path(csv_path).parts
    effective_date = start_date + dt.timedelta(days=1) if is_sleep and start_time.hour < 12 else start_date
    start_ts = pd.Timestamp.combine(effective_date, start_time).tz_localize(tz)
    end_ts = start_ts + pd.Timedelta(seconds=window_seconds)

    df = load_csv(csv_path)
    window = df[(df["timestamp_ist"] >= start_ts) & (df["timestamp_ist"] < end_ts)].reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Window", f"{start_ts.strftime('%H:%M:%S')} → {end_ts.strftime('%H:%M:%S')}")
    c2.metric("Samples", f"{len(window):,}")
    c3.metric("Duration", f"{window_seconds} s")

    if window.empty:
        st.warning("No samples in this window.")
        return

    required = {"green", "accel_x", "accel_y", "accel_z"}
    missing = required - set(window.columns)
    if missing:
        st.error(f"CSV missing required columns: {sorted(missing)}")
        return

    result = compute_peaks(window)
    st.caption(f"Bandpass selected: **{result['filter']}**")

    firmware_hr = 0.0
    if fetch_fw:
        email = email_from_filename(csv_path)
        if email is None:
            st.info("Filename doesn't encode an email (`local_at_domain_..._YYYY-MM-DD`); skipping firmware HR.")
        else:
            st.caption(f"Email: `{email}`")
            try:
                firmware_hr = fetch_firmware_hr_mean(
                    email,
                    start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                )
            except Exception as e:
                st.warning(f"Firmware HR fetch failed: {e}")

    peaks_df = pd.DataFrame(
        [{"signal": k, "peak_bpm": round(v, 2)} for k, v in result["peaks"].items()]
    )
    st.subheader("Dominant peaks")
    st.dataframe(peaks_df, use_container_width=True, hide_index=True)

    st.subheader("Resampled signals (26 Hz)")
    st.plotly_chart(
        resampled_figure(result["resampled"], result["t_ms"]),
        use_container_width=True,
    )

    st.subheader("Spectra")
    fig = spectra_figure(result["spectra"], result["peaks"], result["freq_res"], firmware_hr)
    png_name = f"{Path(csv_path).stem}_{start_ts.strftime('%Y%m%d_%H%M%S')}_fw{firmware_hr:.0f}.png"
    try:
        png_bytes = fig.to_image(format="png", width=1200, height=180 * 5 + 60, scale=2)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=png_name,
            mime="image/png",
            use_container_width=False,
        )
    except Exception:
        pass
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
