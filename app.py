"""Streamlit viewer for AccelHR signals.

Pick a mode (Movement / Sleep), choose a CSV under that mode's data dir, set a
time range, and view raw / filtered / bandpass-only stacked plots.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from processing import (
    BandpassedWindow,
    ProcessedWindow,
    bandpass_window,
    process_window,
)

ROOT = Path(__file__).parent
MODE_DIRS = {
    "Movement": ROOT / "movement" / "data",
    "Sleep": ROOT / "sleep" / "data",
}
USECOLS = ["timestamp", "green", "accel_x", "accel_y", "accel_z"]
MAX_PLOT_POINTS = 50_000
ROW_HEIGHT = 180

PPG_SIGNALS = [("green", "#2ca02c")]
ACCEL_SIGNALS = [
    ("accel_x", "#1f77b4"),
    ("accel_y", "#ff7f0e"),
    ("accel_z", "#9467bd"),
    ("accel_xy", "#17becf"),
    ("accel_xyz", "#8c564b"),
]
DEFAULT_SIGNALS = {"green", "accel_z"}

TIME_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?$")


FILE_RE = re.compile(
    r"^(?P<name>.+?)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>\d{2}-\d{2}-\d{2}))?$"
)


def parse_time(text: str) -> dt.time | None:
    m = TIME_RE.match(text.strip())
    if not m:
        return None
    h, mi = int(m.group("h")), int(m.group("m"))
    s = int(m.group("s") or 0)
    if not (0 <= h < 24 and 0 <= mi < 60 and 0 <= s < 60):
        return None
    return dt.time(h, mi, s)


def fmt_time(t: dt.time) -> str:
    return t.strftime("%H:%M:%S")


def prettify_name(stem: str) -> str:
    """Decode the encoded user stem into something readable."""
    if "_at_" in stem:
        local, _, domain = stem.partition("_at_")
        local = local.replace("_plus_", "+")
        return f"{local}@{domain.replace('_', '.')}"
    return stem.replace("_", " ")


def parse_filename(stem: str) -> tuple[str, str] | None:
    m = FILE_RE.match(stem)
    if not m:
        return None
    return m.group("name"), m.group("date")


def list_csvs(base: Path) -> list[Path]:
    """Return all CSV files under base (recursive). Empty list if base missing."""
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.suffix.lower() == ".csv")


def index_files(files: list[Path]) -> dict[str, dict[str, list[Path]]]:
    """Group files by parsed (name, date). Returns {name: {date: [paths]}}.
    Files that don't match the expected pattern are dropped silently.
    """
    out: dict[str, dict[str, list[Path]]] = {}
    for p in files:
        parsed = parse_filename(p.stem)
        if not parsed:
            continue
        name, date_str = parsed
        out.setdefault(name, {}).setdefault(date_str, []).append(p)
    return out


@st.cache_data(show_spinner="Loading CSV…")
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in USECOLS if c in df.columns]
    df = df[cols]
    df["timestamp_ist"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    ).dt.tz_convert("Asia/Kolkata")
    return df


def decimate(t: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / max_points))
    return t[::step], y[::step]


def stacked_figure(signals: list[tuple[np.ndarray, np.ndarray, str, str]]) -> go.Figure:
    n = len(signals)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.04, 0.6 / max(n, 1)),
        subplot_titles=[s[2] for s in signals],
    )
    for i, (t, y, name, color) in enumerate(signals, start=1):
        t_d, y_d = decimate(t, y, MAX_PLOT_POINTS)
        fig.add_trace(
            go.Scattergl(x=t_d, y=y_d, mode="lines", name=name, line=dict(color=color, width=1)),
            row=i,
            col=1,
        )
    fig.update_layout(
        height=ROW_HEIGHT * n + 60,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
        hovermode="x unified",
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return fig


def collect_raw_signals(
    selected: set[str],
    ppg: pd.DataFrame,
    accel: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    signals: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    if not ppg.empty:
        t = ppg["timestamp_ist"].to_numpy()
        for name, color in PPG_SIGNALS:
            if name in selected and name in ppg.columns:
                signals.append((t, ppg[name].to_numpy(), name, color))
    if not accel.empty:
        t = accel["timestamp_ist"].to_numpy()
        for name, color in ACCEL_SIGNALS:
            if name in selected and name in accel.columns:
                signals.append((t, accel[name].to_numpy(), name, color))
    return signals


def collect_filtered_signals(
    selected: set[str],
    pw: ProcessedWindow,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    signals: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    if pw.t_ms.size == 0:
        return signals
    t_filt = pd.to_datetime(pw.t_ms, unit="ms", utc=True).tz_convert("Asia/Kolkata").to_numpy()
    name_to_arr = {
        "green": pw.green,
        "accel_x": pw.accel_x,
        "accel_y": pw.accel_y,
        "accel_z": pw.accel_z,
        "accel_xy": pw.accel_xy,
        "accel_xyz": pw.accel_xyz,
    }
    for name, color in PPG_SIGNALS + ACCEL_SIGNALS:
        if name in selected and name in name_to_arr:
            signals.append((t_filt, name_to_arr[name], name, color))
    return signals


def collect_bandpassed_signals(
    selected: set[str],
    bw: BandpassedWindow,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    signals: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    if bw.t_ms.size == 0:
        return signals
    t_bp = pd.to_datetime(bw.t_ms, unit="ms", utc=True).tz_convert("Asia/Kolkata").to_numpy()
    name_to_arr = {
        "green": bw.green,
        "accel_x": bw.accel_x,
        "accel_y": bw.accel_y,
        "accel_z": bw.accel_z,
        "accel_xy": bw.accel_xy,
        "accel_xyz": bw.accel_xyz,
    }
    for name, color in PPG_SIGNALS + ACCEL_SIGNALS:
        if name in selected and name in name_to_arr:
            signals.append((t_bp, name_to_arr[name], name, color))
    return signals


def main() -> None:
    st.set_page_config(page_title="AccelHR Viewer", layout="wide")
    st.title("AccelHR Signal Viewer")

    with st.sidebar:
        st.header("Selection")
        mode = st.radio("Mode", list(MODE_DIRS.keys()), horizontal=True, key="mode")
        data_dir = MODE_DIRS[mode]

    files = list_csvs(data_dir)
    if not files:
        st.error(f"No CSV files found in {data_dir}.")
        return

    file_index = index_files(files)
    if not file_index:
        st.error(
            f"No files in {data_dir} matched the expected `name_YYYY-MM-DD[_HH-MM-SS]` pattern."
        )
        return

    with st.sidebar:
        # Picker 1: name (prettified)
        name_to_label = {n: prettify_name(n) for n in sorted(file_index.keys())}
        sel_label = st.selectbox(
            "Name",
            list(name_to_label.values()),
            key=f"{mode.lower()}_name",
        )
        sel_name = next(n for n, lbl in name_to_label.items() if lbl == sel_label)

        # Picker 2: date
        dates = sorted(file_index[sel_name].keys())
        sel_date_str = st.selectbox("Date", dates, key=f"{mode.lower()}_file_date")

        # Picker 3: time (only when several files share the same date)
        candidates = file_index[sel_name][sel_date_str]
        if len(candidates) == 1:
            path = candidates[0]
        else:
            prefix = f"{sel_name}_{sel_date_str}_"
            time_to_path = {
                p.stem[len(prefix):] if p.stem.startswith(prefix) else p.stem: p
                for p in candidates
            }
            sel_time_label = st.selectbox(
                "Time",
                list(time_to_path.keys()),
                key=f"{mode.lower()}_file_time",
            )
            path = time_to_path[sel_time_label]

        # Copyable full path (st.code shows a copy button in the corner).
        st.caption("Path (copy):")
        try:
            display_path = str(path.relative_to(ROOT))
        except ValueError:
            display_path = str(path)
        st.code(display_path, language=None)

    # Reset date/time widgets whenever the active file (or mode) changes so
    # they pick up the new file's t_min/t_max defaults instead of carrying
    # over stale values from the previous selection.
    active_key = (mode, str(path))
    if st.session_state.get("active_file") != active_key:
        for k in (
            "sleep_start_date", "move_date",
            "sleep_start_time", "sleep_end_time",
            "move_start_time", "move_end_time",
        ):
            st.session_state.pop(k, None)
        st.session_state["active_file"] = active_key

    df = load_csv(str(path))
    t_min = df["timestamp_ist"].min()
    t_max = df["timestamp_ist"].max()

    with st.sidebar:
        st.caption(f"Data range:\n{t_min}  →  {t_max}")
        date_min = t_min.date()
        date_max = t_max.date()

        if mode == "Sleep":
            sleep_min = date_min - dt.timedelta(days=1)
            default_night = (
                sleep_min if date_min == date_max and t_min.time().hour < 12 else date_min
            )
            sel_start_date = st.date_input(
                "Night start date", value=default_night,
                min_value=sleep_min, max_value=date_max, key="sleep_start_date",
            )
            sel_end_date = sel_start_date  # may shift to +1 day below if end time wraps past midnight
            default_start = "21:00:00"
            default_end = "09:00:00"
            time_key_prefix = "sleep"
        else:
            sel_start_date = st.date_input(
                "Date", value=date_min,
                min_value=date_min, max_value=date_max, key="move_date",
            )
            sel_end_date = sel_start_date
            default_start = fmt_time(t_min.time().replace(microsecond=0))
            default_end = fmt_time(t_max.time().replace(microsecond=0))
            time_key_prefix = "move"

        st.caption("Format: HH:MM or HH:MM:SS")
        col_a, col_b = st.columns(2)
        with col_a:
            start_text = st.text_input(
                "Start time", value=default_start, key=f"{time_key_prefix}_start_time",
            )
        with col_b:
            end_text = st.text_input(
                "End time", value=default_end, key=f"{time_key_prefix}_end_time",
            )

        start_time = parse_time(start_text)
        end_time = parse_time(end_text)
        if start_time is None or end_time is None:
            st.warning("Use HH:MM or HH:MM:SS (e.g. 14:30:05).")
            return

        st.divider()
        st.subheader("Signals")
        all_names = [n for n, _ in PPG_SIGNALS + ACCEL_SIGNALS]
        available = [n for n in all_names if n in df.columns or n in {"accel_xy", "accel_xyz"}]
        selected: set[str] = set()
        for name in available:
            if st.checkbox(name, value=name in DEFAULT_SIGNALS, key=f"sig_{name}"):
                selected.add(name)

    tz = "Asia/Kolkata"
    if mode == "Sleep":
        # Sleep "start date" anchors 21:00 of the night. Any time past
        # midnight (hour < 12) belongs to the next calendar day; PM times
        # belong to the same date.
        def _sleep_ts(t: dt.time) -> pd.Timestamp:
            d = sel_start_date + dt.timedelta(days=1) if t.hour < 12 else sel_start_date
            return pd.Timestamp.combine(d, t).tz_localize(tz)

        start_ts = _sleep_ts(start_time)
        end_ts = _sleep_ts(end_time)
    else:
        start_ts = pd.Timestamp.combine(sel_start_date, start_time).tz_localize(tz)
        end_ts = pd.Timestamp.combine(sel_end_date, end_time).tz_localize(tz)

    with st.sidebar:
        st.caption(f"Window: {start_ts}  →  {end_ts}")

    if end_ts <= start_ts:
        st.warning("End time must be after start time.")
        return

    window = df[(df["timestamp_ist"] >= start_ts) & (df["timestamp_ist"] < end_ts)]
    if window.empty:
        st.warning("No samples in this window.")
        return

    ppg = window.dropna(subset=["green"]) if "green" in window.columns else window.iloc[0:0]
    accel_cols = [c for c in ("accel_x", "accel_y", "accel_z") if c in window.columns]
    if accel_cols:
        accel = window.dropna(subset=accel_cols).copy()
        if {"accel_x", "accel_y"}.issubset(accel.columns):
            accel["accel_xy"] = np.sqrt(accel["accel_x"] ** 2 + accel["accel_y"] ** 2)
        if {"accel_x", "accel_y", "accel_z"}.issubset(accel.columns):
            accel["accel_xyz"] = np.sqrt(
                accel["accel_x"] ** 2 + accel["accel_y"] ** 2 + accel["accel_z"] ** 2
            )
    else:
        accel = window.iloc[0:0]

    pw = process_window(window)

    c1, c2 = st.columns(2)
    c1.metric("Samples", f"{len(window):,}")
    c2.metric("Window", f"{(end_ts - start_ts).total_seconds():.0f} s")

    tab_raw, tab_filt, tab_bp = st.tabs(["Raw", "Filtered", "Bandpassed"])

    with tab_raw:
        signals = collect_raw_signals(selected, ppg, accel)
        if signals:
            st.plotly_chart(stacked_figure(signals), use_container_width=True, key="raw_chart")
        else:
            st.info("Select at least one signal in the sidebar.")

    with tab_filt:
        signals = collect_filtered_signals(selected, pw)
        if signals:
            st.plotly_chart(stacked_figure(signals), use_container_width=True, key="filt_chart")
        else:
            st.info("Select at least one signal in the sidebar.")

    filter_options = ["Auto (firmware logic)", "Low movement [0.6, 3.6] Hz", "High movement [0.9, 4.0] Hz"]
    filter_map = {
        "Auto (firmware logic)": "auto",
        "Low movement [0.6, 3.6] Hz": "low",
        "High movement [0.9, 4.0] Hz": "high",
    }

    with tab_bp:
        bp_choice_label = st.radio(
            "Bandpass filter",
            filter_options,
            horizontal=True,
            key="bp_choice",
        )
        bw_bp = bandpass_window(window, filter_map[bp_choice_label])
        st.caption(f"Filter selected: **{bw_bp.filter_label}**")
        signals = collect_bandpassed_signals(selected, bw_bp)
        if signals:
            st.plotly_chart(stacked_figure(signals), use_container_width=True, key="bp_chart")
        else:
            st.info("Select at least one signal in the sidebar.")


if __name__ == "__main__":
    main()
