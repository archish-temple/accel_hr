from __future__ import annotations

import os
import re
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent / "sleep" / "data"
TZ = "Asia/Kolkata"
PPG_SIGNALS = ["red", "ir", "green"]
ACCEL_SIGNALS = ["accel_x", "accel_y", "accel_z"]
SIGNALS = PPG_SIGNALS + ACCEL_SIGNALS
DEFAULT_SIGNALS = {"green", "accel_x", "accel_y"}
SIGNAL_COLORS = {
    "red": "#e53935",
    "ir": "#8e24aa",
    "green": "#43a047",
    "accel_x": "#1e88e5",
    "accel_y": "#fb8c00",
    "accel_z": "#00897b",
}

FNAME_RE = re.compile(
    r"^(?P<user>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_(?P<end>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$"
)


def list_sessions() -> list[dict]:
    sessions = []
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        if p.stat().st_size < 100:
            continue
        sessions.append(
            {
                "path": p,
                "name": p.name,
                "user": m.group("user"),
                "start": datetime.fromisoformat(m.group("start")),
                "end": datetime.fromisoformat(m.group("end")),
            }
        )
    return sessions


@st.cache_data(show_spinner=False)
def load_csv(path_str: str, mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path_str)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(TZ)
    ppg = df.dropna(subset=PPG_SIGNALS, how="all")[["time", *PPG_SIGNALS]].reset_index(drop=True)
    accel = df.dropna(subset=ACCEL_SIGNALS, how="all")[["time", *ACCEL_SIGNALS]].reset_index(drop=True)
    return ppg, accel


def parse_time(s: str) -> time | None:
    s = s.strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    return None


def main() -> None:
    st.set_page_config(page_title="Raw Signal Viewer", layout="wide")
    st.title("Raw Signal Viewer")

    sessions = list_sessions()
    if not sessions:
        st.error(f"No usable session files found in {DATA_DIR}")
        return

    with st.sidebar:
        st.header("Settings")

        users = sorted({s["user"] for s in sessions})
        user = st.selectbox("User", users)

        user_sessions = [s for s in sessions if s["user"] == user]
        date_labels = {s["start"].date().isoformat(): s for s in user_sessions}
        date_choice = st.selectbox("Start Date", sorted(date_labels.keys()))
        session = date_labels[date_choice]

        ppg, accel = load_csv(str(session["path"]), session["path"].stat().st_mtime)
        all_times = pd.concat([ppg["time"], accel["time"]])
        file_min = all_times.min().to_pydatetime()
        file_max = all_times.max().to_pydatetime()

        if st.session_state.get("_loaded_session") != session["name"]:
            st.session_state["_loaded_session"] = session["name"]
            st.session_state["sd"] = file_min.date()
            st.session_state["st"] = file_min.time().replace(microsecond=0).strftime("%H:%M:%S")
            st.session_state["ed"] = file_max.date()
            st.session_state["et"] = file_max.time().replace(microsecond=0).strftime("%H:%M:%S")

        st.caption(f"PPG: {len(ppg):,} | Accel: {len(accel):,}")
        st.caption(f"{file_min}  →  {file_max}")

        c1, c2 = st.columns(2)
        with c1:
            sd = st.date_input("Start Date", key="sd")
            stime_str = st.text_input("Start Time", key="st", placeholder="HH:MM or HH:MM:SS")
        with c2:
            ed = st.date_input("End Date", key="ed")
            etime_str = st.text_input("End Time", key="et", placeholder="HH:MM or HH:MM:SS")

        stime = parse_time(stime_str)
        etime = parse_time(etime_str)
        time_ok = stime is not None and etime is not None
        if not time_ok:
            st.warning("Time must be HH:MM or HH:MM:SS.")

        st.markdown("**Signals**")
        selected = []
        for sig in SIGNALS:
            if st.checkbox(sig, value=sig in DEFAULT_SIGNALS, key=f"cb_{sig}"):
                selected.append(sig)

        st.markdown("**Copy**")
        st.caption("Path")
        st.code(str(session["path"].relative_to(Path(__file__).parent)), language=None)
        if time_ok:
            start_dt = datetime.combine(sd, stime)
            end_dt = datetime.combine(ed, etime)
            st.caption("Start")
            st.code(start_dt.strftime("%Y-%m-%d %H:%M:%S"), language=None)
            st.caption("End")
            st.code(end_dt.strftime("%Y-%m-%d %H:%M:%S"), language=None)

    if not time_ok:
        return

    start_dt = datetime.combine(sd, stime)
    end_dt = datetime.combine(ed, etime)

    if start_dt >= end_dt:
        st.warning("Start must be before end.")
        return
    if not selected:
        st.info("Select at least one signal to plot.")
        return

    ts_start = pd.Timestamp(start_dt, tz=TZ)
    ts_end = pd.Timestamp(end_dt, tz=TZ)
    ppg_sub = ppg[(ppg["time"] >= ts_start) & (ppg["time"] <= ts_end)]
    accel_sub = accel[(accel["time"] >= ts_start) & (accel["time"] <= ts_end)]

    if ppg_sub.empty and accel_sub.empty:
        st.warning("No samples in selected range.")
        return

    fig = make_subplots(
        rows=len(selected),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=selected,
    )
    for i, sig in enumerate(selected, start=1):
        src = ppg_sub if sig in PPG_SIGNALS else accel_sub
        fig.add_trace(
            go.Scattergl(
                x=src["time"],
                y=src[sig],
                mode="lines",
                name=sig,
                line=dict(color=SIGNAL_COLORS[sig]),
            ),
            row=i,
            col=1,
        )
    fig.update_layout(
        height=220 * len(selected),
        margin=dict(l=40, r=20, t=40, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
