"""Find dense 15-second sleep windows across multiple users.

For each user:
1. Look up `user_id` from email.
2. Find the most recent main-sleep interval via `get_sleep`.
3. Fetch all signals during that interval.
4. Bin samples into non-overlapping windows; keep windows whose combined
   PPG + accel count meets `MIN_SAMPLES`.
5. Append rows to OUTPUT_CSV.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pull_temple_data import (
    fetch_all_signals,
    get_firmware_hr,
    get_sleep,
    get_user_id_by_email,
)

USERS = [
    # "aniket.jana@temple.com"
    # "gokul@temple.com",
    # 'hardik@temple.com',
    # "kushagra@temple.com",
    "sanchit@temple.com",
]

WINDOW_SECONDS = 60
OUTER_WINDOW_SECONDS = 90  # std(accel) is checked on this outer span; saved window is the centered WINDOW_SECONDS slice
STEP_SECONDS = 1       # sliding-window stride
PPG_FS_HZ = 32         # green sample rate
ACCEL_FS_HZ = 26       # accel sample rate
COVERAGE = 0.95        # require ≥ COVERAGE of expected per stream
MIN_PPG_SAMPLES = int(COVERAGE * PPG_FS_HZ * WINDOW_SECONDS)
MIN_ACCEL_SAMPLES = int(COVERAGE * ACCEL_FS_HZ * WINDOW_SECONDS)
ACCEL_STD_MAX = 250    # reject windows where any accel axis std (over outer span) exceeds this
HR_CONFIDENCE_MIN = 50  # strict >: every fw sample in window must exceed this
SLEEP_LOOKBACK_DAYS = 14
OUTPUT_CSV = Path(f"sleep/dense_windows_{WINDOW_SECONDS}.csv")
SLEEP_DATA_DIR = Path("sleep/data")
MAX_WORKERS = 4


def find_dense_windows(merged: pd.DataFrame) -> pd.DataFrame:
    """Sliding-window scan: every STEP_SECONDS, evaluate an OUTER_WINDOW_SECONDS
    candidate. Keep the centered WINDOW_SECONDS slice when std(accel_x|y|z) over
    the outer span < ACCEL_STD_MAX and the inner slice has PPG >= MIN_PPG_SAMPLES
    AND accel >= MIN_ACCEL_SAMPLES; greedy non-overlapping pick from left.
    """
    if merged.empty:
        return pd.DataFrame()

    ts_ms = merged["timestamp"].to_numpy()
    has_ppg = merged["green"].notna().to_numpy()
    has_acc = merged["accel_z"].notna().to_numpy()

    ppg_ts = np.sort(ts_ms[has_ppg])

    acc_order = np.argsort(ts_ms[has_acc])
    acc_ts = ts_ms[has_acc][acc_order]
    accel_x = merged["accel_x"].to_numpy()[has_acc][acc_order]
    accel_y = merged["accel_y"].to_numpy()[has_acc][acc_order]
    accel_z = merged["accel_z"].to_numpy()[has_acc][acc_order]

    if len(ppg_ts) == 0 or len(acc_ts) == 0:
        return pd.DataFrame()

    inner_ms = WINDOW_SECONDS * 1000
    outer_ms = OUTER_WINDOW_SECONDS * 1000
    pad_ms = (outer_ms - inner_ms) // 2  # offset from outer start to inner start
    step_ms = STEP_SECONDS * 1000
    t_start = int(max(ppg_ts[0], acc_ts[0]))
    t_end = int(min(ppg_ts[-1], acc_ts[-1])) - outer_ms
    if t_end <= t_start:
        return pd.DataFrame()

    outer_starts = np.arange(t_start, t_end + 1, step_ms, dtype=np.int64)
    outer_ends = outer_starts + outer_ms
    inner_starts = outer_starts + pad_ms
    inner_ends = inner_starts + inner_ms

    ppg_counts = (
        np.searchsorted(ppg_ts, inner_ends, side="left")
        - np.searchsorted(ppg_ts, inner_starts, side="left")
    )
    acc_lo_inner = np.searchsorted(acc_ts, inner_starts, side="left")
    acc_hi_inner = np.searchsorted(acc_ts, inner_ends, side="left")
    acc_counts = acc_hi_inner - acc_lo_inner

    acc_lo_outer = np.searchsorted(acc_ts, outer_starts, side="left")
    acc_hi_outer = np.searchsorted(acc_ts, outer_ends, side="left")

    qualifies = (ppg_counts >= MIN_PPG_SAMPLES) & (acc_counts >= MIN_ACCEL_SAMPLES)
    candidates = np.where(qualifies)[0]
    if candidates.size == 0:
        return pd.DataFrame()

    kept: list[int] = []
    kept_std_x: list[float] = []
    kept_std_y: list[float] = []
    kept_std_z: list[float] = []
    last_inner_end_ms = -1
    for i in candidates:
        inner_s = int(inner_starts[i])
        if inner_s < last_inner_end_ms:
            continue
        lo, hi = int(acc_lo_outer[i]), int(acc_hi_outer[i])
        if hi - lo < 2:
            continue
        sx = float(accel_x[lo:hi].std())
        sy = float(accel_y[lo:hi].std())
        sz = float(accel_z[lo:hi].std())
        if sx >= ACCEL_STD_MAX or sy >= ACCEL_STD_MAX or sz >= ACCEL_STD_MAX:
            continue
        kept.append(i)
        kept_std_x.append(sx)
        kept_std_y.append(sy)
        kept_std_z.append(sz)
        last_inner_end_ms = inner_s + inner_ms

    if not kept:
        return pd.DataFrame()

    kept_idx = np.array(kept, dtype=np.int64)
    return pd.DataFrame({
        "window_start": pd.to_datetime(inner_starts[kept_idx], unit="ms", utc=True).tz_convert("Asia/Kolkata"),
        "window_end": pd.to_datetime(inner_ends[kept_idx], unit="ms", utc=True).tz_convert("Asia/Kolkata"),
        "ppg_count": ppg_counts[kept_idx],
        "accel_count": acc_counts[kept_idx],
        "accel_x_std": kept_std_x,
        "accel_y_std": kept_std_y,
        "accel_z_std": kept_std_z,
    })


def process_user(email: str) -> pd.DataFrame:
    user_id = get_user_id_by_email(email)
    if not user_id:
        print(f"[{email}] no user_id")
        return pd.DataFrame()

    start_date = pd.Timestamp('2026-05-16')
    end_date = pd.Timestamp('2026-05-16')
    # end_date = date.today() - timedelta(days=1)
    # start_date = end_date - timedelta(days=SLEEP_LOOKBACK_DAYS)
    sleeps = get_sleep(user_id, start_date.isoformat(), end_date.isoformat())
    if sleeps.empty:
        print(f"[{email}] no sleep records in last {SLEEP_LOOKBACK_DAYS} days")
        return pd.DataFrame()

    print(f"[{email}] {len(sleeps)} sleep records:")
    for _, row in sleeps.iterrows():
        print(f"  {row['sleep_date']}: {row['start_time_local']} → {row['end_time_local']}")

    fw_start = sleeps["start_time_local"].min().strftime("%Y-%m-%d %H:%M:%S")
    fw_end = sleeps["end_time_local"].max().strftime("%Y-%m-%d %H:%M:%S")
    fw = get_firmware_hr(user_id, fw_start, fw_end)
    if not fw.empty:
        fw_t = pd.to_datetime(fw["time_ist"])
        if fw_t.dt.tz is None:
            fw_t = fw_t.dt.tz_localize("Asia/Kolkata")
        else:
            fw_t = fw_t.dt.tz_convert("Asia/Kolkata")
        fw = fw.assign(time_ist=fw_t).sort_values("time_ist").reset_index(drop=True)
    print(f"[{email}] firmware hr {fw_start} → {fw_end}: {len(fw)} samples")

    parts: list[pd.DataFrame] = []
    for _, sleep_row in sleeps.iterrows():
        start_ts = sleep_row["start_time_local"].strftime("%Y-%m-%d %H:%M:%S")
        end_ts = sleep_row["end_time_local"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{email}] fetching {sleep_row['sleep_date']} {start_ts} → {end_ts}")

        merged = fetch_all_signals(user_id, start_ts, end_ts)
        if merged.empty:
            print(f"[{email}] {sleep_row['sleep_date']} no signals returned")
            continue

        SLEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sig_name = (
            f"{email}_"
            f"{sleep_row['start_time_local'].strftime('%Y-%m-%dT%H:%M:%S')}_"
            f"{sleep_row['end_time_local'].strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        sig_path = SLEEP_DATA_DIR / sig_name
        merged.to_csv(sig_path, index=False)
        print(f"[{email}] {sleep_row['sleep_date']} wrote {len(merged)} signal rows → {sig_path}")

        windows = find_dense_windows(merged)
        if windows.empty:
            print(f"[{email}] {sleep_row['sleep_date']} 0 dense windows")
            continue

        sleep_lo = pd.Timestamp(sleep_row["start_time_local"])
        sleep_hi = pd.Timestamp(sleep_row["end_time_local"])
        if sleep_lo.tz is None:
            sleep_lo = sleep_lo.tz_localize("Asia/Kolkata")
        if sleep_hi.tz is None:
            sleep_hi = sleep_hi.tz_localize("Asia/Kolkata")
        fw_sleep = fw[(fw["time_ist"] >= sleep_lo) & (fw["time_ist"] < sleep_hi)] if not fw.empty else fw
        if fw_sleep.empty:
            print(f"[{email}] {sleep_row['sleep_date']} no firmware hr — dropping all {len(windows)} windows")
            continue

        # Match each window to the firmware sample closest to its midpoint.
        win_mid = windows["window_start"] + (windows["window_end"] - windows["window_start"]) / 2
        probe = pd.DataFrame({"time_ist": win_mid}).sort_values("time_ist")
        probe["_orig"] = probe.index
        matched = pd.merge_asof(
            probe, fw_sleep, on="time_ist", direction="nearest"
        ).sort_values("_orig").reset_index(drop=True)

        # ok = (matched["hr_confidence"] > HR_CONFIDENCE_MIN) & (matched["reporting_mitigated"] == 0)
        ok = (matched["hr_confidence"] > 0) # & (matched["reporting_mitigated"] == 0)
        before = len(windows)
        windows = windows.assign(
            firmware_hr=matched["firmware_hr"].to_numpy(),
            hr_confidence=matched["hr_confidence"].to_numpy(),
        ).loc[ok.to_numpy()].reset_index(drop=True)
        print(f"[{email}] {sleep_row['sleep_date']} fw filter kept {len(windows)}/{before} windows")
        if windows.empty:
            continue

        windows.insert(0, "email", email)
        windows.insert(1, "sleep_date", sleep_row["sleep_date"])
        windows.insert(2, "sleep_start", sleep_row["start_time_local"])
        windows.insert(3, "sleep_end", sleep_row["end_time_local"])
        print(f"[{email}] {sleep_row['sleep_date']} {len(windows)} dense windows")
        parts.append(windows)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_user, email): email for email in USERS}
        for fut in as_completed(futures):
            email = futures[fut]
            try:
                df = fut.result()
            except Exception as exc:
                print(f"[{email}] failed: {exc}")
                continue
            if not df.empty:
                parts.append(df)

    if not parts:
        print("No dense windows found.")
        return

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["email", "window_start"]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} windows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
