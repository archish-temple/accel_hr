import numpy as np
import pandas as pd

def find_ppg_periods(df: pd.DataFrame, max_gap_ms: float = 1000.0, min_mean_fs_hz: float = 26.0) -> pd.DataFrame:
    """Continuous spans where green is sampled at ~32 Hz.

    Breaks a period whenever the gap between consecutive green samples
    exceeds `max_gap_ms`. Drops spans whose mean rate is below `min_mean_fs_hz`.

    `df` needs 'timestamp' (ms) and 'green'. Returns one row per period:
    start_ms, end_ms, duration_s.
    """
    g = df.loc[df["green"].notna(), ["timestamp"]].copy()
    g["timestamp"] = pd.to_numeric(g["timestamp"], errors="coerce")
    g = g.dropna().astype({"timestamp": "int64"}).sort_values("timestamp")
    ts = g["timestamp"].to_numpy()
    if ts.size < 2:
        return pd.DataFrame(
            columns=["start_ms", "end_ms" "duration_s"]
        )

    diffs = np.diff(ts)
    break_idx = np.where(diffs > max_gap_ms)[0] + 1
    starts = np.concatenate(([0], break_idx))
    ends = np.concatenate((break_idx, [ts.size]))

    rows = []
    for s, e in zip(starts, ends):
        n = int(e - s)
        if n < 2:
            continue
        dur_s = float((ts[e - 1] - ts[s]) / 1000.0)
        mean_fs = (n - 1) / dur_s if dur_s > 0 else 0.0
        if mean_fs < min_mean_fs_hz:
            continue
        rows.append({
            "start_ms": int(ts[s]),
            "end_ms": int(ts[e - 1]),
            "duration_s": dur_s,
        })
    return pd.DataFrame(rows)

def attach_signals_per_period(periods: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
    """For each row in `periods`, pull green / accel / hr arrays (each on its
    native timestamps).

    Green and hr are sliced to the block's own [start_ms, end_ms]. Accel is
    sliced to [start_ms, next block's start_ms) so it bridges the gaps between
    PPG periods; the last block's accel runs through its own end_ms. The accel
    upper bound is recorded as `accel_end_ms`.

    For periods longer than 1 minute, green is duty-cycled: only the first 30 s
    of every 5-min window (measured from the period start) is retained. This
    applies to green alone — accel and hr keep their full sample sets.

    Adds columns: green_ts, green, accel_ts, accel_x, accel_y, accel_z,
    accel_end_ms, hr_ts, hr, hr_confidence.
    """
    out = periods.copy()
    ts = raw_data['timestamp'].to_numpy()
    starts = periods['start_ms'].to_numpy()
    n_periods = len(periods)

    cols = {
        'green_ts': [], 'green': [],
        'accel_ts': [], 'accel_x': [], 'accel_y': [], 'accel_z': [], 'accel_end_ms': [],
        'hr_ts': [], 'hr': [], 'hr_confidence': [],
    }

    for i, (_, row) in enumerate(periods.iterrows()):
        s, e = row['start_ms'], row['end_ms']
        # Accel bridges into the gap up to the next block's start; last block
        # stops at its own end_ms (+1 so the boundary sample stays inclusive).
        accel_end = int(starts[i + 1]) if i + 1 < n_periods else int(e) + 1
        window = raw_data.loc[(ts >= s) & (ts <= e)]
        accel_window = raw_data.loc[(ts >= s) & (ts < accel_end)]

        g = window.loc[window['green'].notna(), ['timestamp', 'green']]
        g_ts = g['timestamp'].to_numpy()
        g_val = g['green'].to_numpy()
        cols['green_ts'].append(g_ts)
        cols['green'].append(g_val)

        a = accel_window.loc[accel_window['accel_x'].notna(), ['timestamp', 'accel_x', 'accel_y', 'accel_z']]
        cols['accel_ts'].append(a['timestamp'].to_numpy())
        cols['accel_x'].append(a['accel_x'].to_numpy())
        cols['accel_y'].append(a['accel_y'].to_numpy())
        cols['accel_z'].append(a['accel_z'].to_numpy())
        cols['accel_end_ms'].append(accel_end)

        h = window.loc[window['firmware_hr'].notna(), ['timestamp', 'firmware_hr', 'hr_confidence']]
        cols['hr_ts'].append(h['timestamp'].to_numpy())
        cols['hr'].append(h['firmware_hr'].to_numpy())
        cols['hr_confidence'].append(h['hr_confidence'].to_numpy())

    for k, v in cols.items():
        out[k] = v
    return out

def _window_by_time(ts: np.ndarray, values: np.ndarray, start_ms: float, end_ms: float,
                    window_s: float, stride_s: float) -> tuple:
    """Slice `values` (aligned to `ts` in ms) into windows of `window_s`
    seconds advancing by `stride_s` seconds across [start_ms, end_ms].

    Returns (windows, window_ts): list of value-lists and a parallel list of
    the actual sample timestamps (ms) inside each window.
    """
    window_ms = window_s * 1000.0
    stride_ms = stride_s * 1000.0
    windows, win_ts = [], []
    w_start = float(start_ms)
    while w_start + window_ms <= end_ms:
        w_end = w_start + window_ms
        mask = (ts >= w_start) & (ts < w_end)
        windows.append(values[mask].tolist())
        win_ts.append(ts[mask].tolist())
        w_start += stride_ms
    return windows, win_ts

def process_block_data(block_raw_data: pd.DataFrame,
                       ppg_window_s: float = 12.0, ppg_stride_s: float = 1.0,
                       accel_window_s: float = 15.0, accel_stride_s: float = 15.0) -> pd.DataFrame:
    """For each block, window the PPG and accel signals by timestamp.

    PPG (green) → list of lists, `ppg_window_s`-second windows with `ppg_stride_s`
    stride. Accel (x/y/z) → list of lists, `accel_window_s`-second windows with
    `accel_stride_s` stride. Adds `*_window_ts` columns with each window's
    sample timestamps and `*_window_mean` / `*_window_std` columns with
    per-window summary stats.

    Also adds `last_updated_hr` / `last_updated_hr_ts`: the latest firmware HR
    in the block, or — if the block has none — carried forward from the most
    recent prior block. Drops the now-redundant `hr` / `hr_ts` columns.
    """
    out = block_raw_data.copy()
    green_w, green_w_ts, green_mu, green_sd = [], [], [], []
    ax_w, ay_w, az_w, accel_w_ts = [], [], [], []
    ax_mu, ax_sd, ay_mu, ay_sd, az_mu, az_sd = [], [], [], [], [], []
    last_updated_hr, last_updated_hr_ts = [], []
    carry_hr, carry_hr_ts = np.nan, np.nan

    def _stats(windows):
        mus, sds = [], []
        for w in windows:
            if len(w) == 0:
                mus.append(np.nan)
                sds.append(np.nan)
            else:
                arr = np.asarray(w, dtype=float)
                mus.append(float(arr.mean()))
                sds.append(float(arr.std()))
        return mus, sds

    for _, row in block_raw_data.iterrows():
        s, e = row['start_ms'], row['end_ms']

        h_ts = np.asarray(row['hr_ts'], dtype=float)
        h = np.asarray(row['hr'], dtype=float)
        if h_ts.size > 0:
            i = int(np.argmax(h_ts))
            carry_hr, carry_hr_ts = float(h[i]), float(h_ts[i])
        last_updated_hr.append(carry_hr)
        last_updated_hr_ts.append(carry_hr_ts)

        g_ts = np.asarray(row['green_ts'])
        gw, gws = _window_by_time(g_ts, np.asarray(row['green']), s, e, ppg_window_s, ppg_stride_s)
        green_w.append(gw)
        green_w_ts.append(gws)
        gmu, gsd = _stats(gw)
        green_mu.append(gmu)
        green_sd.append(gsd)

        a_ts = np.asarray(row['accel_ts'])
        a_end = row['accel_end_ms']
        axw, aws = _window_by_time(a_ts, np.asarray(row['accel_x']), s, a_end, accel_window_s, accel_stride_s)
        ayw, _ = _window_by_time(a_ts, np.asarray(row['accel_y']), s, a_end, accel_window_s, accel_stride_s)
        azw, _ = _window_by_time(a_ts, np.asarray(row['accel_z']), s, a_end, accel_window_s, accel_stride_s)
        ax_w.append(axw)
        ay_w.append(ayw)
        az_w.append(azw)
        accel_w_ts.append(aws)
        axmu, axsd = _stats(axw)
        aymu, aysd = _stats(ayw)
        azmu, azsd = _stats(azw)
        ax_mu.append(axmu); ax_sd.append(axsd)
        ay_mu.append(aymu); ay_sd.append(aysd)
        az_mu.append(azmu); az_sd.append(azsd)

    out['green_windows'] = green_w
    out['green_window_ts'] = green_w_ts
    out['green_window_mean'] = green_mu
    out['green_window_std'] = green_sd
    out['accel_x_windows'] = ax_w
    out['accel_y_windows'] = ay_w
    out['accel_z_windows'] = az_w
    out['accel_window_ts'] = accel_w_ts
    out['accel_x_window_mean'] = ax_mu
    out['accel_x_window_std'] = ax_sd
    out['accel_y_window_mean'] = ay_mu
    out['accel_y_window_std'] = ay_sd
    out['accel_z_window_mean'] = az_mu
    out['accel_z_window_std'] = az_sd
    out['last_updated_hr'] = last_updated_hr
    out['last_updated_hr_ts'] = last_updated_hr_ts
    # out = out.drop(columns=['hr', 'hr_ts', 'hr_confidence'])
    out = out.dropna(subset=['last_updated_hr']).reset_index(drop=True)
    return out