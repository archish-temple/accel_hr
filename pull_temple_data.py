import pandas as pd
import json
import requests
from datetime import date, timedelta
from pathlib import Path

CH_HOST = "clickhouse.continue.bio"
CH_PORT = 443
CH_USER = "redash_user"
CH_PASSWORD = "qwerty123"


CF_AppSession = "3361ec228a3833fb"
CF_Authorization = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjkxNWYwMzIyZDI4NDU5Njc5NjVlZmU4MDY0NjliMzY4MTE1YzlhZjY5Yzg5YWMxM2IzODQwYWQ4YmVmZjdjMWMifQ.eyJhdWQiOlsiNGM5NjMxMDdmZDFmMmZkMDY5OTMwNDMwYjRmMGNkMGFiN2Y3NGMxYzM4ZGQwMzhiYmM1MmI2MTg4NTZjNGNkYSJdLCJlbWFpbCI6ImFyY2hpc2hAdGVtcGxlLmNvbSIsImV4cCI6MTc4MjQ5MzI2NSwiaWF0IjoxNzc5ODY1MjY1LCJuYmYiOjE3Nzk4NjUyNjUsImlzcyI6Imh0dHBzOi8vY29udGludWUtbGlmZXNjaWVuY2VzLmNsb3VkZmxhcmVhY2Nlc3MuY29tIiwidHlwZSI6ImFwcCIsImlkZW50aXR5X25vbmNlIjoiRGRTbjBvWEF5ajRrMVZlTCIsInN1YiI6IjU1MTUwNzNiLTM3MGEtNTk3NC1hYWQyLTBiOTYzZmEzZjJkNyIsImNvdW50cnkiOiJJTiIsInBvbGljeV9pZCI6IjUwNTQ3ZmEyLWE0OTEtNGRkOS1iZTkwLThjNzc0NzIxMjFhYyJ9.j78c2B6NiIoN2Oy-uoyFM78WCp2PaMaR9gYrQPsWIz67XPx_wCkH4if7Tszjs0IJ4IxXv7K-y6noQmKIAdpvO1692U0xzDc5XlfM4K-mHvBp7Ffbog1q_MKQJYBaQvJsAidbnc0wm49xCHrFeILIhvH5V_FUwBVqmYylgvNQIEpy-TYK-cejlCbysvBuLQJ65CLuQ3uT61zv8obqGloprX9JWw6I3HepkAgqlDIw_P2h1w4ZBIC_RSZnmCCynj6Rjkf1Y4dmPTcDAYH60wYPVZ-V5XTlCsrQfSwDmbbdXoNoft3I6PP7dSXq3abo57gFmRq1r2r24xzBE0VXlvgaJw"

SIGNALS = ("red", "ir", "green", "accel_x", "accel_y", "accel_z")


def run_clickhouse_query(query: str) -> pd.DataFrame:
    url = f"https://{CH_HOST}:{CH_PORT}"

    if "FORMAT" not in query.upper():
        query = f"{query}\nFORMAT JSONEachRow"

    params = {
        "query": query,
        "user": CH_USER,
        "password": CH_PASSWORD,
    }

    cookies = {
        "CF_AppSession": CF_AppSession,
        "CF_Authorization": CF_Authorization,
    }

    r = requests.get(url, params=params, cookies=cookies, timeout=120)
    r.raise_for_status()

    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    return pd.DataFrame(rows)


def generate_day_tuples(start_time: str, end_time: str):
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)

    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    return [(d.day, d.month, d.year) for d in days]


def get_user_id_by_email(email: str) -> str | None:
    """Look up user_id from email via dataplatform.user_profile."""
    query = f"""
    SELECT
      user_id
    FROM dataplatform.user_profile
    WHERE email = '{email}'
    LIMIT 1
    """
    df = run_clickhouse_query(query)
    if df.empty:
        return None
    return str(df.iloc[0]["user_id"])


def get_firmware_hr(user_id: str, start_time: str, end_time: str) -> pd.DataFrame:
    """Return per-second firmware HR (BPM) for `user_id` between start_time and
    end_time (IST) as a DataFrame with columns ['time_ist', 'firmware_hr'].

    `start_time` / `end_time` are 'YYYY-MM-DD HH:MM:SS' strings (Asia/Kolkata).
    The window may span multiple days; partition pruning is added per day.
    """
    days = generate_day_tuples(start_time, end_time)
    day_condition = " OR ".join(
        f"(year = {y} AND month = {m} AND day = {d})" for d, m, y in days
    )

    query = f"""
    SELECT
        toTimezone(fromUnixTimestamp(CAST(timestamp / 1000 AS INT)), 'Asia/Kolkata') AS time_ist,
        avg(data) AS firmware_hr
    FROM `prod-signal-data-service`.fw_l1
    WHERE user_id = '{user_id}'
      AND timestamp >= toUnixTimestamp(toDateTime('{start_time}', 'Asia/Kolkata')) * 1000
      AND timestamp <  toUnixTimestamp(toDateTime('{end_time}', 'Asia/Kolkata')) * 1000
      AND ({day_condition})
      AND name = 'hr'
    GROUP BY 1
    ORDER BY 1
    """
    df = run_clickhouse_query(query)
    if df.empty:
        return df
    df['time_ist'] = pd.to_datetime(df['time_ist'])
    df['firmware_hr'] = pd.to_numeric(df['firmware_hr'], errors='coerce')
    return df


def build_signal_query(user_id, start_time, end_time, day_filters, signal):
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")

    day_condition = " OR ".join(
        f"(day={d} AND month={m} AND year={y})" for d, m, y in day_filters
    )

    return f"""
    SELECT
        timestamp,
        value AS {signal}
    FROM rawdata.{signal}
    WHERE user_id = '{user_id}'
      AND timestamp >= toUnixTimestamp64Milli(
            toDateTime64('{start_time}', 3, 'Asia/Kolkata')
          )
      AND timestamp <  toUnixTimestamp64Milli(
            toDateTime64('{end_time}', 3, 'Asia/Kolkata')
          )
      AND ({day_condition})
    ORDER BY timestamp
    """


def fetch_signal(user_id, start_time, end_time, signal):
    days = generate_day_tuples(start_time, end_time)
    query = build_signal_query(user_id, start_time, end_time, days, signal)
    return run_clickhouse_query(query)


def fetch_all_signals(user_id, start_time, end_time, save_path):
    """Fetch each signal separately, outer-join on timestamp, save a single CSV.

    Outer join means: rows with matching timestamps are merged into one row;
    rows with timestamps unique to a signal are appended with NaN elsewhere.
    """
    dfs = []
    for signal in SIGNALS:
        df = fetch_signal(user_id, start_time, end_time, signal)
        if df.empty:
            continue
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='timestamp', how='outer')

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged['timestamp_ist'] = pd.to_datetime(
        merged['timestamp'], unit='ms', utc=True
    ).dt.tz_convert('Asia/Kolkata')

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"✓ Saved: {out} ({len(merged)} rows)")
    return merged


def fetch_sleep_windows_for_user(
    user_id: str,
    save_path: str,
    start_date: str = "2026-05-01",
    end_date: str = "2026-05-25",
    night_start_hour: int = 21,
    night_end_hour: int = 9,
) -> None:
    """Fetch overnight (night_start_hour → next-day night_end_hour, IST) data
    for a single user across [start_date, end_date] inclusive and save ONE
    combined CSV at `save_path` (all nights concatenated, sorted by timestamp).
    Skips entirely if `save_path` already exists.
    """
    out = Path(save_path)
    if out.exists():
        print(f"↷ {out} (cached)")
        return

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    out.parent.mkdir(parents=True, exist_ok=True)

    night_dfs = []
    night = start
    while night <= end:
        morning = night + timedelta(days=1)
        window_start = f"{night.isoformat()} {night_start_hour:02d}:00:00"
        window_end = f"{morning.isoformat()} {night_end_hour:02d}:00:00"

        print(f"  [{user_id[:8]}] {night} {night_start_hour:02d}:00 → {morning} {night_end_hour:02d}:00")
        try:
            tmp = out.parent / f".tmp_{user_id}_{night.isoformat()}.csv"
            night_df = fetch_all_signals(user_id, window_start, window_end, str(tmp))
            tmp.unlink(missing_ok=True)
            if night_df is not None and not night_df.empty:
                night_dfs.append(night_df)
        except Exception as e:
            print(f"    ✗ [{user_id[:8]}] error: {e}")
        night = morning

    if not night_dfs:
        print(f"✗ no data for {user_id}")
        return

    combined = (
        pd.concat(night_dfs, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined.to_csv(out, index=False)
    print(f"✓ Saved: {out} ({len(combined)} rows across {len(night_dfs)} nights)")
