import pandas as pd
import numpy as np
import json
import requests
from datetime import date, timedelta
from pathlib import Path

CH_HOST = "clickhouse.continue.bio"
CH_PORT = 443
CH_USER = "redash_user"
CH_PASSWORD = "qwerty123"


CF_AppSession = "3361ec228a3833fb"
CF_Authorization = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjkxNWYwMzIyZDI4NDU5Njc5NjVlZmU4MDY0NjliMzY4MTE1YzlhZjY5Yzg5YWMxM2IzODQwYWQ4YmVmZjdjMWMifQ.eyJhdWQiOlsiNGM5NjMxMDdmZDFmMmZkMDY5OTMwNDMwYjRmMGNkMGFiN2Y3NGMxYzM4ZGQwMzhiYmM1MmI2MTg4NTZjNGNkYSJdLCJlbWFpbCI6ImFyY2hpc2hAdGVtcGxlLmNvbSIsImV4cCI6MTc4MjUwMjEzOSwiaWF0IjoxNzc5ODc0MTM5LCJuYmYiOjE3Nzk4NzQxMzksImlzcyI6Imh0dHBzOi8vY29udGludWUtbGlmZXNjaWVuY2VzLmNsb3VkZmxhcmVhY2Nlc3MuY29tIiwidHlwZSI6ImFwcCIsImlkZW50aXR5X25vbmNlIjoiRGRTbjBvWEF5ajRrMVZlTCIsInN1YiI6IjU1MTUwNzNiLTM3MGEtNTk3NC1hYWQyLTBiOTYzZmEzZjJkNyIsImNvdW50cnkiOiJJTiIsInBvbGljeV9pZCI6IjUwNTQ3ZmEyLWE0OTEtNGRkOS1iZTkwLThjNzc0NzIxMjFhYyJ9.l2G59XaYItgHX0TxynWYkB4m_LZWgvLTRM7eOjnqbmwE3JRZscnWvY3voTLvFgyFYErkBnr7DaAaH0b4fnKFUQhvBEoJTJ8tHaYx5IYCImMp0dyWtV9q-UUf1p7Vj_VMiAX05_mky9ub70TU-Nj4naBCqKVySVyYii-E6NiI99f7QLecEZrYYXUn8zVg_1lea7CK3XrB58Ox5Aiy7E0Uh2kYJOXY7Z9RTfVcY-wm4MHR1IvUKHNesO5bFcMw7Y0ue-2BUar69AgckQSzHsuuKuzlEmraDMHK298xS5to1GBZVNLBXKq2IYDZkvTKs61w1CoF_8gFGDQP_1dsRqMu5g"

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
    """Return per-second firmware HR signals for `user_id` between start_time
    and end_time (IST) as a DataFrame with columns
    ['time_ist', 'firmware_hr', 'hr_confidence', 'reporting_mitigated'].

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
        avgIf(data, name = 'hr') AS firmware_hr,
        avgIf(data, name = 'hr_confidence') AS hr_confidence,
        avgIf(data, name = 'reporting_mitigated') AS reporting_mitigated
    FROM `prod-signal-data-service`.fw_l1
    WHERE user_id = '{user_id}'
      AND timestamp >= toUnixTimestamp(toDateTime('{start_time}', 'Asia/Kolkata')) * 1000
      AND timestamp <  toUnixTimestamp(toDateTime('{end_time}', 'Asia/Kolkata')) * 1000
      AND ({day_condition})
      AND name IN ('hr', 'hr_confidence', 'reporting_mitigated')
    GROUP BY 1
    ORDER BY 1
    """
    df = run_clickhouse_query(query)
    if df.empty:
        return df
    df['time_ist'] = pd.to_datetime(df['time_ist'])
    for c in ('firmware_hr', 'hr_confidence', 'reporting_mitigated'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
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


def fetch_minute_sampled_signals(start_time: str, end_time: str,
                                  query_path: str = "query.sql") -> pd.DataFrame:
    """Run the long-format minute-sampled-user query (query.sql) against the same
    ClickHouse cluster used by fetch_signal. Returns a DataFrame with columns
    ['user_id', 'timestamp', 'time_ist', 'signal', 'value'].

    `start_time` / `end_time` are 'YYYY-MM-DD HH:MM:SS' (Asia/Kolkata).
    """
    sql = Path(query_path).read_text()
    sql = sql.replace("{{start_dt}}", start_time).replace("{{end_dt}}", end_time)
    df = run_clickhouse_query(sql)
    if df.empty:
        return df
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')
    df['time_ist'] = pd.to_datetime(df['time_ist'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df


def fetch_all_signals(user_id: str, start_time, end_time) -> pd.DataFrame:
    """Fetch each signal separately, outer-join on timestamp, save a single CSV.

    Outer join means: rows with matching timestamps are merged into one row;
    rows with timestamps unique to a signal are appended with NaN elsewhere.
    Green-bearing rows are augmented with the closest firmware HR sample;
    accel-only rows leave `firmware_hr` as NaN.
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

    g = merged.loc[merged['green'].notna(), ['timestamp', 'green']]
    a = merged.loc[(merged['accel_x'].notna() & merged['accel_y'].notna() & merged['accel_z'].notna()), ['timestamp', 'accel_x', 'accel_y', 'accel_z']]

    matched = pd.merge_asof(
        g,
        a[['timestamp']].rename(columns={'timestamp': 'accel_ts'}),
        left_on='timestamp',
        right_on='accel_ts',
        tolerance=50,
        direction='nearest',
    )
    matched = matched.dropna(subset=['accel_ts'])

    keep_green_ts = set(matched['timestamp'])

    # Keep any accel sample within ±15s of a kept green timestamp so each
    # window has 15s of pre/post accel context for std/preroll analysis.
    PAD_MS = 15_000
    kept_green_arr = np.fromiter(keep_green_ts, dtype=np.int64, count=len(keep_green_ts))
    kept_green_arr.sort()
    a_sorted = a.sort_values('timestamp').reset_index(drop=True)
    a_ts = a_sorted['timestamp'].to_numpy()
    nearest_idx = np.searchsorted(kept_green_arr, a_ts)
    left = np.clip(nearest_idx - 1, 0, len(kept_green_arr) - 1)
    right = np.clip(nearest_idx, 0, len(kept_green_arr) - 1)
    nearest_dist = np.minimum(
        np.abs(a_ts - kept_green_arr[left]),
        np.abs(a_ts - kept_green_arr[right]),
    ) if len(kept_green_arr) else np.full(len(a_ts), PAD_MS + 1, dtype=np.int64)
    keep_accel_ts = set(a_ts[nearest_dist <= PAD_MS].tolist())

    merged = merged[
        (merged['green'].notna() & merged['timestamp'].isin(keep_green_ts))
        | (merged['accel_z'].notna() & merged['timestamp'].isin(keep_accel_ts))
    ].reset_index(drop=True)

    fw = get_firmware_hr(user_id, start_time, end_time)
    if not fw.empty:
        fw_t = pd.to_datetime(fw['time_ist'])
        if fw_t.dt.tz is None:
            fw_t = fw_t.dt.tz_localize('Asia/Kolkata')
        fw_ms = (fw_t.dt.tz_convert('UTC').astype('int64') // 1_000_000)
        fw_small = pd.DataFrame({
            'timestamp': fw_ms.to_numpy(),
            'firmware_hr': fw['firmware_hr'].to_numpy(),
            'hr_confidence': fw['hr_confidence'].to_numpy(),
        }).sort_values('timestamp').reset_index(drop=True)

        green_ts = (merged.loc[merged['green'].notna(), ['timestamp']]
                          .sort_values('timestamp')
                          .reset_index(drop=True))
        if not green_ts.empty:
            green_hr = pd.merge_asof(
                green_ts, fw_small, on='timestamp', direction='nearest'
            )
            merged = merged.merge(green_hr, on='timestamp', how='left')
        else:
            merged['firmware_hr'] = float('nan')
    else:
        merged['firmware_hr'] = float('nan')

    return merged


def get_sleep(user_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Return main-sleep start/end (local time) for `user_id` between
    start_date and end_date (inclusive, IST calendar dates).

    `start_date` / `end_date` are 'YYYY-MM-DD' strings.
    Returns DataFrame with columns ['sleep_date', 'start_time_local', 'end_time_local'].
    """
    query = f"""
    WITH base AS (
        SELECT
            splitByChar('#', d.pk)[2] AS user_id,
            arrayElement(JSONExtractArrayRaw(data, 'Sleeps'), 1) AS s,
            toDate(toTimezone(fromUnixTimestamp64Milli(JSONExtractInt(arrayElement(JSONExtractArrayRaw(data, 'Sleeps'), 1), 'EndTime')), 'Asia/Kolkata')) AS ist_date,
            updated_at
        FROM dynamodb.prod_consumer_service d
        WHERE d.pk = 'USER#{user_id}#SLEEPS'
          AND d.sk LIKE 'SLEEP#%#main_sleep'
    ),
    ranged AS (
        SELECT * FROM base
        WHERE ist_date BETWEEN toDate('{start_date}') AND toDate('{end_date}')
    ),
    dedup AS (
        SELECT
            user_id,
            ist_date,
            argMax(s, updated_at) AS latest_json
        FROM ranged
        GROUP BY user_id, ist_date
    )
    SELECT
        ist_date AS sleep_date,
        toDateTime(
            (toInt64OrNull(JSON_VALUE(latest_json, '$.StartTime')) / 1000) +
            toInt64OrNull(JSON_VALUE(latest_json, '$.TZOffsetSeconds'))
        ) AS start_time_local,
        toDateTime(
            (toInt64OrNull(JSON_VALUE(latest_json, '$.EndTime')) / 1000) +
            toInt64OrNull(JSON_VALUE(latest_json, '$.TZOffsetSeconds'))
        ) AS end_time_local
    FROM dedup
    ORDER BY sleep_date
    """
    df = run_clickhouse_query(query)
    if df.empty:
        return df
    df['sleep_date'] = pd.to_datetime(df['sleep_date']).dt.date
    df['start_time_local'] = pd.to_datetime(df['start_time_local'])
    df['end_time_local'] = pd.to_datetime(df['end_time_local'])
    return df


def get_deep_sleep(user_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Return deep-sleep stage windows for `user_id` between start_date and
    end_date (inclusive, IST calendar dates).

    `start_date` / `end_date` are 'YYYY-MM-DD' strings.
    Returns DataFrame with columns
    ['sleep_date', 'stage_start', 'stage_end', 'duration_s'] — one row per
    deep-sleep span recorded by the firmware sleep metric.
    """
    query = f"""
    WITH base AS (
        SELECT
            pk,
            sk,
            argMax(data, event_time) AS latest_json
        FROM dynamodb.prod_consumer_service
        WHERE pk = 'USER_METRIC#{user_id}#METRIC_TYPE#firmware-sleep_metric'
          AND sk LIKE 'SLEEP#main_sleep#TIMESTAMP#%'
          AND toTimeZone(
                fromUnixTimestamp64Milli(toInt64OrZero(splitByChar('#', sk)[-1])),
                'Asia/Kolkata'
              ) BETWEEN toDateTime('{start_date} 00:00:00', 'Asia/Kolkata')
                    AND toDateTime('{end_date} 23:59:59', 'Asia/Kolkata')
        GROUP BY pk, sk
    ),
    sleep_base AS (
        SELECT
            JSONExtractString(latest_json, 'Date') AS dt,
            JSONExtractRaw(
                arrayFirst(x -> JSONExtractString(x, 'type') = 'deep_sleep',
                           JSONExtractArrayRaw(latest_json, 'Stages')),
                'time_windows'
            ) AS deep_sleep_windows_json
        FROM base
    )
    SELECT
        toDate(stage_start) AS sleep_date,
        stage_start,
        stage_end,
        dateDiff('second', stage_start, stage_end) AS duration_s
    FROM (
        SELECT
            dt,
            toTimeZone(fromUnixTimestamp64Milli(toInt64(JSONExtractString(event, 'start_time'))),
                       'Asia/Kolkata') AS stage_start,
            toTimeZone(fromUnixTimestamp64Milli(toInt64(JSONExtractString(event, 'end_time'))),
                       'Asia/Kolkata') AS stage_end
        FROM sleep_base
        ARRAY JOIN JSONExtractArrayRaw(deep_sleep_windows_json) AS event
    )
    ORDER BY stage_start
    """
    df = run_clickhouse_query(query)
    if df.empty:
        return df

    df['sleep_date'] = pd.to_datetime(df['sleep_date']).dt.date
    df['stage_start'] = pd.to_datetime(df['stage_start'])
    df['stage_end'] = pd.to_datetime(df['stage_end'])
    df['duration_s'] = pd.to_numeric(df['duration_s'], errors='coerce')
    return df

