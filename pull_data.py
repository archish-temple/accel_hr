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


def fetch_hr(user_id: str, start_time: str, end_time: str) -> pd.DataFrame:
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


SIGNALS = ("green", "accel_x", "accel_y", "accel_z")
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

    signals = dfs[0]
    for df in dfs[1:]:
        signals = pd.merge(signals, df, on='timestamp', how='outer')

    fw = fetch_hr(user_id, start_time, end_time)
    if not fw.empty:
        fw = fw.copy()
        fw['timestamp'] = (
            fw['time_ist'].dt.tz_localize('Asia/Kolkata').astype('int64') // 1_000_000
        )
        fw = fw.drop(columns=['time_ist'])
        signals = pd.merge(signals, fw, on='timestamp', how='outer')

    return signals.sort_values('timestamp').reset_index(drop=True)