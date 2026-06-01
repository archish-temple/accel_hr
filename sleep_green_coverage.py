"""
Report green coverage during a user's sleep window on a given date.

Usage:
    python sleep_green_coverage.py <email> <YYYY-MM-DD>

Example:
    python sleep_green_coverage.py prateek@temple.com 2026-05-27
"""

import sys
import pandas as pd
from pull_temple_data import (
    get_user_id_by_email,
    get_sleep,
    generate_day_tuples,
    run_clickhouse_query,
)


def get_green_coverage_for_window(user_id: str, start_time: str, end_time: str) -> dict:
    """
    Return green coverage stats for `user_id` between start_time and end_time (IST).
    start_time / end_time are 'YYYY-MM-DD HH:MM:SS' strings (Asia/Kolkata).
    """
    days = generate_day_tuples(start_time, end_time)
    day_condition = " OR ".join(
        f"(year = {y} AND month = {m} AND day = {d})" for d, m, y in days
    )

    query = f"""
    SELECT
        count()                                          AS total_hr_samples,
        countIf(g.g_count > 0)                           AS samples_with_green,
        count() - countIf(g.g_count > 0)                 AS missing_green_count,
        round(countIf(g.g_count > 0) / count() * 100, 2) AS green_coverage_pct,
        IF(green_coverage_pct >= 95, 'Not Missing', 'Missing') AS missing_flag
    FROM `prod-signal-data-service`.fw_l1 AS l
    LEFT JOIN (
        SELECT
            user_id,
            intDiv(timestamp, 10000) AS bucket,
            count() AS g_count
        FROM rawdata.green
        WHERE user_id = '{user_id}'
          AND timestamp >= toUnixTimestamp(toDateTime('{start_time}', 'Asia/Kolkata')) * 1000
          AND timestamp <  toUnixTimestamp(toDateTime('{end_time}',   'Asia/Kolkata')) * 1000
          AND ({day_condition})
        GROUP BY user_id, bucket
    ) AS g ON g.user_id = l.user_id AND g.bucket = intDiv(l.timestamp, 10000)
    WHERE l.user_id = '{user_id}'
      AND l.name = 'hr'
      AND l.timestamp >= toUnixTimestamp(toDateTime('{start_time}', 'Asia/Kolkata')) * 1000
      AND l.timestamp <  toUnixTimestamp(toDateTime('{end_time}',   'Asia/Kolkata')) * 1000
      AND ({day_condition})
    """

    df = run_clickhouse_query(query)
    if df.empty:
        return {}

    row = df.iloc[0]
    return {
        "total_hr_samples":   int(row["total_hr_samples"]),
        "samples_with_green": int(row["samples_with_green"]),
        "missing_green_count": int(row["missing_green_count"]),
        "green_coverage_pct": float(row["green_coverage_pct"]),
        "missing_flag":       str(row["missing_flag"]),
    }


def report(email: str, date: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Green coverage during sleep")
    print(f"  User : {email}")
    print(f"  Date : {date}")
    print(f"{'='*60}")

    user_id = get_user_id_by_email(email)
    if not user_id:
        print(f"  ERROR: no user found for {email}")
        return

    sleep_df = get_sleep(user_id, date, date)
    if sleep_df.empty:
        print(f"  No sleep record found for {date}")
        return

    row = sleep_df.iloc[0]
    start = row["start_time_local"]
    end   = row["end_time_local"]
    duration_h = (end - start).total_seconds() / 3600

    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str   = end.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  Sleep window  : {start_str}  →  {end_str}  ({duration_h:.2f} h)")

    stats = get_green_coverage_for_window(user_id, start_str, end_str)
    if not stats:
        print("  No HR/green data found in this window.")
        return

    print(f"\n  HR samples total   : {stats['total_hr_samples']}")
    print(f"  With green match   : {stats['samples_with_green']}")
    print(f"  Missing green      : {stats['missing_green_count']}")
    print(f"  Green coverage     : {stats['green_coverage_pct']} %")
    print(f"  Status             : {stats['missing_flag']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sleep_green_coverage.py <email> <YYYY-MM-DD>")
        sys.exit(1)

    report(email=sys.argv[1], date=sys.argv[2])
