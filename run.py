import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


from pull_temple_data import (
    fetch_all_signals,
    fetch_sleep_windows_for_user,
    get_user_id_by_email,
)



EMAILS_PATH = "sleep/source_emails.txt"
START_DATE = "2026-02-01"
END_DATE = "2026-05-25"
SLEEP_OUTPUT_DIR = "sleep/temple"
MAX_WORKERS = 6  # parallel users; ClickHouse instance can handle this

def main():
    emails = [
        line.strip() for line in Path(EMAILS_PATH).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    tasks = []
    for email in emails:
        user_id = get_user_id_by_email(email)
        if not user_id:
            print(f"✗ no user_id for {email}")
            continue
        safe_email = email.replace("@", "_at_").replace("+", "_plus_").replace(".", "_")
        save_path = os.path.join(SLEEP_OUTPUT_DIR, f"{safe_email}.csv")
        tasks.append((email, user_id, save_path))

    print(f"\nFetching {len(tasks)} users with {MAX_WORKERS} workers …\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                fetch_sleep_windows_for_user,
                user_id, save_path, START_DATE, END_DATE,
            ): email
            for email, user_id, save_path in tasks
        }
        for fut in as_completed(futures):
            email = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"✗ {email}: {e}")

    for filepath in glob.glob('sleep/temple/*.csv'):
        temple_df = pd.read_csv(filepath)
        if temple_df.empty:
            continue

        g = temple_df.loc[temple_df['green'].notna(), ['timestamp', 'green']]
        a = temple_df.loc[temple_df['accel_z'].notna(), ['timestamp', 'accel_x', 'accel_y', 'accel_z']]
        ga = pd.merge_asof(
            g.sort_values('timestamp'),
            a.sort_values('timestamp'),
            on='timestamp',
            tolerance=60,
            direction='nearest',
        ).dropna(subset=['accel_z']).reset_index(drop=True)
        if ga.empty:
            continue

        ga.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()