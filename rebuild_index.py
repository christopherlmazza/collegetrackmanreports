"""
rebuild_index.py
================
One-time utility to rebuild data/index.parquet from all existing
parquet files in data/by_date/.

Run this whenever:
- The index is missing dates that exist in by_date/
- SEASON_START was changed and old games dropped out of the index
- You manually added parquet files to by_date/

Usage:
    python rebuild_index.py
"""

import os
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BY_DATE_DIR = os.path.join(DATA_DIR, "by_date")
INDEX_PATH  = os.path.join(DATA_DIR, "index.parquet")


def rebuild():
    if not os.path.exists(BY_DATE_DIR):
        print(f"ERROR: {BY_DATE_DIR} does not exist.")
        return

    files = sorted(f for f in os.listdir(BY_DATE_DIR) if f.endswith(".parquet"))
    if not files:
        print("No parquet files found in data/by_date/")
        return

    print(f"Found {len(files)} date files. Rebuilding index...")

    rows = []
    errors = []

    for fname in files:
        fpath = os.path.join(BY_DATE_DIR, fname)
        try:
            df = pd.read_parquet(fpath, columns=["SessionID", "GameDate", "HomeTeam", "AwayTeam"])
            games = df.drop_duplicates(subset=["GameDate", "HomeTeam", "AwayTeam"])
            for _, row in games.iterrows():
                rows.append({
                    "SessionID": str(row.get("SessionID", "")),
                    "GameDate":  row["GameDate"],
                    "HomeTeam":  row["HomeTeam"],
                    "AwayTeam":  row["AwayTeam"],
                })
            print(f"  {fname}: {len(games)} game(s)")
        except Exception as e:
            print(f"  {fname}: ERROR — {e}")
            errors.append(fname)

    if not rows:
        print("No rows collected — index not written.")
        return

    idx = pd.DataFrame(rows).drop_duplicates()
    idx["GameDate"] = pd.to_datetime(idx["GameDate"], errors="coerce").dt.date
    idx = idx.dropna(subset=["GameDate"]).sort_values("GameDate").reset_index(drop=True)
    idx.to_parquet(INDEX_PATH, index=False)

    print(f"\nIndex rebuilt successfully:")
    print(f"  Games : {len(idx)}")
    print(f"  From  : {idx['GameDate'].min()}")
    print(f"  To    : {idx['GameDate'].max()}")
    print(f"  Saved : {INDEX_PATH}")

    if errors:
        print(f"\nFiles with errors ({len(errors)}):")
        for f in errors:
            print(f"  {f}")


if __name__ == "__main__":
    rebuild()