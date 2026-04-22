import pandas as pd
import os
import glob
import re

by_date_dir = "data/by_date"
dfs = []

files = sorted(glob.glob(os.path.join(by_date_dir, "*.parquet")))
print(f"Found {len(files)} date files. Rebuilding index...")

for f in files:
    # Extract date from filename like "2026-04-18.parquet"
    fname = os.path.basename(f)
    match = re.match(r"(\d{4}-\d{2}-\d{2})\.parquet", fname)
    if not match:
        continue
    file_date = match.group(1)
    
    df = pd.read_parquet(f, columns=["GameDate","HomeTeam","AwayTeam","SessionID"])
    # Fill in missing GameDate from filename
    df["GameDate"] = df["GameDate"].fillna(file_date)
    # Also override any None string with the filename date
    mask = df["GameDate"].isna() | (df["GameDate"].astype(str).str.lower() == "none")
    df.loc[mask, "GameDate"] = file_date
    
    uniq = df.drop_duplicates(subset=["GameDate","HomeTeam","AwayTeam"])
    print(f"  {fname}: {len(uniq)} game(s)")
    dfs.append(df)

if not dfs:
    print("No date files found!")
else:
    combined = pd.concat(dfs, ignore_index=True)
    index = combined.drop_duplicates(subset=["GameDate","HomeTeam","AwayTeam"]).reset_index(drop=True)
    index["GameDate"] = pd.to_datetime(index["GameDate"], errors="coerce").dt.date
    index = index[index["GameDate"].notna()]
    index.to_parquet("data/index.parquet", index=False)
    print(f"\nIndex rebuilt successfully:")
    print(f"  Games : {len(index)}")
    print(f"  From  : {index['GameDate'].min()}")
    print(f"  To    : {index['GameDate'].max()}")
    print(f"  Saved : {os.path.abspath('data/index.parquet')}")