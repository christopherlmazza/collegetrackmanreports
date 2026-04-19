import pandas as pd
import os
import glob
import re

by_date_dir = "data/by_date"
files = sorted(glob.glob(os.path.join(by_date_dir, "*.parquet")))

print(f"Fixing {len(files)} parquet files...")
fixed = 0
for f in files:
    fname = os.path.basename(f)
    match = re.match(r"(\d{4}-\d{2}-\d{2})\.parquet", fname)
    if not match:
        continue
    file_date = match.group(1)
    
    df = pd.read_parquet(f)
    before_none = df["GameDate"].isna().sum()
    # Also treat "None" string and empty string as missing
    mask = df["GameDate"].isna() | (df["GameDate"].astype(str).str.lower().isin(["none", "nan", ""]))
    if mask.any():
        df.loc[mask, "GameDate"] = file_date
        df.to_parquet(f, index=False)
        fixed += 1
        print(f"  {fname}: filled {mask.sum()} missing dates")

print(f"\nFixed {fixed} files")