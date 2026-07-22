"""
config.py — Central path configuration for TrackMan data storage.

Works in BOTH environments automatically:
  - Local dev:       data lives on Google Drive (G:\\My Drive\\trackman_data)
  - Streamlit Cloud: data lives in the repo's data/ folder (committed to GitHub)

DATA_DIR auto-detects: it uses the Google Drive path if that folder exists
(your local machine), otherwise it falls back to the repo's own data/ folder
(what Streamlit Cloud sees at /mount/src/<repo>/data).
"""
import os

# Google Drive location on your local machine.
# If your Drive mounts at a letter other than G:, change it here.
GDRIVE_DATA = r"G:\My Drive\trackman_data"

# The repo's own data/ folder (used on Streamlit Cloud, where G:\ doesn't exist).
REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATA_DIR    = GDRIVE_DATA if os.path.isdir(GDRIVE_DATA) else REPO_DATA
BY_DATE_DIR = os.path.join(DATA_DIR, "by_date")
INDEX_PATH  = os.path.join(DATA_DIR, "index.parquet")
LEGACY_PATH = os.path.join(DATA_DIR, "pitches.parquet")