"""
fetch_data.py
=============
Runs once daily to pull all 2026 D1 college baseball data from the
TrackMan API and write it to partitioned parquet files.

New storage layout:
    data/
        index.parquet          -- lightweight game index (date, teams, session ID)
        by_date/
            2026-02-13.parquet -- all pitches from that date
            2026-02-14.parquet
            ...
        last_updated.json

On first run, automatically migrates existing data/pitches.parquet
into the new by_date/ structure.

Manual run:
    python fetch_data.py
"""

import requests, json, os, sys, time
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from collections import defaultdict

# ===========================================================================
# CREDENTIALS
# ===========================================================================
BASE_URL      = "https://dataapi.trackmanbaseball.com"
TOKEN_URL     = "https://login.trackman.com/connect/token"
CLIENT_ID     = "LongIslandUniversity-02"
CLIENT_SECRET = "3406f40b-d596-41ff-8110-808d7a4ef38d"
SEASON_START  = date(2026, 2, 1)

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BY_DATE_DIR = os.path.join(DATA_DIR, "by_date")
INDEX_PATH  = os.path.join(DATA_DIR, "index.parquet")
LEGACY_PATH = os.path.join(DATA_DIR, "pitches.parquet")

# ===========================================================================
# AUTH
# ===========================================================================
_token_cache = {"token": None, "expires": 0}

def get_token():
    if time.time() < _token_cache["expires"] - 60:
        return _token_cache["token"]
    resp = requests.post(TOKEN_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }, timeout=30)
    if resp.status_code != 200:
        print(f"  Auth failed: {resp.status_code} -- {resp.text[:200]}")
        return None
    data = resp.json()
    _token_cache["token"]   = data["access_token"]
    _token_cache["expires"] = time.time() + data.get("expires_in", 3600)
    return _token_cache["token"]

def get_headers():
    token = get_token()
    if not token: return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }

# ===========================================================================
# HELPERS
# ===========================================================================
def sg(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict): d = d.get(k, default)
        else: return default
    return d

def sf(v):
    if v is None: return np.nan
    try:    return float(v)
    except: return np.nan

def is_d1_session(s):
    lv      = s.get("level", "")
    lv_text = (" ".join(str(v) for v in lv.values()).upper()
               if isinstance(lv, dict) else str(lv).upper())
    return any(kw in lv_text for kw in ["D1","NCAA-D1","DIVISION 1","DIV1","DIV-1"])

# ===========================================================================
# API FETCHERS
# ===========================================================================
def fetch_sessions(date_from_str, date_to_str):
    """
    Try multiple approaches to get sessions:
    1. POST with sessionType (original)
    2. POST without sessionType
    3. POST with different field names
    4. GET request instead of POST
    """
    headers = get_headers()
    if not headers: return []

    payloads = [
        {"sessionType": "All", "utcDateFrom": date_from_str, "utcDateTo": date_to_str},
        {"utcDateFrom": date_from_str, "utcDateTo": date_to_str},
        {"sessionType": "Game", "utcDateFrom": date_from_str, "utcDateTo": date_to_str},
        {"dateFrom": date_from_str, "dateTo": date_to_str},
    ]

    for i, payload in enumerate(payloads):
        for attempt in range(3):
            try:
                headers = get_headers()
                if not headers: return []

                resp = requests.post(
                    f"{BASE_URL}/api/v1/discovery/game/sessions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    print(f"  Rate limited, waiting {wait}s..."); time.sleep(wait); continue
                if resp.status_code == 500:
                    print(f"  500 error with payload variant {i+1}, trying next...")
                    break  # try next payload
                if resp.ok:
                    data = resp.json()
                    result = data if isinstance(data, list) else data.get("sessions", [])
                    if result:
                        print(f"  Success with payload variant {i+1}")
                        return result
                else:
                    print(f"  Error {resp.status_code} with payload variant {i+1}")
                    break
            except Exception as e:
                print(f"  Exception: {e}")
                time.sleep(15)

    # Last resort — try GET with query params
    try:
        headers = get_headers()
        resp = requests.get(
            f"{BASE_URL}/api/v1/discovery/game/sessions",
            headers=headers,
            params={"utcDateFrom": date_from_str, "utcDateTo": date_to_str},
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            result = data if isinstance(data, list) else data.get("sessions", [])
            if result:
                print("  Success with GET request")
                return result
    except Exception as e:
        print(f"  GET attempt failed: {e}")

    print("  All session fetch attempts failed.")
    return []

def fetch_game_data(session_id):
    for attempt in range(4):
        try:
            headers = get_headers()
            if not headers: return [], []
            r = requests.get(f"{BASE_URL}/api/v1/data/game/plays/{session_id}",
                             headers=headers, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                print(f"  Rate limited, waiting {wait}s..."); time.sleep(wait); continue
            if r.status_code != 200: return [], []
            plays = r.json()
            if not isinstance(plays, list): return [], []
            r = requests.get(f"{BASE_URL}/api/v1/data/game/balls/{session_id}",
                             headers=headers, timeout=30)
            if r.status_code != 200: return plays, []
            return plays, r.json()
        except Exception as e:
            wait = 15*(attempt+1)
            print(f"  Connection error (attempt {attempt+1}/4), retrying in {wait}s: {e}")
            time.sleep(wait)
    return [], []

# ===========================================================================
# DATA FLATTENING
# ===========================================================================
def flatten_game(plays_raw, balls_raw, session):
    ht         = session.get("homeTeam", {}).get("name", "")
    at         = session.get("awayTeam", {}).get("name", "")
    game_date  = (session.get("gameDateLocal") or session.get("gameDateUtc") or "")[:10]
    session_id = session.get("sessionId", "")

    rows = []
    for p in plays_raw:
        rows.append({
            "PlayID":          p.get("playID"),
            "SessionID":       session_id,
            "GameDate":        game_date,
            "HomeTeam":        ht,
            "AwayTeam":        at,
            "PitchNo":         sg(p, "taggerBehavior", "pitchNo"),
            "PAofInning":      sg(p, "taggerBehavior", "pAofinning"),
            "PitchofPA":       sg(p, "taggerBehavior", "pitchofPA"),
            "Pitcher":         sg(p, "pitcher",   "name",      default=""),
            "PitcherTeam":     sg(p, "pitcher",   "team",      default=""),
            "PitcherThrows":   sg(p, "pitcher",   "throwHand", default=""),
            "Batter":          sg(p, "batter",    "name",      default=""),
            "BatterSide":      sg(p, "batter",    "side",      default=""),
            "Inning":          sg(p, "gameState", "inning"),
            "TopBottom":       sg(p, "gameState", "topBottom"),
            "Outs":            sg(p, "gameState", "outs"),
            "Balls":           sg(p, "gameState", "balls"),
            "Strikes":         sg(p, "gameState", "strikes"),
            "TaggedPitchType": sg(p, "pitchTag",  "taggedPitchType", default=""),
            "AutoPitchType":   sg(p, "pitchTag",  "autoPitchType",   default=""),
            "PitchCall":       sg(p, "pitchTag",  "pitchCall",       default=""),
            "KorBB":           p.get("korBB", ""),
            "PlayResult":      sg(p, "playResult", "playResult",  default=""),
            "OutsOnPlay":      sf(sg(p, "playResult", "outsOnPlay",  default=0)),
            "RunsScored":      sf(sg(p, "playResult", "runsScored",  default=0)),
        })

    plays_df = pd.DataFrame(rows)
    if plays_df.empty: return plays_df
    plays_df["PitchNo"] = pd.to_numeric(plays_df["PitchNo"], errors="coerce")
    plays_df = plays_df.sort_values("PitchNo").reset_index(drop=True)

    pitch_rows, hit_rows = [], []
    for b in balls_raw:
        kind = b.get("kind",""); pid = b.get("playId")
        if kind == "Pitch":
            pitch_rows.append({
                "PlayID":           pid,
                "RelSpeed":         sf(sg(b,"pitch","release","relSpeed")),
                "SpinRate":         sf(sg(b,"pitch","release","spinRate")),
                "Extension":        sf(sg(b,"pitch","release","extension")),
                "RelHeight":        sf(sg(b,"pitch","release","relHeight")),
                "RelSide":          sf(sg(b,"pitch","release","relSide")),
                "HorzBreak":        sf(sg(b,"pitch","movement","horzBreak")),
                "InducedVertBreak": sf(sg(b,"pitch","movement","inducedVertBreak")),
                "PlateLocHeight":   sf(sg(b,"pitch","location","plateLocHeight")),
                "PlateLocSide":     sf(sg(b,"pitch","location","plateLocSide")),
                "VertApprAngle":    sf(sg(b,"pitch","location","vertApprAngle")),
            })
        elif kind == "Hit":
            hit_rows.append({
                "PlayID":      pid,
                "ExitSpeed":   sf(sg(b,"hit","launch","exitSpeed")),
                "LaunchAngle": sf(sg(b,"hit","launch","angle")),
            })

    pbdf = pd.DataFrame(pitch_rows).drop_duplicates("PlayID",keep="first") if pitch_rows else pd.DataFrame()
    hbdf = pd.DataFrame(hit_rows).drop_duplicates("PlayID",keep="first")   if hit_rows  else pd.DataFrame()

    df = plays_df
    if not pbdf.empty:
        df = df.merge(pbdf, on="PlayID", how="left")
    else:
        for c in ["RelSpeed","SpinRate","Extension","RelHeight","RelSide",
                  "HorzBreak","InducedVertBreak","PlateLocHeight","PlateLocSide","VertApprAngle"]:
            df[c] = np.nan
    if not hbdf.empty:
        df = df.merge(hbdf, on="PlayID", how="left")
    else:
        for c in ["ExitSpeed","LaunchAngle"]:
            df[c] = np.nan

    return df.drop_duplicates("PlayID", keep="first")

def resolve_pt(row):
    t = row.get("TaggedPitchType","") or ""
    a = row.get("AutoPitchType","")   or ""
    for v in [t, a]:
        if v in ("Fastball","FourSeamFastBall"):   return "Fastball"
        if v in ("Sinker","TwoSeamFastBall"):       return "Sinker"
        if v == "Cutter":                           return "Cutter"
        if v in ("Slider","Sweeper"):               return "Slider"
        if v in ("Curveball","CurveBall"):          return "Curveball"
        if v in ("ChangeUp","Changeup","Splitter"): return "ChangeUp"
    return "Other"

def optimize_df(df):
    """Downcast floats and categorize to save memory."""
    # Normalize GameDate to string YYYY-MM-DD to avoid mixed type issues
    if "GameDate" in df.columns:
        df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    cat_cols = ["PitcherTeam","PitcherThrows","BatterSide","TopBottom",
                "TaggedPitchType","AutoPitchType","PitchCall","KorBB",
                "PlayResult","PitchType","HomeTeam","AwayTeam"]
    for col in cat_cols:
        if col in df.columns and df[col].nunique() < 500:
            df[col] = df[col].astype("category")
    return df

# ===========================================================================
# INDEX MANAGEMENT
# ===========================================================================
def load_index():
    if not os.path.exists(INDEX_PATH):
        return pd.DataFrame(columns=["SessionID","GameDate","HomeTeam","AwayTeam"])
    return pd.read_parquet(INDEX_PATH)

def save_index(sessions):
    rows = []
    for s in sessions:
        rows.append({
            "SessionID": s.get("sessionId",""),
            "GameDate":  (s.get("gameDateLocal") or s.get("gameDateUtc") or "")[:10],
            "HomeTeam":  s.get("homeTeam",{}).get("name",""),
            "AwayTeam":  s.get("awayTeam",{}).get("name",""),
        })
    df = pd.DataFrame(rows).drop_duplicates("SessionID")
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
    df.to_parquet(INDEX_PATH, index=False)
    print(f"  Index saved: {len(df)} games")
    return df

# ===========================================================================
# MIGRATION
# ===========================================================================
def migrate_legacy():
    if not os.path.exists(LEGACY_PATH):
        return
    print(f"\n  Migrating legacy pitches.parquet...")
    df = pd.read_parquet(LEGACY_PATH)
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
    os.makedirs(BY_DATE_DIR, exist_ok=True)
    migrated = 0
    for gdate, group in df.groupby("GameDate"):
        if pd.isna(gdate): continue
        # Normalize date to YYYY-MM-DD string safe for filenames
        try:
            gdate_str = pd.Timestamp(gdate).strftime("%Y-%m-%d")
        except Exception:
            gdate_str = str(gdate).replace("/", "-")[:10]
        out_path = os.path.join(BY_DATE_DIR, f"{gdate_str}.parquet")
        if not os.path.exists(out_path):
            optimize_df(group.copy()).to_parquet(out_path, index=False)
            migrated += 1
    print(f"  Migrated {migrated} date files to {BY_DATE_DIR}/")
    os.rename(LEGACY_PATH, LEGACY_PATH + ".bak")
    print(f"  Renamed pitches.parquet to pitches.parquet.bak")

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print(f"FETCH DATA -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(BY_DATE_DIR, exist_ok=True)

    migrate_legacy()

    existing_index       = load_index()
    existing_session_ids = set(existing_index["SessionID"].dropna().astype(str))
    print(f"\n  Existing: {len(existing_session_ids)} sessions")

    print(f"\n[1/3] Fetching sessions {SEASON_START} to today...")
    all_sessions = []
    chunk_start  = SEASON_START
    today        = date.today()

    while chunk_start <= today:
        chunk_end = min(chunk_start + timedelta(days=14), today + timedelta(days=1))
        sessions  = fetch_sessions(f"{chunk_start}T00:00:00Z", f"{chunk_end}T00:00:00Z")
        if sessions:
            all_sessions.extend(sessions)
            print(f"  {chunk_start} to {chunk_end}: {len(sessions)} sessions")
        time.sleep(1.0)
        chunk_start = chunk_end

    seen, unique = set(), []
    for s in all_sessions:
        sid = s.get("sessionId","")
        if sid and sid not in seen:
            seen.add(sid); unique.append(s)

    d1 = [s for s in unique if is_d1_session(s)]
    if not d1:
        print(f"  WARNING: D1 filter returned 0, using all {len(unique)} sessions")
        d1 = unique
    print(f"  D1 sessions: {len(d1)} of {len(unique)} total")

    new_sessions = [s for s in d1 if s.get("sessionId") not in existing_session_ids]
    print(f"  New sessions to fetch: {len(new_sessions)}")

    if not new_sessions:
        print("\n  Already up to date.")
        if d1:
            save_index(d1)
        _write_timestamp()
        return

    print(f"\n[2/3] Fetching pitch data ({len(new_sessions)} games)...")

    sessions_by_date = defaultdict(list)
    for s in new_sessions:
        raw = (s.get("gameDateLocal") or s.get("gameDateUtc") or "")[:10]
        # Normalize to YYYY-MM-DD regardless of source format
        try:
            gdate = pd.to_datetime(raw).strftime("%Y-%m-%d")
        except Exception:
            gdate = raw.replace("/", "-")
        sessions_by_date[gdate].append(s)

    total_new = 0
    processed = 0

    for gdate, date_sessions in sorted(sessions_by_date.items()):
        date_file = os.path.join(BY_DATE_DIR, f"{gdate}.parquet")
        new_dfs   = []

        for session in date_sessions:
            sid = session.get("sessionId")
            plays, balls = fetch_game_data(sid)
            if plays:
                df = flatten_game(plays, balls, session)
                if not df.empty:
                    df["PitchType"] = df.apply(resolve_pt, axis=1)
                    new_dfs.append(df)
            time.sleep(1.5)
            processed += 1
            if processed % 10 == 0:
                print(f"  {processed}/{len(new_sessions)} sessions processed")

        if not new_dfs:
            continue

        new_df = pd.concat(new_dfs, ignore_index=True)

        if os.path.exists(date_file):
            existing = pd.read_parquet(date_file)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates("PlayID", keep="first")
        else:
            combined = new_df

        optimize_df(combined).to_parquet(date_file, index=False)
        total_new += len(new_df)
        print(f"  {gdate}: {len(new_df)} pitches written")

    print(f"\n  Total new pitches: {total_new}")

    print(f"\n[3/3] Saving index...")
    save_index(d1)
    _write_timestamp()
    print(f"\n  Done! {len(new_sessions)} new games added.")

def _write_timestamp():
    ts_path = os.path.join(DATA_DIR, "last_updated.json")
    with open(ts_path, "w") as f:
        json.dump({
            "last_updated":      datetime.now().isoformat(),
            "last_updated_date": date.today().isoformat(),
        }, f, indent=2)

def _git_push():
    import subprocess
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n  Pushing data to GitHub from {repo_dir}...")

    cmds = [
        ["git", "add", "data/by_date/", "data/index.parquet", "data/last_updated.json"],
        ["git", "commit", "-m", f"data update {date.today()}"],
        ["git", "pull", "--no-edit"],
        ["git", "push"],
    ]
    for cmd in cmds:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=repo_dir,
                                capture_output=True, text=True)
        out = result.stdout.strip()
        err = result.stderr.strip()
        if out: print(f"  stdout: {out}")
        if err: print(f"  stderr: {err}")
        if result.returncode != 0:
            if "nothing to commit" in out + err:
                print("  Nothing new to push."); return
            if "Everything up-to-date" in out + err:
                print("  Already up to date."); return
            # Try pulling first then pushing
            print("  Push failed, trying pull then push...")
            pull = subprocess.run(["git", "pull", "--rebase"],
                                   cwd=repo_dir, capture_output=True, text=True)
            print(f"  pull: {pull.stdout.strip()} {pull.stderr.strip()}")
            retry = subprocess.run(["git", "push"],
                                    cwd=repo_dir, capture_output=True, text=True)
            if retry.returncode == 0:
                print("  Pushed to GitHub after pull.")
            else:
                print(f"  Push failed: {retry.stderr.strip()}")
            return
    print("  Pushed to GitHub.")

if __name__ == "__main__":
    main()
    _git_push()