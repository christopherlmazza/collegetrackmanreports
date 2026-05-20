"""
rebuild_heatmap_priors.py — Build league-wide heatmap priors for empirical
Bayes shrinkage in the pitcher report.

Scans every parquet in data/by_date/, groups pitches by
(pitch_type, batter_side), and computes a Gaussian-KDE-smoothed 80x80 grid
per metric (run_value, xwoba, whiff).

Output: d1_heatmap_priors.json  (kept in repo root)

Run weekly or whenever you want the priors refreshed:
    python rebuild_heatmap_priors.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BY_DATE_DIR = os.path.join(SCRIPT_DIR, "data", "by_date")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "d1_heatmap_priors.json")

# Grid resolution (must match what generate_heatmap uses)
GRID_NX = 80
GRID_NY = 80
GRID_X_RANGE = (-2.5, 2.5)
GRID_Y_RANGE = (-0.5, 5.0)

# KDE bandwidth — controls smoothing. Matches generate_heatmap value.
KDE_BANDWIDTH = 0.4
# Spatial weighting sigma used when computing the weighted mean per cell.
SPATIAL_SIGMA = 0.3
# Density threshold below which the prior cell is set to NaN (no data).
DENSITY_THRESH_FRAC = 0.02
# Minimum pitches per (pitch_type, batter_side) to bother computing a prior.
MIN_PITCHES_PER_GROUP = 200

# Pitch type name canonicalization (matches utils.py PT_NORMALIZE)
PT_NORMALIZE = {
    "Four-Seam": "Fastball",
    "Two-Seam":  "Sinker",
    "Changeup":  "ChangeUp",
    "FourSeamFastBall": "Fastball",
    "TwoSeamFastBall":  "Sinker",
}

# Run value mapping (matches utils.py RUN_VALUES)
RUN_VALUES = {
    "StrikeSwinging": -0.065,
    "StrikeCalled":   -0.038,
    "FoulBallNotFieldable": -0.025,
    "BallCalled":     0.032,
    "BallinDirt":     0.032,
    "BallIntentional": 0.032,
    "HitByPitch":     0.035,
}

SWING_CALLS = {"StrikeSwinging", "FoulBallNotFieldable", "InPlay"}


# ── xwOBA helper (matches utils.py calc_xwoba) ──────────────────────────────
def calc_xwoba(ev, la):
    if pd.isna(ev) or pd.isna(la): return np.nan
    if ev < 50: return 0.05
    if la > 60: return 0.05
    if la > 50: return 0.05 + max(0, (ev - 90)) * 0.01
    if la < -10: return 0.05
    if la < 10:
        if ev >= 100: return 0.50
        if ev >= 90: return 0.30
        if ev >= 80: return 0.20
        return 0.10
    if ev >= 105: base = 1.8
    elif ev >= 100: base = 1.4
    elif ev >= 95:  base = 0.9
    elif ev >= 90:  base = 0.55
    elif ev >= 85:  base = 0.35
    elif ev >= 80:  base = 0.25
    elif ev >= 70:  base = 0.15
    else:           base = 0.08
    la_opt = 24.0
    la_penalty = ((la - la_opt) / 20.0) ** 2
    modifier = max(0.3, 1.0 - la_penalty * 0.5)
    return min(base * modifier, 2.1)


# ── Metric extraction ───────────────────────────────────────────────────────
def compute_run_value(row):
    call = row.get("PitchCall", "")
    if call in RUN_VALUES:
        return RUN_VALUES[call]
    if call == "InPlay":
        ev = row.get("ExitSpeed", np.nan)
        la = row.get("LaunchAngle", np.nan)
        xw = calc_xwoba(ev, la)
        if pd.notna(xw):
            return (xw - 0.320) / 1.15
        return 0.0
    return 0.0


def compute_xwoba_val(row):
    call = row.get("PitchCall", "")
    if call == "InPlay":
        ev = row.get("ExitSpeed", np.nan)
        la = row.get("LaunchAngle", np.nan)
        xw = calc_xwoba(ev, la)
        return xw if pd.notna(xw) else np.nan
    elif call == "StrikeSwinging":
        return 0.0
    elif call == "StrikeCalled":
        return 0.05
    elif call in ("BallCalled", "BallinDirt"):
        return 0.4
    return np.nan


def compute_whiff_val(row):
    """1 = whiff, 0 = swing-no-whiff, NaN = no swing."""
    call = row.get("PitchCall", "")
    if call == "StrikeSwinging":
        return 1.0
    if call in SWING_CALLS:
        return 0.0
    return np.nan


# ── Smoothed grid computation ───────────────────────────────────────────────
def smoothed_grid(x, y, vals, Xi, Yi):
    """
    Compute a smoothed grid the same way generate_heatmap does:
    Gaussian-weighted local mean. Returns Zi (grid of values) and
    density (grid of pitch density, useful as a "trust" mask).
    """
    positions = np.vstack([x, y])
    try:
        kde = gaussian_kde(positions, bw_method=KDE_BANDWIDTH)
        density = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    except Exception:
        density = np.zeros_like(Xi)
        return np.full_like(Xi, np.nan), density

    Zi = np.zeros_like(Xi)
    W  = np.zeros_like(Xi)
    sigma2 = 2 * SPATIAL_SIGMA ** 2
    for px, py, pv in zip(x, y, vals):
        if not np.isfinite(pv):
            continue
        dist2 = (Xi - px) ** 2 + (Yi - py) ** 2
        w = np.exp(-dist2 / sigma2)
        Zi += w * pv
        W  += w
    W[W == 0] = 1
    Zi = Zi / W

    # Mask cells where density is too low (sparse areas)
    if density.max() > 0:
        thresh = density.max() * DENSITY_THRESH_FRAC
        Zi[density < thresh] = np.nan
    return Zi, density


# ── Main loop ───────────────────────────────────────────────────────────────
def main():
    if not os.path.isdir(BY_DATE_DIR):
        print(f"ERROR: {BY_DATE_DIR} not found")
        return

    files = sorted(f for f in os.listdir(BY_DATE_DIR) if f.endswith(".parquet"))
    if not files:
        print(f"ERROR: no parquet files in {BY_DATE_DIR}")
        return
    print(f"Found {len(files)} parquet files in by_date/")

    # Build the grid
    xi = np.linspace(*GRID_X_RANGE, GRID_NX)
    yi = np.linspace(*GRID_Y_RANGE, GRID_NY)
    Xi, Yi = np.meshgrid(xi, yi)

    # Collect all pitches into one DataFrame
    cols_needed = [
        "PitchCall", "PlateLocSide", "PlateLocHeight", "BatterSide",
        "TaggedPitchType", "AutoPitchType",
        "ExitSpeed", "LaunchAngle",
    ]
    parts = []
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(BY_DATE_DIR, fname)
        try:
            df = pd.read_parquet(fpath, columns=cols_needed)
            parts.append(df)
            if i % 10 == 0 or i == len(files):
                print(f"  loaded {i}/{len(files)}")
        except Exception as e:
            print(f"  WARN: failed to read {fname}: {e}")
    if not parts:
        print("ERROR: no data loaded")
        return

    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"Total pitches: {len(df):,}")

    # Pre-clean
    df = df.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    df["BatterSide"] = df["BatterSide"].astype(str).str.strip()
    df = df[df["BatterSide"].isin(["Left", "Right"])]

    # Resolve pitch type (Auto first, Tagged fallback), then canonicalize
    auto = df["AutoPitchType"].astype(str).str.strip()
    tag  = df["TaggedPitchType"].astype(str).str.strip()
    df["PitchType"] = auto.where(
        ~auto.isin(["", "Undefined", "nan", "None"]),
        tag.where(~tag.isin(["", "Undefined", "nan", "None"]), "Other")
    )
    df["PitchType"] = df["PitchType"].replace(PT_NORMALIZE)
    df = df[df["PitchType"] != "Other"]
    print(f"After cleanup: {len(df):,} pitches")
    print()

    # Compute metric values for every pitch (vectorized where possible)
    print("Computing metric values...")
    df["run_value"] = df.apply(compute_run_value, axis=1)
    df["xwoba"]     = df.apply(compute_xwoba_val, axis=1)
    df["whiff"]     = df.apply(compute_whiff_val, axis=1)

    # Mirror x to catcher POV (same as generate_heatmap does at render time)
    df["_x"] = -df["PlateLocSide"]
    df["_y"] = df["PlateLocHeight"]

    # Build the priors dict
    priors = {
        "_meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "grid_x_range": list(GRID_X_RANGE),
            "grid_y_range": list(GRID_Y_RANGE),
            "grid_nx": GRID_NX,
            "grid_ny": GRID_NY,
            "metrics": ["run_value", "xwoba", "whiff"],
            "kde_bandwidth": KDE_BANDWIDTH,
            "spatial_sigma": SPATIAL_SIGMA,
            "n_pitches_total": int(len(df)),
        },
    }

    pitch_types = sorted(df["PitchType"].unique())
    for pt in pitch_types:
        pt_df = df[df["PitchType"] == pt]
        priors[pt] = {}
        for side in ("Left", "Right"):
            sd = pt_df[pt_df["BatterSide"] == side]
            if len(sd) < MIN_PITCHES_PER_GROUP:
                print(f"  skip {pt:<12s} vs {side[0]}HB: only {len(sd)} pitches")
                continue
            x = sd["_x"].values
            y = sd["_y"].values
            entry = {"n": int(len(sd))}
            for metric in ("run_value", "xwoba", "whiff"):
                vals = sd[metric].values
                # whiff and xwoba can have NaN; filter to valid pitches
                valid = np.isfinite(vals)
                if valid.sum() < MIN_PITCHES_PER_GROUP:
                    entry[metric] = None
                    continue
                Zi, _ = smoothed_grid(x[valid], y[valid], vals[valid], Xi, Yi)
                # Convert NaN to None so JSON can serialize
                entry[metric] = [
                    [(None if not np.isfinite(v) else round(float(v), 5))
                     for v in row]
                    for row in Zi
                ]
            priors[pt][side] = entry
            print(f"  {pt:<12s} vs {side[0]}HB: {len(sd):>6,} pitches -> grid built")

    print()
    print(f"Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(priors, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()