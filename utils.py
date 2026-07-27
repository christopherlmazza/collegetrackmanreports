"""
utils.py — Shared functions for TrackMan Baseball Reports
All chart functions, data loaders, analytics library, and AI chatbot
are defined here and imported by pages/1_Pitchers.py and pages/2_Hitters.py.
"""
import streamlit as st
import json, os, io, math, warnings, requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from datetime import date, timedelta, datetime

# ───────── STUFF+ MODEL (LightGBM tjStuff+ — matches collegestuff01.py) ──────────
# Model file: stuff_plus_models/stuff_plus_model.pkl  (sklearn Pipeline w/ RobustScaler + LGBMRegressor)
# Scales:     stuff_plus_models/scales.json           ({"mean": float, "std": float, "features": [...]})
# Feature set expected by the trained model:
#   RelSpeed, SpinRate, Extension, InducedVertBreak,
#   ax_mirrored, x0_mirrored, RelHeight,
#   speed_diff, ivb_diff, hb_diff
# Mirroring: LHP → negate HorzBreak (ax), keep RelSide (x0); RHP → keep HorzBreak, negate RelSide.
# Diffs: relative to the pitcher's primary fastball (FF/SI/FC by usage; fallback = fastest pitch).
# Scoring formula:  stuff_plus = 100 - ((pred - mean) / std) * 10
_stuff_model  = None
_stuff_scales = {}
_stuff_status = ""
_stuff_debug  = []

# Display-label → Statcast-code (matches label_map in collegestuff01.py, reversed)
_PT_LABEL_TO_CODE = {
    "Four-Seam": "FF", "Fastball": "FF",
    "Sinker": "SI", "Two-Seam": "SI",
    "Cutter": "FC",
    "Slider": "SL",
    "Curveball": "CU",
    "Changeup": "CH",
    "Splitter": "FS",
}
_FASTBALL_LABELS = {"Four-Seam", "Fastball", "Sinker", "Two-Seam", "Cutter"}

STUFF_FEATURES = [
    "RelSpeed", "SpinRate", "Extension", "InducedVertBreak",
    "ax_mirrored", "x0_mirrored", "RelHeight",
    "speed_diff", "ivb_diff", "hb_diff",
]
# Raw columns that must be present (non-null) to compute Stuff+
_STUFF_RAW_COLS = ["RelSpeed", "SpinRate", "Extension", "InducedVertBreak",
                   "HorzBreak", "RelSide", "RelHeight", "PitchType"]

try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _model_dirs = [
        os.path.join(_script_dir, "stuff_plus_models"),
        os.path.join(os.getcwd(), "stuff_plus_models"),
        "stuff_plus_models",
    ]
    _stuff_debug.append(f"script_dir = {_script_dir}")
    _stuff_debug.append(f"cwd = {os.getcwd()}")
    for _md in _model_dirs:
        _stuff_debug.append(f"checking dir: {_md} -> exists={os.path.isdir(_md)}")
        if os.path.isdir(_md):
            try:
                _stuff_debug.append(f"  contents: {os.listdir(_md)}")
            except Exception as _e:
                _stuff_debug.append(f"  listdir failed: {_e}")

    import joblib
    _stuff_debug.append("joblib imported OK")

    _model_dir = None
    for _md in _model_dirs:
        if os.path.exists(os.path.join(_md, "stuff_plus_model.pkl")) and \
           os.path.exists(os.path.join(_md, "scales.json")):
            _model_dir = _md
            break

    if _model_dir is None:
        _stuff_status = "Stuff+ files not found (need stuff_plus_model.pkl + scales.json)"
        _stuff_debug.append(_stuff_status)
    else:
        _model_path  = os.path.join(_model_dir, "stuff_plus_model.pkl")
        _scales_path = os.path.join(_model_dir, "scales.json")
        _stuff_debug.append(f"loading model from {_model_path}")
        _stuff_debug.append(f"loading scales from {_scales_path}")

        _stuff_model = joblib.load(_model_path)
        with open(_scales_path) as f:
            _stuff_scales = json.load(f)

        # Sanity check the features list matches
        _expected = _stuff_scales.get("features", STUFF_FEATURES)
        if _expected != STUFF_FEATURES:
            _stuff_debug.append(f"WARNING: scales.json features {_expected} differ from code features {STUFF_FEATURES}")

        _stuff_status = (
            f"Stuff+ ready (mean={_stuff_scales['mean']:.4f}, "
            f"std={_stuff_scales['std']:.4f}, "
            f"n={_stuff_scales.get('n_train_pitches','?')})"
        )
        _stuff_debug.append(_stuff_status)
except ImportError as e:
    _stuff_status = f"import error (need joblib + lightgbm + scikit-learn): {e}"
    _stuff_debug.append(_stuff_status)
except Exception as e:
    _stuff_status = f"Stuff+ error: {e}"
    _stuff_debug.append(_stuff_status)

def get_stuff_status():
    """Return a multi-line diagnostic string about Stuff+ model loading."""
    return _stuff_status + "\n\n" + "\n".join(_stuff_debug)

def _infer_hand_from_relside(full_df):
    """Infer pitcher handedness the same way collegestuff01.py does:
    median RelSide < 0 → LHP. Returns 'L' or 'R'."""
    rs = full_df["RelSide"].dropna()
    if len(rs) == 0:
        return "R"
    return "L" if rs.median() < 0 else "R"

def _compute_primary_fastball(full_df, hand):
    """Find the pitcher's primary fastball from their full outing.
    Matches collegestuff01.py logic: highest-usage FF/SI/FC, fallback = fastest pitch.
    Returns (fb_speed, fb_ivb, fb_hb_mirrored) — floats or None if no data at all.
    Works on DISPLAY labels (PitchType column), not Statcast codes."""
    is_lhp = (hand == "L")

    fb_rows = full_df[full_df["PitchType"].isin(_FASTBALL_LABELS)].copy()
    # Apply the same mirror: LHP negate HorzBreak
    if not fb_rows.empty:
        fb_rows = fb_rows.dropna(subset=["RelSpeed", "InducedVertBreak", "HorzBreak"])

    if not fb_rows.empty:
        fb_rows["ax_mirrored"] = np.where(is_lhp, -fb_rows["HorzBreak"], fb_rows["HorzBreak"])
        # Pick the fastball type with the most pitches
        top_pt = fb_rows["PitchType"].value_counts().idxmax()
        grp = fb_rows[fb_rows["PitchType"] == top_pt]
        return (
            float(grp["RelSpeed"].mean()),
            float(grp["InducedVertBreak"].mean()),
            float(grp["ax_mirrored"].mean()),
        )

    # Fallback: no fastball thrown — use all pitches, speed=max, ivb=mean, hb=mean (mirrored)
    fallback = full_df.dropna(subset=["RelSpeed", "InducedVertBreak", "HorzBreak"])
    if fallback.empty:
        return (None, None, None)
    hb_mirrored = (-fallback["HorzBreak"]) if is_lhp else fallback["HorzBreak"]
    return (
        float(fallback["RelSpeed"].max()),
        float(fallback["InducedVertBreak"].mean()),
        float(hb_mirrored.mean()),
    )

def _build_stuff_features(sub, hand, fb_speed, fb_ivb, fb_hb):
    """Engineer the 10 model features exactly like collegestuff01.py."""
    is_lhp = (hand == "L")
    out = pd.DataFrame(index=sub.index)
    out["RelSpeed"]         = sub["RelSpeed"].astype(float)
    out["SpinRate"]         = sub["SpinRate"].astype(float)
    out["Extension"]        = sub["Extension"].astype(float)
    out["InducedVertBreak"] = sub["InducedVertBreak"].astype(float)
    out["ax_mirrored"]      = np.where(is_lhp, -sub["HorzBreak"].astype(float),  sub["HorzBreak"].astype(float))
    out["x0_mirrored"]      = np.where(is_lhp,  sub["RelSide"].astype(float),   -sub["RelSide"].astype(float))
    out["RelHeight"]        = sub["RelHeight"].astype(float)
    out["speed_diff"]       = out["RelSpeed"]         - fb_speed
    out["ivb_diff"]         = out["InducedVertBreak"] - fb_ivb
    out["hb_diff"]          = (out["ax_mirrored"]     - fb_hb).abs()   # TJ uses abs() on ax_diff
    return out[STUFF_FEATURES]

def score_stuff_plus(pitch_df, pitch_type, full_df=None):
    """Return mean Stuff+ for ONE pitch type in a pitcher's outing.

    Parameters
    ----------
    pitch_df : DataFrame filtered to ONE pitch type (e.g. all sliders in the outing)
    pitch_type : display label, e.g. "Four-Seam", "Slider" (not used for scoring — kept for API compat)
    full_df : the full outing DataFrame, needed for primary-fastball computation.
              If None, falls back to pitch_df (less accurate for non-fastballs).

    Returns float Stuff+ (clamped 40-160), or None if unavailable.
    """
    if _stuff_model is None or not _stuff_scales:
        return None
    if pitch_df is None or len(pitch_df) == 0:
        return None

    try:
        sub = pitch_df.dropna(subset=["RelSpeed", "SpinRate", "Extension",
                                      "InducedVertBreak", "HorzBreak",
                                      "RelSide", "RelHeight"])
        if len(sub) == 0:
            return None

        context = full_df if full_df is not None else pitch_df
        hand = _infer_hand_from_relside(context)
        fb_speed, fb_ivb, fb_hb = _compute_primary_fastball(context, hand)
        if fb_speed is None:
            return None

        X = _build_stuff_features(sub, hand, fb_speed, fb_ivb, fb_hb)
        # Drop any rows with NaN in engineered features (shouldn't happen after dropna above,
        # but diffs could be NaN if fb_* is NaN — guarded above)
        X = X.dropna()
        if len(X) == 0:
            return None

        preds = _stuff_model.predict(X)
        z = (preds - _stuff_scales["mean"]) / _stuff_scales["std"]
        sp = 100 - (z * 10)
        sp = np.clip(sp, 40, 160)
        return round(float(np.mean(sp)), 1)
    except Exception as e:
        _stuff_debug.append(f"score error for {pitch_type}: {e}")
        return None

warnings.filterwarnings("ignore")

# ===========================================================================
# PAGE CONFIG
# ===========================================================================

# ===========================================================================
# CONSTANTS
# ===========================================================================
STRIKE_CALLS = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
SWING_CALLS  = {"StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
PITCH_COLORS = {
    "Fastball": "#D32F2F", "FourSeamFastBall": "#D32F2F", "Four-Seam": "#D32F2F",
    "Sinker": "#E65100", "TwoSeamFastBall": "#E65100", "Two-Seam": "#E65100",
    "Cutter": "#B8A000", "Slider": "#00897B", "Curveball": "#1565C0",
    "ChangeUp": "#F9A825", "Changeup": "#F9A825",
    "Splitter": "#00796B", "Sweeper": "#7B1FA2",
    "Knuckleball": "#9E9E9E", "Other": "#888888",
}
BG_COLOR = "#FFFFFF"
PANEL_COLOR = "#F7F8FA"
GRID_COLOR = "#D5D8DC"
TEXT_COLOR = "#1A1A2E"
ACCENT_COLOR = "#1565C0"
MUTED_TEXT = "#6B7280"
AUTO_CORRECT_PITCHES = True
MIN_CLUSTER_SIZE = 3

# ===========================================================================
# D1 PERCENTILE COLOR GRADING
# ===========================================================================
GRADE_CMAP = LinearSegmentedColormap.from_list("grade", [
    (0.0, "#4575B4"), (0.25, "#91BFDB"), (0.5, "#FFFFFF"),
    (0.75, "#FDB863"), (1.0, "#E66101"),
])

@st.cache_data(ttl=3600)
def load_percentiles():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    paths = [
        os.path.join(base, "D1_percentiles.json"),
        os.path.join(os.path.expanduser("~"), "Downloads", "D1_percentiles.json"),
        "D1_percentiles.json",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

D1_PCTLS = load_percentiles()

@st.cache_data(ttl=3600)
def load_heatmap_priors():
    """Load D1 league-wide heatmap priors (run_value/xwoba/whiff smoothed grids
    per pitch type × batter side). Built by rebuild_heatmap_priors.py."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    paths = [
        os.path.join(base, "d1_heatmap_priors.json"),
        os.path.join(os.path.expanduser("~"), "Downloads", "d1_heatmap_priors.json"),
        "d1_heatmap_priors.json",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    raw = json.load(f)
                # Convert each metric grid (list of lists with None for NaN)
                # back into a numpy array with NaN, for use at render time.
                out = {"_meta": raw.get("_meta", {})}
                for pt, sides in raw.items():
                    if pt == "_meta":
                        continue
                    out[pt] = {}
                    for side, entry in sides.items():
                        side_out = {"n": entry.get("n", 0)}
                        for metric in ("run_value", "xwoba", "whiff"):
                            grid = entry.get(metric)
                            if grid is None:
                                side_out[metric] = None
                                continue
                            arr = np.array(
                                [[np.nan if v is None else v for v in row]
                                 for row in grid],
                                dtype=float,
                            )
                            side_out[metric] = arr
                        out[pt][side] = side_out
                return out
            except Exception as e:
                print(f"Failed to load heatmap priors: {e}")
                return {}
    return {}

D1_HEATMAP_PRIORS = load_heatmap_priors()

def get_percentile(pitch_type, stat_name, value):
    if not D1_PCTLS: return None
    pt_map = {"FourSeamFastBall": "Fastball", "TwoSeamFastBall": "Sinker", "Changeup": "ChangeUp"}
    pt = pt_map.get(pitch_type, pitch_type)
    pt_data = D1_PCTLS.get(pt, {}).get(stat_name, {})
    pctls = pt_data.get("percentiles", {})
    if not pctls: return None
    pts_list = sorted([(int(k), v) for k, v in pctls.items()], key=lambda x: x[0])
    if value <= pts_list[0][1]: return pts_list[0][0]
    if value >= pts_list[-1][1]: return pts_list[-1][0]
    for i in range(len(pts_list) - 1):
        p0, v0 = pts_list[i]; p1, v1 = pts_list[i + 1]
        if v0 <= value <= v1:
            if v1 == v0: return (p0 + p1) / 2
            return p0 + (value - v0) / (v1 - v0) * (p1 - p0)
        elif v0 >= value >= v1:
            if v0 == v1: return (p0 + p1) / 2
            return p0 + (v0 - value) / (v0 - v1) * (p1 - p0)
    return None

def grade_color(pitch_type, stat_name, value, higher_is_better=True):
    pctile = get_percentile(pitch_type, stat_name, value)
    if pctile is None: return None
    norm = pctile / 100.0
    if not higher_is_better: norm = 1.0 - norm
    return GRADE_CMAP(norm)

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================
def pc(pt): return PITCH_COLORS.get(pt, "#C8C8C8")

def sg(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict): d = d.get(k, default)
        else: return default
    return d

def sf(v):
    if v is None: return np.nan
    try: return float(v)
    except: return np.nan

# Canonical pitch-type names used everywhere in the app (colors, HAVAA,
# percentiles, season-summary display). Maps every TrackMan Auto/Tagged spelling
# (and common abbreviations) to one canonical label.
PITCH_TYPE_CANON = {
    "fastball": "Fastball", "fourseam": "Fastball", "fourseamfastball": "Fastball",
    "4seam": "Fastball", "ff": "Fastball",
    "sinker": "Sinker", "twoseam": "Sinker", "twoseamfastball": "Sinker",
    "2seam": "Sinker", "si": "Sinker", "ft": "Sinker",
    "cutter": "Cutter", "fc": "Cutter",
    "slider": "Slider", "sl": "Slider", "slurve": "Slider",
    "sweeper": "Sweeper", "st": "Sweeper",
    "curveball": "Curveball", "curve": "Curveball", "cu": "Curveball",
    "kc": "Curveball", "knucklecurve": "Curveball",
    "changeup": "ChangeUp", "ch": "ChangeUp",
    "splitter": "Splitter", "splitfinger": "Splitter", "fs": "Splitter",
    "forkball": "Splitter",
    "knuckleball": "Knuckleball", "kn": "Knuckleball",
}

def _canon_pt(name):
    """Map any pitch-type spelling to a canonical label, or None if unknown."""
    key = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    return PITCH_TYPE_CANON.get(key)

def resolve_pt(row):
    """
    Resolve a row's canonical BASE pitch type.

    Source of truth is the human TaggedPitchType — it's present on ~96% of pitches
    and is the most accurate source. (Validation against 4,225 properly-tagged MLB
    Statcast pitches showed movement-only classification tops out ~79%, and
    TrackMan's AutoPitchType is unreliable in college.) Fall back to AutoPitchType,
    then "Other". Names are normalized to canonical labels ("Four-Seam" ->
    "Fastball", "Changeup" -> "ChangeUp").

    Sweeper / Slurve / Sinker carve-outs are applied later, from pitch movement,
    in classify_pitch_types() — those are the distinctions humans/auto miss.
    """
    t = str(row.get("TaggedPitchType", "") or "").strip()
    a = str(row.get("AutoPitchType", "") or "").strip()
    for v in (t, a):
        if v and v not in ("", "Undefined", "nan", "None"):
            c = _canon_pt(v)
            if c:
                return c
    return "Other"

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
    elif ev >= 95: base = 0.9
    elif ev >= 90: base = 0.55
    elif ev >= 85: base = 0.35
    elif ev >= 80: base = 0.25
    elif ev >= 70: base = 0.15
    else: base = 0.08
    la_opt = 24.0
    la_penalty = ((la - la_opt) / 20.0) ** 2
    modifier = max(0.3, 1.0 - la_penalty * 0.5)
    return min(base * modifier, 2.1)

def calc_ip(pd_):
    total = 0
    for (_, _), grp in pd_.groupby(["Inning", "PAofInning"]):
        last = grp.loc[grp["PitchNo"].idxmax()]
        oop = last["OutsOnPlay"]
        korbb = last.get("KorBB", "")
        result = last.get("PlayResult", "")
        if pd.notna(oop) and oop > 0:
            total += int(oop)
        elif korbb == "Strikeout":
            reached = result in ("Single", "Double", "Triple", "HomeRun",
                                 "Error", "FieldersChoice", "CaughtStealing",
                                 "ReachedOnError")
            if not reached:
                total += 1
    return f"{total // 3}.{total % 3}"

def calc_pa(pd_): return pd_.groupby(["Inning", "PAofInning"]).ngroups

def calc_er(pd_):
    total = 0
    for (_, _), grp in pd_.groupby(["Inning", "PAofInning"]):
        last = grp.loc[grp["PitchNo"].idxmax()]
        rs = last["RunsScored"]
        if pd.notna(rs) and rs > 0:
            total += int(rs)
        elif last["PlayResult"] == "HomeRun":
            total += 1
    return total

def in_zone(s):
    return (s["PlateLocSide"].notna() & s["PlateLocHeight"].notna() &
            (s["PlateLocSide"].abs() <= 0.95) &
            (s["PlateLocHeight"] >= 1.6) & (s["PlateLocHeight"] <= 3.5))

# ===========================================================================
# RULE-BASED PITCH CLASSIFIER
# ===========================================================================
# Replaces TaggedPitchType / AutoPitchType. For each pitcher, we cluster their
# pitches in (velo, IVB, HB, spin) space, identify their primary fastball, and
# label every other cluster RELATIVE to that fastball. This matches how college
# pitchers actually throw — pitch labels are relative to a pitcher's arsenal,
# not absolute movement ranges. RHP rules; HB is mirrored for LHP so the same
# rules work for both.

def _infer_hand_for_pitcher(pdf):
    """LHP if median RelSide < 0, else RHP. Returns 'L' or 'R'."""
    rs = pd.to_numeric(pdf.get("RelSide"), errors="coerce").dropna()
    if len(rs) == 0:
        return "R"
    return "L" if rs.median() < 0 else "R"

def _cluster_pitcher_pitches(pdf, min_cluster_size=5):
    """
    Cluster a single pitcher's pitches in (velo, IVB, HB_mirrored, spin) space.
    Uses DBSCAN for shape-flexible clusters (pitchers don't all have N pitches).
    Returns the input df with a new '_cluster' int column. -1 = noise/uncluster.
    """
    feats = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate"]
    pdf = pdf.copy()
    pdf["_hb_mirror"] = pdf["HorzBreak"]
    is_lhp = _infer_hand_for_pitcher(pdf) == "L"
    if is_lhp:
        pdf["_hb_mirror"] = -pdf["_hb_mirror"]

    valid = pdf[feats].notna().all(axis=1) & pdf["_hb_mirror"].notna()
    pdf["_cluster"] = -1
    if valid.sum() < min_cluster_size:
        return pdf, is_lhp

    X = pdf.loc[valid, ["RelSpeed", "InducedVertBreak", "_hb_mirror", "SpinRate"]].values
    # Normalize: velo and spin have very different scales than IVB/HB.
    # Use scale factors that roughly equalize their importance.
    scales = np.array([2.0, 4.0, 4.0, 250.0])  # velo, ivb, hb, spin
    Xn = X / scales

    try:
        from sklearn.cluster import DBSCAN
        # Moderately tight eps so fastball/changeup mixes (similar movement,
        # 5-10 mph velo gap) get separated into different clusters. The
        # post-clustering merge logic handles any over-splitting downstream
        # by collapsing same-labeled clusters and re-labeling slower duplicates.
        eps = 1.0
        min_samp = max(3, min(5, int(valid.sum() * 0.03)))
        labels = DBSCAN(eps=eps, min_samples=min_samp).fit_predict(Xn)
        pdf.loc[valid, "_cluster"] = labels
    except Exception:
        pdf["_cluster"] = -1
    return pdf, is_lhp

def _label_cluster(cluster_stats, fb_stats, is_primary_fb):
    """
    Assign a pitch label to a single cluster.
    cluster_stats: dict with 'velo','ivb','hb_mirror','spin' (means for cluster)
    fb_stats: same, for the primary fastball cluster (the hardest cluster)
    is_primary_fb: True if this cluster IS the primary fastball
    Returns one of: Fastball, Sinker, Cutter, Changeup, Splitter,
                    Curveball, Slider, Sweeper, Other
    """
    velo  = cluster_stats["velo"]
    ivb   = cluster_stats["ivb"]
    hb    = cluster_stats["hb_mirror"]   # already mirrored: + = arm side
    spin  = cluster_stats["spin"]
    fb_velo = fb_stats["velo"]
    fb_hb   = fb_stats["hb_mirror"]
    fb_ivb  = fb_stats["ivb"]

    # ── 1. PRIMARY FASTBALL CLUSTER LABELING ──
    if is_primary_fb:
        # Sinker: low ride AND arm-side run >= ride (or sub-10 IVB w/ matching HB)
        if ivb < 10 and hb >= ivb:
            return "Sinker"
        # Cutter (as primary): ride 5+, HB within ±5
        if ivb >= 5 and -5 <= hb <= 5:
            return "Cutter"
        # Fastball: default for hardest cluster
        return "Fastball"

    # ── 2. SPLITTER: low spin + slower ──
    if spin < 1500 and velo < fb_velo - 5:
        return "Splitter"

    # ── 3. CURVEBALL: real depth (5+ inches of drop). HB direction matters but
    # threshold is loose — a curveball needs at least slight glove-side break
    # (HB ≤ -2) since a deep pitch with arm-side run would be a splitter/sinker
    # (handled above).
    if ivb <= -5 and hb <= -2:
        return "Curveball"

    # ── 4. SWEEPER: big glove-side sweep WITHOUT significant depth ──
    # Sweepers sit near IVB = 0 (slight ride or slight drop, but not curveball-deep).
    # If a pitch has 10+ sweep AND 5+ depth, the Curveball rule above caught it.
    if fb_hb > 3:
        # Standard pitcher: FB has arm-side run, sweeper goes glove-side
        if hb <= -10 and ivb > -5:
            return "Sweeper"
    else:
        # Cutter-primary pitcher: 10+ inches of glove-side sweep, minimal depth
        if hb <= -10 and ivb > -5:
            return "Sweeper"

    # ── 5. CHANGEUP: 7+ mph slower than FB, arm-side movement matching FB direction ──
    # "Arm-side run similar to primary fastball" — if FB has arm-side HB, CH should too.
    # If FB is cutter-like (low HB), CH just needs arm-side run > 5.
    if velo <= fb_velo - 7:
        if fb_hb > 3:
            # FB has arm-side run — CH should match direction (HB > 0, ideally close to fb_hb)
            if hb > 3:
                return "ChangeUp"
        else:
            # FB is cutter-ish — any arm-side movement is changeup
            if hb > 5:
                return "ChangeUp"

    # ── 6. CUTTER: 5+ IVB, HB between -5 and +5, faster than slider ──
    if ivb >= 5 and -5 <= hb <= 5 and velo >= fb_velo - 8:
        return "Cutter"

    # ── 7. SLIDER: minimal-movement breaking ball sitting around (0,0) on
    # the pitch plot. Includes gyro sliders with slight depth or sweep.
    # Covers: HB between -10 and 0, IVB between -5 and +5.
    # Anything with deeper depth OR more sweep falls through to other categories
    # (Curveball above needs depth+sweep; Sweeper above needs ≥10 sweep).
    if -10 < hb <= 0 and -5 < ivb < 5:
        return "Slider"
    # Also: any glove-side break with no depth is still a slider
    if hb <= -3 and ivb > -5:
        return "Slider"

    # ── 8. Catch-all — make a best-guess based on shape rather than "Other".
    # By this point the pitch doesn't fit any tight rule but it IS a real pitch.
    # Use coarse direction + depth to pick the closest pitch type.
    if ivb <= -3:
        # Has some drop — call it a Curveball
        return "Curveball"
    if hb <= -3:
        # Glove-side break with no significant depth — Slider
        return "Slider"
    if hb >= 8 and velo < fb_velo - 3:
        # Arm-side run, slower than fastball — Changeup
        return "ChangeUp"
    return "Other"

def classify_pitches_for_pitcher(pdf):
    """
    Run the full rule-based classifier on one pitcher's pitches.
    Returns a new df with the 'PitchType' column overwritten with our labels.
    Pitches that couldn't be clustered keep whatever PitchType they came in with
    (which would be the original Auto/Tagged value from resolve_pt).
    """
    pdf = pdf.copy()
    if len(pdf) < 5:
        return pdf

    clustered, is_lhp = _cluster_pitcher_pitches(pdf)

    # Post-process: if any cluster spans more than 6 mph of velocity, it likely
    # contains two different pitch types (e.g. fastball + changeup that DBSCAN
    # merged because they have similar movement). Split such clusters using
    # KMeans(n=2) on velo so the slower group can be labeled separately.
    try:
        from sklearn.cluster import KMeans
        next_cluster_id = max((c for c in clustered["_cluster"].unique() if c != -1), default=-1) + 1
        for c in sorted([c for c in clustered["_cluster"].unique() if c != -1]):
            mask = clustered["_cluster"] == c
            sub = clustered[mask]
            velos = sub["RelSpeed"].dropna()
            if len(velos) < 6:
                continue
            velo_range = velos.max() - velos.min()
            if velo_range > 6:
                # Split this cluster into 2 sub-clusters by velocity
                km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(
                    velos.values.reshape(-1, 1)
                )
                sublabels = km.labels_
                # Map: keep the larger sub-cluster as original c, give the
                # smaller a new ID
                counts = pd.Series(sublabels).value_counts()
                if len(counts) < 2 or counts.min() < 3:
                    continue  # too unbalanced to split safely
                bigger = counts.idxmax()
                indices = sub.dropna(subset=["RelSpeed"]).index.values
                for idx, lbl in zip(indices, sublabels):
                    if lbl != bigger:
                        clustered.at[idx, "_cluster"] = next_cluster_id
                next_cluster_id += 1
    except Exception:
        pass  # KMeans optional; if it fails just use DBSCAN clusters as-is

    # Compute cluster centroids (means)
    clusters = sorted(c for c in clustered["_cluster"].unique() if c != -1)
    if len(clusters) == 0:
        return pdf  # No usable clusters; keep original labels

    stats = {}
    for c in clusters:
        sub = clustered[clustered["_cluster"] == c]
        stats[c] = {
            "velo":       float(sub["RelSpeed"].mean()),
            "ivb":        float(sub["InducedVertBreak"].mean()),
            "hb_mirror":  float(sub["_hb_mirror"].mean()),
            "spin":       float(sub["SpinRate"].mean()),
            "n":          int(len(sub)),
        }

    # Primary fastball = hardest cluster (highest mean velo)
    fb_cluster = max(stats, key=lambda c: stats[c]["velo"])
    fb_stats   = stats[fb_cluster]

    # Label each cluster
    cluster_to_label = {}
    for c, cs in stats.items():
        cluster_to_label[c] = _label_cluster(cs, fb_stats, is_primary_fb=(c == fb_cluster))

    # If two+ clusters got the same label, disambiguate intelligently.
    # E.g. two "Slider" clusters: faster keeps name, slower may become Curveball/Sweeper.
    # E.g. two "ChangeUp" clusters: just merge — they're the same pitch with natural variance.
    label_counts = {}
    for c, lbl in cluster_to_label.items():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    for lbl, cnt in list(label_counts.items()):
        if cnt <= 1:
            continue
        same = [c for c, l in cluster_to_label.items() if l == lbl]

        if lbl == "Slider":
            # Faster stays Slider; slower demoted based on shape
            same_sorted = sorted(same, key=lambda c: stats[c]["velo"], reverse=True)
            for c in same_sorted[1:]:
                if stats[c]["ivb"] <= 0:
                    cluster_to_label[c] = "Curveball"
                elif abs(stats[c]["hb_mirror"]) >= 10:
                    cluster_to_label[c] = "Sweeper"
                # else: keep as Slider, will get merged below
        elif lbl == "Fastball":
            # Two fastball clusters → likely 4-seam vs sinker.
            # Hardest stays Fastball; second one re-labeled by movement.
            same_sorted = sorted(same, key=lambda c: stats[c]["velo"], reverse=True)
            for c in same_sorted[1:]:
                cs = stats[c]
                if cs["ivb"] < 10 and cs["hb_mirror"] >= cs["ivb"]:
                    cluster_to_label[c] = "Sinker"
                elif cs["ivb"] >= 5 and -5 <= cs["hb_mirror"] <= 5:
                    cluster_to_label[c] = "Cutter"
                # else: keep as Fastball, will get merged below

    # FINAL MERGE: any remaining duplicate labels just collapse into the same
    # pitch type. Variance within one pitch is normal and shouldn't split into
    # two reported pitch types in the report.

    # Apply labels back to the dataframe
    pdf = clustered.drop(columns=["_hb_mirror"], errors="ignore")
    new_labels = pdf["_cluster"].map(cluster_to_label)
    # Where clustering produced a label, use it. Otherwise keep original PitchType.
    pdf["PitchType"] = new_labels.where(new_labels.notna(), pdf.get("PitchType", "Other"))
    pdf = pdf.drop(columns=["_cluster"], errors="ignore")
    # CANONICALIZE: ensure every final label matches one canonical name so the
    # same physical pitch doesn't appear as both "Fastball" and "Four-Seam",
    # or "ChangeUp" and "Changeup", in the report.
    pdf["PitchType"] = pdf["PitchType"].replace(PT_NORMALIZE)
    return pdf

# ---------------------------------------------------------------------------
# Pitch classification: base label + movement-based refinement
# ---------------------------------------------------------------------------
# Pitch tagging is NOT a pure if/then. We take the best available label from
# resolve_pt (AutoPitchType, then human tag), then apply HIGH-CONFIDENCE
# movement refinements that surface the distinctions the raw labels miss —
# Sinker, Sweeper, Slurve — only when the pitch's shape is unambiguous. Cutters,
# changeups and splitters are left as-is, so a cutting four-seam stays a fastball
# and a high-slot changeup (with more IVB than usual) stays a changeup.
#
# Prototypes: (dV = velo - pitcher's fastball, IVB, arm-side HB, spin), from D1
# tagged medians. Used ONLY to classify the rare pitch with no usable label.
_PT_PROTO = {
    "Fastball":  (-0.7, 16.7,  10.6, 2215),
    "Sinker":    (-0.9,  8.9,  16.7, 2154),
    "Cutter":    (-6.4,  6.0,  -1.7, 2332),
    "Slider":    (-10.1, 0.5,  -6.4, 2383),
    "Sweeper":   (-11.7, -0.5,-13.9, 2505),
    "Slurve":    (-11.0, -5.0,-13.0, 2450),
    "Curveball": (-13.0, -9.4, -8.8, 2404),
    "ChangeUp":  (-8.5,  6.7,  14.5, 1713),
    "Splitter":  (-9.2,  5.1,   9.1, 1189),
}
_PT_SCALE  = np.array([5.0, 5.0, 5.0, 300.0])
_PT_WEIGHT = np.array([0.8, 1.1, 1.1, 0.6])   # movement > velocity > spin

def _arm_side_hb(df):
    """HorzBreak mirrored so arm-side run is POSITIVE and glove-side sweep is
    NEGATIVE, using each pitcher's fastball as the reference — no handedness
    field required (it is usually empty in the data)."""
    hb = pd.to_numeric(df["HorzBreak"], errors="coerce")
    overall = np.sign(hb.median())
    overall = overall if overall != 0 else 1.0
    if "Pitcher" in df.columns:
        fb = df[df["PitchType"].isin(["Fastball", "Sinker"])]
        med = fb.groupby("Pitcher")["HorzBreak"].median() if len(fb) else pd.Series(dtype=float)
        sign = np.sign(df["Pitcher"].map(med))
        sign = sign.where(sign != 0).fillna(overall)
    else:
        sign = pd.Series(overall, index=df.index)
    return hb * sign

def classify_pitch_types(df):
    """Return df with 'PitchType' set to canonical labels: resolve_pt base label
    plus movement-based Sinker/Sweeper/Slurve refinement. See notes above."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["PitchType"] = df.apply(resolve_pt, axis=1)
    need = {"InducedVertBreak", "HorzBreak", "RelSpeed", "SpinRate"}
    if not need.issubset(df.columns):
        return df  # no movement data — keep base labels
    ivb   = pd.to_numeric(df["InducedVertBreak"], errors="coerce")
    armhb = _arm_side_hb(df)
    ab    = armhb.abs()
    spin  = pd.to_numeric(df["SpinRate"], errors="coerce")
    velo  = pd.to_numeric(df["RelSpeed"], errors="coerce")
    base  = df["PitchType"].astype(str)
    out   = base.copy()

    # Carve-outs are decided at the (pitcher, base-type) GROUP level using the
    # group's MEDIAN movement — so a few outlier pitches can never splinter off
    # into their own type. A whole group flips together, or not at all.
    autoc = df["AutoPitchType"].map(_canon_pt) if "AutoPitchType" in df.columns else pd.Series(index=df.index, dtype=object)
    has_p = "Pitcher" in df.columns
    key = [df["Pitcher"], base] if has_p else [base]
    gI = ivb.groupby(key).transform("median")
    gA = ab.groupby(key).transform("median")
    gH = armhb.groupby(key).transform("median")
    gSink = autoc.eq("Sinker").groupby(key).transform("mean")   # AutoPitchType cross-reference
    gN = base.groupby(key).transform("size")
    tot = base.groupby(df["Pitcher"]).transform("size") if has_p else pd.Series(len(df), index=df.index)
    # a carve-out only fires for a group that is a real part of the arsenal
    elig = (gN >= 6) & (gN / tot >= 0.10)

    br = base.isin(["Slider", "Curveball"])
    # Sweeper: little drop + big glove-side sweep
    out = out.mask(br & elig & (gI >= -3) & (gA >= 13), "Sweeper")
    # Slurve: moderate drop + big sweep (slider/curveball hybrid)
    out = out.mask(br & elig & (out != "Sweeper") & gI.between(-9, -2) & (gA >= 11), "Slurve")
    # Fastball -> Sinker only if the group sinks/runs AND AutoPitchType agrees
    out = out.mask((base == "Fastball") & elig & (gI <= 12) & (gH >= 14) & (gSink >= 0.5), "Sinker")

    df["PitchType"] = out.where(out.notna(), "Other")
    # Consolidation: dissolve tiny per-pitcher groups (stray mis-tags, or a pitch
    # with too little support) into the pitcher's nearest group by movement — so a
    # report never shows random one-off pitch types.
    df = _consolidate_pitch_groups(df, velo, ivb, armhb)
    return df

def _consolidate_pitch_groups(df, velo, ivb, armhb):
    """Merge tiny per-pitcher pitch-type groups into the nearest surviving group
    by movement. A group is 'tiny' if it has < 3 pitches, or < 6 and < 3% usage.
    'Other' is always dissolved when the pitcher has any real group."""
    lab = df["PitchType"].astype(str).copy()
    feat = pd.DataFrame({"v": velo, "i": ivb, "h": armhb})
    scale = np.array([5.0, 5.0, 5.0])
    groups = df.groupby("Pitcher").groups.items() if "Pitcher" in df.columns else [(None, list(df.index))]
    for _, idx in groups:
        idx = list(idx)
        n = len(idx)
        vc = lab.loc[idx].value_counts()
        small = {l for l, c in vc.items() if (c < 3 or (c < 6 and c / n < 0.03))}
        if "Other" in vc.index:
            small.add("Other")
        big = [l for l in vc.index if l not in small and l != "Other"]
        if not big:
            continue
        cent = {l: feat.loc[[i for i in idx if lab.at[i] == l]].mean().values for l in big}
        for i in idx:
            if lab.at[i] in small:
                x = feat.loc[i].values
                lab.at[i] = min(big, key=lambda l: (((x - cent[l]) / scale) ** 2).sum())
    df = df.copy()
    df["PitchType"] = lab
    return df

def auto_correct_pitch_types(pitcher_df):
    """Apply movement-based pitch classification (base label + Sinker/Sweeper/
    Slurve refinement). Returns (df, n_changed) for backward compatibility.

    NOTE: the old unsupervised DBSCAN/KMeans re-clustering was removed — it
    mislabeled pitches. This rule-based refinement trusts the base label and only
    overrides it when pitch shape is unambiguous.
    """
    if pitcher_df is None or len(pitcher_df) == 0:
        return pitcher_df, 0
    orig = pitcher_df["PitchType"].astype(str).values if "PitchType" in pitcher_df.columns else None
    out = classify_pitch_types(pitcher_df)
    n = int((out["PitchType"].astype(str).values != orig).sum()) if orig is not None else 0
    return out, n

def fmt(s, fn="mean", d=1):
    v = s.dropna()
    if v.empty: return "—"
    r = v.mean() if fn == "mean" else v.max()
    return f"{r:.{d}f}"

# ---------------------------------------------------------------------------
# Height-Adjusted Vertical Approach Angle (HAVAA)
# ---------------------------------------------------------------------------
# Raw VAA is heavily confounded by the height at which the pitch crosses the
# plate (a letter-high pitch is naturally flatter than a knee-high one). HAVAA
# removes that bias by normalizing each pitch's VAA to a common plate height
# (HAVAA_REF_HEIGHT) using the pitch type's VAA-vs-height slope:
#     HAVAA = VAA - b1 * (PlateLocHeight - HAVAA_REF_HEIGHT)
# The result is an ACTUAL approach angle (deg) with the high/low bias removed —
# i.e. "what this pitch's VAA would be if it crossed the plate at 2.5 ft."
# Less negative = flatter than a typical pitch of that type.
#
# b1 slopes are per-pitch-type OLS fits (VAA ~ PlateLocHeight) on 222,705
# D1 pitches (2026-02-14, 02-28, 03-14, 04-11, 05-02). Fastball R^2 = 0.76.
# (b0 kept for reference; only b1 is used for the height adjustment.)
HAVAA_REF_HEIGHT = 2.5  # feet, ~ middle of the strike zone

HAVAA_COEFFS = {
    "Fastball":    (-8.1506, 1.0733),
    "Sinker":      (-8.4070, 1.0656),
    "Cutter":      (-9.3298, 1.0705),
    "Slider":      (-10.1431, 1.0521),
    "Sweeper":     (-10.1431, 1.0521),  # slider family
    "Curveball":   (-11.4301, 0.9818),
    "ChangeUp":    (-9.5234, 1.0623),
    "Splitter":    (-9.5234, 1.0623),   # changeup family
    "Knuckleball": (-7.9380, 1.0594),
    "Other":       (-7.9380, 1.0594),
}

def _havaa_mean(s, pt):
    """Mean Height-Adjusted VAA (approach angle, deg) for a pitch-type subset s.
    np.nan if unavailable."""
    co = HAVAA_COEFFS.get(pt)
    if co is None:
        return np.nan
    b0, b1 = co
    sub = s[["VertApprAngle", "PlateLocHeight"]].dropna()
    if sub.empty:
        return np.nan
    adj = sub["VertApprAngle"] - b1 * (sub["PlateLocHeight"] - HAVAA_REF_HEIGHT)
    return float(adj.mean())

def havaa_fmt(s, pt, d=1):
    """Formatted mean HAVAA angle for table cells; '—' if unavailable."""
    v = _havaa_mean(s, pt)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}"

# ===========================================================================
# DRAWING FUNCTIONS
# ===========================================================================
def draw_zone(ax, data, title, pts):
    ax.set_facecolor(PANEL_COLOR)
    ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=False, ec="#333333", lw=1.5, alpha=0.8, zorder=3))
    ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=True, fc="#E8EDF2", alpha=0.3, zorder=2))
    ax.add_patch(Rectangle((-1.4, 1.2), 2.8, 2.7, fill=False, ec="#AAAAAA", lw=0.7, ls="--", alpha=0.4, zorder=2))
    ax.add_patch(Polygon([(-.708, .55), (.708, .55), (.708, .35), (0, .15), (-.708, .35)],
                         closed=True, fc="none", ec=MUTED_TEXT, lw=.7, alpha=0.4))
    outcome_markers = {
        "BallCalled": ("o", False), "BallinDirt": ("o", False), "BallIntentional": ("o", False),
        "StrikeCalled": ("o", True), "StrikeSwinging": ("X", True),
        "FoulBallNotFieldable": ("^", True), "FoulBallFieldable": ("^", True),
        "InPlay": ("s", True), "HitByPitch": ("D", True),
    }
    for pt in pts:
        s = data[data["PitchType"] == pt]
        if s.empty: continue
        color = pc(pt)
        for call, grp in s.groupby("PitchCall"):
            marker, filled = outcome_markers.get(call, ("o", False))
            x = -grp["PlateLocSide"]; y = grp["PlateLocHeight"]  # mirror to catcher POV
            valid = x.notna() & y.notna()
            if not valid.any(): continue
            if filled:
                ax.scatter(x[valid], y[valid], marker=marker, c=color, s=45, alpha=0.9,
                           edgecolors="black", linewidths=0.3, zorder=5)
            else:
                ax.scatter(x[valid], y[valid], marker=marker, c="none", s=45, alpha=0.9,
                           edgecolors=color, linewidths=1.2, zorder=5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def draw_mov(ax, data, pts):
    ax.set_facecolor(PANEL_COLOR)
    ax.axhline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    ax.axvline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    for r in [5, 10, 15, 20]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=GRID_COLOR, lw=0.3, ls="--", alpha=0.3))
    for pt in pts:
        s = data[data["PitchType"] == pt]
        if not s.empty:
            ax.scatter(s["HorzBreak"], s["InducedVertBreak"],
                       c=pc(pt), label=pt, s=40, alpha=.9, edgecolors="black", linewidths=0.3, zorder=5)
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xlabel("HB (in)", fontsize=11, color=MUTED_TEXT)
    ax.set_ylabel("IVB (in)", fontsize=11, color=MUTED_TEXT)
    ax.set_title("Pitch Movement", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=min(len(pts), 5),
              fontsize=10, frameon=False, labelcolor=TEXT_COLOR)
    ax.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax.spines.values(): sp.set_color(GRID_COLOR)

def draw_release(ax, data, pts):
    ax.set_facecolor(PANEL_COLOR)
    all_rs = data["RelSide"].dropna(); all_rh = data["RelHeight"].dropna()
    for pt in pts:
        valid = data.loc[data["PitchType"] == pt, ["RelSide", "RelHeight"]].dropna()
        if not valid.empty:
            ax.scatter(valid["RelSide"].mean(), valid["RelHeight"].mean(),
                       c=pc(pt), s=80, alpha=0.95, edgecolors="black", linewidths=0.8, zorder=5)
    if not all_rs.empty and not all_rh.empty:
        avg_rs, avg_rh = all_rs.mean(), all_rh.mean()
        ax.axhline(avg_rh, color=ACCENT_COLOR, lw=1, alpha=0.5, zorder=3)
        ax.axvline(avg_rs, color=ACCENT_COLOR, lw=1, alpha=0.5, zorder=3)
        ax.text(avg_rs, avg_rh - 0.3, f"Avg ({avg_rs:.1f}, {avg_rh:.1f})",
                ha="center", va="top", fontsize=9, color=ACCENT_COLOR, family="monospace",
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=BG_COLOR,
                ec=ACCENT_COLOR, alpha=0.9, lw=0.5))
    ax.axvline(0, color=MUTED_TEXT, lw=0.5, ls="--", alpha=0.3)
    ax.set_xlabel("Release Side (ft)", fontsize=11, color=MUTED_TEXT)
    ax.set_ylabel("Release Height (ft)", fontsize=11, color=MUTED_TEXT)
    ax.set_title("Release Point", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax.spines.values(): sp.set_color(GRID_COLOR)
    if not all_rs.empty and not all_rh.empty:
        rs_c, rh_c = all_rs.mean(), all_rh.mean()
        rs_std = all_rs.std() if len(all_rs) > 1 else 0.3
        rh_std = all_rh.std() if len(all_rh) > 1 else 0.3
        rs_std = rs_std if (rs_std == rs_std) else 0.3  # NaN check
        rh_std = rh_std if (rh_std == rh_std) else 0.3  # NaN check
        pad = max(rs_std, rh_std, 0.3) * 4 + 0.3
        ax.set_xlim(rs_c - pad, rs_c + pad); ax.set_ylim(rh_c - pad, rh_c + pad)
    ax.set_aspect("equal")

# ===========================================================================
# DATA LOADING — reads from data/by_date/ partitioned parquets
# ===========================================================================
from config import DATA_DIR, BY_DATE_DIR, INDEX_PATH

PT_NORMALIZE = {
    # ── Auto names → canonical ──
    "Four-Seam": "Fastball",
    "Two-Seam":  "Sinker",
    "Changeup":  "ChangeUp",
    # ── Tagged names → canonical (some already match) ──
    "FourSeamFastBall": "Fastball",
    "TwoSeamFastBall":  "Sinker",
    # Fastball, Sinker, Cutter, Slider, Curveball, ChangeUp, Splitter,
    # Sweeper, Knuckleball, Other already canonical and pass through unchanged.
}

def _prep_df(df):
    """Shared post-load cleanup."""
    # GameDate may be string "YYYY-MM-DD", date object, or datetime — normalize all
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
    # Convert category columns back to string for comparisons
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str).replace("nan", "")
    if "PitchType" in df.columns:
        df["PitchType"] = df["PitchType"].replace(PT_NORMALIZE)
    return df

@st.cache_data(ttl=3600)
def load_index(_cache_version="v10"):
    """Load lightweight game index — used for sidebar dropdowns. Tiny and fast.
    _cache_version: bump this string to bust the Streamlit Cloud cache."""
    # Support both new index.parquet and legacy pitches.parquet
    if os.path.exists(INDEX_PATH):
        df = pd.read_parquet(INDEX_PATH)
        df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
        return df
    # Fall back to legacy single parquet
    legacy = os.path.join(DATA_DIR, "pitches.parquet")
    if os.path.exists(legacy):
        df = pd.read_parquet(legacy, columns=["GameDate","HomeTeam","AwayTeam","SessionID"])
        df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
        return df.drop_duplicates(subset=["GameDate","HomeTeam","AwayTeam"])
    return pd.DataFrame()

def _to_date(x):
    """Convert anything datey (date, datetime, pandas Timestamp, string) to a plain date."""
    if x is None:
        return None
    if hasattr(x, "date") and callable(x.date):
        try:
            return x.date()
        except Exception:
            pass
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return x

@st.cache_data(ttl=300)
def load_team_data(team_name, date_from, date_to, _cache_version="v6"):
    """
    Load only the date files where the selected team played in the date range.
    Returns ~5-50MB instead of the full 800MB+ season dataset.
    _cache_version: bump this string to bust the Streamlit Cloud cache.
    """
    _debug_lines = []  # collect diagnostic breadcrumbs

    idx = load_index()
    if idx.empty:
        _debug_lines.append("index is EMPTY")
        with st.sidebar.expander("🔍 load_team_data debug", expanded=True):
            st.code("\n".join(_debug_lines))
        return None

    # Normalize the team name we're searching for (strip whitespace)
    team_name = str(team_name).strip() if team_name else team_name
    _debug_lines.append(f"team_name (normalized) = {team_name!r}")
    _debug_lines.append(f"date_from={date_from}  date_to={date_to}")

    # Find dates this team played in range
    idx = idx.copy()
    idx["GameDate"] = pd.to_datetime(idx["GameDate"], errors="coerce").dt.date
    # Normalize index team columns — cast to string and strip whitespace
    # so "East Carolina Pirates " matches "East Carolina Pirates" etc.
    for col in ("HomeTeam", "AwayTeam"):
        if col in idx.columns:
            idx[col] = idx[col].astype(str).str.strip()
    date_from = _to_date(date_from)
    date_to   = _to_date(date_to)

    # Report index coverage for this team
    team_in_idx_total = ((idx["HomeTeam"] == team_name) | (idx["AwayTeam"] == team_name)).sum()
    _debug_lines.append(f"index rows with this team (any date): {team_in_idx_total}")
    if team_in_idx_total > 0:
        team_dates_all = sorted(
            idx[(idx["HomeTeam"] == team_name) | (idx["AwayTeam"] == team_name)]["GameDate"]
            .dropna().unique()
        )
        _debug_lines.append(f"all dates for this team in index: {[str(d) for d in team_dates_all[-5:]]}")

    team_mask = (
        (idx["HomeTeam"] == team_name) | (idx["AwayTeam"] == team_name)
    ) & (idx["GameDate"] >= date_from) & (idx["GameDate"] <= date_to)
    team_dates = sorted(idx[team_mask]["GameDate"].dropna().unique())
    _debug_lines.append(f"dates matching team+range: {[str(d) for d in team_dates]}")

    if not team_dates:
        with st.sidebar.expander("🔍 load_team_data debug", expanded=True):
            st.code("\n".join(_debug_lines))
        return pd.DataFrame()

    dfs = []

    # New partitioned structure
    if os.path.exists(BY_DATE_DIR):
        _debug_lines.append(f"BY_DATE_DIR exists at {BY_DATE_DIR}")
        for gdate in team_dates:
            fpath = os.path.join(BY_DATE_DIR, f"{gdate}.parquet")
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                # Overwrite GameDate with the filename date — this is the
                # ground truth. Late-night games can have internal GameDate
                # values off by a day due to UTC/local timezone shifts,
                # which breaks downstream date filters.
                df["GameDate"] = gdate
                # Normalize team columns FIRST — parquet files store these as
                # categorical dtype sometimes, and can have trailing whitespace
                # that breaks exact equality comparisons.
                for col in ("HomeTeam", "AwayTeam", "TopBottom"):
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                # Filter to only rows involving this team
                mask = (df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)
                n_match = int(mask.sum())
                _debug_lines.append(f"  {gdate}.parquet: {len(df)} rows, {n_match} match team")
                if n_match == 0 and len(df) > 0:
                    # Show what team names ARE in that file so we can see the mismatch
                    unique_home = df["HomeTeam"].dropna().unique()[:15]
                    unique_away = df["AwayTeam"].dropna().unique()[:15]
                    _debug_lines.append(f"    HomeTeams sample: {list(unique_home)}")
                    _debug_lines.append(f"    AwayTeams sample: {list(unique_away)}")
                dfs.append(df[mask])
            else:
                _debug_lines.append(f"  {gdate}.parquet: FILE MISSING at {fpath}")
    else:
        _debug_lines.append(f"BY_DATE_DIR does NOT exist at {BY_DATE_DIR}")
        # Fall back to legacy single parquet
        legacy = os.path.join(DATA_DIR, "pitches.parquet")
        if os.path.exists(legacy):
            df = pd.read_parquet(legacy)
            for col in ("HomeTeam", "AwayTeam"):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
            mask = (
                ((df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)) &
                (df["GameDate"] >= date_from) & (df["GameDate"] <= date_to)
            )
            dfs.append(df[mask])

    with st.sidebar.expander("🔍 load_team_data debug", expanded=True):
        st.code("\n".join(_debug_lines))

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return _prep_df(combined)

@st.cache_data(ttl=300)
def load_all_pitches():
    """
    Legacy compatibility wrapper.
    Returns (index_df, path) — index_df has team/date info for sidebar.
    Actual pitch data is loaded on demand via load_team_data().
    """
    # Debug: show what paths we're checking
    st.sidebar.caption(f"DEBUG: DATA_DIR={DATA_DIR} | INDEX exists={os.path.exists(INDEX_PATH)} | BY_DATE exists={os.path.exists(BY_DATE_DIR)}")
    idx = load_index()
    if idx.empty:
        return None, None
    return idx, INDEX_PATH if os.path.exists(INDEX_PATH) else DATA_DIR

def get_last_updated():
    ts_path = os.path.join(DATA_DIR, "last_updated.json")
    if not os.path.exists(ts_path):
        return None
    with open(ts_path) as f:
        return json.load(f).get("last_updated_date", "unknown")

def build_team_code_map(df):
    """Build a mapping from full team name (HomeTeam/AwayTeam) to PitcherTeam code."""
    mapping = {}
    for _, row in df[["HomeTeam", "AwayTeam", "PitcherTeam"]].dropna().iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        code = row["PitcherTeam"]
        # We can only map the pitcher's own team code
        # So we check both home/away against the code prefix
        code_prefix = code[:3].upper()
        for team in [home, away]:
            if team and team[:3].upper() == code_prefix:
                mapping[team] = code
    return mapping

def get_teams(df):
    teams = set()
    for col in ["HomeTeam", "AwayTeam"]:
        if col in df.columns:
            # Cast to string and strip whitespace — the parquet files sometimes
            # store team names as categorical dtype with trailing spaces, which
            # breaks downstream equality comparisons (" Pirates " != "Pirates").
            teams.update(df[col].dropna().astype(str).str.strip().unique())
    return sorted(t for t in teams if t)

def get_team_pitches(df, team_name, date_from, date_to):
    """Filter to pitches thrown by pitchers on the selected team. Normalizes date types."""
    df = df.copy()
    # Vectorized GameDate → plain Python date conversion.
    # Handles datetime64[ns], datetime64[ns, UTC], strings, and existing dates.
    # Why not `.dt.date`? On tz-aware series that returns datetime64 still.
    # We cast to int64 nanoseconds (stripping tz if any), then rebuild as naive
    # datetime, then extract .date.
    gd_raw = df["GameDate"]
    try:
        # First try a simple parse
        gd = pd.to_datetime(gd_raw, errors="coerce")
        # If tz-aware, strip tz
        if hasattr(gd, "dt") and gd.dt.tz is not None:
            gd = gd.dt.tz_convert("UTC").dt.tz_localize(None)
        df["GameDate"] = gd.dt.date
    except Exception:
        # Fallback: row-wise convert if the above fails for mixed dtypes
        df["GameDate"] = pd.to_datetime(gd_raw.astype(str), errors="coerce").dt.date

    # Cast EVERY filterable column out of categorical to string — categorical
    # comparisons silently return False when the compared value isn't in the
    # category list. This is the #1 pandas filtering gotcha.
    for col in ("HomeTeam", "AwayTeam", "TopBottom"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    team_name = str(team_name).strip() if team_name else team_name
    # Normalize date bounds into plain date objects
    date_from = _to_date(date_from)
    date_to   = _to_date(date_to)
    date_mask = (df["GameDate"] >= date_from) & (df["GameDate"] <= date_to)
    mask = date_mask & (
        ((df["HomeTeam"] == team_name) & (df["TopBottom"] == "Top")) |
        ((df["AwayTeam"] == team_name) & (df["TopBottom"] == "Bottom"))
    )
    return df[mask].copy()


def flatten_game(plays_raw, balls_raw):
    rows = []
    for p in plays_raw:
        rows.append({
            "PlayID": p.get("playID"),
            "PitchNo": sg(p, "taggerBehavior", "pitchNo"),
            "PAofInning": sg(p, "taggerBehavior", "pAofinning"),
            "PitchofPA": sg(p, "taggerBehavior", "pitchofPA"),
            "Pitcher": sg(p, "pitcher", "name", default=""),
            "PitcherTeam": sg(p, "pitcher", "team", default=""),
            "Batter": sg(p, "batter", "name", default=""),
            "BatterSide": sg(p, "batter", "side", default=""),
            "Inning": sg(p, "gameState", "inning"),
            "TopBottom": sg(p, "gameState", "topBottom"),
            "Outs": sg(p, "gameState", "outs"),
            "Balls": sg(p, "gameState", "balls"),
            "Strikes": sg(p, "gameState", "strikes"),
            "TaggedPitchType": sg(p, "pitchTag", "taggedPitchType", default=""),
            "PitchCall": sg(p, "pitchTag", "pitchCall", default=""),
            "AutoPitchType": sg(p, "pitchTag", "autoPitchType", default=""),
            "KorBB": p.get("korBB", ""),
            "PlayResult": sg(p, "playResult", "playResult", default=""),
            "OutsOnPlay": sf(sg(p, "playResult", "outsOnPlay", default=0)),
            "RunsScored": sf(sg(p, "playResult", "runsScored", default=0)),
        })
    plays_df = pd.DataFrame(rows)
    if plays_df.empty: return plays_df
    plays_df["PitchNo"] = pd.to_numeric(plays_df["PitchNo"], errors="coerce")
    plays_df = plays_df.sort_values("PitchNo").reset_index(drop=True)

    pr, hr_ = [], []
    for b in balls_raw:
        kind = b.get("kind", ""); pid = b.get("playId")
        if kind == "Pitch":
            pr.append({"PlayID": pid,
                "RelSpeed": sf(sg(b, "pitch", "release", "relSpeed")),
                "SpinRate": sf(sg(b, "pitch", "release", "spinRate")),
                "Extension": sf(sg(b, "pitch", "release", "extension")),
                "RelHeight": sf(sg(b, "pitch", "release", "relHeight")),
                "RelSide": sf(sg(b, "pitch", "release", "relSide")),
                "HorzBreak": sf(sg(b, "pitch", "movement", "horzBreak")),
                "InducedVertBreak": sf(sg(b, "pitch", "movement", "inducedVertBreak")),
                "PlateLocHeight": sf(sg(b, "pitch", "location", "plateLocHeight")),
                "PlateLocSide": sf(sg(b, "pitch", "location", "plateLocSide")),
                "VertApprAngle": sf(sg(b, "pitch", "location", "vertApprAngle")),
            })
        elif kind == "Hit":
            hr_.append({"PlayID": pid,
                "ExitSpeed": sf(sg(b, "hit", "launch", "exitSpeed")),
                "LaunchAngle": sf(sg(b, "hit", "launch", "angle")),
            })

    pbdf = pd.DataFrame(pr).drop_duplicates("PlayID", keep="first") if pr else pd.DataFrame()
    hbdf = pd.DataFrame(hr_).drop_duplicates("PlayID", keep="first") if hr_ else pd.DataFrame()

    df = plays_df
    if not pbdf.empty: df = df.merge(pbdf, on="PlayID", how="left")
    else:
        for c in ["RelSpeed","SpinRate","Extension","RelHeight","RelSide",
                   "HorzBreak","InducedVertBreak","PlateLocHeight","PlateLocSide","VertApprAngle"]:
            df[c] = np.nan
    if not hbdf.empty: df = df.merge(hbdf, on="PlayID", how="left")
    else:
        for c in ["ExitSpeed", "LaunchAngle"]: df[c] = np.nan
    df = df.drop_duplicates("PlayID", keep="first")
    return df

def identify_team_code(df, team_name, ht, at):
    if team_name.lower() in ht.lower():
        top_pitchers = df[df["TopBottom"] == "Top"]["PitcherTeam"].value_counts()
        if not top_pitchers.empty: return top_pitchers.index[0]
    elif team_name.lower() in at.lower():
        bot_pitchers = df[df["TopBottom"] == "Bottom"]["PitcherTeam"].value_counts()
        if not bot_pitchers.empty: return bot_pitchers.index[0]
    return None

# ===========================================================================
# GENERATE ONE PITCHER PAGE
# ===========================================================================
def generate_pitcher_page(p, pname, gdate, opp):
    N = len(p)
    if N == 0: return None

    ip = calc_ip(p); pa = calc_pa(p)
    hits = int(p["PlayResult"].isin(["Single", "Double", "Triple"]).sum())
    hr = int((p["PlayResult"] == "HomeRun").sum())
    k = int((p["KorBB"] == "Strikeout").sum())
    bb = int((p["KorBB"] == "Walk").sum())
    hbp = int((p["PitchCall"] == "HitByPitch").sum())
    spct = round(p["PitchCall"].isin(STRIKE_CALLS).sum() / N * 100, 1)

    wh = p["PitchCall"] == "StrikeSwinging"
    sw = p["PitchCall"].isin(SWING_CALLS)
    iz = p["InZone"]
    ooz = ~iz

    zpct = round(iz.sum() / N * 100, 1)
    wpct = round(wh.sum() / sw.sum() * 100, 1) if sw.sum() else 0
    cpct = round((sw & ooz).sum() / ooz.sum() * 100, 1) if ooz.sum() else 0
    iz_sw = (sw & iz).sum()
    iz_wh_ct = (wh & iz).sum()
    izwp = round(iz_wh_ct / iz_sw * 100, 1) if iz_sw else 0

    pts = p["PitchType"].value_counts().index.tolist()

    fig = plt.figure(figsize=(26, 16), facecolor=BG_COLOR)
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[.06, .02, .42, .50],
                  width_ratios=[1, 1, 1, 0.65],
                  hspace=.18, wspace=.18,
                  top=0.96, bottom=0.02, left=0.03, right=0.97)

    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    ax.text(.5, .74, _display_name(pname), ha="center", va="center", fontsize=30,
            fontweight="bold", color="#1a1a1a")
    _sub = "   •   ".join([b for b in [_hand_label(p), f"vs {opp}", f"{gdate:%b %d, %Y}"] if b])
    ax.text(.5, .18, _sub, ha="center", va="center", fontsize=14, color="#555")

    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    stats_str = (f"IP {ip}   ·   PA {pa}   ·   P {N}   ·   "
                 f"H {hits + hr}   ·   K {k}   ·   BB {bb}   ·   HBP {hbp}   ·   HR {hr}   ·   "
                 f"STR% {spct}%")
    ax.text(.5, .65, stats_str, ha="center", va="center", fontsize=14,
            color=TEXT_COLOR, family="monospace", fontweight="bold")
    legend_str = "○ Ball    ● Called Strike    ✕ Swinging Strike    ▲ Foul    ■ In Play"
    ax.text(.5, .05, legend_str, ha="center", va="center", fontsize=11,
            color=TEXT_COLOR, family="monospace")

    lhb = p[p["BatterSide"] == "Left"]
    ax_l = fig.add_subplot(gs[2, 0]); draw_zone(ax_l, lhb, f"vs LHB ({len(lhb)})", pts)
    rhb = p[p["BatterSide"] == "Right"]
    ax_r = fig.add_subplot(gs[2, 1]); draw_zone(ax_r, rhb, f"vs RHB ({len(rhb)})", pts)
    ax_m = fig.add_subplot(gs[2, 2]); _ss_plot_breaks(ax_m, p, pts)
    ax_rp = fig.add_subplot(gs[2, 3]); draw_release(ax_rp, p, pts)

    ax_t = fig.add_subplot(gs[3, :]); ax_t.set_facecolor(BG_COLOR); ax_t.axis("off")
    trows = []
    grade_cells = {}
    for ri, pt in enumerate(pts):
        s = p[p["PitchType"] == pt]; n = len(s)
        s_iz = in_zone(s); _sw = s["PitchCall"].isin(SWING_CALLS)
        _wh = s["PitchCall"] == "StrikeSwinging"
        _ooz = ~s_iz
        _ooz_sw = (_sw & _ooz).sum(); _ooz_n = _ooz.sum()
        _iz_sw = (_sw & s_iz).sum(); _iz_wh = (_wh & s_iz).sum()
        iz_whiff_str = f"{_iz_wh / _iz_sw * 100:.1f}%" if _iz_sw else "—"
        _sw_ct = _sw.sum()
        whiff_val = _wh.sum() / _sw_ct * 100 if _sw_ct else None
        whiff_str = f"{whiff_val:.1f}%" if whiff_val is not None else "—"
        chase_val = _ooz_sw / _ooz_n * 100 if _ooz_n else None
        chase_str = f"{chase_val:.1f}%" if chase_val is not None else "—"
        xw = s["xwOBA"].dropna()
        xwoba_val = xw.mean() if not xw.empty else None
        xwoba_str = f"{xwoba_val:.3f}" if xwoba_val is not None else "—"
        avg_velo_raw = s["RelSpeed"].dropna()
        avg_velo_val = avg_velo_raw.mean() if not avg_velo_raw.empty else None
        zone_val = s_iz.sum() / n * 100 if n else None
        zone_str = f"{zone_val:.1f}%" if zone_val is not None else "—"
        stuff_val = score_stuff_plus(s, pt, p)
        stuff_str = f"{stuff_val:.0f}" if stuff_val is not None else "—"
        trows.append([pt, n, f"{n / N * 100:.1f}%", stuff_str,
                      fmt(s["RelSpeed"]), fmt(s["RelSpeed"], "max"),
                      fmt(s["SpinRate"], d=0),
                      fmt(s["InducedVertBreak"]), fmt(s["HorzBreak"]),
                      fmt(s["Extension"]), fmt(s["RelHeight"]), fmt(s["RelSide"]),
                      havaa_fmt(s, pt),
                      xwoba_str, zone_str, whiff_str, chase_str, iz_whiff_str])

        data_row = ri + 1
        if avg_velo_val is not None:
            grade_cells[(data_row, 3)] = (pt, "velo", avg_velo_val, True)
        if xwoba_val is not None:
            grade_cells[(data_row, 12)] = (pt, "xwoba", xwoba_val, False)
        if zone_val is not None:
            grade_cells[(data_row, 13)] = (pt, "zone_pct", zone_val, True)
        if whiff_val is not None:
            grade_cells[(data_row, 14)] = (pt, "whiff_pct", whiff_val, True)
        if chase_val is not None:
            grade_cells[(data_row, 15)] = (pt, "chase_pct", chase_val, True)

    all_sw_ct = sw.sum()
    all_whiff = f"{wh.sum() / all_sw_ct * 100:.1f}%" if all_sw_ct else "0%"
    all_xw = p["xwOBA"].dropna()
    all_xwoba = f"{all_xw.mean():.3f}" if not all_xw.empty else "—"
    trows.append(["All", N, "100%", "—", "—", "—", "—", "—", "—",
                  fmt(p["Extension"]), "—", "—", "—",
                  all_xwoba, f"{zpct}%", all_whiff, f"{cpct}%", f"{izwp}%"])

    cols = ["Count", "Usage%", "Stuff+", "Avg\nVelo", "Max\nVelo", "Avg\nSpin",
            "IVB", "HB", "Ext", "RelH", "RelS", "HAVAA",
            "xwOBA", "Zone%", "Whiff%", "Chase%", "IZ\nWhiff%"]

    tbl = ax_t.table(cellText=[r[1:] for r in trows], rowLabels=[r[0] for r in trows],
                     colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1E1E2E")
            cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=12)
        elif col == -1:
            pitch_name = cell.get_text().get_text()
            if pitch_name == "All":
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontweight="bold", color=TEXT_COLOR, fontfamily="monospace", fontsize=13)
            else:
                cell.set_facecolor(pc(pitch_name))
                cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=13)
        else:
            graded = False
            if (row, col) in grade_cells and row <= len(pts):
                pt_name, stat_name, raw_val, higher_better = grade_cells[(row, col)]
                gc = grade_color(pt_name, stat_name, raw_val, higher_better)
                if gc is not None:
                    cell.set_facecolor(gc)
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontweight="bold", fontsize=12)
                    graded = True
            if not graded:
                if row == len(trows):
                    cell.set_facecolor("#E8E8E8")
                    cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontfamily="monospace", fontsize=12)
                elif row % 2 == 0:
                    cell.set_facecolor("#F4F6F9")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontsize=12)
                else:
                    cell.set_facecolor("#FFFFFF")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontsize=12)

    return fig

# ===========================================================================
# SEASON SUMMARY FUNCTIONS
# ===========================================================================
def calc_fip(k, bb, hbp_ct, hr_ct, ip_str):
    parts = ip_str.split(".")
    ip = int(parts[0]) + int(parts[1]) / 3 if len(parts) == 2 else float(ip_str)
    if ip == 0: return 0.0
    return (13 * hr_ct + 3 * (bb + hbp_ct) - 2 * k) / ip + 3.10

def generate_season_summary(pitcher_name, outings, date_from, date_to):
    all_dfs = []
    for p_df, gdate, opp in outings:
        df_copy = p_df.copy()
        df_copy["_game_date"] = gdate
        df_copy["_opp"] = opp
        all_dfs.append(df_copy)
    p = pd.concat(all_dfs, ignore_index=True)
    N = len(p)
    if N == 0: return None

    total_ip_outs = 0; total_k = 0; total_bb = 0
    total_hbp = 0; total_hr = 0; total_hits = 0; total_pa = 0
    for p_df, gdate, opp in outings:
        ip_s = calc_ip(p_df)
        parts = ip_s.split(".")
        total_ip_outs += int(parts[0]) * 3 + int(parts[1])
        total_k += int((p_df["KorBB"] == "Strikeout").sum())
        total_bb += int((p_df["KorBB"] == "Walk").sum())
        total_hbp += int((p_df["PitchCall"] == "HitByPitch").sum())
        total_hr += int((p_df["PlayResult"] == "HomeRun").sum())
        total_hits += int(p_df["PlayResult"].isin(["Single", "Double", "Triple"]).sum())
        total_pa += calc_pa(p_df)

    ip_full = total_ip_outs // 3; ip_rem = total_ip_outs % 3
    ip_str = f"{ip_full}.{ip_rem}"
    ip_float = ip_full + ip_rem / 3.0
    whip = ((total_bb + total_hits + total_hr) / ip_float) if ip_float > 0 else 0
    k_pct = (total_k / total_pa * 100) if total_pa > 0 else 0
    bb_pct = (total_bb / total_pa * 100) if total_pa > 0 else 0
    fip = calc_fip(total_k, total_bb, total_hbp, total_hr, ip_str)

    wh = p["PitchCall"] == "StrikeSwinging"
    sw = p["PitchCall"].isin(SWING_CALLS)
    iz = p["InZone"]; ooz = ~iz
    zpct = round(iz.sum() / N * 100, 1)
    wpct = round(wh.sum() / sw.sum() * 100, 1) if sw.sum() else 0
    cpct = round((sw & ooz).sum() / ooz.sum() * 100, 1) if ooz.sum() else 0
    iz_sw = (sw & iz).sum(); iz_wh_ct = (wh & iz).sum()
    izwp = round(iz_wh_ct / iz_sw * 100, 1) if iz_sw else 0
    pts = p["PitchType"].value_counts().index.tolist()

    fig = plt.figure(figsize=(26, 17), facecolor=BG_COLOR)
    gs = GridSpec(4, 3, figure=fig,
                  height_ratios=[.06, .04, .36, .54],
                  width_ratios=[1, 1.2, 1.2],
                  hspace=.22, wspace=.22,
                  top=0.96, bottom=0.02, left=0.04, right=0.97)

    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    ax.text(.5, .7, pitcher_name.upper(), ha="center", va="center", fontsize=34,
            fontweight="bold", color=TEXT_COLOR, family="monospace")
    ax.text(.5, .1, "Season Pitching Summary", ha="center", va="center",
            fontsize=16, color=ACCENT_COLOR, family="monospace")

    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    outing_details = []
    for p_df, gdate, opp in outings:
        o_ip = calc_ip(p_df)
        o_k = int((p_df["KorBB"] == "Strikeout").sum())
        o_bb = int((p_df["KorBB"] == "Walk").sum())
        outing_details.append(f"{gdate} vs {opp}: {o_ip}IP {o_k}K {o_bb}BB")

    banner = (f"IP {ip_str}   ·   FIP {fip:.2f}   ·   WHIP {whip:.2f}   ·   "
              f"K% {k_pct:.1f}%   ·   BB% {bb_pct:.1f}%   ·   K-BB% {k_pct - bb_pct:.1f}%   ·   "
              f"PA {total_pa}   ·   P {N}   ·   H {total_hits + total_hr}   ·   HR {total_hr}   ·   "
              f"K {total_k}   ·   BB {total_bb}   ·   {len(outings)} outing(s)")
    ax.text(.5, .7, banner, ha="center", va="center", fontsize=14, color=TEXT_COLOR, family="monospace", fontweight="bold")
    ax.text(.5, .2, f"{date_from} to {date_to}     |     " + "  /  ".join(outing_details),
            ha="center", va="center", fontsize=11, color=MUTED_TEXT, family="monospace")

    ax_velo = fig.add_subplot(gs[2, 0]); ax_velo.set_facecolor(PANEL_COLOR)
    pt_velo_sorted = sorted(pts, key=lambda x: p[p["PitchType"] == x]["RelSpeed"].median()
                            if not p[p["PitchType"] == x]["RelSpeed"].dropna().empty else 0, reverse=True)
    for i, pt in enumerate(pt_velo_sorted):
        velos = p.loc[p["PitchType"] == pt, "RelSpeed"].dropna()
        if len(velos) < 3: continue
        try:
            kde = gaussian_kde(velos, bw_method=0.3)
            x_range = np.linspace(velos.min() - 3, velos.max() + 3, 200)
            density = kde(x_range)
            density = density / density.max() * 0.38
            ax_velo.fill_betweenx(i + density, x_range, i - density, alpha=0.7, color=pc(pt))
            ax_velo.plot(x_range, i + density, color="black", lw=0.4)
            ax_velo.plot(x_range, i - density, color="black", lw=0.4)
            med = velos.median()
            ax_velo.plot([med, med], [i - 0.3, i + 0.3], color="black", lw=1, ls="--", alpha=0.5)
        except:
            pass
    ax_velo.set_yticks(range(len(pt_velo_sorted)))
    ax_velo.set_yticklabels(pt_velo_sorted, fontsize=12, fontfamily="monospace")
    ax_velo.set_xlabel("Velocity (mph)", fontsize=12, color=MUTED_TEXT)
    ax_velo.set_title("Velocity Distribution", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax_velo.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax_velo.spines.values(): sp.set_color(GRID_COLOR)

    ax_mov = fig.add_subplot(gs[2, 1]); ax_mov.set_facecolor(PANEL_COLOR)
    ax_mov.axhline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    ax_mov.axvline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    for r in [5, 10, 15, 20]:
        ax_mov.add_patch(plt.Circle((0, 0), r, fill=False, ec=GRID_COLOR, lw=0.3, ls="--", alpha=0.3))
    for pt in pts:
        s = p[p["PitchType"] == pt]
        hb = s["HorzBreak"].dropna(); ivb = s["InducedVertBreak"].dropna()
        if not hb.empty and not ivb.empty:
            ax_mov.scatter(hb.mean(), ivb.mean(), c=pc(pt), label=pt, s=120, alpha=0.95,
                          edgecolors="black", linewidths=1, zorder=5, marker="o")
    ax_mov.set_xlim(-25, 25); ax_mov.set_ylim(-25, 25)
    ax_mov.set_xlabel("HB (in)", fontsize=12, color=MUTED_TEXT)
    ax_mov.set_ylabel("IVB (in)", fontsize=12, color=MUTED_TEXT)
    ax_mov.set_title("Avg Pitch Movement", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax_mov.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=min(len(pts), 5),
                  fontsize=12, frameon=False, labelcolor=TEXT_COLOR)
    ax_mov.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax_mov.spines.values(): sp.set_color(GRID_COLOR)

    ax_usage = fig.add_subplot(gs[2, 2]); ax_usage.set_facecolor(PANEL_COLOR)
    lhb_data = p[p["BatterSide"] == "Left"]
    rhb_data = p[p["BatterSide"] == "Right"]
    lhb_total = len(lhb_data); rhb_total = len(rhb_data)
    bar_pts = sorted(pts, key=lambda x: p[p["PitchType"] == x]["RelSpeed"].median()
                     if not p[p["PitchType"] == x]["RelSpeed"].dropna().empty else 0, reverse=True)
    y_pos = np.arange(len(bar_pts)); bar_h = 0.35
    for i, pt in enumerate(bar_pts):
        lhb_pct = len(lhb_data[lhb_data["PitchType"] == pt]) / lhb_total * 100 if lhb_total > 0 else 0
        rhb_pct = len(rhb_data[rhb_data["PitchType"] == pt]) / rhb_total * 100 if rhb_total > 0 else 0
        ax_usage.barh(i + bar_h / 2, -lhb_pct, bar_h, color=pc(pt), alpha=0.8, edgecolor="black", lw=0.3)
        ax_usage.barh(i - bar_h / 2, rhb_pct, bar_h, color=pc(pt), alpha=0.8, edgecolor="black", lw=0.3)
        if lhb_pct > 2:
            ax_usage.text(-lhb_pct / 2, i + bar_h / 2, f"{lhb_pct:.1f}%", ha="center", va="center",
                         fontsize=10, fontweight="bold", color="white")
        if rhb_pct > 2:
            ax_usage.text(rhb_pct / 2, i - bar_h / 2, f"{rhb_pct:.1f}%", ha="center", va="center",
                         fontsize=10, fontweight="bold", color="white")
    ax_usage.set_yticks(y_pos)
    ax_usage.set_yticklabels(bar_pts, fontsize=12, fontfamily="monospace")
    ax_usage.axvline(0, color="black", lw=1)
    max_pct = max(60, max(
        [len(lhb_data[lhb_data["PitchType"] == pt]) / max(lhb_total, 1) * 100 for pt in bar_pts] +
        [len(rhb_data[rhb_data["PitchType"] == pt]) / max(rhb_total, 1) * 100 for pt in bar_pts]
    ) + 10)
    ax_usage.set_xlim(-max_pct, max_pct)
    ticks = ax_usage.get_xticks()
    ax_usage.set_xticklabels([f"{abs(t):.0f}%" for t in ticks], fontsize=10)
    ax_usage.set_title("Pitch Usage", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax_usage.text(-max_pct * 0.5, len(bar_pts) + 0.3, f"vs LHB ({lhb_total})", ha="center", fontsize=12,
                 color=MUTED_TEXT, fontweight="bold")
    ax_usage.text(max_pct * 0.5, len(bar_pts) + 0.3, f"vs RHB ({rhb_total})", ha="center", fontsize=12,
                 color=MUTED_TEXT, fontweight="bold")
    ax_usage.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax_usage.spines.values(): sp.set_color(GRID_COLOR)

    ax_t = fig.add_subplot(gs[3, :]); ax_t.set_facecolor(BG_COLOR); ax_t.axis("off")
    trows = []
    grade_cells = {}
    for ri, pt in enumerate(pts):
        s = p[p["PitchType"] == pt]; n = len(s)
        s_iz = in_zone(s); _sw = s["PitchCall"].isin(SWING_CALLS)
        _wh = s["PitchCall"] == "StrikeSwinging"; _ooz = ~s_iz
        _ooz_sw = (_sw & _ooz).sum(); _ooz_n = _ooz.sum()
        _iz_sw = (_sw & s_iz).sum(); _iz_wh = (_wh & s_iz).sum()
        iz_whiff_str = f"{_iz_wh / _iz_sw * 100:.1f}%" if _iz_sw else "—"
        _sw_ct = _sw.sum()
        whiff_val = _wh.sum() / _sw_ct * 100 if _sw_ct else None
        whiff_str = f"{whiff_val:.1f}%" if whiff_val is not None else "—"
        chase_val = _ooz_sw / _ooz_n * 100 if _ooz_n else None
        chase_str = f"{chase_val:.1f}%" if chase_val is not None else "—"
        xw = s["xwOBA"].dropna()
        xwoba_val = xw.mean() if not xw.empty else None
        xwoba_str = f"{xwoba_val:.3f}" if xwoba_val is not None else "—"
        avg_velo_raw = s["RelSpeed"].dropna()
        avg_velo_val = avg_velo_raw.mean() if not avg_velo_raw.empty else None
        zone_val = s_iz.sum() / n * 100 if n else None
        zone_str = f"{zone_val:.1f}%" if zone_val is not None else "—"
        stuff_val = score_stuff_plus(s, pt, p)
        stuff_str = f"{stuff_val:.0f}" if stuff_val is not None else "—"
        trows.append([pt, n, f"{n / N * 100:.1f}%", stuff_str,
                      fmt(s["RelSpeed"]), fmt(s["RelSpeed"], "max"),
                      fmt(s["SpinRate"], d=0),
                      fmt(s["InducedVertBreak"]), fmt(s["HorzBreak"]),
                      fmt(s["Extension"]), fmt(s["RelHeight"]), fmt(s["RelSide"]),
                      havaa_fmt(s, pt),
                      xwoba_str, zone_str, whiff_str, chase_str, iz_whiff_str])

        data_row = ri + 1
        if avg_velo_val is not None:
            grade_cells[(data_row, 3)] = (pt, "velo", avg_velo_val, True)
        if xwoba_val is not None:
            grade_cells[(data_row, 12)] = (pt, "xwoba", xwoba_val, False)
        if zone_val is not None:
            grade_cells[(data_row, 13)] = (pt, "zone_pct", zone_val, True)
        if whiff_val is not None:
            grade_cells[(data_row, 14)] = (pt, "whiff_pct", whiff_val, True)
        if chase_val is not None:
            grade_cells[(data_row, 15)] = (pt, "chase_pct", chase_val, True)

    all_sw_ct = sw.sum()
    all_whiff = f"{wh.sum() / all_sw_ct * 100:.1f}%" if all_sw_ct else "0%"
    all_xw = p["xwOBA"].dropna()
    all_xwoba = f"{all_xw.mean():.3f}" if not all_xw.empty else "—"
    trows.append(["All", N, "100%", "—", "—", "—", "—", "—", "—",
                  fmt(p["Extension"]), "—", "—", "—",
                  all_xwoba, f"{zpct}%", all_whiff, f"{cpct}%", f"{izwp}%"])

    cols = ["Count", "Usage%", "Stuff+", "Avg\nVelo", "Max\nVelo", "Avg\nSpin",
            "IVB", "HB", "Ext", "RelH", "RelS", "HAVAA",
            "xwOBA", "Zone%", "Whiff%", "Chase%", "IZ\nWhiff%"]
    tbl = ax_t.table(cellText=[r[1:] for r in trows], rowLabels=[r[0] for r in trows],
                     colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1E1E2E")
            cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=12)
        elif col == -1:
            pitch_name = cell.get_text().get_text()
            if pitch_name == "All":
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontweight="bold", color=TEXT_COLOR, fontfamily="monospace", fontsize=13)
            else:
                cell.set_facecolor(pc(pitch_name))
                cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=13)
        else:
            graded = False
            if (row, col) in grade_cells and row <= len(pts):
                pt_name, stat_name, raw_val, higher_better = grade_cells[(row, col)]
                gc = grade_color(pt_name, stat_name, raw_val, higher_better)
                if gc is not None:
                    cell.set_facecolor(gc)
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontweight="bold")
                    graded = True
            if not graded:
                if row == len(trows):
                    cell.set_facecolor("#F0F0F0")
                    cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontfamily="monospace")
                elif row % 2 == 0:
                    cell.set_facecolor("#F7F8FA")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace")
                else:
                    cell.set_facecolor("#FFFFFF")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace")

    return fig

# ===========================================================================
# HEATMAP FUNCTIONS
# ===========================================================================
RUN_VALUES = {
    "StrikeSwinging": -0.065, "StrikeCalled": -0.038, "FoulBallNotFieldable": -0.025,
    "BallCalled": 0.032, "BallinDirt": 0.032, "BallIntentional": 0.032,
    "HitByPitch": 0.035,
}

def compute_pitch_run_value(row):
    call = row.get("PitchCall", "")
    if call in RUN_VALUES:
        return RUN_VALUES[call]
    if call == "InPlay":
        xw = row.get("xwOBA", np.nan)
        if pd.notna(xw):
            return (xw - 0.320) / 1.15
        return 0.0
    return 0.0

HEATMAP_GRID_NX = 80
HEATMAP_GRID_NY = 80
HEATMAP_X_RANGE = (-2.5, 2.5)
HEATMAP_Y_RANGE = (-0.5, 5.0)
HEATMAP_KDE_BW = 0.4
HEATMAP_SPATIAL_SIGMA = 0.3
# Effective sample size for the empirical-Bayes prior. Cells with this many
# pitches contribute equally with the prior; cells with fewer pitches get
# pulled toward the prior. Higher = more aggressive shrinkage.
HEATMAP_PRIOR_K = 25.0
# Map our internal metric name to the key used in d1_heatmap_priors.json
_METRIC_TO_PRIOR_KEY = {
    "run_value": "run_value",
    "whiff":     "whiff",
    "xwoba":     "xwoba",
}

def _prior_grid_for(pitch_type, side, metric):
    """Return the (80,80) numpy prior grid for this metric, or None if missing."""
    if not D1_HEATMAP_PRIORS:
        return None
    key = _METRIC_TO_PRIOR_KEY.get(metric)
    if key is None:
        return None
    pt_block = D1_HEATMAP_PRIORS.get(pitch_type)
    if not pt_block:
        return None
    side_block = pt_block.get(side)
    if not side_block:
        return None
    return side_block.get(key)

def generate_heatmap(p, pitch_type, metric="run_value"):
    sub = p[p["PitchType"] == pitch_type].copy()
    sub = sub[sub["PlateLocSide"].notna() & sub["PlateLocHeight"].notna()]
    if len(sub) < 5:
        return None

    is_density_only = False
    if metric == "location":
        is_density_only = True
        cmap_name = "YlOrRd"
        title_label = "Pitch Location Density"
        vmin, vmax = 0, 1
    elif metric == "run_value":
        sub["_val"] = sub.apply(compute_pitch_run_value, axis=1)
        cmap_name = "RdBu_r"
        title_label = "Run Value"
        vmin, vmax = -0.08, 0.08
    elif metric == "whiff":
        sub["_val"] = (sub["PitchCall"] == "StrikeSwinging").astype(float)
        cmap_name = "YlOrRd"
        title_label = "Whiff Rate"
        vmin, vmax = 0, 0.6
    elif metric == "xwoba":
        def xw_val(row):
            if row["PitchCall"] == "InPlay" and pd.notna(row.get("xwOBA")):
                return row["xwOBA"]
            elif row["PitchCall"] == "StrikeSwinging":
                return 0.0
            elif row["PitchCall"] == "StrikeCalled":
                return 0.05
            elif row["PitchCall"] in ("BallCalled", "BallinDirt"):
                return 0.4
            return np.nan
        sub["_val"] = sub.apply(xw_val, axis=1)
        sub = sub[sub["_val"].notna()]
        cmap_name = "RdYlBu_r"
        title_label = "xwOBA"
        vmin, vmax = 0, 0.8
    else:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG_COLOR)

    # Build the grid once
    xi = np.linspace(*HEATMAP_X_RANGE, HEATMAP_GRID_NX)
    yi = np.linspace(*HEATMAP_Y_RANGE, HEATMAP_GRID_NY)
    Xi, Yi = np.meshgrid(xi, yi)
    sigma2 = 2 * HEATMAP_SPATIAL_SIGMA ** 2

    # Detect whether we'll apply EB shrinkage on this metric (not for density)
    use_eb = (not is_density_only)

    for idx, (side, label) in enumerate([("Left", "vs LHB"), ("Right", "vs RHB")]):
        ax = axes[idx]
        ax.set_facecolor(PANEL_COLOR)
        side_data = sub[sub["BatterSide"] == side]

        ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=False, ec="black", lw=1.5, zorder=10))
        ax.add_patch(Polygon([(-.708, .55), (.708, .55), (.708, .35), (0, .15), (-.708, .35)],
                             closed=True, fc="#CCCCCC", ec="black", lw=.5, alpha=0.5, zorder=10))

        # Pull the D1 prior grid for this (pitch type, batter side, metric)
        prior_grid = _prior_grid_for(pitch_type, side, metric) if use_eb else None

        if len(side_data) >= 5:
            x = -side_data["PlateLocSide"].values  # mirror to catcher POV
            y = side_data["PlateLocHeight"].values
            try:
                positions = np.vstack([x, y])
                kde = gaussian_kde(positions, bw_method=HEATMAP_KDE_BW)
                density = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

                if is_density_only:
                    Zi = density / density.max() if density.max() > 0 else density
                    Zi[Zi < 0.05] = np.nan
                else:
                    vals = side_data["_val"].values
                    # Gaussian-weighted local mean (same as before)
                    Zi_local = np.zeros_like(Xi)
                    W = np.zeros_like(Xi)
                    for px, py, pv in zip(x, y, vals):
                        if not np.isfinite(pv):
                            continue
                        dist2 = (Xi - px) ** 2 + (Yi - py) ** 2
                        w = np.exp(-dist2 / sigma2)
                        Zi_local += w * pv
                        W += w
                    # n_local = "effective sample size" at each cell. Use the
                    # weight sum directly — it's the sum of Gaussian weights,
                    # which approximates how many pitches contribute.
                    n_local = W.copy()
                    safe_W = np.where(W == 0, 1, W)
                    local_mean = Zi_local / safe_W

                    if prior_grid is not None:
                        # ── Empirical Bayes shrinkage ──
                        # shrunk = (n * local + k * prior) / (n + k)
                        # Where prior is NaN (sparse league cells), fall back
                        # to the local mean.
                        prior = prior_grid
                        valid_prior = np.isfinite(prior)
                        # Default: just local mean
                        Zi = local_mean.copy()
                        # Apply shrinkage only where prior is valid
                        denom = n_local + HEATMAP_PRIOR_K
                        shrunk = (n_local * local_mean +
                                  HEATMAP_PRIOR_K * np.where(valid_prior, prior, 0)) / denom
                        Zi = np.where(valid_prior, shrunk, local_mean)
                    else:
                        # No prior available — fall back to old behavior
                        Zi = local_mean

                    # Mask very-low-density cells so we don't show garbage
                    density_thresh = density.max() * 0.05
                    Zi[density < density_thresh] = np.nan

                im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap_name, vmin=vmin, vmax=vmax,
                                   shading="gouraud", zorder=1)
            except Exception:
                pass
            ax.scatter(x, y, c="black", s=8, alpha=0.5, zorder=6)

        n_side = len(side_data)
        prior_note = ""
        if use_eb and prior_grid is not None:
            prior_note = "  ·  EB shrunk"
        ax.set_title(f"{pitch_type} {label} ({n_side}){prior_note}",
                     fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=8)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 5.0)
        ax.set_xlabel("Plate Side (ft)", fontsize=12, color=MUTED_TEXT)
        if idx == 0:
            ax.set_ylabel("Plate Height (ft)", fontsize=12, color=MUTED_TEXT)
        ax.tick_params(labelsize=7, colors=MUTED_TEXT)
        for sp in ax.spines.values(): sp.set_color(GRID_COLOR)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(title_label, fontsize=12)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{title_label} Heatmap — {pitch_type}", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, y=0.98)
    return fig

# ===========================================================================
# HELPER: Parse game date from session
# ===========================================================================
def parse_session_date(session, fallback_date):
    # API schema confirms gameDateLocal and gameDateUtc exist on session objects
    for field in ["gameDateLocal", "gameDateUtc"]:
        val = session.get(field, "")
        if val and isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "").split("+")[0]).date()
            except:
                pass
    return fallback_date


# ===========================================================================
# HITTER CARD FUNCTIONS
# ===========================================================================

# ── Hitter percentiles ────────────────────────────────────────────────────────
def load_hitter_percentiles():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    paths = [
        os.path.join(base, "D1_hitter_percentiles.json"),
        os.path.join(os.path.expanduser("~"), "Downloads", "D1_hitter_percentiles.json"),
        "D1_hitter_percentiles.json",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

D1_HITTER_PCTLS = load_hitter_percentiles()

def get_hitter_percentile(stat_name, value):
    """Return 0-100 percentile for a hitter stat."""
    if not D1_HITTER_PCTLS or pd.isna(value): return None
    data = D1_HITTER_PCTLS.get(stat_name, {})
    pctls = data.get("percentiles", {})
    if not pctls: return None
    pts = sorted([(int(k), v) for k, v in pctls.items()])
    if value <= pts[0][1]:  return pts[0][0]
    if value >= pts[-1][1]: return pts[-1][0]
    for i in range(len(pts) - 1):
        p0, v0 = pts[i]; p1, v1 = pts[i+1]
        if v0 <= value <= v1:
            if v1 == v0: return (p0+p1)/2
            return p0 + (value-v0)/(v1-v0)*(p1-p0)
        elif v0 >= value >= v1:
            if v0 == v1: return (p0+p1)/2
            return p0 + (v0-value)/(v0-v1)*(p1-p0)
    return None

def hitter_grade_color(stat_name, value, higher_is_better=True):
    pctile = get_hitter_percentile(stat_name, value)
    if pctile is None: return None
    norm = pctile / 100.0
    if not higher_is_better: norm = 1.0 - norm
    return GRADE_CMAP(norm)

# ── Spray angle estimation ────────────────────────────────────────────────────
def estimate_spray_angle(row):
    """
    Estimate spray angle from PlateLocSide + BatterSide.
    Returns degrees: negative = pull side, 0 = center, positive = oppo side.
    This is a heuristic — positive PlateLocSide = arm side of plate.
    For RHB: arm side = inside = pull (left field)
    For LHB: arm side = inside = pull (right field)
    """
    loc = row.get("PlateLocSide", np.nan)
    side = str(row.get("BatterSide", "Right"))
    if pd.isna(loc): return np.nan
    # Invert for LHB since field is mirrored
    if side == "Left":
        angle = loc * 30    # positive = pull right for LHB
    else:
        angle = -loc * 30   # negative = pull left for RHB
    return float(np.clip(angle, -45, 45))

def spray_direction(angle):
    """Classify spray angle into Pull / Center / Oppo."""
    if pd.isna(angle): return "Unknown"
    if angle < -15:  return "Pull"
    if angle > 15:   return "Oppo"
    return "Center"

# ── Get team batting data ─────────────────────────────────────────────────────
def get_team_batting(df, team_name, date_from, date_to):
    """Filter to at-bats where the selected team was BATTING."""
    df = df.copy()
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
    # Cast out of categorical so comparisons work
    for col in ("HomeTeam", "AwayTeam", "TopBottom"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    team_name = str(team_name).strip() if team_name else team_name
    date_from = _to_date(date_from)
    date_to   = _to_date(date_to)
    date_mask = (df["GameDate"] >= date_from) & (df["GameDate"] <= date_to)
    mask = date_mask & (
        ((df["HomeTeam"] == team_name) & (df["TopBottom"] == "Bottom")) |
        ((df["AwayTeam"] == team_name) & (df["TopBottom"] == "Top"))
    )
    return df[mask].copy()

def get_batters(df):
    """Return sorted list of unique batters."""
    return sorted(df["Batter"].dropna().unique().tolist())

# ── Per-batter stats ──────────────────────────────────────────────────────────
def compute_batter_stats(df):
    """Compute all hitter card stats for a batter DataFrame."""
    SWING_CALLS = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay"]
    bip = df[df["PitchCall"] == "InPlay"].copy()
    bip["xwOBA_val"] = bip.apply(lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"]), axis=1)
    bip["spray_angle"] = bip.apply(estimate_spray_angle, axis=1)

    ev   = bip["ExitSpeed"].dropna()
    la   = bip["LaunchAngle"].dropna()
    xw   = bip["xwOBA_val"].dropna()

    swings    = df["PitchCall"].isin(SWING_CALLS).sum()
    whiffs    = (df["PitchCall"] == "StrikeSwinging").sum()
    iz_mask   = in_zone(df)
    ooz       = df[~iz_mask]
    ooz_swings = ooz["PitchCall"].isin(SWING_CALLS).sum()
    iz_swings  = df[iz_mask]["PitchCall"].isin(SWING_CALLS).sum()
    iz_total   = iz_mask.sum()

    return {
        # Contact quality
        "avg_ev":       ev.mean()                  if not ev.empty else np.nan,
        "max_ev":       ev.max()                   if not ev.empty else np.nan,
        "ev90":         float(np.percentile(ev, 90)) if len(ev) >= 5 else np.nan,
        "barrel_pct":   float(((bip["ExitSpeed"] >= 98) & (bip["LaunchAngle"].between(26,30))).sum()
                              / len(bip) * 100)    if not bip.empty else np.nan,
        "sweet_spot_pct": float(((la >= 8) & (la <= 32)).sum() / len(la) * 100)
                              if not la.empty else np.nan,
        "avg_la":       la.mean()                  if not la.empty else np.nan,
        "gb_pct":       float((la < 10).sum() / len(la) * 100) if not la.empty else np.nan,
        "ld_pct":       float(((la >= 10) & (la <= 25)).sum() / len(la) * 100) if not la.empty else np.nan,
        "fb_pct":       float((la > 25).sum() / len(la) * 100) if not la.empty else np.nan,
        # Swing decisions
        "whiff_pct":    float(whiffs / swings * 100)    if swings > 0 else np.nan,
        "chase_pct":    float(ooz_swings / len(ooz) * 100) if len(ooz) > 0 else np.nan,
        "zone_sw_pct":  float(iz_swings / iz_total * 100)  if iz_total > 0 else np.nan,
        "contact_pct":  float((swings - whiffs) / swings * 100) if swings > 0 else np.nan,
        # Outcomes
        "xwoba":        xw.mean()                  if not xw.empty else np.nan,
        "hard_hit_pct": float((ev >= 95).sum() / len(ev) * 100) if not ev.empty else np.nan,
        "pa":           df.groupby(["GameDate","Inning","PAofInning"]).ngroups,
        "bip":          len(bip),
        # Spray
        "pull_pct":     float((bip["spray_angle"] < -15).sum() / len(bip) * 100) if not bip.empty else np.nan,
        "center_pct":   float((bip["spray_angle"].between(-15,15)).sum() / len(bip) * 100) if not bip.empty else np.nan,
        "oppo_pct":     float((bip["spray_angle"] > 15).sum() / len(bip) * 100) if not bip.empty else np.nan,
        "bip_df":       bip,
    }

# ── Drawing helpers ───────────────────────────────────────────────────────────
def _hax(ax):
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values(): sp.set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(False)

def draw_ev_la_scatter(ax, bip):
    """EV vs Launch Angle scatter coloured by xwOBA."""
    if bip.empty:
        ax.text(0.5, 0.5, "No BIP data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=8)
        ax.set_facecolor(PANEL_COLOR); return
    _hax(ax)
    sc = ax.scatter(bip["LaunchAngle"], bip["ExitSpeed"],
                    c=bip["xwOBA_val"].clip(0, 1.5), cmap="RdYlGn",
                    vmin=0, vmax=1.5, s=40, alpha=0.9, edgecolors="#333333", linewidths=0.5)
    ax.axvspan(8, 32, color=ACCENT_COLOR, alpha=0.06, label="Sweet spot")
    ax.axhline(95, color="#CC3333", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Launch Angle (°)", fontsize=7)
    ax.set_ylabel("Exit Velocity (mph)", fontsize=7)
    ax.set_title("EV vs Launch Angle", fontsize=9, fontweight="bold", color=TEXT_COLOR)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("xwOBA", color=MUTED_TEXT, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=MUTED_TEXT, labelsize=6)

def draw_ev_distribution(ax, bip):
    """EV distribution with percentile markers."""
    if bip.empty or bip["ExitSpeed"].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=8)
        ax.set_facecolor(PANEL_COLOR); return
    _hax(ax)
    ev = bip["ExitSpeed"].dropna()
    n_bins = min(20, max(5, len(ev) // 2))
    ax.hist(ev, bins=n_bins, color=ACCENT_COLOR, alpha=0.75, edgecolor="white", lw=0.8)
    ymax = ax.get_ylim()[1]
    for p, col, lbl in [(50, "#333333", "Avg"), (90, "#B8860B", "90th")]:
        val = np.percentile(ev, p)
        ax.axvline(val, color=col, lw=2.0, ls="--")
        ax.text(val + 0.5, ymax * 0.88, f"{lbl}\n{val:.1f}", color=col,
                fontsize=7, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85, lw=0.8))
    ax.set_xlabel("Exit Velocity (mph)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.set_title("EV Distribution", fontsize=9, fontweight="bold", color=TEXT_COLOR)

def draw_zone_heatmap(ax, df, stat="ev", title="EV Heatmap", filter_df=None):
    """
    Grid-based zone heatmap with numbered cells, bold strike zone,
    zone thirds lines, and home plate.
    """
    SWING_CALLS = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay"]
    _hax(ax)
    use_df = filter_df if filter_df is not None else df
    if use_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=10)
        return

    # Build 6x6 grid
    x_bins = np.linspace(-1.5, 1.5, 7)
    y_bins = np.linspace(1.0, 4.0, 7)
    grid   = np.full((6, 6), np.nan)

    bip = use_df[use_df["PitchCall"] == "InPlay"].copy()
    if not bip.empty:
        bip["xwOBA_val"] = bip.apply(
            lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"]), axis=1)

    for i in range(6):
        for j in range(6):
            x0, x1 = x_bins[j], x_bins[j+1]
            y0, y1 = y_bins[i], y_bins[i+1]
            # Mirror PlateLocSide to catcher POV (negate)
            loc_mask = ((-use_df["PlateLocSide"]).between(x0, x1) &
                        use_df["PlateLocHeight"].between(y0, y1))
            cell_all = use_df[loc_mask]
            cell_bip = bip[bip["PlateLocSide"].between(x0, x1) &
                           bip["PlateLocHeight"].between(y0, y1)] if not bip.empty else bip

            if stat == "ev" and len(cell_bip) >= 1:
                v = cell_bip["ExitSpeed"].dropna()
                if not v.empty: grid[i, j] = v.mean()
            elif stat == "xwoba" and len(cell_bip) >= 1:
                v = cell_bip["xwOBA_val"].dropna()
                if not v.empty: grid[i, j] = v.mean()
            elif stat == "whiff":
                sw = cell_all["PitchCall"].isin(SWING_CALLS).sum()
                wh = (cell_all["PitchCall"] == "StrikeSwinging").sum()
                if sw >= 1: grid[i, j] = wh / sw * 100
            elif stat == "swing":
                total = len(cell_all)
                sw    = cell_all["PitchCall"].isin(SWING_CALLS).sum()
                if total >= 1: grid[i, j] = sw / total * 100

    # Color ranges
    if stat == "ev":
        vmin, vmax, cmap = 65, 105, "RdYlGn_r"   # red = hard contact
    elif stat == "xwoba":
        vmin, vmax, cmap = 0.0, 1.2, "RdYlGn_r"
    elif stat == "whiff":
        vmin, vmax, cmap = 0, 60, "RdYlGn_r"
    else:  # swing
        vmin, vmax, cmap = 0, 100, "RdYlGn"

    im = ax.imshow(grid, extent=[-1.5, 1.5, 1.0, 4.0], origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", alpha=0.88)

    # Cell value labels
    for i in range(6):
        for j in range(6):
            v = grid[i, j]
            if not np.isnan(v):
                fmt = f"{v:.0f}" if stat in ("ev", "whiff", "swing") else f"{v:.2f}"
                norm_v = (v - vmin) / max(vmax - vmin, 1)
                txt_col = "black" if 0.25 < norm_v < 0.75 else "white"
                ax.text(x_bins[j] + 0.25, y_bins[i] + 0.25, fmt,
                        ha="center", va="center", fontsize=11,
                        color=txt_col, fontweight="bold")

    # Bold strike zone box
    ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2.0,
                                lw=3.0, ec="#111111", fc="none", zorder=5))
    # Zone thirds dashed lines
    for yline in [2.167, 2.833]:
        ax.plot([-0.83, 0.83], [yline, yline],
                color="#111111", lw=1.0, ls="--", alpha=0.6, zorder=4)
    # Home plate — tip points DOWN (toward viewer = catcher POV)
    ax.add_patch(Polygon([(-.708,.55),(.708,.55),(.708,.35),(0,.15),(-.708,.35)],
                 closed=True, fc="#CCCCCC", ec="#222222", lw=1.2, alpha=0.85, zorder=5))

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(0.8, 4.0)
    ax.set_xlabel("Plate Side (ft)", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Plate Height (ft)", fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for sp in ax.spines.values(): sp.set_color("#CCCCCC")
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.ax.tick_params(labelsize=9, colors=TEXT_COLOR)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=TEXT_COLOR)

def draw_swing_zones(ax, df):
    """Swing rate zone heatmap."""
    _hax(ax)
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=8); return
    draw_zone_heatmap(ax, df, stat="swing", title="Swing Rate by Zone")

def draw_spray_chart(ax, bip):
    """
    Estimated spray chart.
    FIX: spray_angle=0 → center field (straight up on chart).
    Positive angle = right side (oppo for RHB / pull for LHB).
    x = dist * sin(angle_rad)  — horizontal spread
    y = dist * cos(angle_rad)  — depth into field
    """
    ax.set_facecolor("#F0F4F0")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    if bip is None or bip.empty:
        ax.text(0.5, 0.5, "No BIP data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=8); return

    bip = bip.copy()
    if "spray_angle" not in bip.columns or bip["spray_angle"].isna().all():
        ax.text(0.5, 0.5, "No location data\nfor spray chart",
                transform=ax.transAxes, ha="center", va="center",
                color=MUTED_TEXT, fontsize=8); return

    batter_side = bip["BatterSide"].mode()[0] if "BatterSide" in bip.columns and not bip["BatterSide"].empty else "Right"

    # ── Field outline ──
    # Foul lines at 45° left and right of center
    r_out = 320
    for sign in [-1, 1]:
        fx = sign * r_out * np.sin(np.radians(45))
        fy = r_out * np.cos(np.radians(45))
        ax.plot([0, fx], [0, fy], color="#888888", lw=1.0, zorder=1)

    # Outfield arc
    theta = np.linspace(-np.radians(45), np.radians(45), 100)
    ax.plot(r_out * np.sin(theta), r_out * np.cos(theta), color="#888888", lw=1.2, zorder=1)

    # Infield arc
    r_in = 95
    ax.plot(r_in * np.sin(theta), r_in * np.cos(theta),
            color="#AAAAAA", lw=0.7, ls="--", zorder=1)

    # Bases: 3B left, 2B center, 1B right, home plate bottom
    base_coords = [(-63, 63), (0, 126), (63, 63), (0, 0)]
    for bx, by in base_coords:
        ax.plot(bx, by, "s", color="#CC9900", ms=6, zorder=4, markeredgecolor="#333333", markeredgewidth=0.5)

    # ── Plot BIPs ──
    valid = bip.dropna(subset=["spray_angle", "ExitSpeed"])
    for _, row in valid.iterrows():
        ang  = float(row["spray_angle"])   # degrees: neg=left, 0=center, pos=right
        ev   = float(row["ExitSpeed"])
        dist = min(max(ev * 2.5, 60), 310)

        # FIXED: x = lateral (sin), y = depth (cos)
        ang_rad = np.radians(ang)
        x = dist * np.sin(ang_rad)
        y = dist * np.cos(ang_rad)

        result = row.get("PlayResult", "")
        if result in ("HomeRun", "Triple", "Double"):
            color, ms = "#FF4444", 40
        elif result == "Single":
            color, ms = "#44FF88", 35
        else:
            color, ms = "#8888AA", 25

        ax.scatter(x, y, c=color, s=ms, alpha=0.85, edgecolors="white",
                   linewidths=0.4, zorder=3)

    ax.set_xlim(-340, 340)
    ax.set_ylim(-25, 340)
    ax.set_aspect("equal")
    hand_lbl = "LHB" if batter_side == "Left" else "RHB"
    ax.set_title(f"Spray Chart (est.) — {hand_lbl}",
                 fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.axis("off")

    for col, lbl in [("#FF4444", "XBH"), ("#44FF88", "Single"), ("#8888AA", "Out")]:
        ax.scatter([], [], c=col, s=25, label=lbl, edgecolors="white", linewidths=0.4)
    ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_COLOR,
              loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

def draw_batted_ball_profile(ax, stats):
    """GB / LD / FB bar chart."""
    _hax(ax)
    labels = ["GB\n(<10°)", "LD\n(10-25°)", "FB\n(>25°)"]
    values = [stats.get("gb_pct") or 0, stats.get("ld_pct") or 0, stats.get("fb_pct") or 0]
    colors = ["#4488FF", "#44FF88", "#FF6644"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", lw=0.8)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color="#111111", padding=3, fontweight="bold")
    ax.set_ylim(0, max(max(values) * 1.35, 10))
    ax.set_ylabel("%", fontsize=7, color="#111111")
    ax.set_title("Batted Ball Profile", fontsize=9, fontweight="bold", color=TEXT_COLOR)

def draw_pull_oppo(ax, stats):
    """Pull / Center / Oppo bar chart."""
    _hax(ax)
    labels = ["Pull", "Center", "Oppo"]
    values = [stats.get("pull_pct") or 0, stats.get("center_pct") or 0, stats.get("oppo_pct") or 0]
    colors = ["#E84040", "#F0B800", "#3399EE"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", lw=0.8)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color="#111111", padding=3, fontweight="bold")
    ax.set_ylim(0, max(max(values) * 1.35, 10))
    ax.set_ylabel("%", fontsize=7, color="#111111")
    ax.set_title("Spray Direction", fontsize=9, fontweight="bold", color=TEXT_COLOR)

# ── Stats banner ──────────────────────────────────────────────────────────────
def draw_hitter_stats_banner(ax, stats, batter_name):
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    def _bg(stat, val, hib):
        c = hitter_grade_color(stat, val, hib) if D1_HITTER_PCTLS else None
        return c if c else (0.94, 0.94, 0.97, 1.0)

    def _text_color_for_bg(bg_rgba):
        """Return black or white depending on background luminance — strict threshold."""
        if isinstance(bg_rgba, tuple) and len(bg_rgba) >= 3:
            r, g, b = bg_rgba[:3]
            # Perceptual luminance
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            return "#111111" if lum > 0.45 else "#FFFFFF"
        return "#111111"

    metrics = [
        ("Avg EV",   stats.get("avg_ev"),          "ev90",          True,  ".1f", " mph"),
        ("90th EV",  stats.get("ev90"),             "ev90",          True,  ".1f", " mph"),
        ("Max EV",   stats.get("max_ev"),            "ev90",          True,  ".1f", " mph"),
        ("Barrel%",  stats.get("barrel_pct"),       "sweet_spot_pct",True,  ".1f", "%"),
        ("Swt Spt%", stats.get("sweet_spot_pct"),   "sweet_spot_pct",True,  ".1f", "%"),
        ("xwOBA",    stats.get("xwoba"),             "xwoba",         True,  ".3f", ""),
        ("Whiff%",   stats.get("whiff_pct"),        "whiff_pct",     False, ".1f", "%"),
        ("Chase%",   stats.get("chase_pct"),        "chase_pct",     False, ".1f", "%"),
        ("Contact%", stats.get("contact_pct"),      "whiff_pct",     True,  ".1f", "%"),
        ("HardHit%", stats.get("hard_hit_pct"),     "ev90",          True,  ".1f", "%"),
        ("Avg LA",   stats.get("avg_la"),            "sweet_spot_pct",True, ".1f", "°"),
        ("PA",       stats.get("pa"),               None,            True,  "d",   ""),
    ]

    n     = len(metrics)
    col_w = 1.0 / n

    for i, (label, val, pct_key, hib, fmt, unit) in enumerate(metrics):
        x   = (i + 0.5) * col_w
        missing = val is None or (isinstance(val, float) and np.isnan(val))

        if missing:
            disp   = "—"
            bg     = (0.91, 0.91, 0.93, 1.0)
            t_col  = "#888899"
            p_txt  = ""
        else:
            disp  = f"{val:{fmt}}{unit}"
            bg    = _bg(pct_key, val, hib) if pct_key else (0.15, 0.15, 0.25, 1.0)
            t_col = _text_color_for_bg(bg)
            p = get_hitter_percentile(pct_key, val) if pct_key else None
            p_txt = f"{int(round(p))}th" if p is not None else ""

        ax.add_patch(plt.Rectangle((i * col_w + 0.004, 0.05), col_w - 0.008, 0.90,
                                    transform=ax.transAxes, color=bg,
                                    clip_on=False, zorder=2))
        # Label — slightly muted version of contrast color
        label_col = t_col if missing else t_col
        ax.text(x, 0.76, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=7,
                color=label_col, fontweight="bold", zorder=3, alpha=0.75)
        # Value — full contrast
        ax.text(x, 0.44, disp, transform=ax.transAxes,
                ha="center", va="center", fontsize=11,
                color=t_col, fontweight="bold", zorder=3)
        # Percentile
        if p_txt:
            ax.text(x, 0.13, p_txt, transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5,
                    color=t_col, alpha=0.80, zorder=3)


# ── Standalone hitter heatmap ─────────────────────────────────────────────────
def generate_hitter_heatmap(batter_df, metric="ev", pitch_type=None,
                             pitcher_hand=None, count=None):
    """
    KDE-smoothed single-panel hitter zone heatmap.
    PitcherThrows is blank in the parquet so no LHP/RHP split —
    show one combined panel. Filters: pitch_type, pitcher_hand, count.
    """
    SWING_CALLS = {"StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay"}

    df = batter_df.copy()
    df = df[df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()]

    # ── Filters ──
    if pitch_type:
        df = df[df["PitchType"] == pitch_type]
    if count:
        if count == "2-strike":
            df = df[df["Strikes"] == 2]
        elif count == "ahead":
            df = df[df["Balls"] > df["Strikes"]]
        elif count == "behind":
            df = df[df["Strikes"] > df["Balls"]]
        elif "-" in str(count):
            try:
                b, s = count.split("-")
                df = df[(df["Balls"] == int(b)) & (df["Strikes"] == int(s))]
            except Exception:
                pass

    if len(df) < 3:
        return None

    # ── Metric setup ──
    is_density = False
    if metric == "location":
        is_density  = True
        cmap_name   = "YlOrRd"
        title_label = "Pitch Location Density"
        vmin, vmax  = 0, 1
        plot_df     = df

    elif metric == "ev":
        plot_df = df[(df["PitchCall"] == "InPlay") & df["ExitSpeed"].notna()].copy()
        plot_df["_val"] = plot_df["ExitSpeed"]
        cmap_name   = "RdYlGn_r"   # red=hard, green=soft — flipped so RED = danger
        title_label = "Exit Velocity (mph)"
        vmin, vmax  = 65, 105

    elif metric == "xwoba":
        df["_val"] = df.apply(
            lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"])
            if r["PitchCall"] == "InPlay" else (
                0.0  if r["PitchCall"] == "StrikeSwinging" else
                0.05 if r["PitchCall"] == "StrikeCalled"   else np.nan), axis=1)
        plot_df   = df[df["_val"].notna()].copy()
        cmap_name   = "RdYlGn_r"
        title_label = "xwOBA"
        vmin, vmax  = 0.0, 1.2

    elif metric == "whiff":
        plot_df = df[df["PitchCall"].isin(SWING_CALLS)].copy()
        plot_df["_val"] = (plot_df["PitchCall"] == "StrikeSwinging").astype(float)
        cmap_name   = "RdYlGn"    # green=whiff (bad for hitter), red=contact
        title_label = "Whiff Rate"
        vmin, vmax  = 0, 0.6

    elif metric == "swing":
        plot_df = df.copy()
        plot_df["_val"] = plot_df["PitchCall"].isin(SWING_CALLS).astype(float)
        cmap_name   = "RdYlGn"
        title_label = "Swing Rate"
        vmin, vmax  = 0, 1.0
    else:
        return None

    if len(plot_df) < 3:
        return None

    # ── Title ──
    batter_name = plot_df["Batter"].iloc[0] if "Batter" in plot_df.columns else "Batter"
    batter_side = plot_df["BatterSide"].mode()[0] if "BatterSide" in plot_df.columns else "?"
    hand_lbl    = "LHB" if batter_side == "Left" else "RHB" if batter_side == "Right" else batter_side

    parts = []
    if pitch_type: parts.append(pitch_type)
    if count:      parts.append(f"{count} count")
    filter_str = "  ·  " + "  ·  ".join(parts) if parts else ""

    date_range = f"{batter_df['GameDate'].min()} – {batter_df['GameDate'].max()}"

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # Strike zone
    ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9,
                            fill=False, ec="#333333", lw=2.0, zorder=10))
    # Zone thirds
    for yline in [2.167, 2.833]:
        ax.plot([-0.95, 0.95], [yline, yline],
                color="#333333", lw=0.5, ls="--", alpha=0.5, zorder=9)
    # Home plate — tip points DOWN (toward viewer = catcher POV)
    ax.add_patch(Polygon([(-.708,.55),(.708,.55),(.708,.35),(0,.15),(-.708,.35)],
                         closed=True, fc="#CCCCCC", ec="#333333", lw=.8,
                         alpha=0.7, zorder=10))

    x = -plot_df["PlateLocSide"].values  # mirror to catcher POV
    y = plot_df["PlateLocHeight"].values
    xi = np.linspace(-2.5, 2.5, 100)
    yi = np.linspace(-0.5, 5.0, 100)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        positions = np.vstack([x, y])
        kde = gaussian_kde(positions, bw_method=0.35)
        density = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        if is_density:
            Zi = density / density.max() if density.max() > 0 else density
            Zi[Zi < 0.05] = np.nan
        else:
            vals = plot_df["_val"].values
            Zi = np.zeros_like(Xi)
            W  = np.zeros_like(Xi)
            for px, py, pv in zip(x, y, vals):
                dist2   = (Xi - px)**2 + (Yi - py)**2
                weights = np.exp(-dist2 / (2 * 0.3**2))
                Zi += weights * pv
                W  += weights
            W[W == 0] = 1
            Zi = Zi / W
            Zi[density < density.max() * 0.04] = np.nan

        im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap_name,
                           vmin=vmin, vmax=vmax, shading="gouraud", zorder=1)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(title_label, fontsize=10, color=TEXT_COLOR)
        cb.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=TEXT_COLOR)

    except Exception as e:
        ax.text(0.5, 0.5, f"KDE failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", color="red")

    ax.scatter(x, y, c="#333333", s=10, alpha=0.35, zorder=6)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 5.0)
    ax.set_xlabel("Plate Side (ft)  ←  Arm Side | Glove Side  →",
                  fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("Plate Height (ft)", fontsize=10, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID_COLOR)

    fig.suptitle(
        f"{title_label} Heatmap\n{batter_name}  ·  {hand_lbl}  ·  n={len(plot_df)}{filter_str}  ·  {date_range}",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.patch.set_facecolor(BG_COLOR)
    return fig

# ── Main hitter page generator ────────────────────────────────────────────────
def generate_hitter_page(batter_df, batter_name, game_date, opponent,
                          filter_count=None, filter_pitch_hand=None, filter_pitch_type=None):
    """
    Generate a one-page hitter card.
    Optional filters applied to heatmaps only (not banner stats):
      filter_count       : e.g. "0-0", "2-strike", "ahead" — None = all counts
      filter_pitch_hand  : "L", "R", or None
      filter_pitch_type  : pitch type string or None
    """
    stats = compute_batter_stats(batter_df)
    bip   = stats.pop("bip_df")

    # Build filtered DataFrame for heatmaps
    heat_df = batter_df.copy()
    filter_labels = []

    if filter_pitch_hand and "PitcherThrows" in heat_df.columns:
        # PitcherThrows is "Left"/"Right" — handle both short and long form
        hand_map = {"L": "Left", "R": "Right", "Left": "Left", "Right": "Right"}
        hand_val = hand_map.get(filter_pitch_hand, filter_pitch_hand)
        heat_df = heat_df[heat_df["PitcherThrows"] == hand_val]
        filter_labels.append(f"vs {'LHP' if filter_pitch_hand=='L' else 'RHP'}")

    if filter_pitch_type and "PitchType" in heat_df.columns:
        heat_df = heat_df[heat_df["PitchType"] == filter_pitch_type]
        filter_labels.append(filter_pitch_type)

    if filter_count and "Balls" in heat_df.columns:
        if filter_count == "2-strike":
            heat_df = heat_df[heat_df["Strikes"] == 2]
            filter_labels.append("2-strike")
        elif filter_count == "ahead":
            heat_df = heat_df[heat_df["Balls"] > heat_df["Strikes"]]
            filter_labels.append("hitter ahead")
        elif filter_count == "behind":
            heat_df = heat_df[heat_df["Strikes"] > heat_df["Balls"]]
            filter_labels.append("hitter behind")
        elif "-" in str(filter_count):
            try:
                b, s = filter_count.split("-")
                heat_df = heat_df[(heat_df["Balls"] == int(b)) & (heat_df["Strikes"] == int(s))]
                filter_labels.append(f"{filter_count} count")
            except Exception:
                pass

    filter_suffix = f" ({', '.join(filter_labels)})" if filter_labels else ""

    # Layout: 5 rows x 3 cols — no spray chart, bigger everything
    # Row 0: title | Row 1: banner | Row 2: EV scatter, EV dist, Zone EV
    # Row 3: Zone xwOBA, Swing rate, Whiff% | Row 4: BB profile, Spray dir, blank
    fig = plt.figure(figsize=(26, 28), facecolor=BG_COLOR)
    gs  = fig.add_gridspec(5, 3,
                            height_ratios=[0.06, 0.10, 1.0, 1.0, 0.55],
                            hspace=0.45, wspace=0.35,
                            left=0.05, right=0.97, top=0.97, bottom=0.03)

    # ── Title bar ──
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(BG_COLOR); ax_title.axis("off")
    ax_title.text(0.01, 0.70, batter_name,
                  transform=ax_title.transAxes, fontsize=36, fontweight="bold",
                  color=TEXT_COLOR, va="center")
    side     = batter_df["BatterSide"].mode()[0] if not batter_df["BatterSide"].empty else "?"
    hand_lbl = "LHB" if side == "Left" else "RHB" if side == "Right" else side
    info     = f"{game_date}  ·  vs {opponent}  ·  {hand_lbl}  ·  {stats['pa']} PA  ·  {stats['bip']} BIP"
    if filter_suffix:
        info += f"  ·  Heatmaps filtered{filter_suffix}"
    ax_title.text(0.01, 0.15, info, transform=ax_title.transAxes,
                  fontsize=15, color=MUTED_TEXT, va="center")

    # ── Stats banner ──
    ax_banner = fig.add_subplot(gs[1, :])
    draw_hitter_stats_banner(ax_banner, stats, batter_name)

    # ── Row 2: EV scatter | EV distribution | Zone EV ──
    ax_scatter = fig.add_subplot(gs[2, 0]); draw_ev_la_scatter(ax_scatter, bip)
    ax_evdist  = fig.add_subplot(gs[2, 1]); draw_ev_distribution(ax_evdist, bip)
    ax_zone_ev = fig.add_subplot(gs[2, 2])
    draw_zone_heatmap(ax_zone_ev, batter_df, "ev", f"Zone EV (Catcher POV){filter_suffix}", heat_df)

    # ── Row 3: Zone xwOBA | Swing rate | Whiff% ──
    ax_zone_xw = fig.add_subplot(gs[3, 0])
    draw_zone_heatmap(ax_zone_xw, batter_df, "xwoba", f"Zone xwOBA (Catcher POV){filter_suffix}", heat_df)
    ax_swing = fig.add_subplot(gs[3, 1])
    draw_zone_heatmap(ax_swing, batter_df, "swing", f"Swing Rate (Catcher POV){filter_suffix}", heat_df)
    ax_whiff = fig.add_subplot(gs[3, 2])
    draw_zone_heatmap(ax_whiff, batter_df, "whiff", f"Whiff% (Catcher POV){filter_suffix}", heat_df)

    # ── Row 4: BB profile | Spray direction | blank ──
    ax_bb   = fig.add_subplot(gs[4, 0]); draw_batted_ball_profile(ax_bb, stats)
    ax_pull = fig.add_subplot(gs[4, 1]); draw_pull_oppo(ax_pull, stats)
    ax_blank = fig.add_subplot(gs[4, 2]); ax_blank.axis("off")

    return fig


# ============================================================================
# ===========  MUNN-STYLE SEASON SUMMARY (matches reference PDF)  =============
# ===========  Defined LAST so generate_season_summary overrides the  ========
# ===========  older version above via `from utils import *`.         ========
# ============================================================================

def _display_name(name):
    """Convert 'Last, First' -> 'First Last' for display."""
    try:
        s = str(name)
        if "," in s:
            last, first = [x.strip() for x in s.split(",", 1)]
            if first:
                return f"{first} {last}"
    except Exception:
        pass
    return str(name)

def _hand_label(df):
    """Return 'LHP' or 'RHP' inferred from release side."""
    try:
        return "LHP" if _infer_hand_from_relside(df) == "L" else "RHP"
    except Exception:
        return ""

# Display-name overrides for the season summary (matches the Munn PDF look)
SS_PITCH_DISPLAY = {
    "Fastball":  "4-Seam",
    "Cutter":    "Cutter",
    "Curveball": "Curveball",
    "Slider":    "Slider",
    "ChangeUp":  "Changeup",
    "Splitter":  "Splitter",
    "Sinker":    "Sinker",
    "Sweeper":   "Sweeper",
    "Knuckleball": "Knuckleball",
    "Other":     "Other",
}

def _ss_is_swing(df):
    return df["PitchCall"].isin(SWING_CALLS)

def _ss_plot_velo_distribution(ax, sm, pitch_order):
    """Ridge-style KDE per pitch type (Munn style)."""
    n = len(pitch_order)
    offset_step = 1.0
    all_velos = sm["RelSpeed"].dropna()
    x_min = max(60, int(all_velos.min()) - 3) if len(all_velos) > 0 else 70
    x_max = min(105, int(all_velos.max()) + 3) if len(all_velos) > 0 else 100

    for i, pt in enumerate(pitch_order):
        velos = sm[sm["PitchType"] == pt]["RelSpeed"].dropna().values
        base_y = (n - 1 - i) * offset_step
        label = SS_PITCH_DISPLAY.get(pt, pt)
        if len(velos) < 2:
            if len(velos):
                ax.plot([velos[0], velos[0]], [base_y, base_y + 0.6],
                        color=pc(pt), lw=1.2)
            ax.text(x_min - 1, base_y + 0.15, label,
                    fontsize=7, ha="right", va="center", color="#333")
            continue
        try:
            kde = gaussian_kde(velos, bw_method=0.35)
        except Exception:
            continue
        xs = np.linspace(x_min, x_max, 300)
        ys = kde(xs); ys = ys / ys.max() * 0.85
        ax.fill_between(xs, base_y, base_y + ys, color=pc(pt), alpha=0.55,
                        edgecolor=pc(pt), linewidth=0.9)
        mean_v = velos.mean()
        idx = np.argmin(np.abs(xs - mean_v))
        ax.plot([mean_v, mean_v], [base_y, base_y + ys[idx]],
                color=pc(pt), lw=0.8, linestyle=":")
        ax.text(x_min - 1, base_y + 0.15, label,
                fontsize=7, ha="right", va="center", color="#333")

    ax.set_xlim(x_min - 4, x_max)
    ax.set_ylim(-0.2, n * offset_step)
    ax.set_xlabel("Velocity (mph)", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Pitch Velocity Distribution", fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.4)

def _ss_plot_breaks(ax, sm, pitch_order):
    """One large dot per pitch type at the avg movement (Munn style)."""
    for pt in pitch_order:
        sub = sm[sm["PitchType"] == pt]
        if len(sub) == 0:
            continue
        ax.scatter(sub["HorzBreak"].mean(), sub["InducedVertBreak"].mean(),
                   color=pc(pt), s=420, alpha=0.92,
                   edgecolors="white", linewidths=1.6, zorder=3,
                   label=SS_PITCH_DISPLAY.get(pt, pt))

    ax.axhline(0, color="#8B7355", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(0, color="#8B7355", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("Horizontal Break (in)", fontsize=8)
    ax.set_ylabel("Induced Vertical Break (in)", fontsize=8)
    ax.set_title("Pitch Breaks", fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(axis="both", labelsize=8, colors="#222")
    ax.grid(True, alpha=0.18, linewidth=0.4)
    ax.set_aspect("equal")
    ax.text(-23, -22, "← Glove Side", fontsize=7, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#aaa", linewidth=0.4))
    ax.text(11, -22, "Arm Side →", fontsize=7, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#aaa", linewidth=0.4))

def _ss_plot_usage(ax, sm, pitch_order):
    """Back-to-back horizontal bars — vs LHH | vs RHH (Munn style)."""
    lhh = sm[sm["BatterSide"] == "Left"]
    rhh = sm[sm["BatterSide"] == "Right"]
    pitches_present = [p for p in pitch_order if (sm["PitchType"] == p).any()]
    y_positions = np.arange(len(pitches_present))[::-1]

    bar_h = 0.7
    for i, pt in enumerate(pitches_present):
        y = y_positions[i]
        color = pc(pt)
        lp = 100 * (lhh["PitchType"] == pt).sum() / len(lhh) if len(lhh) else 0
        rp = 100 * (rhh["PitchType"] == pt).sum() / len(rhh) if len(rhh) else 0
        ax.barh(y, -lp, height=bar_h, color=color, edgecolor="white", linewidth=0.6)
        ax.barh(y,  rp, height=bar_h, color=color, edgecolor="white", linewidth=0.6)
        if lp > 0:
            ax.text(-lp - 2, y, f"{lp:.1f}%", va="center", ha="right",
                    fontsize=7, color="#333")
        if rp > 0:
            ax.text(rp + 2, y, f"{rp:.1f}%", va="center", ha="left",
                    fontsize=7, color="#333")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-0.7, len(pitches_present) + 0.3)
    ax.set_yticks([])
    ax.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%", "25%", "50%", "75%", "100%"],
                       fontsize=7)
    ax.set_xlabel("Usage (%)", fontsize=8)
    ax.set_title("Pitch Usage", fontsize=11, fontweight="bold", pad=18)
    ax.text(-50, len(pitches_present) - 0.15, "vs LHH", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#555")
    ax.text(50, len(pitches_present) - 0.15, "vs RHH", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#555")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.4)

def _ss_build_table(sm, pitch_order):
    rows = []
    total = len(sm)
    for pt in pitch_order:
        sub = sm[sm["PitchType"] == pt]
        if len(sub) == 0:
            continue
        cnt = len(sub)
        iz_full = in_zone(sub)
        oz_full = sub[~iz_full]
        chase_pct = 100 * _ss_is_swing(oz_full).sum() / len(oz_full) if len(oz_full) > 0 else 0.0
        swings_full = _ss_is_swing(sub)
        n_sw = swings_full.sum()
        whiff_pct = 100 * (sub[swings_full]["PitchCall"] == "StrikeSwinging").sum() / n_sw if n_sw > 0 else 0.0
        zone_pct = 100 * iz_full.sum() / cnt if cnt else np.nan
        rows.append({
            "Pitch":    SS_PITCH_DISPLAY.get(pt, pt),
            "Color":    pc(pt),
            "Count":    cnt,
            "Pitch%":   100 * cnt / total,
            "Velocity": sub["RelSpeed"].mean(),
            "Max":      sub["RelSpeed"].max(),
            "iVB":      sub["InducedVertBreak"].mean(),
            "HB":       sub["HorzBreak"].mean(),
            "Spin":     sub["SpinRate"].mean(),
            "HAVAA":    _havaa_mean(sub, pt),
            "vRel":     sub["RelHeight"].mean(),
            "hRel":     sub["RelSide"].mean(),
            "Ext":      sub["Extension"].mean(),
            "Zone%":    zone_pct,
            "Chase%":   chase_pct,
            "Whiff%":   whiff_pct,
        })

    iz_all = in_zone(sm)
    oz_all = sm[~iz_all]
    chase_all = 100 * _ss_is_swing(oz_all).sum() / len(oz_all) if len(oz_all) > 0 else 0.0
    swings_all = _ss_is_swing(sm)
    n_sw_all = swings_all.sum()
    whiff_all = 100 * (sm[swings_all]["PitchCall"] == "StrikeSwinging").sum() / n_sw_all if n_sw_all > 0 else 0.0
    rows.append({
        "Pitch": "All", "Color": "#FFFFFF", "Count": total, "Pitch%": 100.0,
        "Velocity": np.nan, "Max": np.nan, "iVB": np.nan, "HB": np.nan,
        "Spin": np.nan, "HAVAA": np.nan, "vRel": np.nan, "hRel": np.nan,
        "Ext": sm["Extension"].mean(),
        "Zone%": 100 * iz_all.sum() / total if total else np.nan,
        "Chase%": chase_all, "Whiff%": whiff_all,
    })
    return pd.DataFrame(rows)

def _ss_draw_table(ax, df):
    ax.axis("off")
    cols = ["Pitch Name", "Count", "Pitch%", "Velocity", "Max", "iVB", "HB", "Spin",
            "HAVAA", "vRel", "hRel", "Ext.", "Zone%", "Chase%", "Whiff%"]
    df_keys = ["Pitch", "Count", "Pitch%", "Velocity", "Max", "iVB", "HB", "Spin",
               "HAVAA", "vRel", "hRel", "Ext", "Zone%", "Chase%", "Whiff%"]
    fmts = {
        "Count":    lambda v: f"{int(v)}",
        "Pitch%":   lambda v: f"{v:.1f}%",
        "Velocity": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Max":      lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "iVB":      lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "HB":       lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Spin":     lambda v: f"{v:.0f}" if pd.notna(v) else "—",
        "HAVAA":    lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "vRel":     lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "hRel":     lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Ext":      lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Zone%":    lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
        "Chase%":   lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
        "Whiff%":   lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    }

    cell_text, cell_colors = [], []
    for _, row in df.iterrows():
        line, clrs = [], []
        for c, kk in zip(cols, df_keys):
            if kk == "Pitch":
                line.append(row["Pitch"])
                clrs.append(row["Color"] if row["Pitch"] != "All" else "#FFFFFF")
            else:
                v = row[kk]
                line.append(fmts[kk](v) if kk in fmts else str(v))
                clrs.append("#FFFFFF")
        cell_text.append(line)
        cell_colors.append(clrs)

    header_colors = ["#2C2C2C"] * len(cols)
    table = ax.table(cellText=cell_text, colLabels=cols,
                     cellColours=cell_colors, colColours=header_colors,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)

    for j in range(len(cols)):
        cell = table[(0, j)]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#1a1a1a"); cell.set_linewidth(0.5)

    n_rows = len(df)
    for i in range(1, n_rows + 1):
        for j in range(len(cols)):
            cell = table[(i, j)]
            cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.4)
            if j == 0:
                is_all = (df.iloc[i - 1]["Pitch"] == "All")
                cell.set_text_props(fontweight="bold",
                                    color="#1a1a1a" if is_all else "white")

    widths = [1.5, 0.65, 0.75, 0.95, 0.7, 0.65, 0.65, 0.75, 0.65, 0.65, 0.65,
              0.65, 0.8, 0.85, 0.85]
    total_w = sum(widths)
    for j, w in enumerate(widths):
        for i in range(n_rows + 1):
            table[(i, j)].set_width(w / total_w)

def generate_season_summary(pitcher_name, outings, date_from, date_to):
    """
    One-page Munn-style season pitching summary:
    header / velocity ridges / avg-movement breaks / usage bars / pitch table.
    """
    all_dfs = [p_df.copy() for p_df, gdate, opp in outings]
    if not all_dfs:
        return None
    sm = pd.concat(all_dfs, ignore_index=True)
    if len(sm) == 0:
        return None

    # Order pitch types fastest -> slowest (matches the reference layout)
    pts_present = [pt for pt in sm["PitchType"].dropna().unique() if pt]
    pitch_order = sorted(
        pts_present,
        key=lambda x: sm.loc[sm["PitchType"] == x, "RelSpeed"].median()
        if not sm.loc[sm["PitchType"] == x, "RelSpeed"].dropna().empty else 0,
        reverse=True,
    )

    hand = _hand_label(sm)
    display = _display_name(pitcher_name)
    table_df = _ss_build_table(sm, pitch_order)

    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    gs = GridSpec(nrows=3, ncols=3, figure=fig,
                  height_ratios=[0.9, 3.0, 2.5],
                  width_ratios=[1, 1, 1],
                  left=0.045, right=0.965, top=0.96, bottom=0.045,
                  hspace=0.45, wspace=0.30)

    ax_hdr = fig.add_subplot(gs[0, :]); ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.92, display, ha="center", va="top",
                fontsize=22, fontweight="bold", color="#1a1a1a")
    sub_bits = [b for b in [hand, f"{len(outings)} outings"] if b]
    ax_hdr.text(0.5, 0.58, "  •  ".join(sub_bits), ha="center", va="top",
                fontsize=10, color="#555")
    ax_hdr.text(0.5, 0.38, "Season Pitching Summary", ha="center", va="top",
                fontsize=13, fontweight="bold", color="#1a1a1a")
    try:
        season_label = f"{date_from:%b %d} – {date_to:%b %d, %Y}"
    except Exception:
        season_label = f"{date_from} – {date_to}"
    ax_hdr.text(0.5, 0.18, season_label, ha="center", va="top",
                fontsize=9, style="italic", color="#777")

    _ss_plot_velo_distribution(fig.add_subplot(gs[1, 0]), sm, pitch_order)
    _ss_plot_breaks(fig.add_subplot(gs[1, 1]), sm, pitch_order)
    _ss_plot_usage(fig.add_subplot(gs[1, 2]), sm, pitch_order)
    _ss_draw_table(fig.add_subplot(gs[2, :]), table_df)

    return fig


# ===========================================================================