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
warnings.filterwarnings("ignore")

# ===========================================================================
# CONSTANTS
# ===========================================================================
STRIKE_CALLS = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
SWING_CALLS  = {"StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
PITCH_COLORS = {
    "Fastball": "#D32F2F", "FourSeamFastBall": "#D32F2F",
    "Sinker": "#E65100", "TwoSeamFastBall": "#E65100",
    "Cutter": "#B8A000", "Slider": "#00897B", "Curveball": "#1565C0",
    "ChangeUp": "#F9A825", "Changeup": "#F9A825",
    "Splitter": "#00796B", "Sweeper": "#7B1FA2", "Other": "#888888",
}
BG_COLOR     = "#FFFFFF"
PANEL_COLOR  = "#F7F8FA"
GRID_COLOR   = "#D5D8DC"
TEXT_COLOR   = "#1A1A2E"
ACCENT_COLOR = "#1565C0"
MUTED_TEXT   = "#6B7280"
AUTO_CORRECT_PITCHES = True
MIN_CLUSTER_SIZE     = 3

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

def resolve_pt(row):
    t = row.get("TaggedPitchType", ""); a = row.get("AutoPitchType", "")
    if t and t not in ("", "Undefined"): return t
    if a and a not in ("", "Undefined"): return a
    return "Other"

def calc_xwoba(ev, la):
    if pd.isna(ev) or pd.isna(la): return np.nan
    if ev < 50: return 0.05
    if la > 60: return 0.05
    if la > 50: return 0.05 + max(0, (ev - 90)) * 0.01
    if la < -10: return 0.05
    if la < 10:
        if ev >= 100: return 0.50
        if ev >= 90:  return 0.30
        if ev >= 80:  return 0.20
        return 0.10
    if ev >= 105:   base = 1.8
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

def auto_correct_pitch_types(pitcher_df):
    if not AUTO_CORRECT_PITCHES: return pitcher_df, 0
    df = pitcher_df.copy(); corrections = 0
    features = ["RelSpeed", "InducedVertBreak", "HorzBreak"]
    for pname in df["Pitcher"].unique():
        pmask = df["Pitcher"] == pname; pdf = df[pmask].copy()
        valid = pdf[features].notna().all(axis=1)
        if valid.sum() < 5: continue
        centroids, stds = {}, {}
        for pt in pdf.loc[valid, "PitchType"].unique():
            if pt in ("Other", "Undefined", ""): continue
            ptmask = (pdf["PitchType"] == pt) & valid
            if ptmask.sum() >= MIN_CLUSTER_SIZE:
                centroids[pt] = pdf.loc[ptmask, features].mean().values
                stds[pt] = pdf.loc[ptmask, features].std().values
                stds[pt] = np.where(stds[pt] < 0.5, 2.0, stds[pt])
        if len(centroids) < 2: continue
        for idx in pdf[valid].index:
            row_vals = pdf.loc[idx, features].values.astype(float)
            tagged = pdf.loc[idx, "PitchType"]
            if tagged not in centroids: continue
            own_max_sd = (np.abs(row_vals - centroids[tagged]) / stds[tagged]).max()
            if own_max_sd > 2.5:
                best_type, best_dist = tagged, own_max_sd
                for other_pt, other_cent in centroids.items():
                    if other_pt == tagged: continue
                    other_max_sd = (np.abs(row_vals - other_cent) / stds[other_pt]).max()
                    if other_max_sd < best_dist and other_max_sd < 1.5:
                        best_dist = other_max_sd; best_type = other_pt
                if best_type != tagged:
                    df.loc[idx, "PitchType"] = best_type; corrections += 1
    return df, corrections

def fmt(s, fn="mean", d=1):
    v = s.dropna()
    if v.empty: return "—"
    r = v.mean() if fn == "mean" else v.max()
    return f"{r:.{d}f}"

# ===========================================================================
# UI / STYLING FUNCTIONS
# ===========================================================================

def inject_app_styles():
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at top left, rgba(21, 101, 192, 0.08), transparent 28%),
          linear-gradient(180deg, #F8FBFF 0%, #FFFFFF 16%, #FFFFFF 100%);
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F6F8FB 0%, #FDFDFE 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
      }
      .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.25rem;
      }

      /* ── Hero banner ── */
      .tm-banner {
        padding: 1.3rem 1.4rem;
        border-radius: 24px;
        border: 1px solid rgba(37, 99, 235, 0.16);
        background: linear-gradient(135deg, rgba(21,101,192,0.12) 0%, rgba(255,255,255,0.98) 62%);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
      }
      .tm-banner-top {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #2563EB;
        font-weight: 700;
        margin-bottom: 0.35rem;
      }
      .tm-banner h2 {
        margin: 0;
        color: #0F172A;
        font-size: 2rem;
        line-height: 1.1;
      }
      .tm-banner p {
        margin: 0.55rem 0 0 0;
        color: #475569;
        line-height: 1.55;
        max-width: 60rem;
      }

      /* ── Chips ── */
      .tm-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.9rem;
      }
      .tm-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.34rem 0.72rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #334155;
        font-size: 0.84rem;
        font-weight: 500;
      }

      /* ── Section titles ── */
      .tm-section {
        margin: 1rem 0 0.25rem 0;
      }
      .tm-section h3 {
        margin: 0;
        color: #0F172A;
        font-size: 1.18rem;
      }
      .tm-section p {
        margin: 0.2rem 0 0 0;
        color: #64748B;
        font-size: 0.93rem;
      }

      /* ── Stat cards ── */
      .tm-card {
        height: 100%;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 20px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
      }
      .tm-card-label {
        margin-bottom: 0.35rem;
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .tm-card-value {
        color: #0F172A;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.15;
      }
      .tm-card-help {
        margin-top: 0.35rem;
        color: #64748B;
        font-size: 0.88rem;
        line-height: 1.4;
      }

      /* ── Info panels ── */
      .tm-panel {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
      }
      .tm-panel h4 {
        margin: 0 0 0.55rem 0;
        color: #0F172A;
        font-size: 1rem;
      }
      .tm-panel ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #475569;
      }
      .tm-panel li {
        margin: 0.22rem 0;
      }

      /* ── Tabs ── */
      .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
      }
      .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 999px;
        padding: 0.45rem 0.95rem;
      }
      .stTabs [aria-selected="true"] {
        background: rgba(37, 99, 235, 0.10) !important;
        border-color: rgba(37, 99, 235, 0.35) !important;
        color: #1D4ED8 !important;
      }

      /* ── Misc ── */
      div[data-testid="stExpander"] {
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 18px;
        background: rgba(255,255,255,0.88);
      }
      div[data-testid="stDataFrame"] {
        border-radius: 16px;
      }
      .stButton > button,
      .stDownloadButton > button {
        border-radius: 12px;
      }
    </style>
    """, unsafe_allow_html=True)


def render_page_intro(title, description, chips=None, eyebrow="TrackMan reports"):
    chip_html = ""
    if chips:
        chip_html = '<div class="tm-chip-row">' + "".join(
            f'<div class="tm-chip">{c}</div>' for c in chips if c
        ) + "</div>"
    st.markdown(
        f"""
        <div class="tm-banner">
          <div class="tm-banner-top">{eyebrow}</div>
          <h2>{title}</h2>
          <p>{description}</p>
          {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label, value, help_text=None):
    help_html = f'<div class="tm-card-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="tm-card">
          <div class="tm-card-label">{label}</div>
          <div class="tm-card-value">{value}</div>
          {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_panel(title, lines):
    """Bullet-point info box. Pass a list of strings as `lines`."""
    items = "".join(f"<li>{line}</li>" for line in lines if line)
    st.markdown(
        f"""
        <div class="tm-panel">
          <h4>{title}</h4>
          <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, caption=None):
    caption_html = f"<p>{caption}</p>" if caption else ""
    st.markdown(
        f"""
        <div class="tm-section">
          <h3>{title}</h3>
          {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_number(value, digits=1, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}{suffix}"


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
            x = -grp["PlateLocSide"]; y = grp["PlateLocHeight"]
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
        rs_std = rs_std if (rs_std == rs_std) else 0.3
        rh_std = rh_std if (rh_std == rh_std) else 0.3
        pad = max(rs_std, rh_std, 0.3) * 4 + 0.3
        ax.set_xlim(rs_c - pad, rs_c + pad); ax.set_ylim(rh_c - pad, rh_c + pad)
    ax.set_aspect("equal")

# ===========================================================================
# DATA LOADING — reads from data/by_date/ partitioned parquets
# ===========================================================================
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(), "data")
BY_DATE_DIR = os.path.join(DATA_DIR, "by_date")
INDEX_PATH  = os.path.join(DATA_DIR, "index.parquet")

PT_NORMALIZE = {"FourSeamFastBall": "Fastball", "TwoSeamFastBall": "Sinker", "Changeup": "ChangeUp"}

def _prep_df(df):
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str).replace("nan", "")
    if "PitchType" in df.columns:
        df["PitchType"] = df["PitchType"].replace(PT_NORMALIZE)
    return df

@st.cache_data(ttl=60)
def load_index():
    if os.path.exists(INDEX_PATH):
        df = pd.read_parquet(INDEX_PATH)
        df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
        return df
    legacy = os.path.join(DATA_DIR, "pitches.parquet")
    if os.path.exists(legacy):
        df = pd.read_parquet(legacy, columns=["GameDate", "HomeTeam", "AwayTeam", "SessionID"])
        df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
        return df.drop_duplicates(subset=["GameDate", "HomeTeam", "AwayTeam"])
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_team_data(team_name, date_from, date_to):
    idx = load_index()
    if idx.empty:
        return None
    team_mask = (
        (idx["HomeTeam"] == team_name) | (idx["AwayTeam"] == team_name)
    ) & (idx["GameDate"] >= date_from) & (idx["GameDate"] <= date_to)
    team_dates = sorted(idx[team_mask]["GameDate"].dropna().unique())
    if not team_dates:
        return pd.DataFrame()
    dfs = []
    if os.path.exists(BY_DATE_DIR):
        for gdate in team_dates:
            fpath = os.path.join(BY_DATE_DIR, f"{gdate}.parquet")
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                mask = (df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)
                dfs.append(df[mask])
    else:
        legacy = os.path.join(DATA_DIR, "pitches.parquet")
        if os.path.exists(legacy):
            df = pd.read_parquet(legacy)
            df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
            mask = (
                ((df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)) &
                (df["GameDate"] >= date_from) & (df["GameDate"] <= date_to)
            )
            dfs.append(df[mask])
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    return _prep_df(combined)

@st.cache_data(ttl=60)
def load_all_pitches():
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
    mapping = {}
    for _, row in df[["HomeTeam", "AwayTeam", "PitcherTeam"]].dropna().iterrows():
        home = row["HomeTeam"]; away = row["AwayTeam"]; code = row["PitcherTeam"]
        code_prefix = code[:3].upper()
        for team in [home, away]:
            if team and team[:3].upper() == code_prefix:
                mapping[team] = code
    return mapping

def get_teams(df):
    teams = set()
    for col in ["HomeTeam", "AwayTeam"]:
        if col in df.columns:
            teams.update(df[col].dropna().unique())
    return sorted(t for t in teams if t)

def get_team_pitches(df, team_name, date_from, date_to):
    """Normalizes GameDate type to prevent datetime/date comparison errors."""
    df = df.copy()
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
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


def draw_count_tree(ax, p, pts):
    """Pie-chart count tree showing pitch mix at each count."""
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.set_title("Pitch Usage by Count", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=6)

    # Use inset axes for each count so pies are always circular
    # Map (balls, strikes) to normalized figure position (x, y) as fractions of axes
    positions = {
        (0,0): (0.50, 0.90),
        (1,0): (0.64, 0.76), (0,1): (0.36, 0.76),
        (2,0): (0.78, 0.62), (1,1): (0.50, 0.62), (0,2): (0.22, 0.62),
        (3,0): (0.90, 0.48), (2,1): (0.64, 0.48), (1,2): (0.36, 0.48),
        (3,1): (0.77, 0.32), (2,2): (0.50, 0.32),
        (3,2): (0.50, 0.14),
    }
    edges = [
        ((0,0),(1,0)),((0,0),(0,1)),
        ((1,0),(2,0)),((1,0),(1,1)),
        ((0,1),(1,1)),((0,1),(0,2)),
        ((2,0),(3,0)),((2,0),(2,1)),
        ((1,1),(2,1)),((1,1),(1,2)),
        ((0,2),(1,2)),
        ((3,0),(3,1)),((2,1),(3,1)),
        ((2,1),(2,2)),((1,2),(2,2)),
        ((3,1),(3,2)),((2,2),(3,2)),
    ]
    # Draw connecting lines in axes coordinates
    for (b1,s1),(b2,s2) in edges:
        x1,y1 = positions[(b1,s1)]; x2,y2 = positions[(b2,s2)]
        ax.plot([x1,x2],[y1,y2], color=GRID_COLOR, lw=0.8,
                transform=ax.transAxes, zorder=1, clip_on=False)

    pie_size = 0.14  # fraction of axes width/height for each pie inset

    for (balls, strikes), (cx, cy) in positions.items():
        sub = p[(p["Balls"] == balls) & (p["Strikes"] == strikes)]
        n = len(sub)

        # Create inset axes centered at (cx, cy) in axes fraction coords
        ins = ax.inset_axes([cx - pie_size/2, cy - pie_size/2, pie_size, pie_size])
        ins.set_aspect("equal")
        ins.axis("off")

        if n == 0:
            ins.pie([1], colors=["#E8E8E8"], startangle=90)
        else:
            counts = sub["PitchType"].value_counts()
            sizes  = [counts.get(pt, 0) for pt in pts]
            colors = [pc(pt) for pt in pts]
            filtered = [(s, c) for s, c in zip(sizes, colors) if s > 0]
            if filtered:
                szf, clf = zip(*filtered)
                ins.pie(szf, colors=clf, startangle=90,
                        wedgeprops={"linewidth": 0.4, "edgecolor": "white"})

        # Count label above
        ax.text(cx, cy + pie_size/2 + 0.012, f"{balls}-{strikes}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=TEXT_COLOR, fontweight="bold")
        # n label below
        if n > 0:
            ax.text(cx, cy - pie_size/2 - 0.012, f"n={n}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.5, color=MUTED_TEXT)

    # Legend
    for pt in pts:
        ax.scatter([], [], c=pc(pt), s=80, label=pt)
    ax.legend(loc="lower left", fontsize=11, frameon=False,
              labelcolor=TEXT_COLOR, ncol=min(len(pts), 6),
              bbox_to_anchor=(0.0, 0.0))

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

    fig = plt.figure(figsize=(26, 22), facecolor=BG_COLOR)
    gs = GridSpec(5, 4, figure=fig,
                  height_ratios=[.045, .015, .28, .22, .435],
                  width_ratios=[1, 1, 1, 0.65],
                  hspace=.22, wspace=.18,
                  top=0.96, bottom=0.02, left=0.03, right=0.97)

    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    ax.text(.5, .78, pname.upper(), ha="center", va="center", fontsize=34,
            fontweight="bold", color=TEXT_COLOR, family="monospace")
    ax.text(.5, .22, f"{gdate:%B %d, %Y}   ·   vs {opp}",
            ha="center", va="center", fontsize=16, color=ACCENT_COLOR, family="monospace")

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
    ax_m = fig.add_subplot(gs[2, 2]); draw_mov(ax_m, p, pts)
    ax_rp = fig.add_subplot(gs[2, 3]); draw_release(ax_rp, p, pts)

    ax_tree = fig.add_subplot(gs[3, :]); draw_count_tree(ax_tree, p, pts)
    ax_t = fig.add_subplot(gs[4, :]); ax_t.set_facecolor(BG_COLOR); ax_t.axis("off")
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
        trows.append([pt, n, f"{n / N * 100:.1f}%",
                      fmt(s["RelSpeed"]), fmt(s["RelSpeed"], "max"),
                      fmt(s["SpinRate"], d=0),
                      fmt(s["InducedVertBreak"]), fmt(s["HorzBreak"]),
                      fmt(s["Extension"]), fmt(s["RelHeight"]), fmt(s["RelSide"]),
                      fmt(s["VertApprAngle"]),
                      xwoba_str, zone_str, whiff_str, chase_str, iz_whiff_str])

        data_row = ri + 1
        if avg_velo_val is not None:
            grade_cells[(data_row, 2)] = (pt, "velo", avg_velo_val, True)
        if xwoba_val is not None:
            grade_cells[(data_row, 11)] = (pt, "xwoba", xwoba_val, False)
        if zone_val is not None:
            grade_cells[(data_row, 12)] = (pt, "zone_pct", zone_val, True)
        if whiff_val is not None:
            grade_cells[(data_row, 13)] = (pt, "whiff_pct", whiff_val, True)
        if chase_val is not None:
            grade_cells[(data_row, 14)] = (pt, "chase_pct", chase_val, True)

    all_sw_ct = sw.sum()
    all_whiff = f"{wh.sum() / all_sw_ct * 100:.1f}%" if all_sw_ct else "0%"
    all_xw = p["xwOBA"].dropna()
    all_xwoba = f"{all_xw.mean():.3f}" if not all_xw.empty else "—"
    trows.append(["All", N, "100%", "—", "—", "—", "—", "—",
                  fmt(p["Extension"]), "—", "—", "—",
                  all_xwoba, f"{zpct}%", all_whiff, f"{cpct}%", f"{izwp}%"])

    cols = ["Count", "Usage%", "Avg\nVelo", "Max\nVelo", "Avg\nSpin",
            "IVB", "HB", "Ext", "RelH", "RelS", "VAA",
            "xwOBA", "Zone%", "Whiff%", "Chase%", "IZ\nWhiff%"]

    tbl = ax_t.table(cellText=[r[1:] for r in trows], rowLabels=[r[0] for r in trows],
                     colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1, 3.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1E1E2E")
            cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=15)
        elif col == -1:
            pitch_name = cell.get_text().get_text()
            if pitch_name == "All":
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontweight="bold", color=TEXT_COLOR, fontfamily="monospace", fontsize=16)
            else:
                cell.set_facecolor(pc(pitch_name))
                cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=16)
        else:
            graded = False
            if (row, col) in grade_cells and row <= len(pts):
                pt_name, stat_name, raw_val, higher_better = grade_cells[(row, col)]
                gc = grade_color(pt_name, stat_name, raw_val, higher_better)
                if gc is not None:
                    cell.set_facecolor(gc)
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontweight="bold", fontsize=15)
                    graded = True
            if not graded:
                if row == len(trows):
                    cell.set_facecolor("#E8E8E8")
                    cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontfamily="monospace", fontsize=15)
                elif row % 2 == 0:
                    cell.set_facecolor("#F4F6F9")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontsize=15)
                else:
                    cell.set_facecolor("#FFFFFF")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontsize=15)

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

    fig = plt.figure(figsize=(26, 24), facecolor=BG_COLOR)
    gs = GridSpec(5, 3, figure=fig,
                  height_ratios=[.045, .03, .24, .20, .485],
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
    ax.text(.5, .2, f"2026 Season   ·   {date_from} to {date_to}   ·   {len(outings)} outing(s)",
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

    ax_tree = fig.add_subplot(gs[3, :]); draw_count_tree(ax_tree, p, pts)
    ax_t = fig.add_subplot(gs[4, :]); ax_t.set_facecolor(BG_COLOR); ax_t.axis("off")
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
        trows.append([pt, n, f"{n / N * 100:.1f}%",
                      fmt(s["RelSpeed"]), fmt(s["RelSpeed"], "max"),
                      fmt(s["SpinRate"], d=0),
                      fmt(s["InducedVertBreak"]), fmt(s["HorzBreak"]),
                      fmt(s["Extension"]), fmt(s["RelHeight"]), fmt(s["RelSide"]),
                      fmt(s["VertApprAngle"]),
                      xwoba_str, zone_str, whiff_str, chase_str, iz_whiff_str])

        data_row = ri + 1
        if avg_velo_val is not None:
            grade_cells[(data_row, 2)] = (pt, "velo", avg_velo_val, True)
        if xwoba_val is not None:
            grade_cells[(data_row, 11)] = (pt, "xwoba", xwoba_val, False)
        if zone_val is not None:
            grade_cells[(data_row, 12)] = (pt, "zone_pct", zone_val, True)
        if whiff_val is not None:
            grade_cells[(data_row, 13)] = (pt, "whiff_pct", whiff_val, True)
        if chase_val is not None:
            grade_cells[(data_row, 14)] = (pt, "chase_pct", chase_val, True)

    all_sw_ct = sw.sum()
    all_whiff = f"{wh.sum() / all_sw_ct * 100:.1f}%" if all_sw_ct else "0%"
    all_xw = p["xwOBA"].dropna()
    all_xwoba = f"{all_xw.mean():.3f}" if not all_xw.empty else "—"
    trows.append(["All", N, "100%", "—", "—", "—", "—", "—",
                  fmt(p["Extension"]), "—", "—", "—",
                  all_xwoba, f"{zpct}%", all_whiff, f"{cpct}%", f"{izwp}%"])

    cols = ["Count", "Usage%", "Avg\nVelo", "Max\nVelo", "Avg\nSpin",
            "IVB", "HB", "Ext", "RelH", "RelS", "VAA",
            "xwOBA", "Zone%", "Whiff%", "Chase%", "IZ\nWhiff%"]
    tbl = ax_t.table(cellText=[r[1:] for r in trows], rowLabels=[r[0] for r in trows],
                     colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1, 3.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1E1E2E")
            cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=15)
        elif col == -1:
            pitch_name = cell.get_text().get_text()
            if pitch_name == "All":
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontweight="bold", color=TEXT_COLOR, fontfamily="monospace", fontsize=16)
            else:
                cell.set_facecolor(pc(pitch_name))
                cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=16)
        else:
            graded = False
            if (row, col) in grade_cells and row <= len(pts):
                pt_name, stat_name, raw_val, higher_better = grade_cells[(row, col)]
                gc = grade_color(pt_name, stat_name, raw_val, higher_better)
                if gc is not None:
                    cell.set_facecolor(gc)
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontweight="bold", fontsize=15)
                    graded = True
            if not graded:
                if row == len(trows):
                    cell.set_facecolor("#F0F0F0")
                    cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontfamily="monospace", fontsize=15)
                elif row % 2 == 0:
                    cell.set_facecolor("#F7F8FA")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontsize=15)
                else:
                    cell.set_facecolor("#FFFFFF")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontsize=15)

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

    for idx, (side, label) in enumerate([("Left", "vs LHB"), ("Right", "vs RHB")]):
        ax = axes[idx]
        ax.set_facecolor(PANEL_COLOR)
        side_data = sub[sub["BatterSide"] == side]

        ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=False, ec="black", lw=1.5, zorder=10))
        ax.add_patch(Polygon([(-.708, .55), (.708, .55), (.708, .35), (0, .15), (-.708, .35)],
                             closed=True, fc="#CCCCCC", ec="black", lw=.5, alpha=0.5, zorder=10))

        if len(side_data) >= 5:
            x = -side_data["PlateLocSide"].values
            y = side_data["PlateLocHeight"].values
            xi = np.linspace(-2.5, 2.5, 80)
            yi = np.linspace(-0.5, 5.0, 80)
            Xi, Yi = np.meshgrid(xi, yi)
            try:
                positions = np.vstack([x, y])
                kde = gaussian_kde(positions, bw_method=0.4)
                density = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                if is_density_only:
                    Zi = density / density.max() if density.max() > 0 else density
                    density_thresh = 0.05
                    Zi[Zi < density_thresh] = np.nan
                else:
                    vals = side_data["_val"].values
                    Zi = np.zeros_like(Xi)
                    for px, py, pv in zip(x, y, vals):
                        dist2 = (Xi - px) ** 2 + (Yi - py) ** 2
                        weights = np.exp(-dist2 / (2 * 0.3 ** 2))
                        Zi += weights * pv
                    weight_sum = np.zeros_like(Xi)
                    for px, py in zip(x, y):
                        dist2 = (Xi - px) ** 2 + (Yi - py) ** 2
                        weight_sum += np.exp(-dist2 / (2 * 0.3 ** 2))
                    weight_sum[weight_sum == 0] = 1
                    Zi = Zi / weight_sum
                    density_thresh = density.max() * 0.05
                    Zi[density < density_thresh] = np.nan
                im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap_name, vmin=vmin, vmax=vmax,
                                   shading="gouraud", zorder=1)
            except:
                pass
            ax.scatter(x, y, c="black", s=8, alpha=0.5, zorder=6)

        n_side = len(side_data)
        ax.set_title(f"{pitch_type} {label} ({n_side})", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=8)
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

def estimate_spray_angle(row):
    loc = row.get("PlateLocSide", np.nan)
    side = str(row.get("BatterSide", "Right"))
    if pd.isna(loc): return np.nan
    if side == "Left":
        angle = loc * 30
    else:
        angle = -loc * 30
    return float(np.clip(angle, -45, 45))

def spray_direction(angle):
    if pd.isna(angle): return "Unknown"
    if angle < -15:  return "Pull"
    if angle > 15:   return "Oppo"
    return "Center"

def get_team_batting(df, team_name, date_from, date_to):
    """Normalizes GameDate type to prevent datetime/date comparison errors."""
    df = df.copy()
    df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce").dt.date
    date_mask = (df["GameDate"] >= date_from) & (df["GameDate"] <= date_to)
    mask = date_mask & (
        ((df["HomeTeam"] == team_name) & (df["TopBottom"] == "Bottom")) |
        ((df["AwayTeam"] == team_name) & (df["TopBottom"] == "Top"))
    )
    return df[mask].copy()

def get_batters(df):
    return sorted(df["Batter"].dropna().unique().tolist())

def compute_batter_stats(df):
    SWING_CALLS_LOCAL = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay"]
    bip = df[df["PitchCall"] == "InPlay"].copy()
    bip["xwOBA_val"] = bip.apply(lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"]), axis=1)
    bip["spray_angle"] = bip.apply(estimate_spray_angle, axis=1)

    ev   = bip["ExitSpeed"].dropna()
    la   = bip["LaunchAngle"].dropna()
    xw   = bip["xwOBA_val"].dropna()

    swings     = df["PitchCall"].isin(SWING_CALLS_LOCAL).sum()
    whiffs     = (df["PitchCall"] == "StrikeSwinging").sum()
    iz_mask    = in_zone(df)
    ooz        = df[~iz_mask]
    ooz_swings = ooz["PitchCall"].isin(SWING_CALLS_LOCAL).sum()
    iz_swings  = df[iz_mask]["PitchCall"].isin(SWING_CALLS_LOCAL).sum()
    iz_total   = iz_mask.sum()

    return {
        "avg_ev":         ev.mean()                    if not ev.empty else np.nan,
        "max_ev":         ev.max()                     if not ev.empty else np.nan,
        "ev90":           float(np.percentile(ev, 90)) if len(ev) >= 5 else np.nan,
        "barrel_pct":     float(((bip["ExitSpeed"] >= 98) & (bip["LaunchAngle"].between(26,30))).sum()
                                / len(bip) * 100)      if not bip.empty else np.nan,
        "sweet_spot_pct": float(((la >= 8) & (la <= 32)).sum() / len(la) * 100)
                                if not la.empty else np.nan,
        "avg_la":         la.mean()                    if not la.empty else np.nan,
        "gb_pct":         float((la < 10).sum() / len(la) * 100) if not la.empty else np.nan,
        "ld_pct":         float(((la >= 10) & (la <= 25)).sum() / len(la) * 100) if not la.empty else np.nan,
        "fb_pct":         float((la > 25).sum() / len(la) * 100) if not la.empty else np.nan,
        "whiff_pct":      float(whiffs / swings * 100)    if swings > 0 else np.nan,
        "chase_pct":      float(ooz_swings / len(ooz) * 100) if len(ooz) > 0 else np.nan,
        "zone_sw_pct":    float(iz_swings / iz_total * 100)  if iz_total > 0 else np.nan,
        "contact_pct":    float((swings - whiffs) / swings * 100) if swings > 0 else np.nan,
        "xwoba":          xw.mean()                    if not xw.empty else np.nan,
        "hard_hit_pct":   float((ev >= 95).sum() / len(ev) * 100) if not ev.empty else np.nan,
        "pa":             df.groupby(["GameDate","Inning","PAofInning"]).ngroups,
        "bip":            len(bip),
        "pull_pct":       float((bip["spray_angle"] < -15).sum() / len(bip) * 100) if not bip.empty else np.nan,
        "center_pct":     float((bip["spray_angle"].between(-15,15)).sum() / len(bip) * 100) if not bip.empty else np.nan,
        "oppo_pct":       float((bip["spray_angle"] > 15).sum() / len(bip) * 100) if not bip.empty else np.nan,
        "bip_df":         bip,
    }

def _hax(ax):
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values(): sp.set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(False)

def draw_ev_la_scatter(ax, bip):
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
    SWING_CALLS_LOCAL = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay"]
    _hax(ax)
    use_df = filter_df if filter_df is not None else df
    if use_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=10)
        return

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
                sw = cell_all["PitchCall"].isin(SWING_CALLS_LOCAL).sum()
                wh = (cell_all["PitchCall"] == "StrikeSwinging").sum()
                if sw >= 1: grid[i, j] = wh / sw * 100
            elif stat == "swing":
                total = len(cell_all)
                sw    = cell_all["PitchCall"].isin(SWING_CALLS_LOCAL).sum()
                if total >= 1: grid[i, j] = sw / total * 100

    if stat == "ev":
        vmin, vmax, cmap = 65, 105, "RdYlGn_r"
    elif stat == "xwoba":
        vmin, vmax, cmap = 0.0, 1.2, "RdYlGn_r"
    elif stat == "whiff":
        vmin, vmax, cmap = 0, 60, "RdYlGn_r"
    else:
        vmin, vmax, cmap = 0, 100, "RdYlGn"

    im = ax.imshow(grid, extent=[-1.5, 1.5, 1.0, 4.0], origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", alpha=0.88)

    for i in range(6):
        for j in range(6):
            v = grid[i, j]
            if not np.isnan(v):
                fmt_str = f"{v:.0f}" if stat in ("ev", "whiff", "swing") else f"{v:.2f}"
                norm_v = (v - vmin) / max(vmax - vmin, 1)
                txt_col = "black" if 0.25 < norm_v < 0.75 else "white"
                ax.text(x_bins[j] + 0.25, y_bins[i] + 0.25, fmt_str,
                        ha="center", va="center", fontsize=11,
                        color=txt_col, fontweight="bold")

    ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2.0,
                                lw=3.0, ec="#111111", fc="none", zorder=5))
    for yline in [2.167, 2.833]:
        ax.plot([-0.83, 0.83], [yline, yline],
                color="#111111", lw=1.0, ls="--", alpha=0.6, zorder=4)
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
    _hax(ax)
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT, fontsize=8); return
    draw_zone_heatmap(ax, df, stat="swing", title="Swing Rate by Zone")

def draw_spray_chart(ax, bip):
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

    r_out = 320
    for sign in [-1, 1]:
        fx = sign * r_out * np.sin(np.radians(45))
        fy = r_out * np.cos(np.radians(45))
        ax.plot([0, fx], [0, fy], color="#888888", lw=1.0, zorder=1)

    theta = np.linspace(-np.radians(45), np.radians(45), 100)
    ax.plot(r_out * np.sin(theta), r_out * np.cos(theta), color="#888888", lw=1.2, zorder=1)
    r_in = 95
    ax.plot(r_in * np.sin(theta), r_in * np.cos(theta),
            color="#AAAAAA", lw=0.7, ls="--", zorder=1)

    base_coords = [(-63, 63), (0, 126), (63, 63), (0, 0)]
    for bx, by in base_coords:
        ax.plot(bx, by, "s", color="#CC9900", ms=6, zorder=4, markeredgecolor="#333333", markeredgewidth=0.5)

    valid = bip.dropna(subset=["spray_angle", "ExitSpeed"])
    for _, row in valid.iterrows():
        ang  = float(row["spray_angle"])
        ev   = float(row["ExitSpeed"])
        dist = min(max(ev * 2.5, 60), 310)
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
    _hax(ax)
    labels = ["Pull", "Center", "Oppo"]
    values = [stats.get("pull_pct") or 0, stats.get("center_pct") or 0, stats.get("oppo_pct") or 0]
    colors = ["#E84040", "#F0B800", "#3399EE"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", lw=0.8)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color="#111111", padding=3, fontweight="bold")
    ax.set_ylim(0, max(max(values) * 1.35, 10))
    ax.set_ylabel("%", fontsize=7, color="#111111")
    ax.set_title("Spray Direction", fontsize=9, fontweight="bold", color=TEXT_COLOR)

def draw_hitter_stats_banner(ax, stats, batter_name):
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    def _bg(stat, val, hib):
        c = hitter_grade_color(stat, val, hib) if D1_HITTER_PCTLS else None
        return c if c else (0.94, 0.94, 0.97, 1.0)

    def _text_color_for_bg(bg_rgba):
        if isinstance(bg_rgba, tuple) and len(bg_rgba) >= 3:
            r, g, b = bg_rgba[:3]
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

    for i, (label, val, pct_key, hib, fmt_str, unit) in enumerate(metrics):
        x   = (i + 0.5) * col_w
        missing = val is None or (isinstance(val, float) and np.isnan(val))

        if missing:
            disp   = "—"
            bg     = (0.91, 0.91, 0.93, 1.0)
            t_col  = "#888899"
            p_txt  = ""
        else:
            disp  = f"{val:{fmt_str}}{unit}"
            bg    = _bg(pct_key, val, hib) if pct_key else (0.15, 0.15, 0.25, 1.0)
            t_col = _text_color_for_bg(bg)
            p = get_hitter_percentile(pct_key, val) if pct_key else None
            p_txt = f"{int(round(p))}th" if p is not None else ""

        ax.add_patch(plt.Rectangle((i * col_w + 0.004, 0.05), col_w - 0.008, 0.90,
                                    transform=ax.transAxes, color=bg,
                                    clip_on=False, zorder=2))
        ax.text(x, 0.76, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=7,
                color=t_col, fontweight="bold", zorder=3, alpha=0.75)
        ax.text(x, 0.44, disp, transform=ax.transAxes,
                ha="center", va="center", fontsize=11,
                color=t_col, fontweight="bold", zorder=3)
        if p_txt:
            ax.text(x, 0.13, p_txt, transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5,
                    color=t_col, alpha=0.80, zorder=3)

def generate_hitter_heatmap(batter_df, metric="ev", pitch_type=None,
                             pitcher_hand=None, count=None):
    SWING_CALLS_LOCAL = {"StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay"}

    df = batter_df.copy()
    df = df[df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()]

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
        cmap_name   = "RdYlGn_r"
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
        plot_df = df[df["PitchCall"].isin(SWING_CALLS_LOCAL)].copy()
        plot_df["_val"] = (plot_df["PitchCall"] == "StrikeSwinging").astype(float)
        cmap_name   = "RdYlGn"
        title_label = "Whiff Rate"
        vmin, vmax  = 0, 0.6
    elif metric == "swing":
        plot_df = df.copy()
        plot_df["_val"] = plot_df["PitchCall"].isin(SWING_CALLS_LOCAL).astype(float)
        cmap_name   = "RdYlGn"
        title_label = "Swing Rate"
        vmin, vmax  = 0, 1.0
    else:
        return None

    if len(plot_df) < 3:
        return None

    batter_name = plot_df["Batter"].iloc[0] if "Batter" in plot_df.columns else "Batter"
    batter_side = plot_df["BatterSide"].mode()[0] if "BatterSide" in plot_df.columns else "?"
    hand_lbl    = "LHB" if batter_side == "Left" else "RHB" if batter_side == "Right" else batter_side

    parts = []
    if pitch_type: parts.append(pitch_type)
    if count:      parts.append(f"{count} count")
    filter_str = "  ·  " + "  ·  ".join(parts) if parts else ""

    date_range = f"{batter_df['GameDate'].min()} – {batter_df['GameDate'].max()}"

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9,
                            fill=False, ec="#333333", lw=2.0, zorder=10))
    for yline in [2.167, 2.833]:
        ax.plot([-0.95, 0.95], [yline, yline],
                color="#333333", lw=0.5, ls="--", alpha=0.5, zorder=9)
    ax.add_patch(Polygon([(-.708,.55),(.708,.55),(.708,.35),(0,.15),(-.708,.35)],
                         closed=True, fc="#CCCCCC", ec="#333333", lw=.8,
                         alpha=0.7, zorder=10))

    x = -plot_df["PlateLocSide"].values
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

def generate_hitter_page(batter_df, batter_name, game_date, opponent,
                          filter_count=None, filter_pitch_hand=None, filter_pitch_type=None):
    stats = compute_batter_stats(batter_df)
    bip   = stats.pop("bip_df")

    heat_df = batter_df.copy()
    filter_labels = []

    if filter_pitch_hand and "PitcherThrows" in heat_df.columns:
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

    fig = plt.figure(figsize=(26, 28), facecolor=BG_COLOR)
    gs  = fig.add_gridspec(5, 3,
                            height_ratios=[0.06, 0.10, 1.0, 1.0, 0.55],
                            hspace=0.45, wspace=0.35,
                            left=0.05, right=0.97, top=0.97, bottom=0.03)

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

    ax_banner = fig.add_subplot(gs[1, :])
    draw_hitter_stats_banner(ax_banner, stats, batter_name)

    ax_scatter = fig.add_subplot(gs[2, 0]); draw_ev_la_scatter(ax_scatter, bip)
    ax_evdist  = fig.add_subplot(gs[2, 1]); draw_ev_distribution(ax_evdist, bip)
    ax_zone_ev = fig.add_subplot(gs[2, 2])
    draw_zone_heatmap(ax_zone_ev, batter_df, "ev", f"Zone EV (Catcher POV){filter_suffix}", heat_df)

    ax_zone_xw = fig.add_subplot(gs[3, 0])
    draw_zone_heatmap(ax_zone_xw, batter_df, "xwoba", f"Zone xwOBA (Catcher POV){filter_suffix}", heat_df)
    ax_swing = fig.add_subplot(gs[3, 1])
    draw_zone_heatmap(ax_swing, batter_df, "swing", f"Swing Rate (Catcher POV){filter_suffix}", heat_df)
    ax_whiff = fig.add_subplot(gs[3, 2])
    draw_zone_heatmap(ax_whiff, batter_df, "whiff", f"Whiff% (Catcher POV){filter_suffix}", heat_df)

    ax_bb   = fig.add_subplot(gs[4, 0]); draw_batted_ball_profile(ax_bb, stats)
    ax_pull = fig.add_subplot(gs[4, 1]); draw_pull_oppo(ax_pull, stats)
    ax_blank = fig.add_subplot(gs[4, 2]); ax_blank.axis("off")

    return fig

# ===========================================================================
