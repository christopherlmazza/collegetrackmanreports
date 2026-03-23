"""
Hitters page — TrackMan Baseball Reports
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

st.set_page_config(page_title="Hitters — TrackMan", layout="wide", page_icon="🏏")
st.title("🏏 Hitter Reports")

# ── Load index ──
idx_df, parquet_path = load_all_pitches()

with st.sidebar:
    st.header("Report Settings")

    if idx_df is None or idx_df.empty:
        st.error("❌ No data found. Run fetch_data.py first.")
        st.stop()

    last_updated = get_last_updated()
    n_games = len(idx_df.drop_duplicates(subset=["GameDate","HomeTeam","AwayTeam"]))
    st.success(f"✅ Data loaded\n\n{n_games} games · Updated {last_updated or 'unknown'}")

    all_dates = idx_df["GameDate"].dropna()
    from datetime import date as _date
    min_date  = all_dates.min() if not all_dates.empty else _date(2026, 2, 1)
    max_date  = all_dates.max() if not all_dates.empty else _date.today()
    # Ensure they are proper date objects not NaT
    if hasattr(min_date, "date"): min_date = min_date.date()
    if hasattr(max_date, "date"): max_date = max_date.date()
    # Clamp default value so it never goes below min_date
    default_from = max(min_date, max_date - timedelta(days=7))
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From", value=default_from,
                                   min_value=min_date, max_value=max_date)
    with col2:
        date_to = st.date_input("To", value=max_date,
                                 min_value=min_date, max_value=max_date)

    if date_from > date_to:
        st.error("'From' date must be before 'To' date")
        st.stop()

    teams_list = get_teams(idx_df)
    if not teams_list:
        st.error("No teams found in data.")
        st.stop()

    prev_team   = st.session_state.get("selected_team_h", None)
    default_idx = teams_list.index(prev_team) if prev_team and prev_team in teams_list else 0
    selected_team = st.selectbox("Select Team", teams_list, index=default_idx)
    st.session_state["selected_team_h"] = selected_team

    if st.button("🔄 Reset", use_container_width=True):
        for k in ["_hitter_hm_bytes","_hitter_hm_label"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    if D1_HITTER_PCTLS:
        hmeta = D1_HITTER_PCTLS.get("_meta", {})
        st.success(f"🏏 Hitter percentiles loaded\n\n{hmeta.get('qualifying_batters',0)} batters")
    else:
        st.warning("No D1_hitter_percentiles.json found.")

# ── Load team data on demand ──
team_name = selected_team
with st.spinner(f"Loading {team_name} data..."):
    df_all = load_team_data(team_name, date_from, date_to)

if df_all is None or df_all.empty:
    st.warning(f"No data found for {team_name} between {date_from} and {date_to}")
    st.stop()

bat_df = get_team_batting(df_all, team_name, date_from, date_to)

if bat_df.empty:
    st.warning(f"No batting data found for {team_name} between {date_from} and {date_to}")
    st.stop()

bat_df["xwOBA"] = bat_df.apply(
    lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"])
    if r["PitchCall"] == "InPlay" else np.nan, axis=1)
bat_df["InZone"] = in_zone(bat_df)

# PitcherThrows is often blank on batting rows — enrich from pitching rows
_all_throws = df_all["PitcherThrows"].dropna().unique().tolist()
_all_throws_nonempty = df_all[df_all["PitcherThrows"].str.strip().ne("") & df_all["PitcherThrows"].notna()]
_hand_lookup = (
    _all_throws_nonempty
    .groupby("Pitcher")["PitcherThrows"]
    .agg(lambda x: x.mode()[0] if not x.empty else "")
    .to_dict()
)
bat_df["PitcherThrows"] = bat_df["Pitcher"].map(_hand_lookup).fillna("")

# Games summary
games_h = (bat_df.groupby(["GameDate","HomeTeam","AwayTeam"])
           .size().reset_index().sort_values("GameDate"))
st.subheader(f"{team_name} — {len(games_h)} game(s)")
for _, row in games_h.iterrows():
    st.text(f"  📅 {row['GameDate']} — {row['HomeTeam']} vs {row['AwayTeam']}")

# Batter + game pickers
batter_list = get_batters(bat_df)
if not batter_list:
    st.warning("No batters found."); st.stop()

selected_batter = st.selectbox("Select Batter", batter_list)
b_df = bat_df[bat_df["Batter"] == selected_batter]

batter_games = sorted(b_df.groupby(["GameDate","HomeTeam","AwayTeam"]).groups.keys())
game_options = []
for gdate, ht, at in batter_games:
    opp = at if team_name.lower() in ht.lower() else ht
    game_options.append((f"{gdate} vs {opp}", gdate, opp))

if not game_options:
    st.warning("No games found for this batter."); st.stop()

if len(game_options) == 1:
    _, sel_gdate, sel_opp = game_options[0]
else:
    all_opt = ("All games (season)", None, "Season")
    all_game_opts = [all_opt] + game_options
    all_labels = [g[0] for g in all_game_opts]
    sel_idx = st.selectbox("Select Game", range(len(all_labels)),
                           format_func=lambda i: all_labels[i])
    _, sel_gdate, sel_opp = all_game_opts[sel_idx]

if sel_gdate is None:
    page_df   = b_df
    gdate_lbl = "Full Season"
    opp_lbl   = team_name
else:
    page_df   = b_df[b_df["GameDate"] == sel_gdate]
    gdate_lbl = str(sel_gdate)
    opp_lbl   = sel_opp

if page_df.empty:
    st.warning("No data for this selection."); st.stop()

# ── Tabs ──

tab_card, tab_heatmaps = st.tabs(["📄 Hitter Card", "🔥 Heatmaps"])

with tab_card:
    st.caption("Stats banner uses all data. Heatmaps on the card can be filtered below.")
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        ph_opts = ["All"] + sorted([h for h in page_df["PitcherThrows"].dropna().unique() if h])
        card_hand  = st.selectbox("vs Pitcher Hand", ph_opts, key="card_hand")
    with cf2:
        cpt_opts = ["All"] + sorted([p for p in page_df["PitchType"].dropna().unique()
                                     if p and p != "Other"])
        card_pt    = st.selectbox("Pitch Type", cpt_opts, key="card_pt")
    with cf3:
        cnt_opts = ["All","0-0","1-0","2-0","3-0","0-1","1-1","2-1","3-1",
                    "0-2","1-2","2-2","3-2","2-strike","ahead","behind"]
        card_count = st.selectbox("Count", cnt_opts, key="card_count")

    f_hand  = None if card_hand  == "All" else card_hand
    f_pt    = None if card_pt    == "All" else card_pt
    f_count = None if card_count == "All" else card_count

    if st.button("📄 Generate Hitter Card", use_container_width=True,
                 type="primary", key="btn_hitter_card"):
        with st.spinner(f"Generating card for {selected_batter}..."):
            try:
                hfig = generate_hitter_page(
                    page_df, selected_batter, gdate_lbl, opp_lbl,
                    filter_count=f_count,
                    filter_pitch_hand=f_hand,
                    filter_pitch_type=f_pt,
                )
                st.pyplot(hfig, use_container_width=True)
                buf = io.BytesIO()
                hfig.savefig(buf, format="png", bbox_inches="tight",
                             dpi=150, facecolor=BG_COLOR)
                buf.seek(0)
                st.download_button("📥 Download Card", data=buf,
                                   file_name=f"HitterCard_{selected_batter}_{gdate_lbl}.png",
                                   mime="image/png", use_container_width=True)
                plt.close(hfig)
            except Exception as e:
                st.error(f"Error generating hitter card: {e}")
                import traceback; st.code(traceback.format_exc())

with tab_heatmaps:
    st.caption("KDE-smoothed zone heatmaps — same style as pitcher heatmaps.")

    # Filter controls
    hm_col1, hm_col2, hm_col3 = st.columns(3)
    with hm_col1:
        hm_metric = st.selectbox(
            "Metric",
            ["Exit Velocity", "xwOBA", "Whiff Rate", "Swing Rate", "Location Density"],
            key="hm_metric")
    with hm_col2:
        avail_pt = ["All"] + sorted([p for p in page_df["PitchType"].dropna().unique()
                                      if p and p != "Other"])
        hm_pt = st.selectbox("Pitch Type", avail_pt, key="hm_pt_hitter")
    with hm_col3:
        avail_hand = ["All"] + sorted([h for h in page_df["PitcherThrows"].dropna().unique() if h])
        hm_hand = st.selectbox("vs Pitcher Hand", avail_hand, key="hm_hand_hitter")

    hm_cnt_opts = ["All","0-0","1-0","2-0","3-0","0-1","1-1","2-1","3-1",
                   "0-2","1-2","2-2","3-2","2-strike","ahead","behind"]
    hm_count = st.selectbox("Count", hm_cnt_opts, key="hm_count_hitter")

    metric_map = {
        "Exit Velocity":    "ev",
        "xwOBA":            "xwoba",
        "Whiff Rate":       "whiff",
        "Swing Rate":       "swing",
        "Location Density": "location",
    }

    if st.button("🔥 Generate Heatmap", type="primary",
                 use_container_width=True, key="btn_hitter_hm"):
        with st.spinner(f"Generating {hm_metric} heatmap..."):
            try:
                hmfig = generate_hitter_heatmap(
                    page_df,
                    metric     = metric_map[hm_metric],
                    pitch_type = None if hm_pt    == "All" else hm_pt,
                    count      = None if hm_count == "All" else hm_count,
                )
                if hmfig:
                    st.pyplot(hmfig, use_container_width=True)
                    buf = io.BytesIO()
                    hmfig.savefig(buf, format="png", bbox_inches="tight",
                                  dpi=150, facecolor=BG_COLOR)
                    buf.seek(0)
                    st.session_state["_hitter_hm_bytes"] = buf.getvalue()
                    st.session_state["_hitter_hm_label"] = (
                        f"{hm_metric}_{selected_batter}_{gdate_lbl}")
                    plt.close(hmfig)
                else:
                    st.warning("Not enough data for this filter combination (need 3+ pitches).")
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
                import traceback; st.code(traceback.format_exc())

    if st.session_state.get("_hitter_hm_bytes"):
        st.download_button(
            "📥 Download Heatmap",
            data=st.session_state["_hitter_hm_bytes"],
            file_name=f"Heatmap_{st.session_state.get('_hitter_hm_label','hitter')}.png",
            mime="image/png", use_container_width=True)

