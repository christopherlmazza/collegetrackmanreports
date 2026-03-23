"""
Hitters page — TrackMan Baseball Reports
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

st.set_page_config(page_title="Hitters — TrackMan", layout="wide", page_icon="🏏")
inject_app_styles()
render_page_intro(
    "🏏 Hitter Reports",
    "This rebuild keeps every existing hitter feature but reorganizes the layout so coaches and players can understand coverage, select a batter, and reach the right visual faster.",
    chips=["Hitter cards", "Single-batter heatmaps", "Pitch-type filters", "Count filters"],
    eyebrow="Hitting view",
)

# ── Load index ──
idx_df, parquet_path = load_all_pitches()

with st.sidebar:
    st.header("Report Settings")

    if idx_df is None or idx_df.empty:
        st.error("❌ No data found. Run fetch_data.py first.")
        st.stop()

    last_updated = get_last_updated()
    n_games = len(idx_df.drop_duplicates(subset=["GameDate", "HomeTeam", "AwayTeam"]))
    st.success(f"✅ Data loaded\n\n{n_games} games · Updated {last_updated or 'unknown'}")

    all_dates = idx_df["GameDate"].dropna()
    from datetime import date as _date
    min_date = all_dates.min() if not all_dates.empty else _date(2026, 2, 1)
    max_date = all_dates.max() if not all_dates.empty else _date.today()
    if hasattr(min_date, "date"): min_date = min_date.date()
    if hasattr(max_date, "date"): max_date = max_date.date()
    default_from = max(min_date, max_date - timedelta(days=7))
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From", value=default_from, min_value=min_date, max_value=max_date)
    with col2:
        date_to = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

    teams = sorted(idx_df["HomeTeam"].dropna().unique().tolist()
                   + idx_df["AwayTeam"].dropna().unique().tolist())
    teams = sorted(set(teams))
    selected_team = st.selectbox("Team", teams)

    perc_data = load_percentiles()
    if perc_data:
        meta = perc_data.get("_meta", {})
        st.success(f"D1 Percentiles loaded\n\n{meta.get('sessions_scanned', 0)} sessions · {meta.get('generated', '?')[:10]}")
    else:
        st.warning("No D1_percentiles.json found.\nColor grading disabled.")

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

# Enrich PitcherThrows from pitching rows
_all_throws_nonempty = df_all[df_all["PitcherThrows"].str.strip().ne("") & df_all["PitcherThrows"].notna()]
_hand_lookup = (
    _all_throws_nonempty
    .groupby("Pitcher")["PitcherThrows"]
    .agg(lambda x: x.mode()[0] if not x.empty else "")
    .to_dict()
)
bat_df["PitcherThrows"] = bat_df["Pitcher"].map(_hand_lookup).fillna("")

# ── Team coverage overview ──
games_h = (bat_df.groupby(["GameDate", "HomeTeam", "AwayTeam"])
           .size().reset_index().sort_values("GameDate"))

section_title(f"{team_name} coverage", "The overview stays team-level so you can confirm the data set before drilling into one batter.")
hov1, hov2, hov3, hov4, hov5 = st.columns(5)
with hov1:
    render_stat_card("Games", len(games_h), f"{date_from} to {date_to}")
with hov2:
    render_stat_card("Batters", bat_df["Batter"].dropna().nunique(), "Players with batting events in range")
with hov3:
    render_stat_card("Plate Appearances", calc_pa(bat_df), "Team batting PAs in the selected span")
with hov4:
    team_xwoba = bat_df["xwOBA"].dropna().mean()
    render_stat_card("Team xwOBA", format_number(team_xwoba, 3), "In-play xwOBA estimate")
with hov5:
    start_lbl = str(games_h["GameDate"].min()) if not games_h.empty else "—"
    end_lbl = str(games_h["GameDate"].max()) if not games_h.empty else "—"
    render_stat_card("Window", f"{start_lbl} → {end_lbl}", "First and last tracked game")

cover_left, cover_right = st.columns([1.6, 1.0])
with cover_left:
    games_display = games_h.copy()
    games_display["Matchup"] = games_display["HomeTeam"] + " vs " + games_display["AwayTeam"]
    games_display["GameDate"] = games_display["GameDate"].astype(str)
    with st.expander("Game list for this range", expanded=False):
        st.dataframe(
            games_display[["GameDate", "Matchup", 0]].rename(columns={0: "Tracked pitches"}),
            use_container_width=True,
            hide_index=True,
        )
with cover_right:
    render_info_panel("What changed visually", [
        "Team-level coverage sits above the batter tools for faster orientation.",
        "Batter selection and game selection are grouped together in one cleaner block.",
        "The hitter card and heatmap features behave the same as before.",
    ])

# ── Batter + game pickers ──
batter_list = get_batters(bat_df)
if not batter_list:
    st.warning("No batters found.")
    st.stop()

section_title("Choose a batter", "Select the batter first, then choose either one tracked game or the full season roll-up.")
pick_col1, pick_col2 = st.columns([1.45, 1.0])

# BUG FIX: only one selectbox for selected_batter — inside pick_col1
with pick_col1:
    selected_batter = st.selectbox("Select Batter", batter_list)

b_df = bat_df[bat_df["Batter"] == selected_batter]

batter_games = sorted(b_df.groupby(["GameDate", "HomeTeam", "AwayTeam"]).groups.keys())
game_options = []
for gdate, ht, at in batter_games:
    opp = at if team_name.lower() in ht.lower() else ht
    game_options.append((f"{gdate} vs {opp}", gdate, opp))

if not game_options:
    st.warning("No games found for this batter.")
    st.stop()

if len(game_options) == 1:
    _, sel_gdate, sel_opp = game_options[0]
    with pick_col2:
        render_info_panel("Current split", [
            game_options[0][0],
            "Only one tracked game is available for this batter in the selected date range.",
        ])
else:
    all_opt = ("All games (season)", None, "Season")
    all_game_opts = [all_opt] + game_options
    all_labels = [g[0] for g in all_game_opts]
    # BUG FIX: only one selectbox for sel_idx — inside pick_col2
    with pick_col2:
        sel_idx = st.selectbox("Select Game", range(len(all_labels)),
                               format_func=lambda i: all_labels[i])
    _, sel_gdate, sel_opp = all_game_opts[sel_idx]

if sel_gdate is None:
    page_df = b_df
    gdate_lbl = "Full Season"
    opp_lbl = team_name
else:
    page_df = b_df[b_df["GameDate"] == sel_gdate]
    gdate_lbl = str(sel_gdate)
    opp_lbl = sel_opp

if page_df.empty:
    st.warning("No data for this selection.")
    st.stop()

# ── Batter stat summary ──
batter_stats = compute_batter_stats(page_df)
summary_col, helper_col = st.columns([1.6, 1.0])
with helper_col:
    side = page_df["BatterSide"].mode()[0] if not page_df["BatterSide"].dropna().empty else "Unknown"
    render_info_panel("Current batter view", [
        f"Batter: {selected_batter}",
        f"Split: {gdate_lbl} vs {opp_lbl}",
        f"Side: {side}",
        f"Pitch types seen: {page_df['PitchType'].dropna().replace('', np.nan).dropna().nunique()}",
    ])
with summary_col:
    sel1, sel2, sel3, sel4 = st.columns(4)
    with sel1:
        render_stat_card("PA", batter_stats["pa"], "Plate appearances in current view")
    with sel2:
        render_stat_card("BIP", batter_stats["bip"], "Balls in play in current view")
    with sel3:
        render_stat_card("Avg EV", format_number(batter_stats["avg_ev"], 1, " mph"), "Average exit velocity on BIP")
    with sel4:
        render_stat_card("Whiff%", format_number(batter_stats["whiff_pct"], 1, "%"), "Whiffs per swing")

# ── Tabs ──
tab_card, tab_heatmaps = st.tabs(["📄 Hitter Card", "🔥 Heatmaps"])

with tab_card:
    info_left, info_right = st.columns([1.25, 1.0])
    with info_left:
        st.caption("The hitter card still uses the current batter/game selection. The optional filters below only change the heatmap panels shown on the card.")
    with info_right:
        render_info_panel("Use this tab for", [
            "One-page player review visuals.",
            "Filtered heatmaps inside the card layout.",
            "Downloading a shareable PNG card.",
        ])
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        ph_opts = ["All"] + sorted([h for h in page_df["PitcherThrows"].dropna().unique() if h])
        card_hand = st.selectbox("vs Pitcher Hand", ph_opts, key="card_hand")
    with cf2:
        cpt_opts = ["All"] + sorted([p for p in page_df["PitchType"].dropna().unique()
                                     if p and p != "Other"])
        card_pt = st.selectbox("Pitch Type", cpt_opts, key="card_pt")
    with cf3:
        cnt_opts = ["All", "0-0", "1-0", "2-0", "3-0", "0-1", "1-1", "2-1", "3-1",
                    "0-2", "1-2", "2-2", "3-2", "2-strike", "ahead", "behind"]
        card_count = st.selectbox("Count", cnt_opts, key="card_count")

    f_hand = None if card_hand == "All" else card_hand
    f_pt = None if card_pt == "All" else card_pt
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
    info_left, info_right = st.columns([1.25, 1.0])
    with info_left:
        st.caption("Generate a single heatmap for the current batter selection. Tighter filters usually create more useful visual reads.")
    with info_right:
        render_info_panel("Use this tab for", [
            "A focused zone view for one metric at a time.",
            "Exploring pitch type and count splits.",
            "Saving the current heatmap as a PNG.",
        ])

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

    hm_cnt_opts = ["All", "0-0", "1-0", "2-0", "3-0", "0-1", "1-1", "2-1", "3-1",
                   "0-2", "1-2", "2-2", "3-2", "2-strike", "ahead", "behind"]
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
        hm_data = page_df.copy()
        if hm_pt != "All":
            hm_data = hm_data[hm_data["PitchType"] == hm_pt]
        if hm_hand != "All":
            hm_data = hm_data[hm_data["PitcherThrows"] == hm_hand]
        if hm_count != "All":
            hm_data = filter_by_count(hm_data, hm_count)

        if hm_data.empty:
            st.warning("No pitches match the current filter combination.")
        else:
            with st.spinner("Generating heatmap..."):
                hm_fig = generate_hitter_heatmap(
                    hm_data, selected_batter, metric=metric_map[hm_metric])
                if hm_fig:
                    st.pyplot(hm_fig, use_container_width=True)
                    buf = io.BytesIO()
                    hm_fig.savefig(buf, format="png", bbox_inches="tight",
                                   dpi=150, facecolor=BG_COLOR)
                    buf.seek(0)
                    st.download_button(
                        "📥 Download Heatmap",
                        data=buf,
                        file_name=f"HitterHeatmap_{selected_batter}_{hm_metric}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
