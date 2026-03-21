"""
Pitchers page — TrackMan Baseball Reports
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

st.set_page_config(page_title="Pitchers — TrackMan", layout="wide", page_icon="⚾")
st.title("⚾ Pitcher Reports")

# ── Load index (tiny — just team names and dates) ──
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

    prev_team   = st.session_state.get("selected_team", None)
    default_idx = teams_list.index(prev_team) if prev_team and prev_team in teams_list else 0
    selected_team = st.selectbox("Select Team", teams_list, index=default_idx)
    st.session_state["selected_team"] = selected_team

    if st.button("🔄 Reset Selections", use_container_width=True):
        for k in ["pitcher_outings","pitcher_names","_team_key",
                  "_hm_loaded","_pm_active_key","_pm_pitchers"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    if D1_PCTLS:
        meta = D1_PCTLS.get("_meta", {})
        st.success(f"D1 Percentiles loaded\n\n{meta.get('sessions_scanned',0)} sessions · {meta.get('generated','?')[:10]}")
    else:
        st.warning("No D1_percentiles.json found.\nColor grading disabled.")

# ── Load this team's pitch data on demand ──
team_name = selected_team
with st.spinner(f"Loading {team_name} data..."):
    team_df = load_team_data(team_name, date_from, date_to)

if team_df is None or team_df.empty:
    st.warning(f"No pitches found for {team_name} between {date_from} and {date_to}")
    st.stop()

# Filter to pitches thrown BY this team
team_df = get_team_pitches(team_df, team_name, date_from, date_to)

if team_df.empty:
    st.warning(f"No pitching data found for {team_name}")
    st.stop()

df_all = team_df  # alias for compatibility

# Show games found
games = (team_df.groupby(["GameDate", "HomeTeam", "AwayTeam"])
         .size().reset_index().sort_values("GameDate"))
st.subheader(f"{team_name} — {len(games)} game(s)")
for _, row in games.iterrows():
    st.text(f"  📅 {row['GameDate']} — {row['HomeTeam']} vs {row['AwayTeam']}")

# Build pitcher list from parquet
cache_key = f"{team_name}_{date_from}_{date_to}"
if st.session_state.get("_team_key") != cache_key:
    pitcher_meta = {}
    for pn in sorted(team_df["Pitcher"].dropna().unique()):
        p_games = (team_df[team_df["Pitcher"] == pn]
                   .groupby(["GameDate", "HomeTeam", "AwayTeam"]))
        outings = []
        for (gdate, ht, at), _ in p_games:
            opp = at if team_name.lower() in ht.lower() else ht
            outings.append((gdate, opp))
        if outings:
            pitcher_meta[pn] = sorted(outings, key=lambda x: x[0])
    st.session_state["pitcher_meta_parquet"] = pitcher_meta
    st.session_state["pitcher_names"] = sorted(pitcher_meta.keys())
    st.session_state["_team_key"] = cache_key
    st.session_state.pop("pitcher_outings", None)

def load_full_pitcher_data(pitcher_names_to_load):
    """Build per-pitcher outing DataFrames from parquet — no API calls."""
    if "pitcher_outings" not in st.session_state:
        st.session_state["pitcher_outings"] = {}
    needed = [pn for pn in pitcher_names_to_load
              if pn not in st.session_state["pitcher_outings"]]
    if not needed:
        return
    for pn in needed:
        p_df_all = team_df[team_df["Pitcher"] == pn].copy()
        p_df_all = p_df_all.sort_values(["GameDate", "PitchNo"]).reset_index(drop=True)
        p_df_all["PitchType"] = p_df_all.apply(resolve_pt, axis=1)
        p_df_all, _ = auto_correct_pitch_types(p_df_all)
        p_df_all["xwOBA"] = p_df_all.apply(
            lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"])
            if r["PitchCall"] == "InPlay" else np.nan, axis=1)
        p_df_all["InZone"] = in_zone(p_df_all)
        outings = []
        for gdate, g_df in p_df_all.groupby("GameDate"):
            ht = g_df["HomeTeam"].iloc[0] if "HomeTeam" in g_df.columns else ""
            at = g_df["AwayTeam"].iloc[0] if "AwayTeam" in g_df.columns else ""
            opp = at if team_name.lower() in ht.lower() else ht
            outings.append((g_df.reset_index(drop=True), gdate, opp))
        st.session_state["pitcher_outings"][pn] = sorted(outings, key=lambda x: x[1])

if "pitcher_names" in st.session_state and st.session_state["pitcher_names"]:
    st.divider()
    pitcher_meta = st.session_state.get("pitcher_meta_parquet", {})
    outing_labels = []
    for pn in st.session_state["pitcher_names"]:
        outings = pitcher_meta.get(pn, [])
        if len(outings) > 1:
            outing_labels.append(f"{pn}  ({len(outings)} outings)")
        elif outings:
            gd, op = outings[0]
            outing_labels.append(f"{pn}  ({gd} vs {op})")
        else:
            outing_labels.append(pn)

    selected_labels = st.multiselect("Select Pitcher(s)", outing_labels, default=None)
    label_to_name = dict(zip(outing_labels, st.session_state["pitcher_names"]))
    selected_names = [label_to_name[lbl] for lbl in selected_labels]

    if selected_names:
        tab_reports, tab_summary, tab_heatmaps, tab_debug = st.tabs(
            ["📄 Game Reports", "📊 Season Summary", "🔥 Heatmaps", "🔧 Debug"])

        # ========== TAB 1: GAME REPORTS ==========
        with tab_reports:
            if st.button("⚾ Generate Game Reports", type="primary",
                         use_container_width=True, key="btn_reports"):
                figures = []
                with st.spinner("Generating reports..."):
                    load_full_pitcher_data(selected_names)
                    for pname in selected_names:
                        if pname not in st.session_state.get("pitcher_outings", {}):
                            continue
                        for p_data, gdate, opp in st.session_state["pitcher_outings"][pname]:
                            if len(p_data) == 0:
                                continue
                            fig = generate_pitcher_page(p_data, pname, gdate, opp)
                            if fig:
                                figures.append((f"{pname} ({gdate} vs {opp})", fig))
                if figures:
                    for label, fig in figures:
                        st.pyplot(fig, use_container_width=True)
                        st.divider()
                    pdf_buffer = io.BytesIO()
                    with PdfPages(pdf_buffer) as pdf:
                        for label, fig in figures:
                            pdf.savefig(fig, bbox_inches="tight", facecolor=BG_COLOR)
                            plt.close(fig)
                    pdf_buffer.seek(0)
                    safe_team = team_name.replace(" ", "")[:15]
                    st.download_button("📥 Download Game Reports PDF", data=pdf_buffer,
                                       file_name=f"GameReports_{safe_team}_{date_from}_to_{date_to}.pdf",
                                       mime="application/pdf", type="primary",
                                       use_container_width=True)
                else:
                    st.error("No reports generated")

        # ========== TAB 2: SEASON SUMMARY ==========
        with tab_summary:
            if st.button("📊 Generate Season Summaries", type="primary",
                         use_container_width=True, key="btn_summary"):
                figures = []
                with st.spinner("Building season summaries from local data..."):
                    # Always load full season regardless of date filter
                    _all_dates = idx_df["GameDate"].dropna()
                    full_season_df = load_team_data(team_name, _all_dates.min(), _all_dates.max())
                    if full_season_df is None or full_season_df.empty:
                        full_season_df = df_all
                    season_df = get_team_pitches(
                        full_season_df, team_name,
                        full_season_df["GameDate"].min(), full_season_df["GameDate"].max())
                    for pname in selected_names:
                        p_season = season_df[season_df["Pitcher"] == pname].copy()
                        if p_season.empty:
                            st.warning(f"No season data for {pname}")
                            continue
                        p_season["PitchType"] = p_season.apply(resolve_pt, axis=1)
                        p_season, _ = auto_correct_pitch_types(p_season)
                        p_season["xwOBA"] = p_season.apply(
                            lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"])
                            if r["PitchCall"] == "InPlay" else np.nan, axis=1)
                        p_season["InZone"] = in_zone(p_season)
                        outings = []
                        for gdate, g_df in p_season.groupby("GameDate"):
                            ht = g_df["HomeTeam"].iloc[0] if "HomeTeam" in g_df.columns else ""
                            at = g_df["AwayTeam"].iloc[0] if "AwayTeam" in g_df.columns else ""
                            opp = at if team_name.lower() in ht.lower() else ht
                            outings.append((g_df.reset_index(drop=True), gdate, opp))
                        outings = sorted(outings, key=lambda x: x[1])
                        fig = generate_season_summary(
                            pname, outings,
                            season_df["GameDate"].min(), season_df["GameDate"].max())
                        if fig:
                            figures.append((pname, fig))
                if figures:
                    for label, fig in figures:
                        st.pyplot(fig, use_container_width=True)
                        st.divider()
                    pdf_buffer = io.BytesIO()
                    with PdfPages(pdf_buffer) as pdf:
                        for label, fig in figures:
                            pdf.savefig(fig, bbox_inches="tight", facecolor=BG_COLOR)
                            plt.close(fig)
                    pdf_buffer.seek(0)
                    safe_team = team_name.replace(" ", "")[:15]
                    st.download_button("📥 Download Season Summary PDF", data=pdf_buffer,
                                       file_name=f"SeasonSummary_{safe_team}_2026.pdf",
                                       mime="application/pdf", type="primary",
                                       use_container_width=True)
                else:
                    st.error("No summaries generated")

        # ========== TAB 3: HEATMAPS ==========
        with tab_heatmaps:
            hm_pitcher = (selected_names[0] if len(selected_names) == 1
                          else st.selectbox("Select pitcher for heatmaps",
                                            selected_names, key="hm_pitcher"))
            if st.button("📂 Load Pitch Data", type="primary",
                         use_container_width=True, key="btn_load_hm"):
                load_full_pitcher_data([hm_pitcher])
                st.session_state["_hm_loaded"] = hm_pitcher

            if (st.session_state.get("_hm_loaded") == hm_pitcher and
                    hm_pitcher in st.session_state.get("pitcher_outings", {})):
                all_outing_dfs = [p_df for p_df, _, _
                                  in st.session_state["pitcher_outings"][hm_pitcher]]
                hm_data = (pd.concat(all_outing_dfs, ignore_index=True)
                           if all_outing_dfs else pd.DataFrame())
                if not hm_data.empty:
                    avail_types = hm_data["PitchType"].value_counts()
                    avail_types = avail_types[avail_types >= 5].index.tolist()
                    if avail_types:
                        hm_pitch_type = st.selectbox("Select Pitch Type", avail_types,
                                                      key="hm_pt_select")
                        hm_metric = st.selectbox(
                            "Select Metric",
                            ["Location", "Run Value", "Whiff Rate", "xwOBA"],
                            key="hm_metric_select")
                        metric_map = {"Location": "location", "Run Value": "run_value",
                                      "Whiff Rate": "whiff", "xwOBA": "xwoba"}
                        if st.button("🔥 Generate Heatmap", type="primary",
                                     use_container_width=True, key="btn_gen_hm"):
                            with st.spinner(f"Generating {hm_metric} heatmap..."):
                                fig = generate_heatmap(hm_data, hm_pitch_type,
                                                       metric_map[hm_metric])
                            if fig:
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", bbox_inches="tight",
                                            dpi=150, facecolor=BG_COLOR)
                                buf.seek(0)
                                st.session_state["_hm_fig_bytes"] = buf.getvalue()
                                st.session_state["_hm_fig_label"] = (
                                    f"{hm_metric}_{hm_pitcher}_{hm_pitch_type}")
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.warning(f"Not enough {hm_pitch_type} data for heatmap")
                        if st.session_state.get("_hm_fig_bytes"):
                            st.download_button(
                                "📥 Download Heatmap",
                                data=st.session_state["_hm_fig_bytes"],
                                file_name=f"Heatmap_{st.session_state.get('_hm_fig_label','heatmap')}.png",
                                mime="image/png", use_container_width=True)
                    else:
                        st.warning("No pitch types with enough data (need 5+)")
                else:
                    st.warning("No pitch data available")
            elif (st.session_state.get("_hm_loaded") and
                  st.session_state.get("_hm_loaded") != hm_pitcher):
                st.info("Click **Load Pitch Data** to load data for this pitcher")

        # ========== TAB 4: DEBUG ==========
        with tab_debug:
            st.caption("Raw PA-level data for verifying IP/ER calculations.")
            if st.button("🔧 Load & Show Debug Data", type="primary",
                         use_container_width=True, key="btn_debug"):
                load_full_pitcher_data(selected_names)
                for pname in selected_names:
                    if pname not in st.session_state.get("pitcher_outings", {}):
                        st.warning(f"No data for {pname}")
                        continue
                    st.subheader(f"🔍 {pname}")
                    for p_df, gdate, opp in st.session_state["pitcher_outings"][pname]:
                        st.write(f"**{gdate} vs {opp}** — {len(p_df)} pitches")
                        pa_rows = []
                        debug_outs = 0
                        reached_results = ("Single", "Double", "Triple", "HomeRun",
                                           "Error", "FieldersChoice", "CaughtStealing",
                                           "ReachedOnError")
                        for (inn, pa_num), grp in p_df.groupby(["Inning", "PAofInning"]):
                            last = grp.loc[grp["PitchNo"].idxmax()]
                            oop = last.get("OutsOnPlay", "")
                            korbb = last.get("KorBB", "")
                            result = last.get("PlayResult", "")
                            out_src = ""
                            if pd.notna(oop) and float(oop) > 0:
                                debug_outs += int(float(oop))
                                out_src = f"+{int(float(oop))} (OutsOnPlay)"
                            elif korbb == "Strikeout":
                                if result in reached_results:
                                    out_src = f"K but reached ({result}) → NO out"
                                else:
                                    debug_outs += 1
                                    out_src = "+1 (K)"
                            pa_rows.append({
                                "Inn": inn, "PA#": pa_num, "#P": len(grp),
                                "Batter": last.get("Batter", ""),
                                "LastCall": last.get("PitchCall", ""),
                                "KorBB": korbb, "PlayResult": result,
                                "OutsOnPlay": oop, "RunsScored": last.get("RunsScored", ""),
                                "OutCredit": out_src,
                            })
                        pa_table = pd.DataFrame(pa_rows)
                        st.dataframe(pa_table, use_container_width=True, hide_index=True)
                        ip_s = calc_ip(p_df)
                        er_s = calc_er(p_df)
                        pa_s = calc_pa(p_df)
                        k_s = int((p_df["KorBB"] == "Strikeout").sum())
                        bb_s = int((p_df["KorBB"] == "Walk").sum())
                        st.write(f"**Computed:** IP={ip_s}, ER={er_s}, PA={pa_s}, "
                                 f"K={k_s}, BB={bb_s}, Debug outs={debug_outs}")
                        st.write("**Raw OutsOnPlay:**",
                                 p_df["OutsOnPlay"].value_counts().to_dict())
                        st.write("**Raw RunsScored:**",
                                 p_df["RunsScored"].value_counts().to_dict())
                        st.divider()

st.divider()
st.title("🎯 Pitch Mix Analysis")

if "selected_team" in st.session_state:
    pm_team = st.session_state["selected_team"]

    pm_col1, pm_col2 = st.columns(2)
    with pm_col1:
        pm_from = st.date_input("From", value=date(2026, 1, 1), key="pm_from")
    with pm_col2:
        pm_to = st.date_input("To", value=date.today(), key="pm_to")

    pm_cache_key = f"_pm_data_{pm_team}_{pm_from}_{pm_to}"

    if st.button("📂 Load Pitchers", type="secondary", use_container_width=True, key="btn_pm_load"):
        with st.spinner("Loading pitch mix data..."):
            pm_df = get_team_pitches(df_all, pm_team, pm_from, pm_to)
            pm_all_outings = {}
            if not pm_df.empty:
                pm_df = pm_df.copy()
                pm_df["PitchType"] = pm_df.apply(resolve_pt, axis=1)
                pm_df, _ = auto_correct_pitch_types(pm_df)
                pm_df["InZone"] = in_zone(pm_df)
                for pn in pm_df["Pitcher"].dropna().unique():
                    p_all = pm_df[pm_df["Pitcher"] == pn]
                    for gdate, g_df in p_all.groupby("GameDate"):
                        ht = g_df["HomeTeam"].iloc[0] if "HomeTeam" in g_df.columns else ""
                        at = g_df["AwayTeam"].iloc[0] if "AwayTeam" in g_df.columns else ""
                        opp = at if pm_team.lower() in ht.lower() else ht
                        p_df = g_df.sort_values("PitchNo").reset_index(drop=True)
                        if pn not in pm_all_outings:
                            pm_all_outings[pn] = []
                        pm_all_outings[pn].append((p_df, gdate, opp))
            st.session_state[pm_cache_key] = pm_all_outings
            st.session_state["_pm_pitchers"] = sorted(pm_all_outings.keys())
            st.session_state["_pm_active_key"] = pm_cache_key

    if st.session_state.get("_pm_active_key") == pm_cache_key and "_pm_pitchers" in st.session_state:
        pm_outings = st.session_state.get(pm_cache_key, {})

        if not st.session_state["_pm_pitchers"]:
            st.warning("No pitchers found")
        else:
            pm_pitcher = st.selectbox("Select Pitcher", st.session_state["_pm_pitchers"], key="pm_pitcher")
            fc1, fc2 = st.columns(2)
            with fc1:
                pm_hand = st.selectbox("Batter Hand", ["All", "vs RHH", "vs LHH"], key="pm_hand")
            with fc2:
                pm_tto = st.selectbox("Time Through Order",
                                      ["All", "1st Time", "2nd Time", "3rd Time", "4th+ Time"], key="pm_tto")

            if st.button("🎯 Generate Pitch Mix", type="primary", use_container_width=True, key="btn_pitchmix"):
                if pm_pitcher not in pm_outings or not pm_outings[pm_pitcher]:
                    st.warning(f"No data found for {pm_pitcher}")
                else:
                    all_dfs = []
                    for p_df, gdate, opp in pm_outings[pm_pitcher]:
                        df_copy = p_df.copy()
                        df_copy["_game_date"] = gdate
                        df_copy["_opp"] = opp
                        all_dfs.append(df_copy)
                    pitch_data = pd.concat(all_dfs, ignore_index=True)

                    pitch_data["_TTO"] = 0
                    for gd in pitch_data["_game_date"].unique():
                        game_mask = pitch_data["_game_date"] == gd
                        game_df = pitch_data[game_mask].copy()
                        pa_order = game_df.groupby(["Inning", "PAofInning"]).first().reset_index()
                        pa_order = pa_order.sort_values("PitchNo")
                        batter_count = {}
                        pa_tto = {}
                        for _, pa_row in pa_order.iterrows():
                            batter = pa_row["Batter"]
                            if batter not in batter_count:
                                batter_count[batter] = 0
                            batter_count[batter] += 1
                            pa_key = (pa_row["Inning"], pa_row["PAofInning"])
                            pa_tto[pa_key] = batter_count[batter]
                        for idx in pitch_data[game_mask].index:
                            row = pitch_data.loc[idx]
                            pa_key = (row["Inning"], row["PAofInning"])
                            pitch_data.loc[idx, "_TTO"] = pa_tto.get(pa_key, 1)

                    filtered = pitch_data.copy()
                    if pm_hand == "vs RHH":
                        filtered = filtered[filtered["BatterSide"] == "Right"]
                    elif pm_hand == "vs LHH":
                        filtered = filtered[filtered["BatterSide"] == "Left"]
                    if pm_tto == "1st Time":
                        filtered = filtered[filtered["_TTO"] == 1]
                    elif pm_tto == "2nd Time":
                        filtered = filtered[filtered["_TTO"] == 2]
                    elif pm_tto == "3rd Time":
                        filtered = filtered[filtered["_TTO"] == 3]
                    elif pm_tto == "4th+ Time":
                        filtered = filtered[filtered["_TTO"] >= 4]

                    if len(filtered) == 0:
                        st.warning("No pitches match these filters")
                    else:
                        N = len(filtered)
                        pts = filtered["PitchType"].value_counts().index.tolist()

                        def get_count_cat(balls, strikes):
                            cats = []
                            if (balls, strikes) in [(0, 0), (1, 0), (0, 1)]:
                                cats.append("early")
                            if strikes > balls and strikes >= 1:
                                cats.append("ahead")
                            if balls > strikes:
                                cats.append("behind")
                            if strikes < 2:
                                cats.append("pre2k")
                            if strikes == 2:
                                cats.append("twok")
                            return cats

                        filtered["_count_cats"] = filtered.apply(
                            lambda r: get_count_cat(
                                int(r["Balls"]) if pd.notna(r["Balls"]) else 0,
                                int(r["Strikes"]) if pd.notna(r["Strikes"]) else 0
                            ), axis=1)

                        situations = {
                            "All Counts": filtered,
                            "Early Count": filtered[filtered["_count_cats"].apply(lambda x: "early" in x)],
                            "Pitcher Ahead": filtered[filtered["_count_cats"].apply(lambda x: "ahead" in x)],
                            "Pitcher Behind": filtered[filtered["_count_cats"].apply(lambda x: "behind" in x)],
                            "Pre Two Strikes": filtered[filtered["_count_cats"].apply(lambda x: "pre2k" in x)],
                            "Two Strikes": filtered[filtered["_count_cats"].apply(lambda x: "twok" in x)],
                        }

                        table_data = []
                        all_counts_pcts = {}
                        for pt in pts:
                            all_pct = len(filtered[filtered["PitchType"] == pt]) / N * 100 if N > 0 else 0
                            all_counts_pcts[pt] = all_pct

                        for pt in pts:
                            row_d = {"Pitch Type": pt}
                            for sit_name, sit_df in situations.items():
                                sit_n = len(sit_df)
                                pct = len(sit_df[sit_df["PitchType"] == pt]) / sit_n * 100 if sit_n > 0 else 0
                                row_d[sit_name] = pct
                            table_data.append(row_d)

                        hand_label = pm_hand if pm_hand != "All" else "vs All"
                        tto_label = pm_tto if pm_tto != "All" else "All ABs"
                        st.markdown(f"### {pm_pitcher}")
                        st.caption(f"Pitch Mix {hand_label} · {tto_label} · {pm_from} to {pm_to} · {N} pitches")

                        fig, ax = plt.subplots(figsize=(14, max(2.5, 0.6 * len(pts) + 1.2)),
                                               facecolor="#1a1d23")
                        ax.axis("off")
                        col_labels = list(situations.keys())
                        cell_text = []
                        for rd in table_data:
                            cell_text.append([f"{rd[c]:.0f}%" for c in col_labels])

                        tbl = ax.table(
                            cellText=cell_text,
                            rowLabels=[r["Pitch Type"] for r in table_data],
                            colLabels=[c.upper().replace(" ", "\n") for c in col_labels],
                            loc="center", cellLoc="center"
                        )
                        tbl.auto_set_font_size(False)
                        tbl.set_fontsize(10)
                        tbl.scale(1, 2.0)

                        for (row, col), cell in tbl.get_celld().items():
                            cell.set_edgecolor("#2a2d35")
                            cell.set_linewidth(0.5)
                            if row == 0:
                                cell.set_facecolor("#2a2d35")
                                cell.set_text_props(fontweight="bold", color="#8890a0",
                                                    fontfamily="monospace", fontsize=8)
                            elif col == -1:
                                pt_name = cell.get_text().get_text()
                                cell.set_facecolor("#1a1d23")
                                cell.get_text().set_text(f"● {pt_name}")
                                cell.get_text().set_color(pc(pt_name))
                                cell.set_text_props(fontweight="bold", fontfamily="monospace", fontsize=10)
                            else:
                                cell.set_facecolor("#1e2128")
                                pt_name = table_data[row - 1]["Pitch Type"]
                                col_name = col_labels[col]
                                val = table_data[row - 1][col_name]
                                base = all_counts_pcts.get(pt_name, 0)
                                diff = val - base
                                if col_name != "All Counts" and abs(diff) >= 5:
                                    if diff >= 5:
                                        cell.set_facecolor("#1a3a2a")
                                        cell.set_text_props(color="#4ade80", fontweight="bold",
                                                            fontfamily="monospace", fontsize=10)
                                    else:
                                        cell.set_facecolor("#3a1a1a")
                                        cell.set_text_props(color="#f87171", fontweight="bold",
                                                            fontfamily="monospace", fontsize=10)
                                else:
                                    cell.set_text_props(color="white", fontfamily="monospace", fontsize=10)

                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#1a1d23")
                        buf.seek(0)
                        st.download_button("📥 Download Pitch Mix", data=buf,
                                           file_name=f"PitchMix_{pm_pitcher}_{pm_hand}_{pm_tto}.png",
                                           mime="image/png", use_container_width=True)
                        plt.close(fig)

                        with st.expander("📋 Sample sizes per situation"):
                            size_data = {name: len(sdf) for name, sdf in situations.items()}
                            st.write(size_data)
else:
    st.info("Select a team above to use Pitch Mix")
