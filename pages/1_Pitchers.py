"""
Pitchers page — TrackMan Baseball Reports
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

st.set_page_config(page_title="Pitchers — TrackMan", layout="wide", page_icon="⚾")
inject_app_styles()
render_page_intro(
    "⚾ Pitcher Reports",
    "All existing report tools are still here—this refresh only changes the presentation so the page is easier to scan, easier to select from, and easier to use during review sessions.",
    chips=["Outing PDFs", "Season summaries", "Pitch heatmaps", "Pitch mix analysis"],
    eyebrow="Pitching view",
)

# ── Load index (tiny — just team names and dates) ──
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

# ── Team coverage overview ──
games = (team_df.groupby(["GameDate", "HomeTeam", "AwayTeam"])
         .size().reset_index().sort_values("GameDate"))

section_title(f"{team_name} coverage", "The cards below show how much pitching data is available for the current team and date range.")
ov1, ov2, ov3, ov4, ov5 = st.columns(5)
with ov1:
    render_stat_card("Games", len(games), f"{date_from} to {date_to}")
with ov2:
    render_stat_card("Pitchers", team_df["Pitcher"].dropna().nunique(), "Unique pitchers in range")
with ov3:
    render_stat_card("Pitches", f"{len(team_df):,}", "All pitches thrown by the selected team")
with ov4:
    strike_rate = team_df["PitchCall"].isin(STRIKE_CALLS).mean() * 100 if len(team_df) else np.nan
    render_stat_card("Strike Rate", format_number(strike_rate, 1, "%"), "Team-level strike percentage")
with ov5:
    start_lbl = str(games["GameDate"].min()) if not games.empty else "—"
    end_lbl = str(games["GameDate"].max()) if not games.empty else "—"
    render_stat_card("Window", f"{start_lbl} → {end_lbl}", "First and last tracked game")

cover_left, cover_right = st.columns([1.6, 1.0])
with cover_left:
    games_display = games.copy()
    games_display["Matchup"] = games_display["HomeTeam"] + " vs " + games_display["AwayTeam"]
    games_display["GameDate"] = games_display["GameDate"].astype(str)
    with st.expander("Game list for this range", expanded=False):
        st.dataframe(
            games_display[["GameDate", "Matchup", 0]].rename(columns={0: "Pitches"}),
            use_container_width=True,
            hide_index=True,
        )
with cover_right:
    render_info_panel("What changed visually", [
        "High-level team coverage is shown before you make any pitcher selections.",
        "Selection controls stay the same, but the page now gives more context up front.",
        "Reports, summaries, heatmaps, and pitch mix keep the same underlying behavior.",
    ])

# ── Build pitcher list from parquet ──
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
    section_title("Choose pitchers", "The multiselect and reports work exactly as before—this section is just organized to surface useful context faster.")

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

    # BUG FIX: only one multiselect — inside picker_col
    picker_col, helper_col = st.columns([1.55, 1.0])
    with picker_col:
        selected_labels = st.multiselect(
            "Select Pitcher(s)",
            outing_labels,
            default=None,
            placeholder="Choose one or more pitchers to unlock reports",
        )
    label_to_name = dict(zip(outing_labels, st.session_state["pitcher_names"]))
    selected_names = [label_to_name[lbl] for lbl in selected_labels]

    with helper_col:
        render_info_panel("Selection tips", [
            "Pick 1 pitcher for the cleanest heatmap experience.",
            "Pick multiple pitchers when you want multiple outing PDFs or season summaries at once.",
            f"{len(st.session_state['pitcher_names'])} pitchers are available in this date range.",
        ])

    if selected_names:
        sel1, sel2, sel3, sel4 = st.columns(4)
        with sel1:
            render_stat_card("Selected", len(selected_names), "Pitchers currently active")
        with sel2:
            total_outings = sum(len(pitcher_meta.get(name, [])) for name in selected_names)
            render_stat_card("Outings", total_outings, "Total outings across your selection")
        with sel3:
            selected_pitch_count = len(team_df[team_df["Pitcher"].isin(selected_names)])
            render_stat_card("Pitches", f"{selected_pitch_count:,}", "Pitches available for reports")
        with sel4:
            selected_games = team_df[team_df["Pitcher"].isin(selected_names)]["GameDate"].nunique()
            render_stat_card("Games", selected_games, "Games represented by selected pitchers")

        tab_reports, tab_summary, tab_heatmaps, tab_debug = st.tabs(
            ["📄 Game Reports", "📊 Season Summary", "🔥 Heatmaps", "🔧 Debug"])

        # ========== TAB 1: GAME REPORTS ==========
        with tab_reports:
            split_a, split_b = st.columns([1.25, 1.0])
            with split_a:
                st.caption("Game reports generate one page per outing for every selected pitcher in the current date range.")
            with split_b:
                render_info_panel("Use this tab for", [
                    "Series prep or postgame review.",
                    "Quick comparison between multiple outings.",
                    "Downloading one combined PDF for staff review.",
                ])
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
            split_a, split_b = st.columns([1.25, 1.0])
            with split_a:
                st.caption("Season summaries still ignore the sidebar date filter and use the full available season for each selected pitcher.")
            with split_b:
                render_info_panel("Use this tab for", [
                    "Big-picture pitch usage and shape review.",
                    "Checking full-season performance trends.",
                    "Exporting a polished one-page season summary.",
                ])
            if st.button("📊 Generate Season Summaries", type="primary",
                         use_container_width=True, key="btn_summary"):
                figures = []
                with st.spinner("Building season summaries from local data..."):
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
            split_a, split_b = st.columns([1.25, 1.0])
            with split_a:
                st.caption("Heatmaps work best with one pitcher at a time so the visual remains clean and easy to interpret.")
            with split_b:
                render_info_panel("Use this tab for", [
                    "Pitch-location tendencies by pitch type.",
                    "Run value and whiff visual review.",
                    "Saving a single heatmap image after generation.",
                ])
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
                        hm_fig = generate_heatmap(
                            hm_data, hm_pitcher, hm_pitch_type,
                            metric=metric_map[hm_metric])
                        if hm_fig:
                            st.pyplot(hm_fig, use_container_width=True)
                            buf = io.BytesIO()
                            hm_fig.savefig(buf, format="png", bbox_inches="tight",
                                           dpi=150, facecolor=BG_COLOR)
                            buf.seek(0)
                            st.download_button(
                                "📥 Download Heatmap",
                                data=buf,
                                file_name=f"Heatmap_{hm_pitcher}_{hm_pitch_type}_{hm_metric}.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                    else:
                        st.info("Not enough pitches (< 5) for any pitch type to generate a heatmap.")

        # ========== TAB 4: DEBUG ==========
        with tab_debug:
            st.caption("Raw PA-level breakdown for verifying IP, ER, K, BB calculations.")
            load_full_pitcher_data(selected_names)
            for pname in selected_names:
                if pname not in st.session_state.get("pitcher_outings", {}):
                    continue
                st.subheader(pname)
                for p_df, gdate, opp in st.session_state["pitcher_outings"][pname]:
                    st.write(f"**{gdate} vs {opp}**")
                    if p_df.empty:
                        st.write("No data"); continue
                    debug_outs = 0
                    pa_rows = []
                    for (inn, pa_num), grp in p_df.groupby(["Inning", "PAofInning"]):
                        last = grp.iloc[-1]
                        korbb = last.get("KorBB", "")
                        result = last.get("PlayResult", "")
                        oop = last.get("OutsOnPlay", 0)
                        try:
                            oop = int(float(oop)) if oop not in (None, "", "nan") else 0
                        except Exception:
                            oop = 0
                        out_src = ""
                        if korbb == "Strikeout":
                            debug_outs += 1; out_src = "+1 (K)"
                        elif oop > 0:
                            debug_outs += oop
                            out_src = f"+{oop} (play)"
                        elif result in ("Out", "FieldersChoice", "Error"):
                            debug_outs += 1; out_src = "+1 (K)"
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

# ── Pitch Mix Analysis ──
st.divider()
section_title("🎯 Pitch Mix Analysis", "Same functionality as before, but visually grouped as a separate workflow for sequencing review.")

if "selected_team" in st.session_state:
    pm_team = st.session_state["selected_team"]

    pm_col1, pm_col2 = st.columns(2)
    with pm_col1:
        pm_from = st.date_input("From", value=date(2026, 1, 1), key="pm_from")
    with pm_col2:
        pm_to = st.date_input("To", value=date.today(), key="pm_to")

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
                        key = pn
                        if key not in pm_all_outings:
                            pm_all_outings[key] = []
                        pm_all_outings[key].append((g_df.reset_index(drop=True), gdate, opp))
                st.session_state["pm_outings"] = pm_all_outings
                st.session_state["pm_pitcher_list"] = sorted(pm_all_outings.keys())

    if "pm_pitcher_list" in st.session_state and st.session_state["pm_pitcher_list"]:
        pm_selected = st.selectbox("Select Pitcher", st.session_state["pm_pitcher_list"], key="pm_pitcher_select")
        if pm_selected and pm_selected in st.session_state.get("pm_outings", {}):
            pm_outings = st.session_state["pm_outings"][pm_selected]
            pm_outing_labels = [f"{gdate} vs {opp}" for _, gdate, opp in pm_outings]
            pm_sel_label = st.selectbox("Select Outing", pm_outing_labels, key="pm_outing_select")
            pm_sel_idx = pm_outing_labels.index(pm_sel_label)
            pm_outing_df, _, _ = pm_outings[pm_sel_idx]

            if not pm_outing_df.empty:
                pm_fig = generate_pitch_mix(pm_outing_df, pm_selected)
                if pm_fig:
                    st.pyplot(pm_fig, use_container_width=True)
                    buf = io.BytesIO()
                    pm_fig.savefig(buf, format="png", bbox_inches="tight",
                                   dpi=150, facecolor=BG_COLOR)
                    buf.seek(0)
                    st.download_button("📥 Download Pitch Mix PNG", data=buf,
                                       file_name=f"PitchMix_{pm_selected}_{pm_sel_label}.png",
                                       mime="image/png", use_container_width=True)
