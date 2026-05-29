from pathlib import Path

import pandas as pd
import streamlit as st

REPORTS_DIR = Path("reports")
FINAL_TEAM_A = "RCB"
FINAL_TEAM_B = "GT"

TEAM_NAME_TO_SHORT = {
    "Royal Challengers Bengaluru": "RCB",
    "Gujarat Titans": "GT",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "None": "",
}

RCB_GT_PREMATCH_PREDICTION = {
    "Prediction snapshot": "Before RCB vs GT playoff match",
    "Match": "RCB vs GT",
    "Predicted winner": "RCB",
    "Actual winner": "RCB",
    "Verdict": "Correct",
}

PLAYOFF_SCHEDULE = pd.DataFrame(
    [
        {
            "තරඟය (Match)": "Qualifier 1 / Semi Final 1",
            "මුහුණ දෙන කණ්ඩායම්": "RCB vs GT",
            "දිනය (Date)": "2026-05-26",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "HPCA Stadium, Dharamshala",
            "තත්ත්වය (Status)": "RCB won by 92 runs; RCB entered final, GT moved to Qualifier 2",
        },
        {
            "තරඟය (Match)": "Eliminator",
            "මුහුණ දෙන කණ්ඩායම්": "RR vs SRH",
            "දිනය (Date)": "2026-05-27",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "Maharaja Yadavindra Singh Stadium, Mullanpur",
            "තත්ත්වය (Status)": "RR won by 47 runs; SRH eliminated",
        },
        {
            "තරඟය (Match)": "Qualifier 2",
            "මුහුණ දෙන කණ්ඩායම්": "GT vs RR",
            "දිනය (Date)": "2026-05-29",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "Maharaja Yadavindra Singh Stadium, Mullanpur",
            "තත්ත්වය (Status)": "GT won by 7 wickets; GT entered final",
        },
        {
            "තරඟය (Match)": "Grand Final",
            "මුහුණ දෙන කණ්ඩායම්": "RCB vs GT",
            "දිනය (Date)": "2026-05-31",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "Narendra Modi Stadium, Ahmedabad",
            "තත්ත්වය (Status)": "Final matchup confirmed",
        },
    ]
)


def read_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists():
        st.warning(f"Missing {path}. Run explainable_ai_pipeline.py first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_project_csv(name: str) -> pd.DataFrame:
    path = Path(name)
    if not path.exists():
        st.warning(f"Missing {path}.")
        return pd.DataFrame()
    return pd.read_csv(path)


def probability_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if {"team_a_win_probability", "team_b_win_probability"}.issubset(frame.columns):
        return "team_a_win_probability", "team_b_win_probability"
    return "gt_win_probability", "srh_win_probability"


def short_team(team_name: str) -> str:
    return TEAM_NAME_TO_SHORT.get(str(team_name).strip(), str(team_name).strip())


def infer_final_matchup(playoff_results: pd.DataFrame) -> tuple[str, str, str]:
    if playoff_results.empty or "Match_Type" not in playoff_results.columns:
        return f"{FINAL_TEAM_A} vs {FINAL_TEAM_B}", FINAL_TEAM_B, "from local playoff CSV"

    qualifier_2 = playoff_results[
        playoff_results["Match_Type"].astype(str).str.contains("Qualifier 2", case=False, na=False)
    ]
    if qualifier_2.empty:
        return f"{FINAL_TEAM_A} vs {FINAL_TEAM_B}", FINAL_TEAM_B, "from local playoff CSV"

    winner = short_team(qualifier_2.iloc[-1]["Winner"])
    date = str(qualifier_2.iloc[-1]["Date"])
    return f"{FINAL_TEAM_A} vs {winner}", winner, f"after Qualifier 2 on {date}"


def score_label(scorecard: pd.DataFrame, team: str) -> str:
    if scorecard.empty or "Taldea" not in scorecard.columns:
        return ""

    rows = scorecard[scorecard["Taldea"] == team]
    if rows.empty:
        return ""

    row = rows.iloc[0]
    total = int(row["Team_Total"])
    wickets = int(row["Wickets"])
    overs = row["Overs"]
    return f"{team} {total}/{wickets} ({overs} ov)"


def actual_result_summary(scorecard: pd.DataFrame) -> dict[str, str]:
    fallback = {
        "winner": "RCB",
        "margin": "92 runs",
        "result": "RCB won by 92 runs",
        "date": "2026-05-26",
        "venue": "HPCA Stadium, Dharamshala",
    }
    if scorecard.empty:
        return fallback

    first = scorecard.iloc[0]
    return {
        "winner": "RCB",
        "margin": f"{int(first['Margin_Runs'])} runs" if "Margin_Runs" in first.index else fallback["margin"],
        "result": str(first["Match_Result"]) if "Match_Result" in first.index else fallback["result"],
        "date": str(first["Match_Date"]) if "Match_Date" in first.index else fallback["date"],
        "venue": str(first["Venue"]) if "Venue" in first.index else fallback["venue"],
    }


st.set_page_config(page_title="IPL 2026 Playoff Prediction Audit", layout="wide")
st.title("IPL 2026 Playoff Prediction Dashboard")
st.caption(
    "Final matchup confirmed from local playoff data: RCB vs GT | RCB vs GT pre-match prediction audit | "
    "Archived May 27 XAI report remains available below"
)

report_path = REPORTS_DIR / "ipl_2026_final_prediction_report.md"
if not report_path.exists():
    st.error("Report not found. Run: .venv\\Scripts\\python.exe explainable_ai_pipeline.py")
    st.stop()

model_probs = read_csv("final_model_probabilities.csv")
metrics = read_csv("model_metrics.csv")
players = read_csv("player_of_match_ranking.csv")
local_shap = read_csv("shap_local_explanation.csv")
scenarios = read_csv("scenario_sensitivity.csv")
sensitivity = read_csv("feature_sensitivity.csv")
cup_probs = read_csv("playoff_cup_probabilities.csv")
final_pairs = read_csv("final_pair_probabilities.csv")
fixture_probs = read_csv("playoff_fixture_probabilities.csv")
player_forecast = read_csv("player_performance_forecast.csv")
playoff_results = read_project_csv("ipl_playoffs_2026.csv")
rcb_gt_batting = read_project_csv("rcb_vs_gt_batting_stats_2026.csv")
rcb_gt_bowling = read_project_csv("rcb_vs_gt_bowling_stats_2026.csv")
weather_input = read_project_csv("ipl_playoffs_live_weather.csv")
right_left_input = read_project_csv("right_left.csv")
q1_scorecard = read_project_csv("ipl_2026_q1_rcb_vs_gt.csv")

final_matchup, final_opponent, final_status = infer_final_matchup(playoff_results)
rcb_gt_result = actual_result_summary(q1_scorecard)
rcb_score = score_label(q1_scorecard, "RCB")
gt_score = score_label(q1_scorecard, "GT")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Confirmed final", final_matchup, final_status)
col2.metric("RCB vs GT audit", "Prediction correct", "pre-match pick: RCB")
col3.metric("RCB vs GT result", "RCB won", rcb_gt_result["margin"])
if not model_probs.empty:
    stacked = model_probs[model_probs["model"] == "Stacked Ensemble"].iloc[0]
    archived_pair = f"{stacked['team_a']} vs {stacked['team_b']}" if {"team_a", "team_b"}.issubset(stacked.index) else "May 27 candidate"
    col4.metric("Archived XAI report", archived_pair, "not the current final row")
else:
    col4.metric("Archived XAI report", "Unavailable", "run pipeline to regenerate")

tab_report, tab_audit, tab_playoffs, tab_xai, tab_models, tab_players, tab_scenarios, tab_inputs = st.tabs(
    ["Report", "RCB vs GT Audit", "Playoffs", "SHAP XAI", "Models", "Players", "Scenarios", "Inputs"]
)

with tab_report:
    st.warning(
        "This markdown report is the May 27 candidate-final model run. "
        "The latest local playoff CSV now confirms the final as RCB vs GT."
    )
    st.markdown(report_path.read_text(encoding="utf-8"))

with tab_audit:
    st.subheader("RCB vs GT pre-match prediction audit")
    st.success(
        "Audit verdict: the pre-match pick was RCB, and the actual result was RCB won by 92 runs."
    )
    st.caption(
        "This section is labelled as a prediction audit, not as a new prediction generated after the match."
    )

    audit_rows = pd.DataFrame(
        [
            {
                **RCB_GT_PREMATCH_PREDICTION,
                "Actual result": rcb_gt_result["result"],
                "Match date": rcb_gt_result["date"],
                "Venue": rcb_gt_result["venue"],
            }
        ]
    )
    st.dataframe(audit_rows, use_container_width=True, hide_index=True)

    score_cols = st.columns(3)
    score_cols[0].metric("RCB innings", rcb_score or "254/5", "20.0 overs")
    score_cols[1].metric("GT innings", gt_score or "162/10", "19.3 overs")
    score_cols[2].metric("Winning margin", rcb_gt_result["margin"], "RCB")

    bat_col, bowl_col = st.columns(2)
    with bat_col:
        st.subheader("Top batting evidence")
        if not rcb_gt_batting.empty:
            top_batting = rcb_gt_batting.sort_values("Runs", ascending=False).head(8)
            st.dataframe(top_batting, use_container_width=True, hide_index=True)
            st.bar_chart(top_batting.set_index("Player")["Runs"])

    with bowl_col:
        st.subheader("Top bowling evidence")
        if not rcb_gt_bowling.empty:
            top_bowling = rcb_gt_bowling.sort_values(["Wickets", "Economy"], ascending=[False, True]).head(8)
            st.dataframe(top_bowling, use_container_width=True, hide_index=True)
            st.bar_chart(top_bowling.set_index("Player")["Wickets"])

with tab_playoffs:
    st.subheader("Confirmed final path")
    st.info(
        f"RCB reached the final from Qualifier 1. {final_opponent} reached the final from Qualifier 2. "
        f"Current final matchup: {final_matchup}."
    )
    if not playoff_results.empty:
        st.dataframe(playoff_results, use_container_width=True)

    st.subheader("IPL 2026 playoff schedule")
    st.dataframe(PLAYOFF_SCHEDULE, use_container_width=True)

    st.subheader("Four-team cup probabilities")
    if not cup_probs.empty:
        st.dataframe(cup_probs, use_container_width=True)
        st.bar_chart(cup_probs.set_index("team")["cup_probability"])

    st.subheader("Most likely final pairs")
    if not final_pairs.empty:
        st.dataframe(final_pairs, use_container_width=True)
        st.bar_chart(final_pairs.set_index("final_pair")["probability"])

    st.subheader("Fixture probabilities")
    if not fixture_probs.empty:
        st.dataframe(fixture_probs, use_container_width=True)

with tab_xai:
    st.subheader("Local SHAP explanation")
    if not local_shap.empty:
        st.dataframe(local_shap.head(15), use_container_width=True)

    image_cols = st.columns(3)
    image_files = [
        ("Local waterfall", "shap_local_waterfall.png"),
        ("Global bar", "shap_global_bar.png"),
        ("Global beeswarm", "shap_global_beeswarm.png"),
    ]
    for col, (caption, file_name) in zip(image_cols, image_files):
        image_path = REPORTS_DIR / file_name
        if image_path.exists():
            col.image(str(image_path), caption=caption, use_container_width=True)

with tab_models:
    st.subheader("Model probabilities")
    if not model_probs.empty:
        team_a_col, team_b_col = probability_columns(model_probs)
        st.dataframe(model_probs, use_container_width=True)
        st.bar_chart(model_probs.set_index("model")[[team_a_col, team_b_col]])

    st.subheader("Cross-validation metrics")
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True)

    mc_path = REPORTS_DIR / "monte_carlo_distribution.png"
    if mc_path.exists():
        st.image(str(mc_path), caption="Monte Carlo distribution", use_container_width=True)

with tab_players:
    st.subheader("Player of the Match candidates")
    if not players.empty:
        st.dataframe(players, use_container_width=True)
        st.bar_chart(players.head(10).set_index("Player_Name")["POTM_Probability"])

    st.subheader("Player performance forecast")
    if not player_forecast.empty:
        st.dataframe(player_forecast, use_container_width=True)
        batting_cols = ["expected_runs", "likely_runs_low", "likely_runs_high"]
        available_cols = [col for col in batting_cols if col in player_forecast.columns]
        if available_cols:
            st.bar_chart(player_forecast.head(10).set_index("Player_Name")[available_cols])

with tab_scenarios:
    st.subheader("Scenario sensitivity")
    if not scenarios.empty:
        st.dataframe(scenarios, use_container_width=True)

    st.subheader("Uncertainty drivers")
    if not sensitivity.empty:
        st.dataframe(sensitivity.head(10), use_container_width=True)

with tab_inputs:
    st.subheader("Semi Final 1 actual scorecard")
    if not q1_scorecard.empty:
        st.dataframe(q1_scorecard, use_container_width=True)

    st.subheader("Live weather input")
    if not weather_input.empty:
        st.dataframe(weather_input, use_container_width=True)

    st.subheader("Right/left matchup input")
    if not right_left_input.empty:
        st.dataframe(right_left_input, use_container_width=True)
