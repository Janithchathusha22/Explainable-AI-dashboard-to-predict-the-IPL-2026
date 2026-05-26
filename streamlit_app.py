from pathlib import Path

import pandas as pd
import streamlit as st

REPORTS_DIR = Path("reports")

PLAYOFF_SCHEDULE = pd.DataFrame(
    [
        {
            "තරඟය (Match)": "Qualifier 1",
            "මුහුණ දෙන කණ්ඩායම්": "RCB vs GT",
            "දිනය (Date)": "මැයි 26 (අඟහරුවාදා)",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "HPCA Stadium, Dharamshala",
        },
        {
            "තරඟය (Match)": "Eliminator",
            "මුහුණ දෙන කණ්ඩායම්": "SRH vs RR",
            "දිනය (Date)": "මැයි 27 (බදාදා)",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "Maharaja Yadavindra Singh Stadium, New Chandigarh",
        },
        {
            "තරඟය (Match)": "Qualifier 2",
            "මුහුණ දෙන කණ්ඩායම්": "Q1 පරාජිතයා vs Eliminator ජයග්‍රාහකයා",
            "දිනය (Date)": "මැයි 29 (සිකුරාදා)",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "Maharaja Yadavindra Singh Stadium, New Chandigarh",
        },
        {
            "තරඟය (Match)": "Grand Final",
            "මුහුණ දෙන කණ්ඩායම්": "Q1 ජයග්‍රාහකයා vs Q2 ජයග්‍රාහකයා",
            "දිනය (Date)": "මැයි 31 (ඉරිදා)",
            "වේලාව (Time - IST)": "7:30 PM",
            "ක්‍රීඩාංගණය (Venue)": "Narendra Modi Stadium, Ahmedabad",
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


st.set_page_config(page_title="IPL 2026 Final XAI Prediction", layout="wide")
st.title("IPL 2026 Final Explainable AI Prediction")
st.caption("GT vs SRH | Final: May 31, 2026 | Prediction as of May 26, 2026 | Stacked ML ensemble + SHAP + Monte Carlo")

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
weather_input = read_project_csv("ipl_playoffs_live_weather.csv")
right_left_input = read_project_csv("right_left.csv")

col1, col2, col3, col4 = st.columns(4)
if not model_probs.empty:
    stacked = model_probs[model_probs["model"] == "Stacked Ensemble"].iloc[0]
    col1.metric("GT win probability", f"{stacked['gt_win_probability'] * 100:.1f}%")
    col2.metric("SRH win probability", f"{stacked['srh_win_probability'] * 100:.1f}%")
if not players.empty:
    top = players.iloc[0]
    col3.metric("Most likely POTM", f"{top['Player_Name']} ({top['Team']})", f"{top['POTM_Probability']:.1f}%")
if not cup_probs.empty:
    cup_top = cup_probs.iloc[0]
    col4.metric("Top cup odds", cup_top["team"], f"{cup_top['cup_probability'] * 100:.1f}%")

tab_report, tab_playoffs, tab_xai, tab_models, tab_players, tab_scenarios, tab_inputs = st.tabs(
    ["Report", "Playoffs", "SHAP XAI", "Models", "Players", "Scenarios", "Inputs"]
)

with tab_report:
    st.markdown(report_path.read_text(encoding="utf-8"))

with tab_playoffs:
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
        st.dataframe(model_probs, use_container_width=True)
        st.bar_chart(model_probs.set_index("model")[["gt_win_probability", "srh_win_probability"]])

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
    st.subheader("Live weather input")
    if not weather_input.empty:
        st.dataframe(weather_input, use_container_width=True)

    st.subheader("Right/left matchup input")
    if not right_left_input.empty:
        st.dataframe(right_left_input, use_container_width=True)
