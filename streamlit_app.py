import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ASSETS_DIR = Path("assets")
PLAYOFF_HERO_IMAGE = ASSETS_DIR / "IPL_2026_Playoffs_dashboard_202605270234.jpeg"
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


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def final_model_summary(model_probs: pd.DataFrame) -> dict[str, object]:
    fallback = {
        "pair": "Unavailable",
        "winner": "No model",
        "probability": "",
        "team_a": "RCB",
        "team_b": "GT",
        "team_a_probability": np.nan,
        "team_b_probability": np.nan,
    }
    if model_probs.empty or "model" not in model_probs.columns:
        return fallback

    stacked_rows = model_probs[model_probs["model"] == "Stacked Ensemble"]
    if stacked_rows.empty:
        return fallback

    stacked = stacked_rows.iloc[0]
    team_a_col, team_b_col = probability_columns(model_probs)
    team_a = str(stacked["team_a"]) if "team_a" in stacked.index else FINAL_TEAM_A
    team_b = str(stacked["team_b"]) if "team_b" in stacked.index else FINAL_TEAM_B
    team_a_probability = float(stacked[team_a_col])
    team_b_probability = float(stacked[team_b_col])

    if team_a_probability >= team_b_probability:
        winner = team_a
        probability = team_a_probability
    else:
        winner = team_b
        probability = team_b_probability

    return {
        "pair": f"{team_a} vs {team_b}",
        "winner": winner,
        "probability": f"{probability * 100:.1f}%",
        "team_a": team_a,
        "team_b": team_b,
        "team_a_probability": team_a_probability,
        "team_b_probability": team_b_probability,
    }


def confidence_label(summary: dict[str, object]) -> str:
    team_a_probability = summary.get("team_a_probability", np.nan)
    if pd.isna(team_a_probability):
        return "Unavailable"

    margin = abs(float(team_a_probability) - 0.5) * 100
    if margin < 7:
        return "LOW"
    if margin < 12:
        return "MEDIUM"
    return "HIGH"


def render_hero(final_matchup: str, final_status: str, prediction: dict[str, object]) -> None:
    hero_uri = image_data_uri(PLAYOFF_HERO_IMAGE)
    image_layer = f"url('{hero_uri}')" if hero_uri else "none"
    st.markdown(
        f"""
        <style>
            .block-container {{
                max-width: 1240px;
                padding-top: 1.4rem;
            }}

            .stApp {{
                background:
                    linear-gradient(180deg, rgba(4, 12, 24, 0.98) 0%, rgba(6, 18, 35, 0.98) 48%, rgba(8, 14, 25, 1) 100%);
                color: #eaf7ff;
            }}

            [data-testid="stHeader"] {{
                background: rgba(4, 12, 24, 0.86);
                backdrop-filter: blur(8px);
            }}

            .playoff-hero {{
                min-height: min(440px, 62vh);
                border: 1px solid rgba(103, 232, 249, 0.28);
                border-radius: 8px;
                background-image:
                    linear-gradient(90deg, rgba(2, 8, 20, 0.94) 0%, rgba(2, 8, 20, 0.66) 46%, rgba(2, 8, 20, 0.24) 100%),
                    linear-gradient(180deg, rgba(2, 8, 20, 0.22), rgba(2, 8, 20, 0.92)),
                    {image_layer};
                background-position: center;
                background-size: cover;
                box-shadow: 0 28px 80px rgba(0, 0, 0, 0.42);
                display: flex;
                align-items: flex-end;
                margin-bottom: 1rem;
                overflow: hidden;
                padding: clamp(1.1rem, 3vw, 2.4rem);
                animation: heroRise 0.8s ease-out both;
            }}

            .hero-copy {{
                max-width: 760px;
            }}

            .hero-kicker {{
                color: #7dd3fc;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0;
                margin-bottom: 0.55rem;
                text-transform: uppercase;
            }}

            .hero-title {{
                color: #f8fbff;
                font-size: clamp(2.2rem, 5.5vw, 4.8rem);
                font-weight: 900;
                letter-spacing: 0;
                line-height: 0.95;
                margin: 0;
                text-shadow: 0 12px 34px rgba(0, 0, 0, 0.72);
            }}

            .hero-subtitle {{
                color: #d8ecff;
                font-size: clamp(1rem, 1.8vw, 1.28rem);
                line-height: 1.55;
                margin: 1rem 0 1.1rem;
                max-width: 690px;
            }}

            .hero-strip {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
            }}

            .hero-chip {{
                border: 1px solid rgba(125, 211, 252, 0.4);
                border-radius: 999px;
                color: #e9fbff;
                background: rgba(3, 13, 28, 0.72);
                font-size: 0.86rem;
                font-weight: 700;
                padding: 0.52rem 0.78rem;
                animation: chipIn 0.5s ease-out both;
            }}

            .hero-chip strong {{
                color: #67e8f9;
            }}

            div[data-testid="stMetric"] {{
                background: rgba(7, 20, 38, 0.72);
                border: 1px solid rgba(125, 211, 252, 0.18);
                border-radius: 8px;
                padding: 0.95rem 1rem;
                animation: cardIn 0.55s ease-out both;
                transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            }}

            div[data-testid="stMetric"]:hover {{
                transform: translateY(-2px);
                border-color: rgba(125, 211, 252, 0.42);
                box-shadow: 0 10px 26px rgba(3, 13, 28, 0.45);
            }}

            div[data-testid="stMetric"] label {{
                color: rgba(226, 245, 255, 0.82) !important;
            }}

            div[data-testid="stMetricValue"] {{
                color: #ffffff;
                font-weight: 800;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.25rem;
                animation: cardIn 0.65s ease-out both;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px;
                color: #c7e5f7;
                padding: 0.6rem 0.82rem;
            }}

            .stTabs [aria-selected="true"] {{
                background: rgba(34, 211, 238, 0.14);
                color: #ffffff;
            }}

            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {{
                border: 1px solid rgba(125, 211, 252, 0.16);
                border-radius: 8px;
                overflow: hidden;
                animation: cardIn 0.65s ease-out both;
            }}

            .prob-shell {{
                border: 1px solid rgba(125, 211, 252, 0.25);
                border-radius: 8px;
                background: rgba(3, 13, 28, 0.64);
                padding: 0.9rem;
                margin: 0.3rem 0 1rem;
                animation: cardIn 0.75s ease-out both;
            }}

            .prob-head {{
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                margin-bottom: 0.5rem;
                color: #e9fbff;
                font-weight: 700;
                font-size: 0.9rem;
            }}

            .prob-track {{
                height: 16px;
                background: rgba(10, 26, 46, 0.9);
                border-radius: 999px;
                overflow: hidden;
                position: relative;
                border: 1px solid rgba(125, 211, 252, 0.2);
            }}

            .prob-rcb {{
                position: absolute;
                inset: 0 auto 0 0;
                background: linear-gradient(90deg, #17d6ea 0%, #2ba5f7 100%);
                box-shadow: 0 0 18px rgba(43, 165, 247, 0.35);
                animation: widthGrow 1s ease-out both;
            }}

            .prob-shimmer {{
                position: absolute;
                inset: 0;
                background: linear-gradient(110deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0.22) 45%, rgba(255, 255, 255, 0) 90%);
                transform: translateX(-120%);
                animation: shimmer 2.5s linear infinite;
                pointer-events: none;
            }}

            @keyframes heroRise {{
                from {{
                    opacity: 0;
                    transform: translateY(16px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            @keyframes chipIn {{
                from {{
                    opacity: 0;
                    transform: translateY(8px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            @keyframes cardIn {{
                from {{
                    opacity: 0;
                    transform: translateY(10px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            @keyframes shimmer {{
                to {{
                    transform: translateX(120%);
                }}
            }}

            @keyframes widthGrow {{
                from {{
                    width: 0;
                }}
            }}
        </style>

        <section class="playoff-hero">
            <div class="hero-copy">
                <div class="hero-kicker">XAI Analytics - IPL 2026</div>
                <h1 class="hero-title">Playoff Prediction Lab</h1>
                <div class="hero-subtitle">
                    Confirmed final is {final_matchup}. The stacked model predicts
                    {prediction["winner"]} at {prediction["probability"]}, with a low-confidence statistical edge.
                </div>
                <div class="hero-strip">
                    <div class="hero-chip"><strong>Final</strong> {final_matchup}</div>
                    <div class="hero-chip"><strong>Status</strong> {final_status}</div>
                    <div class="hero-chip"><strong>Pick</strong> {prediction["winner"]}</div>
                    <div class="hero-chip"><strong>Model</strong> Stacked ensemble</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_probability_meter(rcb_probability: float, gt_probability: float) -> None:
    rcb_pct = max(0.0, min(100.0, rcb_probability * 100.0))
    gt_pct = max(0.0, min(100.0, gt_probability * 100.0))
    st.markdown(
        f"""
        <div class="prob-shell">
            <div class="prob-head">
                <span>Final Win Split</span>
                <span>RCB {rcb_pct:.2f}% | GT {gt_pct:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-rcb" style="width:{rcb_pct:.2f}%"></div>
                <div class="prob-shimmer"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="IPL 2026 Playoff Prediction Audit", layout="wide")

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
prediction = final_model_summary(model_probs)
confidence = confidence_label(prediction)

final_fixture = pd.DataFrame()
if not fixture_probs.empty and "match_type" in fixture_probs.columns:
    final_mask = fixture_probs["match_type"].astype(str).str.strip().str.lower() == "final"
    final_fixture = fixture_probs[final_mask]
weighted_rcb = float(final_fixture.iloc[0]["team1_win_probability"]) if not final_fixture.empty else prediction["team_a_probability"]
weighted_gt = float(final_fixture.iloc[0]["team2_win_probability"]) if not final_fixture.empty else prediction["team_b_probability"]

render_hero(final_matchup, final_status, prediction)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Confirmed final", final_matchup, final_status)
col2.metric("Predicted winner", prediction["winner"], prediction["probability"])
col3.metric("RCB win chance", f"{float(prediction['team_a_probability']) * 100:.1f}%")
col4.metric("GT win chance", f"{float(prediction['team_b_probability']) * 100:.1f}%", f"{confidence} confidence")

tab_prediction, tab_evidence, tab_path, tab_model = st.tabs(
    ["Prediction", "Key Evidence", "Playoff Path", "Model Detail"]
)

with tab_prediction:
    st.subheader("RCB vs GT final winner prediction")
    st.success(
        f"Statistical pick: {prediction['winner']} at {prediction['probability']} from the stacked ensemble."
    )
    st.caption(
        "This is a low-confidence edge, not a lock. The model is using known playoff results through "
        "Qualifier 2, season/team stats, venue/weather inputs, player impact, SHAP, and Monte Carlo uncertainty."
    )

    summary_rows = pd.DataFrame(
        [
            {
                "Signal": "Stacked ensemble",
                "RCB": f"{float(prediction['team_a_probability']) * 100:.2f}%",
                "GT": f"{float(prediction['team_b_probability']) * 100:.2f}%",
                "Leader": prediction["winner"],
            },
            {
                "Signal": "Toss-weighted final scenarios",
                "RCB": f"{weighted_rcb * 100:.2f}%",
                "GT": f"{weighted_gt * 100:.2f}%",
                "Leader": "RCB" if weighted_rcb >= weighted_gt else "GT",
            },
            {
                "Signal": "Qualifier 1 audit",
                "RCB": "Won by 92 runs",
                "GT": "Lost",
                "Leader": "RCB",
            },
        ]
    )
    st.dataframe(summary_rows, width="stretch", hide_index=True)
    render_probability_meter(weighted_rcb, weighted_gt)

    if not cup_probs.empty:
        final_cup = cup_probs[cup_probs["team"].isin(["RCB", "GT"])].copy()
        st.bar_chart(final_cup.set_index("team")["cup_probability"])

    if not players.empty:
        st.subheader("Most likely Player of the Match")
        potm = players.head(5)[["Player_Name", "Team", "Role", "POTM_Probability"]].copy()
        st.dataframe(potm, width="stretch", hide_index=True)

with tab_evidence:
    shap_col, score_col = st.columns(2)
    with shap_col:
        st.subheader("Top deciding factors")
        if not local_shap.empty:
            factor_cols = ["label", "supports", "approx_probability_points"]
            st.dataframe(local_shap.head(8)[factor_cols], width="stretch", hide_index=True)

    with score_col:
        st.subheader("RCB vs GT latest evidence")
        score_cols = st.columns(3)
        score_cols[0].metric("RCB innings", rcb_score or "254/5", "20.0 overs")
        score_cols[1].metric("GT innings", gt_score or "162/10", "19.3 overs")
        score_cols[2].metric("Qualifier 1", "RCB won", rcb_gt_result["margin"])

    st.subheader("Prediction audit")
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
    st.dataframe(audit_rows, width="stretch", hide_index=True)

    bat_col, bowl_col = st.columns(2)
    with bat_col:
        st.subheader("Top batting")
        if not rcb_gt_batting.empty:
            top_batting = rcb_gt_batting.sort_values("Runs", ascending=False).head(6)
            st.dataframe(top_batting, width="stretch", hide_index=True)

    with bowl_col:
        st.subheader("Top bowling")
        if not rcb_gt_bowling.empty:
            top_bowling = rcb_gt_bowling.sort_values(["Wickets", "Economy"], ascending=[False, True]).head(6)
            st.dataframe(top_bowling, width="stretch", hide_index=True)

with tab_path:
    st.subheader("Confirmed final path")
    st.info(
        f"RCB reached the final from Qualifier 1. {final_opponent} reached the final from Qualifier 2. "
        f"Current final matchup: {final_matchup}."
    )
    if not playoff_results.empty:
        st.dataframe(playoff_results, width="stretch")

    st.subheader("Fixture probabilities")
    if not fixture_probs.empty:
        st.dataframe(fixture_probs, width="stretch")

with tab_model:
    st.subheader("Model probabilities")
    if not model_probs.empty:
        team_a_col, team_b_col = probability_columns(model_probs)
        st.dataframe(model_probs, width="stretch")
        st.bar_chart(model_probs.set_index("model")[[team_a_col, team_b_col]])

    with st.expander("Cross-validation metrics"):
        if not metrics.empty:
            st.dataframe(metrics, width="stretch")

    with st.expander("Scenario sensitivity"):
        if not scenarios.empty:
            st.dataframe(scenarios, width="stretch")

    with st.expander("Uncertainty drivers"):
        if not sensitivity.empty:
            st.dataframe(sensitivity.head(10), width="stretch")

    with st.expander("XAI charts"):
        image_cols = st.columns(3)
        image_files = [
            ("Local waterfall", "shap_local_waterfall.png"),
            ("Global bar", "shap_global_bar.png"),
            ("Global beeswarm", "shap_global_beeswarm.png"),
        ]
        for col, (caption, file_name) in zip(image_cols, image_files):
            image_path = REPORTS_DIR / file_name
            if image_path.exists():
                col.image(str(image_path), caption=caption, width="stretch")

    with st.expander("Generated report"):
        st.markdown(report_path.read_text(encoding="utf-8"))
