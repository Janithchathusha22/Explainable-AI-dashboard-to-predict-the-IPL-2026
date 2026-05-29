# IPL 2026 Final Explainable AI Prediction

A Streamlit dashboard and explainable machine-learning pipeline for forecasting the IPL 2026 playoff/final path.

The project combines historical IPL match data, 2025/2026 ball-by-ball data, playoff venue effects, live-weather inputs, right/left matchup features, model ensembling, Monte Carlo simulation, and SHAP explainability.

## Live Demo

```text
https://lnkd.in/gPurrJVM
```

## Current Prediction

Prediction target date: `2026-05-30`

Known playoff results:

| Match | Teams | Result | Venue |
| --- | --- | --- | --- |
| Qualifier 1 | RCB vs GT | RCB won by 92 runs | HPCA Stadium, Dharamshala |
| Eliminator | RR vs SRH | RR won by 47 runs | Maharaja Yadavindra Singh Stadium, Mullanpur |
| Qualifier 2 | GT vs RR | GT won by 7 wickets | Maharaja Yadavindra Singh Stadium, Mullanpur |

Confirmed final:

| Match | Teams | Final Date | Venue |
| --- | --- | --- | --- |
| Grand Final | RCB vs GT | 2026-05-31 | Narendra Modi Stadium, Ahmedabad |

Latest regenerated result:

| Team | Win Probability |
| --- | ---: |
| RCB | 46.2% |
| GT | 53.8% |

Predicted winner for this final: **GT**

Confidence level: **LOW**

The low-confidence label is intentional. The stacked ensemble favors GT, but the edge is narrow and the cross-validation signal is modest, so the dashboard exposes uncertainty instead of presenting the forecast as a lock.

Confirmed path and cup probability:

| Team | Cup Probability | Final Appearance |
| --- | ---: | ---: |
| GT | 54.2% | 100.0% |
| RCB | 45.8% | 100.0% |
| RR | 0.0% | 0.0% |
| SRH | 0.0% | 0.0% |

Most likely final pairs:

| Final Pair | Probability |
| --- | ---: |
| RCB vs GT | 100.0% |

## Dashboard

The Streamlit frontend includes:

- Confirmed final winner prediction
- Key evidence only: top SHAP factors, POTM ranking, playoff path, and compact model detail
- Player of the Match ranking
- Playoff path and fixture probabilities
- SHAP local and global explanations
- Monte Carlo uncertainty distribution
- Scenario sensitivity

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows with the included virtual environment:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Algorithm Pattern

The project follows this pipeline:

```text
Raw CSV data
  -> feature engineering
  -> train/test split by prediction date
  -> imputation and scaling
  -> base ML models
  -> stacked meta-model
  -> Monte Carlo simulation
  -> SHAP explainability
  -> report artifacts
  -> Streamlit dashboard
```

## Data Inputs

Main input files:

- `ipl_matches_clean.csv`
- `ipl_2025_deliveries.csv`
- `ipl_2026_deliveries.csv`
- `ipl_2026_recent_matches.csv`
- `ipl_2026_q1_rcb_vs_gt.csv`
- `ipl_2026_team_standings.csv`
- `ipl_2026_playoff_venues.csv`
- `ipl_2026_playoffs_toss_and_ground_effects.csv`
- `ipl_2026_playoff_players_full_stats.csv`
- `ipl_playoffs_live_weather.csv`
- `right_left.csv`

Generated feature files:

- `match_features.csv`
- `player_impact_scores.csv`

Generated report files live in `reports/`.

## Feature Engineering

The model uses match-level, venue-level, team-form, player-impact, and condition-based features.

Examples:

- Overall win-rate difference
- Recent form over last five matches
- Head-to-head win rate
- Finals experience
- Venue win-rate difference
- Toss and field-first advantage
- Bat-first and chasing win-rate difference
- Net run-rate difference
- Captain form difference
- Playoff path advantage
- Powerplay, middle-over, and death-over batting differences
- Powerplay and death-over wicket-taking differences
- Bowling economy difference
- Live temperature, humidity, dew point, rain probability, dew index, and heat index
- Left-hand/right-hand batting balance
- Left-arm/right-arm bowling matchup difference

Important leakage control:

- The final is played on `2026-05-31`.
- The prediction is made as of `2026-05-30`.
- Training excludes matches on or after `2026-05-30`, so the final outcome is not leaked into the prediction.
- Known playoff results through Qualifier 2 are included, including RCB 254/5 vs GT 162 in Qualifier 1.

## Models Used

Base learners:

- Random Forest
- XGBoost
- LightGBM
- Logistic Regression for benchmark metrics

Final ensemble:

- Stacked model using Random Forest, XGBoost, and LightGBM outputs
- Logistic Regression meta-learner

Validation:

- Stratified cross-validation
- 10 folds when class balance allows it
- Recency weighting for newer seasons

Explainability:

- SHAP is applied to the XGBoost base learner
- Local SHAP explains the confirmed RCB vs GT final row
- Global SHAP ranks the strongest historical factors

Uncertainty:

- 10,000 Monte Carlo simulations perturb final-match features
- Scenario sensitivity tests toss and player-availability changes
- Playoff bracket simulation uses 30,000 runs

## How To Train

Install training dependencies:

```bash
pip install -r requirements-ml.txt
```

Build features:

```bash
python feature_engineering.py
```

Train models and regenerate all XAI reports:

```bash
python explainable_ai_pipeline.py
```

This creates or updates:

- `models/*.joblib`
- `reports/final_model_probabilities.csv`
- `reports/model_metrics.csv`
- `reports/shap_local_explanation.csv`
- `reports/player_of_match_ranking.csv`
- `reports/player_performance_forecast.csv`
- `reports/playoff_cup_probabilities.csv`
- `reports/final_pair_probabilities.csv`
- `reports/playoff_fixture_probabilities.csv`
- `reports/scenario_sensitivity.csv`
- `reports/*.png`
- `reports/ipl_2026_final_prediction_report.md`

## Deploy

This is a Streamlit app, so use an app host that can run Python.

Recommended:

1. Push this folder to a public GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repo.
4. Set the main file path to `streamlit_app.py`.
5. Deploy.

GitHub Pages is not suitable for this project because it hosts static sites, while Streamlit needs a running Python process.

## Project Structure

```text
.
|-- streamlit_app.py
|-- feature_engineering.py
|-- explainable_ai_pipeline.py
|-- train_stacked_model.py
|-- requirements.txt
|-- requirements-ml.txt
|-- runtime.txt
|-- reports/
|-- models/
|-- *.csv
```

## Responsible Use

This is a data-science and explainability project, not betting advice. Cricket outcomes are uncertain, and the current RCB vs GT final prediction is explicitly low-confidence because the statistical edge is narrow.
