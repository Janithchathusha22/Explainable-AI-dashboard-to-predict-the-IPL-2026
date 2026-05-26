# IPL 2026 Final Explainable AI Prediction

A Streamlit dashboard and explainable machine-learning pipeline for predicting the IPL 2026 final between Gujarat Titans (GT) and Sunrisers Hyderabad (SRH).

The project combines historical IPL match data, 2025/2026 ball-by-ball data, playoff venue effects, live-weather inputs, right/left matchup features, model ensembling, Monte Carlo simulation, and SHAP explainability.

## Live Demo

Add the deployed Streamlit URL here after publishing:

```text
https://your-app-name.streamlit.app
```

## Current Prediction

Prediction target date: `2026-05-26`

Final fixture:

| Match | Teams | Final Date | Venue |
| --- | --- | --- | --- |
| Grand Final | GT vs SRH | 2026-05-31 | Narendra Modi Stadium, Ahmedabad |

Latest regenerated result:

| Team | Win Probability |
| --- | ---: |
| GT | 47.5% |
| SRH | 52.5% |

Predicted winner: **SRH**

Confidence level: **LOW**

The low-confidence label is intentional. The stacked ensemble favors SRH, while the tree-based base models lean heavily toward GT, so the dashboard exposes the disagreement instead of hiding it.

## Dashboard

The Streamlit frontend includes:

- Final winner probability
- Player of the Match ranking
- Playoff bracket and fixture probabilities
- Sinhala playoff schedule table
- SHAP local and global explanations
- Monte Carlo uncertainty distribution
- Scenario sensitivity
- Player performance forecast
- Weather and right/left matchup input tables

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
- The prediction is made as of `2026-05-26`.
- Training excludes matches on or after `2026-05-26`, so future playoff outcomes are not leaked into the final prediction.

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
- Local SHAP explains the GT vs SRH final row
- Global SHAP ranks the strongest historical factors

Uncertainty:

- 10,000 Monte Carlo simulations perturb final-match features
- Scenario sensitivity tests toss and player-availability changes
- Four-team playoff bracket simulation uses 30,000 runs

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

This is a data-science and explainability project, not betting advice. Cricket outcomes are uncertain, and the current prediction is explicitly low-confidence because model families disagree.
