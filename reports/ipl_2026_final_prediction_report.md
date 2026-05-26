# IPL 2026 Final Explainable AI Prediction Report

Featured final candidate: RCB vs SRH
Venue: Narendra Modi Stadium, Ahmedabad
Final match date: 2026-05-31
Prediction target date: 2026-05-27
Known playoff update: RCB beat GT by 92 runs in Semi Final 1 (RCB 254/5, GT 162), so RCB is locked as the first finalist.

## 1. Predicted Winner

Predicted winner: **SRH**
RCB win probability: **48.1%**
SRH win probability: **51.9%**
Monte Carlo mean: **47.2%** for RCB
Monte Carlo 95% CI: **45.2% to 49.8%**
Confidence level: **LOW**
Model disagreement across base learners: **18.4 percentage points**
Mean cross-validation AUC across base learners: **0.502**

## 2. Four-Team Playoff Cup Probabilities

- RCB: Cup 46.9%, final appearance 100.0%
- RR: Cup 28.6%, final appearance 53.0%
- SRH: Cup 24.5%, final appearance 47.0%

Most likely final pairings from bracket simulation:

- RCB vs RR: 53.0%
- RCB vs SRH: 47.0%

Note: these odds lock the known Semi Final 1 result, then simulate Semi Final 2 and the Final. The featured final section is one candidate final row until the other finalist is confirmed.

## 3. Model Confidence

- Random Forest: 63.0% RCB, 37.0% SRH
- XGBoost: 73.5% RCB, 26.5% SRH
- LightGBM: 55.1% RCB, 44.9% SRH
- Stacked Ensemble: 48.1% RCB, 51.9% SRH

Stacking meta-learner weights:

- Random Forest: +0.4584
- XGBoost: -0.0843
- LightGBM: -0.2424

Model agreement note: the base tree models favor the opposite side from the stacked output. The stacked meta-learner is therefore treated as a low-confidence calibration layer, not a strong cricket signal.

## 4. SHAP Key Deciding Factors

Important: the SHAP explanation below is for the XGBoost tree model, as required for tree SHAP. In this run XGBoost favors a different side than the final stacked probability, so these factors explain the tree-model push, while the final stacked winner remains the calibrated ensemble output.

- Batting-first win-rate advantage: raw value -0.017; supports SRH; approx impact -7.03 percentage points
- Specific toss advantage for Team A: raw value 0.141; supports RCB; approx impact +6.02 percentage points
- Head-to-head win rate for Team A: raw value 0.444; supports RCB; approx impact +5.25 percentage points
- Powerplay wicket-taking advantage: raw value 0.300; supports RCB; approx impact +4.68 percentage points
- Overall IPL win-rate advantage: raw value 0.013; supports RCB; approx impact +4.25 percentage points
- Chasing win-rate advantage: raw value 0.043; supports RCB; approx impact +3.56 percentage points
- Net run-rate advantage: raw value 0.259; supports RCB; approx impact +3.10 percentage points

Top global historical factors from XGBoost SHAP:

- Head-to-head win rate for Team A: mean |SHAP| 0.2090
- Venue-specific win-rate advantage: mean |SHAP| 0.2029
- Batting-first win-rate advantage: mean |SHAP| 0.1963
- Chasing win-rate advantage: mean |SHAP| 0.1929
- Overall IPL win-rate advantage: mean |SHAP| 0.1777

## 5. Player Of The Match Ranking

Most likely POTM: **Bhuvneshwar Kumar (RCB)** - 13.2%
Reason: 24 wickets at economy 8.07; impact score 78.5/100; team win probability input 48.1%.

1. Bhuvneshwar Kumar (RCB) - 13.2% [bat win 16.6%, bowl win 45.6%]
2. Abhishek Sharma (SRH) - 12.6% [bat win 49.9%, bowl win 32.1%]
3. Eshan Malinga (SRH) - 11.0% [bat win 7.8%, bowl win 41.4%]
4. Heinrich Klaasen (SRH) - 10.3% [bat win 48.4%, bowl win 5.2%]
5. Ishan Kishan (SRH) - 9.9% [bat win 48.1%, bowl win 5.2%]

## 6. Bhuvneshwar Kumar Performance Forecast

Expected batting runs: **6.4**
Likely runs band: **6 to 10**
Expected balls faced: **5.6**
30+ probability: **16.7%**
50+ probability: **16.7%**
Dismissal probability: **6.2%**
Not-out probability: **93.8%**
Expected wickets if he bowls: **1.75**
Probability of taking at least one wicket: **68.8%**

Interpretation: Bhuvneshwar Kumar is the current top POTM candidate for this featured final matchup.

## 7. Scenario Sensitivity

- Baseline final input: RCB 48.1%, SRH 51.9% (+0.00 pts)
- RCB wins toss and fields first: RCB 48.1%, SRH 51.9% (+0.00 pts)
- SRH wins toss and fields first: RCB 48.4%, SRH 51.6% (+0.34 pts)
- RCB wins toss and bats first: RCB 48.4%, SRH 51.6% (+0.34 pts)
- Bhuvneshwar Kumar unavailable (approx feature shock): RCB 48.1%, SRH 51.9% (+0.01 pts)

## 8. Uncertainty Factors

- Head-to-head win rate for Team A: probability range 3.07 pts under feature shock
- Venue-specific win-rate advantage: probability range 3.17 pts under feature shock
- Chasing win-rate advantage: probability range 2.67 pts under feature shock
- Powerplay wicket-taking advantage: probability range 1.89 pts under feature shock
- Batting-first win-rate advantage: probability range 2.21 pts under feature shock

## 9. XAI Method

The stacked ensemble is used for final probability. SHAP is applied to the XGBoost tree model, not directly to the stacked logistic meta-learner, because tree SHAP gives stable feature-level explanations for the base model. The local SHAP rows explain which inputs push the RCB-vs-SRH candidate final prediction toward either side. The feature matrix includes playoff live-weather inputs, right/left matchup inputs, and the known Semi Final 1 result when those CSV rows are available.

Generated files:
- reports/shap_global_bar.png
- reports/shap_global_beeswarm.png
- reports/shap_local_waterfall.png
- reports/monte_carlo_distribution.png
- reports/shap_local_explanation.csv
- reports/player_of_match_ranking.csv
- reports/player_performance_forecast.csv
- reports/playoff_cup_probabilities.csv
- reports/final_pair_probabilities.csv
- reports/playoff_fixture_probabilities.csv
- reports/scenario_sensitivity.csv
- reports/stacking_meta_weights.csv
