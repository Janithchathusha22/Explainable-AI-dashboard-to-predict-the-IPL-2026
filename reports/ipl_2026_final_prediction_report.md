# IPL 2026 Final Explainable AI Prediction Report

Final: GT vs SRH
Venue: Narendra Modi Stadium, Ahmedabad
Final match date: 2026-05-31
Prediction target date: 2026-05-26

## 1. Predicted Winner

Predicted winner: **SRH**
GT win probability: **47.5%**
SRH win probability: **52.5%**
Monte Carlo mean: **47.0%** for GT
Monte Carlo 95% CI: **45.4% to 48.3%**
Confidence level: **LOW**
Model disagreement across base learners: **17.9 percentage points**
Mean cross-validation AUC across base learners: **0.504**

## 2. Four-Team Playoff Cup Probabilities

- GT: Cup 36.4%, final appearance 74.9%
- RCB: Cup 35.3%, final appearance 72.1%
- RR: Cup 15.0%, final appearance 28.0%
- SRH: Cup 13.2%, final appearance 25.1%

Most likely final pairings from bracket simulation:

- GT vs RCB: 46.9%
- GT vs RR: 14.8%
- RCB vs RR: 13.2%
- GT vs SRH: 13.1%
- RCB vs SRH: 12.0%

Note: these four-team odds simulate the playoff bracket from the top-four state and remove manually injected playoff/final results to reduce leakage. The GT vs SRH final section uses the known-finalist final row.

## 3. Model Confidence

- Random Forest: 78.0% GT, 22.0% SRH
- XGBoost: 93.5% GT, 6.5% SRH
- LightGBM: 95.9% GT, 4.1% SRH
- Stacked Ensemble: 47.5% GT, 52.5% SRH

Stacking meta-learner weights:

- Random Forest: -0.3222
- XGBoost: +0.2921
- LightGBM: -0.0622

Model agreement note: the base tree models favor the opposite side from the stacked output. The stacked meta-learner is therefore treated as a low-confidence calibration layer, not a strong cricket signal.

## 4. SHAP Key Deciding Factors

Important: the SHAP explanation below is for the XGBoost tree model, as required for tree SHAP. In this run XGBoost favors a different side than the final stacked probability, so these factors explain the tree-model push, while the final stacked winner remains the calibrated ensemble output.

- Venue-specific win-rate advantage: raw value 0.600; supports GT; approx impact +13.65 percentage points
- Powerplay wicket-taking advantage: raw value 0.400; supports GT; approx impact +13.25 percentage points
- Overall IPL win-rate advantage: raw value 0.136; supports GT; approx impact +10.88 percentage points
- Chasing win-rate advantage: raw value 0.167; supports GT; approx impact +8.20 percentage points
- Net run-rate advantage: raw value 0.171; supports GT; approx impact +6.31 percentage points
- Batting-first win-rate advantage: raw value 0.099; supports GT; approx impact +6.17 percentage points
- Specific toss advantage for Team A: raw value 0.141; supports GT; approx impact +3.92 percentage points

Top global historical factors from XGBoost SHAP:

- Head-to-head win rate for Team A: mean |SHAP| 0.2199
- Venue-specific win-rate advantage: mean |SHAP| 0.2034
- Chasing win-rate advantage: mean |SHAP| 0.1958
- Batting-first win-rate advantage: mean |SHAP| 0.1862
- Overall IPL win-rate advantage: mean |SHAP| 0.1777

## 5. Player Of The Match Ranking

Most likely POTM: **Abhishek Sharma (SRH)** - 11.9%
Reason: 563 runs at SR 206.2; 11 wickets at economy 9.12; impact score 73.0/100; team win probability input 52.5%.

1. Abhishek Sharma (SRH) - 11.9% [bat win 49.2%, bowl win 33.3%]
2. Kagiso Rabada (GT) - 11.8% [bat win 17.3%, bowl win 44.2%]
3. Eshan Malinga (SRH) - 10.3% [bat win 7.9%, bowl win 42.7%]
4. Rashid Khan (GT) - 10.3% [bat win 22.4%, bowl win 39.4%]
5. Heinrich Klaasen (SRH) - 9.7% [bat win 47.5%, bowl win 5.3%]

## 6. Abhishek Sharma Performance Forecast

Expected batting runs: **38.0**
Likely runs band: **9 to 57**
Expected balls faced: **18.4**
30+ probability: **56.2%**
50+ probability: **37.5%**
Dismissal probability: **87.5%**
Not-out probability: **12.5%**
Expected wickets if he bowls: **0.35**
Probability of taking at least one wicket: **16.7%**

Interpretation: Abhishek is projected mainly as a batting/POTM candidate. His bowling contribution is forecast as secondary unless SRH use him for matchup overs.

## 7. Scenario Sensitivity

- Baseline final input: GT 47.5%, SRH 52.5% (+0.00 pts)
- GT wins toss and fields first: GT 47.5%, SRH 52.5% (+0.00 pts)
- SRH wins toss and fields first: GT 47.3%, SRH 52.7% (-0.12 pts)
- GT wins toss and bats first: GT 47.3%, SRH 52.7% (-0.12 pts)
- Abhishek Sharma unavailable (approx feature shock): GT 47.7%, SRH 52.3% (+0.28 pts)

## 8. Uncertainty Factors

- Chasing win-rate advantage: probability range 0.75 pts under feature shock
- Right-hand batting matchup difference: probability range 0.37 pts under feature shock
- Playoff experience advantage: probability range 0.41 pts under feature shock
- Powerplay wicket-taking advantage: probability range 0.48 pts under feature shock
- Venue-specific win-rate advantage: probability range 0.79 pts under feature shock

## 9. XAI Method

The stacked ensemble is used for final probability. SHAP is applied to the XGBoost tree model, not directly to the stacked logistic meta-learner, because tree SHAP gives stable feature-level explanations for the base model. The local SHAP rows explain which inputs push the GT-vs-SRH prediction toward GT or SRH. The feature matrix includes playoff live-weather inputs and right/left matchup inputs when those CSV rows are available.

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