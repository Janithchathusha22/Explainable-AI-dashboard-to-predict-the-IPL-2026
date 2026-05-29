# IPL 2026 Final Explainable AI Prediction Report

Confirmed final matchup: RCB vs GT
Venue: Narendra Modi Stadium, Ahmedabad
Final match date: 2026-05-31
Prediction target date: 2026-05-30
Known playoff update: RCB beat GT by 92 runs in Qualifier 1, RR beat SRH in the Eliminator, and GT beat RR in Qualifier 2. RCB and GT are locked as the finalists.

## 1. Predicted Winner

Predicted winner: **GT**
RCB win probability: **46.2%**
GT win probability: **53.8%**
Monte Carlo mean: **47.4%** for RCB
Monte Carlo 95% CI: **44.6% to 50.3%**
Confidence level: **LOW**
Model disagreement across base learners: **10.5 percentage points**
Mean cross-validation AUC across base learners: **0.508**

## 2. Confirmed Playoff Path And Cup Probabilities

- GT: Cup 54.2%, final appearance 100.0%
- RCB: Cup 45.8%, final appearance 100.0%
- RR: Cup 0.0%, final appearance 0.0%
- SRH: Cup 0.0%, final appearance 0.0%

Most likely final pairings from bracket simulation:

- RCB vs GT: 100.0%

Note: these odds lock all known playoff results through Qualifier 2. The only remaining modeled match is the RCB vs GT final.

## 3. Model Confidence

- Random Forest: 35.8% RCB, 64.2% GT
- XGBoost: 37.5% RCB, 62.5% GT
- LightGBM: 46.3% RCB, 53.7% GT
- Stacked Ensemble: 46.2% RCB, 53.8% GT

Stacking meta-learner weights:

- Random Forest: +0.0658
- XGBoost: +0.5018
- LightGBM: -0.3671

## 4. SHAP Key Deciding Factors

- Head-to-head win rate for Team A: raw value 0.556; supports GT; approx impact -8.04 percentage points
- Overall IPL win-rate advantage: raw value -0.120; supports RCB; approx impact +8.17 percentage points
- Playoff experience advantage: raw value 4.000; supports GT; approx impact -3.92 percentage points
- Captain form advantage: raw value -0.200; supports GT; approx impact -2.85 percentage points
- Chasing win-rate advantage: raw value -0.115; supports RCB; approx impact +1.81 percentage points
- Powerplay wicket-taking advantage: raw value -0.100; supports GT; approx impact -1.50 percentage points
- Venue toss-win/match-win tendency: raw value 0.641; supports GT; approx impact -1.48 percentage points

Top global historical factors from XGBoost SHAP:

- Head-to-head win rate for Team A: mean |SHAP| 0.2097
- Venue-specific win-rate advantage: mean |SHAP| 0.2024
- Chasing win-rate advantage: mean |SHAP| 0.1889
- Overall IPL win-rate advantage: mean |SHAP| 0.1864
- Batting-first win-rate advantage: mean |SHAP| 0.1863

## 5. Player Of The Match Ranking

Most likely POTM: **Kagiso Rabada (GT)** - 12.9%
Reason: 24 wickets at economy 9.19; impact score 75.2/100; team win probability input 53.8%.

1. Kagiso Rabada (GT) - 12.9% [bat win 20.8%, bowl win 49.1%]
2. Bhuvneshwar Kumar (RCB) - 12.8% [bat win 16.8%, bowl win 43.9%]
3. Rashid Khan (GT) - 11.3% [bat win 26.9%, bowl win 43.7%]
4. Mohammed Siraj (GT) - 10.4% [bat win 16.1%, bowl win 41.5%]
5. Sai Sudharsan (GT) - 10.1% [bat win 51.4%, bowl win 5.4%]

## 6. Kagiso Rabada Performance Forecast

Expected batting runs: **11.7**
Likely runs band: **6 to 18**
Expected balls faced: **8.3**
30+ probability: **25.0%**
50+ probability: **25.0%**
Dismissal probability: **12.5%**
Not-out probability: **87.5%**
Expected wickets if he bowls: **1.71**
Probability of taking at least one wicket: **75.0%**

Interpretation: Kagiso Rabada is the current top POTM candidate for this featured final matchup.

## 7. Scenario Sensitivity

- Baseline final input: RCB 46.2%, GT 53.8% (+0.00 pts)
- RCB wins toss and fields first: RCB 46.2%, GT 53.8% (+0.00 pts)
- GT wins toss and fields first: RCB 45.3%, GT 54.7% (-0.96 pts)
- RCB wins toss and bats first: RCB 45.3%, GT 54.7% (-0.96 pts)
- Kagiso Rabada unavailable (approx feature shock): RCB 46.1%, GT 53.9% (-0.13 pts)

## 8. Uncertainty Factors

- Venue-specific win-rate advantage: probability range 2.84 pts under feature shock
- Overall IPL win-rate advantage: probability range 4.03 pts under feature shock
- Batting-first win-rate advantage: probability range 3.32 pts under feature shock
- Head-to-head win rate for Team A: probability range 2.90 pts under feature shock
- Recent form, last five matches: probability range 2.14 pts under feature shock

## 9. XAI Method

The stacked ensemble is used for final probability. SHAP is applied to the XGBoost tree model, not directly to the stacked logistic meta-learner, because tree SHAP gives stable feature-level explanations for the base model. The local SHAP rows explain which inputs push the confirmed RCB-vs-GT final prediction toward either side. The feature matrix includes playoff live-weather inputs, right/left matchup inputs, and known playoff results through Qualifier 2 when those CSV rows are available.

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