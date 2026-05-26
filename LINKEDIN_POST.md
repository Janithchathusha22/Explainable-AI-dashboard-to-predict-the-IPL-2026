# LinkedIn Post Draft

I built an Explainable AI dashboard to forecast the IPL 2026 playoff/final path.

Live app: https://lnkd.in/gPurrJVM

Project goal:

Build a transparent cricket prediction system that does more than output a winner. The dashboard shows model probabilities, playoff simulations, player impact, weather and venue effects, uncertainty, and SHAP-based explanations.

Current playoff update:

- Semi Final 1 result: RCB beat GT by 92 runs
- Score: RCB 254/5, GT 162
- Status: RCB confirmed as a finalist, GT eliminated
- Final date: 2026-05-31
- Venue: Narendra Modi Stadium, Ahmedabad
- Prediction target date: 2026-05-27

Current model output:

- Featured final candidate: RCB vs SRH
- RCB win probability: 48.1%
- SRH win probability: 51.9%
- Predicted winner for this candidate final: SRH
- Confidence: Low

Playoff simulation:

- RCB final appearance probability: 100.0%
- Most likely final pairs:
- RCB vs RR: 53.0%
- RCB vs SRH: 47.0%
- Cup probabilities:
- RCB: 46.9%
- RR: 28.6%
- SRH: 24.5%

Why low confidence?

The stacked ensemble slightly favors SRH in the RCB-vs-SRH candidate final, but the tree-based base models lean toward RCB. Instead of hiding this conflict, the dashboard exposes it through model-comparison charts and confidence notes. For me, that is the important part of this project: prediction with uncertainty, not prediction as a black box.

Technologies used:

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Random Forest
- XGBoost
- LightGBM
- Logistic Regression stacking
- SHAP for explainability
- Monte Carlo simulation

Model and algorithm stack:

- Base learners: Random Forest, XGBoost, LightGBM
- Meta learner: Logistic Regression, stacked on base-model probabilities
- Cross-validation: Stratified 10-fold CV
- Recency weighting: newer season samples are up-weighted during training
- Final probability rule: `p_final = sigmoid(w1*p_rf + w2*p_xgb + w3*p_lgb + b)`
- Uncertainty estimation: 10,000-run Monte Carlo simulation
- Explainability method: Tree SHAP on XGBoost for local and global factor attribution

Dashboard sections:

- Final/candidate-final prediction
- Playoff schedule and bracket probabilities
- SHAP local/global explanations
- Model metrics
- Player of the Match ranking
- Player performance forecast
- Scenario sensitivity
- Weather and right/left matchup inputs

This project demonstrates an end-to-end sports analytics workflow: feature engineering, machine learning, uncertainty modeling, explainability, and a deployed Streamlit product.

Note: This is a data-science and explainability project, not betting advice. Cricket outcomes are uncertain, and the dashboard is designed to show that uncertainty clearly.

#MatchForecasting #WinProbability #SportsAnalytics #RiskAnalysis #PredictiveAnalytics #MatchOutcome #OddsAnalytics #ScenarioModeling #DataDrivenDecisions #IPL2026Final #QuantSports #MachineLearning #DataScience #Streamlit #Python #XAI #SHAP #IPL #CricketAnalytics
