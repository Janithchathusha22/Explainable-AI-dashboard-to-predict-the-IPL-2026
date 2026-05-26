# LinkedIn Post Draft

I built an Explainable AI dashboard to predict the IPL 2026 final: Gujarat Titans vs Sunrisers Hyderabad.

Live app: https://your-app-name.streamlit.app  
GitHub: https://github.com/YOUR_USERNAME/YOUR_REPO_NAME

Project goal:

Build a transparent cricket prediction system that does more than output a winner. The dashboard shows model probabilities, playoff simulations, player impact, weather/venue effects, uncertainty, and SHAP-based explanations.

Current model output:

- Final: GT vs SRH
- Venue: Narendra Modi Stadium, Ahmedabad
- Prediction target date: 2026-05-26
- Final match date: 2026-05-31
- GT win probability: 47.5%
- SRH win probability: 52.5%
- Predicted winner: SRH
- Confidence: Low

Why low confidence?

The stacked ensemble slightly favors SRH, but the tree-based base models lean toward GT. Instead of hiding that conflict, the app exposes it through model-comparison charts and confidence notes. For me, that is the most important part of this project: prediction with uncertainty, not prediction as a black box.

What I used:

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

Feature groups:

- Historical IPL win rates
- Recent team form
- Head-to-head record
- Venue-specific performance
- Toss and chasing advantage
- Net run rate
- Playoff path advantage
- Powerplay/middle/death-over phase features
- Bowling economy and wicket-taking indicators
- Live weather: temperature, humidity, dew point, rain probability, dew index
- Right-hand/left-hand batting and bowling matchups
- Player impact and Player of the Match probability

Training pattern:

Raw CSV data -> feature engineering -> prediction-date train/test split -> imputation/scaling -> base models -> stacked ensemble -> Monte Carlo uncertainty -> SHAP explainability -> Streamlit dashboard.

Key engineering choice:

The final is dated 2026-05-31, but the prediction is made as of 2026-05-26. I used that prediction date as the training cutoff to reduce future-result leakage from later playoff fixtures.

Dashboard sections:

- Final prediction
- Playoff schedule and bracket probabilities
- SHAP local/global explanations
- Model metrics
- Player of the Match ranking
- Player performance forecast
- Scenario sensitivity
- Weather and right/left matchup inputs

This was a fun project because it connects sports analytics, machine learning, explainability, and product presentation in one deployable app.

Note: This is a data-science project, not betting advice. Cricket outcomes are uncertain, and the model clearly shows uncertainty where it exists.

#MachineLearning #DataScience #Streamlit #Python #XAI #SHAP #SportsAnalytics #IPL #CricketAnalytics #PortfolioProject
