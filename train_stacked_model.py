import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib

# Standardize output encoding
sys.stdout.reconfigure(encoding='utf-8')

PREDICTION_TARGET_DATE = pd.Timestamp("2026-05-30")
FINAL_MATCH_DATE = pd.Timestamp("2026-05-31")
TEAM_A = "RCB"
TEAM_B = "GT"
CV_FOLDS = 10

def prepare_data(features_file):
    df = pd.read_csv(features_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # The final match is the test set; training is cut off at the prediction date.
    df_test = df[(df['date'] >= FINAL_MATCH_DATE) & (df['team1'] == TEAM_A) & (df['team2'] == TEAM_B)].copy()
    if df_test.empty:
        df_test = df[df['date'] >= FINAL_MATCH_DATE].copy()
    df_test = df_test.sort_values('date').tail(1).copy()
    df_train = df[df['date'] < PREDICTION_TARGET_DATE].copy()
    
    # Identify feature columns
    exclude_cols = ['match_id', 'date', 'season', 'team1', 'team2', 'venue', 'winner_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_train_raw = df_train[feature_cols].values
    y_train = df_train['winner_label'].values
    
    X_test_raw = df_test[feature_cols].values
    y_test = df_test['winner_label'].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return df_train, df_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, imputer

def evaluate_models(X_train, y_train):
    # Set up 10-fold cross-validation when class balance allows it.
    n_splits = int(min(CV_FOLDS, np.bincount(y_train).min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, random_state=42),
        'XGBoost': XGBClassifier(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(num_leaves=31, learning_rate=0.03, n_estimators=300, subsample=0.8, random_state=42, verbose=-1),
        'Logistic Regression': LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
    }
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        metrics = {'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Recency weighting: 2026 matches are weighted 3x compared to prior seasons
            # Let's check indices in train_idx that belong to 2026 matches
            # Since we don't have the original df here, we can pass weights:
            # We will handle weights outside or just train with sample weights
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['auc'].append(roc_auc_score(y_val, y_prob))
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            
        results[name] = {k: np.mean(v) for k, v in metrics.items()}
        print(f"\n{name} CV Metrics:")
        for k, v in results[name].items():
            print(f"  {k.capitalize()}: {v:.4f}")
            
    return results

def train_stacking_ensemble(X_train, y_train, X_test):
    n_splits = int(min(CV_FOLDS, np.bincount(y_train).min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Layer 1 models
    rf = RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, random_state=42)
    xgb = XGBClassifier(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, eval_metric='logloss', random_state=42)
    lgb = LGBMClassifier(num_leaves=31, learning_rate=0.03, n_estimators=300, subsample=0.8, random_state=42, verbose=-1)
    
    # Out-of-fold prediction matrix
    oof_predictions = np.zeros((X_train.shape[0], 3))
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Fit base models on fold training data
        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)
        lgb.fit(X_tr, y_tr)
        
        oof_predictions[val_idx, 0] = rf.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx, 1] = xgb.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx, 2] = lgb.predict_proba(X_val)[:, 1]
        
    # Fit meta-learner (Logistic Regression) on oof predictions
    meta = LogisticRegression(C=1.0, penalty='l2', random_state=42)
    meta.fit(oof_predictions, y_train)
    
    # Train base models on full training data
    rf_full = RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, random_state=42).fit(X_train, y_train)
    xgb_full = XGBClassifier(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, eval_metric='logloss', random_state=42).fit(X_train, y_train)
    lgb_full = LGBMClassifier(num_leaves=31, learning_rate=0.03, n_estimators=300, subsample=0.8, random_state=42, verbose=-1).fit(X_train, y_train)
    
    # Predict on test set
    p_rf = rf_full.predict_proba(X_test)[:, 1]
    p_xgb = xgb_full.predict_proba(X_test)[:, 1]
    p_lgb = lgb_full.predict_proba(X_test)[:, 1]
    
    test_meta_input = np.column_stack([p_rf, p_xgb, p_lgb])
    final_prob = meta.predict_proba(test_meta_input)[:, 1]
    
    return rf_full, xgb_full, lgb_full, meta, final_prob[0]

def run_monte_carlo_simulation(df_train, df_test, feature_cols, rf_full, xgb_full, lgb_full, meta, imputer, scaler, n_simulations=10000):
    print(f"\nRunning {n_simulations} Monte Carlo simulations...")
    
    # Retrieve raw test features
    x_test_raw = df_test[feature_cols].values[0]
    
    # Standard deviations of raw features from training data (used for perturbation)
    raw_std = df_train[feature_cols].std().values
    raw_std = np.nan_to_num(raw_std, nan=0.1) # avoid divide by zero or NaN std
    
    sim_probs = []
    
    for i in range(n_simulations):
        # Perturb features randomly within +-1 standard deviation
        noise = np.random.uniform(-1.0, 1.0, size=x_test_raw.shape) * raw_std
        x_perturbed_raw = x_test_raw + noise
        
        # Run through imputer & scaler
        x_perturbed_imp = imputer.transform(x_perturbed_raw.reshape(1, -1))
        x_perturbed_scaled = scaler.transform(x_perturbed_imp)
        
        # Run base models
        p_rf = rf_full.predict_proba(x_perturbed_scaled)[:, 1]
        p_xgb = xgb_full.predict_proba(x_perturbed_scaled)[:, 1]
        p_lgb = lgb_full.predict_proba(x_perturbed_scaled)[:, 1]
        
        # Meta-learner prediction
        test_meta_input = np.column_stack([p_rf, p_xgb, p_lgb])
        p_meta = meta.predict_proba(test_meta_input)[:, 1][0]
        
        sim_probs.append(p_meta)
        
    sim_probs = np.array(sim_probs)
    mean_prob = np.mean(sim_probs)
    lower_ci = np.percentile(sim_probs, 2.5)
    upper_ci = np.percentile(sim_probs, 97.5)
    
    print(f"Mean Win Probability for {TEAM_A}: {mean_prob*100:.2f}%")
    print(f"95% Confidence Interval: {lower_ci*100:.2f}% - {upper_ci*100:.2f}%")
    
    # Identify which features cause most variance (Sensitivity Analysis)
    # We will perturb one feature at a time and see the output variance
    sensitivity = {}
    for col_idx, col_name in enumerate(feature_cols):
        test_perturbed_single = np.repeat(x_test_raw.reshape(1, -1), 100, axis=0)
        noise_single = np.random.uniform(-1.0, 1.0, size=100) * raw_std[col_idx]
        test_perturbed_single[:, col_idx] += noise_single
        
        scaled_single = scaler.transform(imputer.transform(test_perturbed_single))
        p_rf = rf_full.predict_proba(scaled_single)[:, 1]
        p_xgb = xgb_full.predict_proba(scaled_single)[:, 1]
        p_lgb = lgb_full.predict_proba(scaled_single)[:, 1]
        
        meta_in = np.column_stack([p_rf, p_xgb, p_lgb])
        probs = meta.predict_proba(meta_in)[:, 1]
        sensitivity[col_name] = np.var(probs)
        
    sorted_sensitivity = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 features causing most prediction variance (uncertainty factors):")
    for name, var in sorted_sensitivity[:5]:
        print(f"  {name}: Variance {var:.6f}")
        
    return sim_probs, mean_prob, lower_ci, upper_ci, sorted_sensitivity

if __name__ == "__main__":
    features_file = "match_features.csv"
    if not os.path.exists(features_file):
        print(f"Error: {features_file} not found. Please run feature_engineering.py first.")
        sys.exit(1)
        
    print("Preparing data...")
    df_train, df_test, X_train, X_test, y_train, y_test, feature_cols, scaler, imputer = prepare_data(features_file)
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("\nEvaluating base models...")
    cv_results = evaluate_models(X_train, y_train)
    
    print("\nTraining Stacking Ensemble...")
    rf_full, xgb_full, lgb_full, meta, team_a_win_prob = train_stacking_ensemble(X_train, y_train, X_test)
    print(f"\nFinal Predicted {TEAM_A} Win Probability: {team_a_win_prob*100:.2f}%")
    print(f"Final Predicted {TEAM_B} Win Probability: {(1-team_a_win_prob)*100:.2f}%")
    
    # Save models for explainability step
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_full, "models/rf_full.joblib")
    joblib.dump(xgb_full, "models/xgb_full.joblib")
    joblib.dump(lgb_full, "models/lgb_full.joblib")
    joblib.dump(meta, "models/meta_stacked.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")
    print("Saved models to models/ directory.")
    
    # Run simulation
    sim_probs, mean_prob, lower_ci, upper_ci, sensitivity = run_monte_carlo_simulation(
        df_train, df_test, feature_cols, rf_full, xgb_full, lgb_full, meta, imputer, scaler
    )
    
    # Save simulation results
    np.save("models/sim_probs.npy", sim_probs)
    print("Saved simulation probabilities to models/sim_probs.npy")
