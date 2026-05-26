from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
PREDICTION_TARGET_DATE = pd.Timestamp("2026-05-27")
FINAL_MATCH_DATE = pd.Timestamp("2026-05-31")
FINAL_DATE = FINAL_MATCH_DATE
CV_FOLDS = 10
TEAM_A = "RCB"
TEAM_B = "SRH"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

FEATURE_LABELS = {
    "overall_win_rate_diff": "Overall IPL win-rate advantage",
    "win_rate_last_5_diff": "Recent form, last five matches",
    "win_rate_finals_diff": "Historical final-match performance",
    "win_rate_venue_diff": "Venue-specific win-rate advantage",
    "head_to_head_win_rate": "Head-to-head win rate for Team A",
    "win_rate_chasing_diff": "Chasing win-rate advantage",
    "win_rate_bat_first_diff": "Batting-first win-rate advantage",
    "toss_win_match_win_corr": "Venue toss-win/match-win tendency",
    "bat_first_win_rate_venue": "Venue batting-first win rate",
    "team1_toss_advantage": "Specific toss advantage for Team A",
    "avg_1st_inn_score_venue": "Venue average first-innings score",
    "dew_factor_impact": "Dew impact",
    "pitch_type_encoded": "Pitch type encoding",
    "boundary_size_factor": "Boundary-size factor",
    "nrr_diff": "Net run-rate advantage",
    "cap_form_diff": "Captain form advantage",
    "team_exp_diff": "Playoff experience advantage",
    "playoff_stage_path_diff": "Playoff path advantage",
    "pp_bat_avg_diff": "Powerplay batting-score advantage",
    "mid_bat_avg_diff": "Middle-overs batting-score advantage",
    "death_bat_avg_diff": "Death-overs batting-score advantage",
    "dot_ball_bat_diff": "Batting dot-ball pressure difference",
    "pp_bowl_wkts_diff": "Powerplay wicket-taking advantage",
    "death_bowl_wkts_diff": "Death-overs wicket-taking advantage",
    "bowl_economy_diff": "Bowling economy difference",
    "weather_temperature_c": "Live weather temperature",
    "weather_humidity_pct": "Live weather humidity",
    "weather_dew_point_c": "Live weather dew point",
    "weather_rain_probability_pct": "Live weather rain probability",
    "weather_dew_index": "Weather dew index",
    "weather_heat_index": "Weather heat index",
    "team1_left_hand_batters": "Team A left-hand batters listed",
    "team2_left_hand_batters": "Team B left-hand batters listed",
    "left_hand_batter_diff": "Left-hand batting matchup difference",
    "right_hand_batter_diff": "Right-hand batting matchup difference",
    "left_arm_bowler_diff": "Left-arm bowling matchup difference",
    "right_arm_bowler_diff": "Right-arm bowling matchup difference",
    "batting_hand_balance_diff": "Batting hand balance difference",
}


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def ensure_feature_files() -> None:
    from feature_engineering import build_player_impact_scores, calculate_match_features, load_all_matches

    print("Building match features from CSV inputs...")
    all_matches = load_all_matches()
    features = calculate_match_features(all_matches)
    features.to_csv("match_features.csv", index=False)
    print(f"Saved match_features.csv: {features.shape[0]} rows, {features.shape[1]} columns")

    players = build_player_impact_scores()
    players.to_csv("player_impact_scores.csv", index=False)
    print(f"Saved player_impact_scores.csv: {players.shape[0]} players")


def recency_weight(season_value: object) -> float:
    text = str(season_value)
    if "2026" in text:
        return 3.0
    if "2025" in text:
        return 2.0
    if "2024" in text:
        return 1.5
    return 1.0


def prepare_data() -> dict:
    df = pd.read_csv("match_features.csv")
    df["date"] = pd.to_datetime(df["date"])

    df_test = df[(df["date"] >= FINAL_MATCH_DATE) & (df["team1"] == TEAM_A) & (df["team2"] == TEAM_B)].copy()
    if df_test.empty:
        df_test = df[df["date"] >= FINAL_MATCH_DATE].copy()
    if df_test.empty:
        raise RuntimeError("Could not find the IPL 2026 final row in match_features.csv")

    df_test = df_test.sort_values("date").tail(1).copy()
    df_train = df[df["date"] < PREDICTION_TARGET_DATE].copy()

    exclude_cols = {"match_id", "date", "season", "team1", "team2", "venue", "winner_label"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    for col in feature_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    x_train_raw = df_train[feature_cols]
    x_test_raw = df_test[feature_cols]
    y_train = df_train["winner_label"].astype(int).to_numpy()
    y_test = df_test["winner_label"].astype(int).to_numpy()
    sample_weight = df_train["season"].map(recency_weight).astype(float).to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_test_imp = imputer.transform(x_test_raw)
    x_train = scaler.fit_transform(x_train_imp)
    x_test = scaler.transform(x_test_imp)

    return {
        "df_train": df_train,
        "df_test": df_test,
        "feature_cols": feature_cols,
        "x_train_raw": x_train_raw,
        "x_test_raw": x_test_raw,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "sample_weight": sample_weight,
        "imputer": imputer,
        "scaler": scaler,
    }


def model_catalog() -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=450,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=5,
            learning_rate=0.03,
            n_estimators=350,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            objective="binary",
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=350,
            subsample=0.85,
            feature_fraction=0.85,
            min_child_samples=10,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }


def fit_with_optional_weight(model: object, x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray | None = None) -> object:
    try:
        if weights is None:
            return model.fit(x_values, y_values)
        return model.fit(x_values, y_values, sample_weight=weights)
    except TypeError:
        return model.fit(x_values, y_values)


def metric_row(model_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    auc_value = np.nan
    if len(np.unique(y_true)) == 2:
        auc_value = roc_auc_score(y_true, y_prob)
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": auc_value,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_base_models(x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray) -> pd.DataFrame:
    class_counts = np.bincount(y_train)
    n_splits = int(min(CV_FOLDS, class_counts.min()))
    if n_splits < 2:
        raise RuntimeError("Not enough class variety for cross-validation")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in model_catalog().items():
        fold_rows = []
        for train_idx, val_idx in cv.split(x_train, y_train):
            fitted = clone(model)
            fit_with_optional_weight(fitted, x_train[train_idx], y_train[train_idx], sample_weight[train_idx])
            prob = fitted.predict_proba(x_train[val_idx])[:, 1]
            fold_rows.append(metric_row(name, y_train[val_idx], prob))

        averaged = {"model": name}
        for metric_name in ["accuracy", "auc_roc", "precision", "recall", "f1"]:
            averaged[metric_name] = float(np.nanmean([row[metric_name] for row in fold_rows]))
        rows.append(averaged)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    return metrics


def train_stacked_models(x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray, x_test: np.ndarray) -> dict:
    class_counts = np.bincount(y_train)
    n_splits = int(min(CV_FOLDS, class_counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    base_template = model_catalog()
    stack_names = ["Random Forest", "XGBoost", "LightGBM"]
    oof = np.zeros((x_train.shape[0], len(stack_names)))

    for fold, (train_idx, val_idx) in enumerate(cv.split(x_train, y_train), start=1):
        print(f"Training stacking fold {fold}/{n_splits}...")
        for col_idx, name in enumerate(stack_names):
            fitted = clone(base_template[name])
            fit_with_optional_weight(fitted, x_train[train_idx], y_train[train_idx], sample_weight[train_idx])
            oof[val_idx, col_idx] = fitted.predict_proba(x_train[val_idx])[:, 1]

    meta = LogisticRegression(C=1.0, penalty="l2", max_iter=2000, random_state=RANDOM_STATE)
    fit_with_optional_weight(meta, oof, y_train, sample_weight)

    fitted_base = {}
    final_base_probs = {}
    for name in stack_names:
        fitted = clone(base_template[name])
        fit_with_optional_weight(fitted, x_train, y_train, sample_weight)
        fitted_base[name] = fitted
        final_base_probs[name] = float(fitted.predict_proba(x_test)[:, 1][0])

    test_meta = np.column_stack([final_base_probs[name] for name in stack_names]).reshape(1, -1)
    stacked_prob = float(meta.predict_proba(test_meta)[:, 1][0])
    oof_stacked_prob = meta.predict_proba(oof)[:, 1]
    stacked_metrics = metric_row("Stacked Ensemble", y_train, oof_stacked_prob)

    return {
        "base_models": fitted_base,
        "meta_model": meta,
        "stack_names": stack_names,
        "oof": oof,
        "final_base_probs": final_base_probs,
        "stacked_prob": stacked_prob,
        "stacked_metrics": stacked_metrics,
    }


def save_model_artifacts(bundle: dict, data: dict) -> None:
    joblib.dump(bundle["base_models"]["Random Forest"], MODELS_DIR / "rf_full.joblib")
    joblib.dump(bundle["base_models"]["XGBoost"], MODELS_DIR / "xgb_full.joblib")
    joblib.dump(bundle["base_models"]["LightGBM"], MODELS_DIR / "lgb_full.joblib")
    joblib.dump(bundle["meta_model"], MODELS_DIR / "meta_stacked.joblib")
    joblib.dump(data["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(data["imputer"], MODELS_DIR / "imputer.joblib")
    joblib.dump(data["feature_cols"], MODELS_DIR / "feature_cols.joblib")


def predict_stacked_from_raw(raw_values: np.ndarray, data: dict, bundle: dict) -> np.ndarray:
    raw_2d = np.asarray(raw_values, dtype=float)
    if raw_2d.ndim == 1:
        raw_2d = raw_2d.reshape(1, -1)
    x_values = data["scaler"].transform(data["imputer"].transform(raw_2d))
    base_probs = [bundle["base_models"][name].predict_proba(x_values)[:, 1] for name in bundle["stack_names"]]
    meta_input = np.column_stack(base_probs)
    return bundle["meta_model"].predict_proba(meta_input)[:, 1]


def run_monte_carlo(data: dict, bundle: dict, n_simulations: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    feature_cols = data["feature_cols"]
    raw_final = data["x_test_raw"].iloc[0].to_numpy(dtype=float)
    raw_std = data["x_train_raw"].std(numeric_only=True).reindex(feature_cols).fillna(0.0).to_numpy(dtype=float)
    replacement = np.nanmedian(raw_std[raw_std > 0]) if np.any(raw_std > 0) else 0.1
    raw_std = np.where(raw_std <= 0, replacement, raw_std)

    noise = rng.uniform(-1.0, 1.0, size=(n_simulations, len(feature_cols))) * raw_std
    simulated_raw = raw_final.reshape(1, -1) + noise
    probs = predict_stacked_from_raw(simulated_raw, data, bundle)

    sim_df = pd.DataFrame({"team_a_win_probability": probs})
    sim_df.to_csv(REPORTS_DIR / "monte_carlo_simulations.csv", index=False)
    np.save(MODELS_DIR / "sim_probs.npy", probs)

    plt.figure(figsize=(8, 5))
    plt.hist(probs * 100.0, bins=35, color="#2878b5", alpha=0.85, edgecolor="white")
    plt.axvline(probs.mean() * 100.0, color="#b52626", linewidth=2, label="Mean")
    plt.xlabel(f"{TEAM_A} win probability (%)")
    plt.ylabel("Simulation count")
    plt.title("Monte Carlo win probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "monte_carlo_distribution.png", dpi=160)
    plt.close()

    sensitivity_rows = []
    for idx, col in enumerate(feature_cols):
        single = np.repeat(raw_final.reshape(1, -1), 400, axis=0)
        single[:, idx] += rng.uniform(-1.0, 1.0, size=400) * raw_std[idx]
        single_probs = predict_stacked_from_raw(single, data, bundle)
        sensitivity_rows.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "probability_variance": float(np.var(single_probs)),
                "probability_range_points": float((single_probs.max() - single_probs.min()) * 100.0),
            }
        )

    sensitivity = pd.DataFrame(sensitivity_rows).sort_values("probability_variance", ascending=False)
    sensitivity.to_csv(REPORTS_DIR / "feature_sensitivity.csv", index=False)
    return sim_df, sensitivity


def select_class_one(values: object) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def select_expected_value(value: object) -> float:
    if isinstance(value, list):
        return float(value[1])
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.size > 1:
        return float(arr.ravel()[1])
    return float(arr.ravel()[0])


def build_shap_outputs(data: dict, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = data["feature_cols"]
    x_train_df = pd.DataFrame(data["x_train"], columns=feature_cols)
    x_test_df = pd.DataFrame(data["x_test"], columns=feature_cols)
    xgb_model = bundle["base_models"]["XGBoost"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_train = select_class_one(explainer.shap_values(x_train_df))
    shap_test = select_class_one(explainer.shap_values(x_test_df))
    expected_value = select_expected_value(explainer.expected_value)

    global_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "label": [FEATURE_LABELS.get(col, col) for col in feature_cols],
            "mean_abs_shap": np.abs(shap_train).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)

    final_raw = data["x_test_raw"].iloc[0]
    local_rows = []
    for idx, col in enumerate(feature_cols):
        shap_value = float(shap_test[0, idx])
        approx_points = (sigmoid(expected_value + shap_value) - sigmoid(expected_value)) * 100.0
        local_rows.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "raw_value": float(final_raw[col]),
                "scaled_value": float(x_test_df.iloc[0, idx]),
                "shap_log_odds": shap_value,
                "approx_probability_points": approx_points,
                "supports": TEAM_A if shap_value >= 0 else TEAM_B,
            }
        )

    local = pd.DataFrame(local_rows)
    local["abs_shap"] = local["shap_log_odds"].abs()
    local = local.sort_values("abs_shap", ascending=False)
    local.to_csv(REPORTS_DIR / "shap_local_explanation.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_train, x_train_df, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_global_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_train, x_train_df, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_global_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=shap_test[0],
        base_values=expected_value,
        data=x_test_df.iloc[0].to_numpy(),
        feature_names=feature_cols,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_local_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close()

    return global_importance, local


def build_player_of_match_ranking(stacked_prob: float) -> pd.DataFrame:
    players = pd.read_csv("player_impact_scores.csv")
    players = players[players["Team"].isin([TEAM_A, TEAM_B])].copy()
    players["Team_Win_Probability"] = np.where(players["Team"] == TEAM_A, stacked_prob, 1.0 - stacked_prob)

    batting_strength = (
        players["Batting_Runs"].fillna(0) / max(players["Batting_Runs"].max(), 1) * 0.55
        + players["Batting_Strike_Rate"].fillna(0) / max(players["Batting_Strike_Rate"].max(), 1) * 0.30
    )
    bowling_strength = (
        players["Wickets_Taken"].fillna(0) / max(players["Wickets_Taken"].max(), 1) * 0.55
        + (1.0 / (players["Bowling_Economy"].replace(0, np.nan))).fillna(0)
        / max((1.0 / players["Bowling_Economy"].replace(0, np.nan)).fillna(0).max(), 1e-9)
        * 0.30
    )
    players["Batting_Match_Winner_Probability"] = np.clip((batting_strength + 0.15) * players["Team_Win_Probability"], 0, 0.85)
    players["Bowling_Match_Winner_Probability"] = np.clip((bowling_strength + 0.10) * players["Team_Win_Probability"], 0, 0.85)
    players["Raw_POTM_Score"] = players["Player_Impact_Score"] * (0.45 + 0.55 * players["Team_Win_Probability"])
    players["POTM_Probability"] = players["Raw_POTM_Score"] / players["Raw_POTM_Score"].sum() * 100.0
    players = players.sort_values("POTM_Probability", ascending=False)
    players.to_csv(REPORTS_DIR / "player_of_match_ranking.csv", index=False)
    return players


def player_reason(row: pd.Series) -> str:
    parts = []
    if row["Batting_Runs"] > 250:
        parts.append(f"{int(row['Batting_Runs'])} runs at SR {row['Batting_Strike_Rate']:.1f}")
    if row["Wickets_Taken"] > 0:
        parts.append(f"{int(row['Wickets_Taken'])} wickets at economy {row['Bowling_Economy']:.2f}")
    parts.append(f"impact score {row['Player_Impact_Score']:.1f}/100")
    parts.append(f"team win probability input {row['Team_Win_Probability'] * 100:.1f}%")
    return "; ".join(parts)


def build_scenarios(data: dict, bundle: dict, top_player: pd.Series) -> pd.DataFrame:
    feature_cols = data["feature_cols"]
    raw = data["x_test_raw"].iloc[0].to_numpy(dtype=float)
    raw_series = pd.Series(raw, index=feature_cols)
    raw_std = data["x_train_raw"].std(numeric_only=True).reindex(feature_cols).fillna(0.0)
    baseline = float(predict_stacked_from_raw(raw, data, bundle)[0])
    rows = [{"scenario": "Baseline final input", "team_a_win_probability": baseline}]

    if "team1_toss_advantage" in feature_cols:
        toss_corr = raw_series.get("toss_win_match_win_corr", 0.5)
        bat_first_rate = raw_series.get("bat_first_win_rate_venue", 0.5)

        gt_fields = raw_series.copy()
        gt_fields["team1_toss_advantage"] = toss_corr - 0.5
        rows.append({"scenario": f"{TEAM_A} wins toss and fields first", "team_a_win_probability": float(predict_stacked_from_raw(gt_fields.to_numpy(), data, bundle)[0])})

        team_b_fields = raw_series.copy()
        team_b_fields["team1_toss_advantage"] = 0.5 - toss_corr
        rows.append({"scenario": f"{TEAM_B} wins toss and fields first", "team_a_win_probability": float(predict_stacked_from_raw(team_b_fields.to_numpy(), data, bundle)[0])})

        team_a_bats = raw_series.copy()
        team_a_bats["team1_toss_advantage"] = bat_first_rate - 0.5
        rows.append({"scenario": f"{TEAM_A} wins toss and bats first", "team_a_win_probability": float(predict_stacked_from_raw(team_a_bats.to_numpy(), data, bundle)[0])})

    unavailable = raw_series.copy()
    direction = -1.0 if top_player["Team"] == TEAM_A else 1.0
    if "Batsman" in str(top_player["Role"]) or "All-Rounder" in str(top_player["Role"]):
        for col in ["pp_bat_avg_diff", "mid_bat_avg_diff", "death_bat_avg_diff"]:
            if col in unavailable:
                unavailable[col] += direction * 0.35 * raw_std.get(col, 0.0)
    if "Bowler" in str(top_player["Role"]) or "All-Rounder" in str(top_player["Role"]):
        if "death_bowl_wkts_diff" in unavailable:
            unavailable["death_bowl_wkts_diff"] += direction * 0.35 * raw_std.get("death_bowl_wkts_diff", 0.0)
        if "bowl_economy_diff" in unavailable:
            unavailable["bowl_economy_diff"] -= direction * 0.35 * raw_std.get("bowl_economy_diff", 0.0)

    rows.append(
        {
            "scenario": f"{top_player['Player_Name']} unavailable (approx feature shock)",
            "team_a_win_probability": float(predict_stacked_from_raw(unavailable.to_numpy(), data, bundle)[0]),
        }
    )

    scenarios = pd.DataFrame(rows)
    scenarios["team_b_win_probability"] = 1.0 - scenarios["team_a_win_probability"]
    scenarios["team_a_probability_shift_points"] = (scenarios["team_a_win_probability"] - baseline) * 100.0
    scenarios.to_csv(REPORTS_DIR / "scenario_sensitivity.csv", index=False)
    return scenarios


def playoff_base_matches() -> pd.DataFrame:
    from feature_engineering import load_all_matches

    base = load_all_matches()
    base["date"] = pd.to_datetime(base["date"])
    # Keep actual results known at the prediction date, including Semi Final 1.
    # Future playoff rows are excluded so remaining matches are still simulated.
    return base[base["date"] < PREDICTION_TARGET_DATE].copy()


def future_match_feature_row(
    base_matches: pd.DataFrame,
    team1: str,
    team2: str,
    venue: str,
    date_value: str,
    match_type: str,
    toss_winner: str,
    toss_decision: str,
) -> pd.Series:
    from feature_engineering import calculate_match_features, normalize_toss_decision

    date = pd.to_datetime(date_value)
    match_id = 900000 + abs(hash((team1, team2, venue, date_value, toss_winner, toss_decision))) % 99999
    future = pd.DataFrame(
        [
            {
                "match_id": match_id,
                "date": date,
                "season": "2026",
                "city": venue.split(",")[-1].strip(),
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": normalize_toss_decision(toss_decision),
                "winner": team1,
                "player_of_match": "Unknown",
                "match_type": match_type,
                "overs": 20,
                "balls_per_over": 6,
            }
        ]
    )
    frame = pd.concat([base_matches[base_matches["date"] < date], future], ignore_index=True)
    features = calculate_match_features(frame).sort_values("date")
    return features.tail(1).iloc[0]


def venue_decision_weights(venue: str) -> dict[str, float]:
    name = venue.lower()
    if "ahmedabad" in name or "narendra modi" in name:
        return {"field": 0.70, "bat": 0.30}
    if "dharamshala" in name or "hpca" in name:
        return {"field": 0.72, "bat": 0.28}
    if "chandigarh" in name or "yadavindra" in name:
        return {"field": 0.55, "bat": 0.45}
    return {"field": 0.60, "bat": 0.40}


def predict_fixture_probability(
    base_matches: pd.DataFrame,
    data: dict,
    bundle: dict,
    team1: str,
    team2: str,
    venue: str,
    date_value: str,
    match_type: str,
) -> dict:
    rows = []
    weighted_prob = 0.0
    decision_weights = venue_decision_weights(venue)
    for toss_winner in [team1, team2]:
        for decision, decision_weight in decision_weights.items():
            scenario_weight = 0.5 * decision_weight
            feature_row = future_match_feature_row(
                base_matches=base_matches,
                team1=team1,
                team2=team2,
                venue=venue,
                date_value=date_value,
                match_type=match_type,
                toss_winner=toss_winner,
                toss_decision=decision,
            )
            raw = feature_row[data["feature_cols"]].astype(float).to_numpy()
            stacked_prob = float(predict_stacked_from_raw(raw, data, bundle)[0])
            xgb_raw = data["scaler"].transform(data["imputer"].transform(raw.reshape(1, -1)))
            xgb_prob = float(bundle["base_models"]["XGBoost"].predict_proba(xgb_raw)[:, 1][0])
            weighted_prob += scenario_weight * stacked_prob
            rows.append(
                {
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "date": date_value,
                    "match_type": match_type,
                    "toss_winner": toss_winner,
                    "toss_decision": decision,
                    "scenario_weight": scenario_weight,
                    "team1_stacked_probability": stacked_prob,
                    "team1_xgb_probability": xgb_prob,
                }
            )
    return {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "date": date_value,
        "match_type": match_type,
        "team1_win_probability": weighted_prob,
        "team2_win_probability": 1.0 - weighted_prob,
        "scenarios": rows,
    }


def build_playoff_cup_probabilities(data: dict, bundle: dict, n_simulations: int = 30000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    base = playoff_base_matches()
    teams = ["RCB", "SRH", "RR"]
    venues = {
        "semi_1": "HPCA Stadium, Dharamshala",
        "semi_2": "Maharaja Yadavindra Singh Stadium, New Chandigarh",
        "final": "Narendra Modi Stadium, Ahmedabad",
    }

    fixture_cache: dict[tuple[str, str, str], dict] = {}

    def fixture(stage: str, team1: str, team2: str) -> dict:
        key = (stage, team1, team2)
        if key not in fixture_cache:
            date_map = {"semi_1": "2026-05-26", "semi_2": "2026-05-27", "final": "2026-05-31"}
            type_map = {"semi_1": "Semi Final 1", "semi_2": "Semi Final 2", "final": "Final"}
            fixture_cache[key] = predict_fixture_probability(
                base,
                data,
                bundle,
                team1,
                team2,
                venues[stage],
                date_map[stage],
                type_map[stage],
            )
        return fixture_cache[key]

    semi_1 = {
        "team1": "RCB",
        "team2": "GT",
        "venue": venues["semi_1"],
        "date": "2026-05-26",
        "match_type": "Semi Final 1",
        "team1_win_probability": 1.0,
        "team2_win_probability": 0.0,
        "scenarios": [
            {
                "team1": "RCB",
                "team2": "GT",
                "venue": venues["semi_1"],
                "date": "2026-05-26",
                "match_type": "Semi Final 1",
                "toss_winner": "GT",
                "toss_decision": "field",
                "scenario_weight": 1.0,
                "team1_stacked_probability": 1.0,
                "team1_xgb_probability": 1.0,
                "actual_result": "RCB won by 92 runs",
            }
        ],
    }
    fixture_cache[("semi_1", "RCB", "GT")] = semi_1
    semi_2 = fixture("semi_2", "SRH", "RR")

    champion_counts = {team: 0 for team in teams}
    final_counts = {team: 0 for team in teams}
    final_pair_counts: dict[str, int] = {}

    for _ in range(n_simulations):
        finalist_1 = "RCB"
        finalist_2 = "SRH" if rng.random() < semi_2["team1_win_probability"] else "RR"
        final_counts[finalist_1] += 1
        final_counts[finalist_2] += 1
        pair_name = f"{finalist_1} vs {finalist_2}"
        final_pair_counts[pair_name] = final_pair_counts.get(pair_name, 0) + 1

        final = fixture("final", finalist_1, finalist_2)
        champion = finalist_1 if rng.random() < final["team1_win_probability"] else finalist_2
        champion_counts[champion] += 1

    team_rows = []
    for team in teams:
        team_rows.append(
            {
                "team": team,
                "cup_probability": champion_counts[team] / n_simulations,
                "final_appearance_probability": final_counts[team] / n_simulations,
                "semi_1_win_probability": 1.0 if team == "RCB" else np.nan,
                "semi_2_win_probability": semi_2["team1_win_probability"] if team == "SRH" else (semi_2["team2_win_probability"] if team == "RR" else np.nan),
            }
        )
    team_summary = pd.DataFrame(team_rows).sort_values("cup_probability", ascending=False)
    team_summary.to_csv(REPORTS_DIR / "playoff_cup_probabilities.csv", index=False)

    pair_summary = (
        pd.DataFrame(
            [{"final_pair": pair, "probability": count / n_simulations} for pair, count in final_pair_counts.items()]
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    pair_summary.to_csv(REPORTS_DIR / "final_pair_probabilities.csv", index=False)

    fixture_rows = []
    for cached in fixture_cache.values():
        fixture_rows.append(
            {
                "match_type": cached["match_type"],
                "team1": cached["team1"],
                "team2": cached["team2"],
                "venue": cached["venue"],
                "team1_win_probability": cached["team1_win_probability"],
                "team2_win_probability": cached["team2_win_probability"],
            }
        )
    fixture_probs = pd.DataFrame(fixture_rows).sort_values(["match_type", "team1", "team2"])
    fixture_probs.to_csv(REPORTS_DIR / "playoff_fixture_probabilities.csv", index=False)

    scenario_rows = []
    for cached in fixture_cache.values():
        scenario_rows.extend(cached["scenarios"])
    pd.DataFrame(scenario_rows).to_csv(REPORTS_DIR / "playoff_toss_scenario_probabilities.csv", index=False)

    return team_summary, pair_summary, fixture_probs


def legal_ball_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["wide"].fillna(0).astype(float).eq(0)


def build_player_performance_forecast(stacked_prob: float) -> pd.DataFrame:
    deliveries = pd.read_csv("ipl_2026_deliveries.csv")
    players = pd.read_csv("player_impact_scores.csv")
    top_batsmen = pd.read_csv("ipl_2026_top_batsmen.csv")
    top_bowlers = pd.read_csv("ipl_2026_top_bowlers.csv")

    finalists = players[players["Team"].isin([TEAM_A, TEAM_B])].copy()
    potm = read_if_exists(REPORTS_DIR / "player_of_match_ranking.csv")
    if not potm.empty:
        finalists = finalists.merge(potm[["Player_Name", "POTM_Probability"]], on="Player_Name", how="left")
    else:
        finalists["POTM_Probability"] = np.nan

    forecast_rows = []
    wicket_types_excluded = {"run out", "retired hurt", "retired out", "obstructing the field"}

    for _, player in finalists.iterrows():
        name = player["Player_Name"]
        team = player["Team"]
        team_win_probability = stacked_prob if team == TEAM_A else 1.0 - stacked_prob

        batting = deliveries[deliveries["striker"] == name].copy()
        if batting.empty:
            match_runs = pd.Series(dtype=float)
            match_balls = pd.Series(dtype=float)
        else:
            match_runs = batting.groupby("match_id")["runs_of_bat"].sum().astype(float)
            match_balls = batting[legal_ball_mask(batting)].groupby("match_id").size().astype(float)

        top_bat = top_batsmen[top_batsmen["Player"] == name]
        season_runs_per_match = float(top_bat["Runs"].iloc[0] / max(top_bat["Matches"].iloc[0], 1)) if not top_bat.empty else float(player["Batting_Runs"] / 14.0)
        season_sr = float(top_bat["Strike_Rate"].iloc[0]) if not top_bat.empty else float(player["Batting_Strike_Rate"])
        empirical_mean = float(match_runs.mean()) if len(match_runs) else season_runs_per_match
        recent_mean = float(match_runs.tail(3).mean()) if len(match_runs) >= 3 else empirical_mean
        expected_runs = 0.45 * empirical_mean + 0.35 * season_runs_per_match + 0.20 * recent_mean
        run_floor = max(0.0, float(match_runs.quantile(0.25)) if len(match_runs) >= 4 else expected_runs * 0.55)
        run_ceiling = max(run_floor + 1.0, float(match_runs.quantile(0.75)) if len(match_runs) >= 4 else expected_runs * 1.55)
        expected_balls = expected_runs / max(season_sr / 100.0, 0.75)

        matches_batted = max(len(match_runs), int(player.get("Matches_Played", 14) if "Matches_Played" in player else 14))
        dismissals = deliveries[deliveries["player_dismissed"] == name]["match_id"].nunique()
        dismissal_probability = (dismissals + 1.0) / (matches_batted + 2.0)

        bowling = deliveries[deliveries["bowler"] == name].copy()
        if bowling.empty:
            match_wickets = pd.Series(dtype=float)
            match_overs = pd.Series(dtype=float)
        else:
            legal_bowling = bowling[legal_ball_mask(bowling)]
            wickets = bowling[
                bowling["player_dismissed"].notna()
                & ~bowling["wicket_type"].fillna("").str.lower().isin(wicket_types_excluded)
            ]
            match_wickets = wickets.groupby("match_id").size().astype(float)
            match_overs = legal_bowling.groupby("match_id").size().astype(float) / 6.0

        top_bowl = top_bowlers[top_bowlers["Player"] == name]
        season_wickets_per_match = float(top_bowl["Wickets"].iloc[0] / max(top_bowl["Matches"].iloc[0], 1)) if not top_bowl.empty else float(player["Wickets_Taken"] / 14.0)
        bowling_matches = max(bowling["match_id"].nunique(), 1)
        empirical_wickets = float(match_wickets.sum() / bowling_matches) if len(match_wickets) else 0.0
        expected_wickets = 0.55 * empirical_wickets + 0.45 * season_wickets_per_match
        wicket_probability = (float((match_wickets.reindex(bowling["match_id"].unique(), fill_value=0) >= 1).sum()) + 1.0) / (bowling_matches + 2.0) if len(bowling) else 0.0

        forecast_rows.append(
            {
                "Player_Name": name,
                "Team": team,
                "Role": player["Role"],
                "team_win_probability": team_win_probability,
                "expected_runs": expected_runs,
                "likely_runs_low": run_floor,
                "likely_runs_high": run_ceiling,
                "expected_balls_faced": expected_balls,
                "probability_30_plus": ((match_runs >= 30).sum() + 1.0) / (len(match_runs) + 2.0) if len(match_runs) else np.nan,
                "probability_50_plus": ((match_runs >= 50).sum() + 1.0) / (len(match_runs) + 2.0) if len(match_runs) else np.nan,
                "dismissal_probability": dismissal_probability,
                "not_out_probability": 1.0 - dismissal_probability,
                "expected_wickets": expected_wickets,
                "probability_takes_wicket": wicket_probability,
                "expected_overs_bowled": float(match_overs.mean()) if len(match_overs) else 0.0,
                "POTM_Probability": player.get("POTM_Probability", np.nan),
            }
        )

    forecast = pd.DataFrame(forecast_rows)
    forecast = forecast.sort_values(["POTM_Probability", "expected_runs"], ascending=False)
    forecast.to_csv(REPORTS_DIR / "player_performance_forecast.csv", index=False)
    return forecast


def read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_final_report(
    data: dict,
    bundle: dict,
    metrics: pd.DataFrame,
    sim_df: pd.DataFrame,
    sensitivity: pd.DataFrame,
    global_shap: pd.DataFrame,
    local_shap: pd.DataFrame,
    players: pd.DataFrame,
    scenarios: pd.DataFrame,
    cup_probs: pd.DataFrame,
    final_pairs: pd.DataFrame,
    player_forecast: pd.DataFrame,
) -> Path:
    final_row = data["df_test"].iloc[0]
    team_a_prob = bundle["stacked_prob"]
    team_b_prob = 1.0 - team_a_prob
    mean_prob = sim_df["team_a_win_probability"].mean()
    ci_low = sim_df["team_a_win_probability"].quantile(0.025)
    ci_high = sim_df["team_a_win_probability"].quantile(0.975)
    ci_width = (ci_high - ci_low) * 100.0
    base_probs = np.array(list(bundle["final_base_probs"].values()), dtype=float)
    model_disagreement = float(base_probs.max() - base_probs.min())
    mean_auc = float(metrics["auc_roc"].mean())
    if abs(team_a_prob - 0.5) < 0.07 or model_disagreement > 0.25 or mean_auc < 0.56:
        confidence = "LOW"
    elif abs(team_a_prob - 0.5) >= 0.12 and ci_width <= 12:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"
    predicted = TEAM_A if team_a_prob >= 0.5 else TEAM_B
    top_player = players.iloc[0]

    model_probs = pd.DataFrame(
        [
            {
                "model": name,
                "team_a": TEAM_A,
                "team_b": TEAM_B,
                "team_a_win_probability": prob,
                "team_b_win_probability": 1.0 - prob,
            }
            for name, prob in bundle["final_base_probs"].items()
        ]
        + [
            {
                "model": "Stacked Ensemble",
                "team_a": TEAM_A,
                "team_b": TEAM_B,
                "team_a_win_probability": team_a_prob,
                "team_b_win_probability": team_b_prob,
            }
        ]
    )
    model_probs.to_csv(REPORTS_DIR / "final_model_probabilities.csv", index=False)
    meta_weights = pd.DataFrame(
        {
            "base_model": bundle["stack_names"],
            "meta_weight": bundle["meta_model"].coef_[0],
        }
    )
    meta_weights["intercept"] = bundle["meta_model"].intercept_[0]
    meta_weights.to_csv(REPORTS_DIR / "stacking_meta_weights.csv", index=False)

    metrics_out = pd.concat([metrics, pd.DataFrame([bundle["stacked_metrics"]])], ignore_index=True)
    metrics_out.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    lines = [
        "# IPL 2026 Final Explainable AI Prediction Report",
        "",
        f"Featured final candidate: {TEAM_A} vs {TEAM_B}",
        f"Venue: {final_row['venue']}",
        f"Final match date: {pd.to_datetime(final_row['date']).date()}",
        f"Prediction target date: {PREDICTION_TARGET_DATE.date()}",
        "Known playoff update: RCB beat GT by 92 runs in Semi Final 1 (RCB 254/5, GT 162), so RCB is locked as the first finalist.",
        "",
        "## 1. Predicted Winner",
        "",
        f"Predicted winner: **{predicted}**",
        f"{TEAM_A} win probability: **{team_a_prob * 100:.1f}%**",
        f"{TEAM_B} win probability: **{team_b_prob * 100:.1f}%**",
        f"Monte Carlo mean: **{mean_prob * 100:.1f}%** for {TEAM_A}",
        f"Monte Carlo 95% CI: **{ci_low * 100:.1f}% to {ci_high * 100:.1f}%**",
        f"Confidence level: **{confidence}**",
        f"Model disagreement across base learners: **{model_disagreement * 100:.1f} percentage points**",
        f"Mean cross-validation AUC across base learners: **{mean_auc:.3f}**",
        "",
        "## 2. Four-Team Playoff Cup Probabilities",
        "",
    ]

    for _, row in cup_probs.iterrows():
        lines.append(
            f"- {row['team']}: Cup {row['cup_probability'] * 100:.1f}%, "
            f"final appearance {row['final_appearance_probability'] * 100:.1f}%"
        )

    lines.extend(["", "Most likely final pairings from bracket simulation:", ""])
    for _, row in final_pairs.head(5).iterrows():
        lines.append(f"- {row['final_pair']}: {row['probability'] * 100:.1f}%")

    lines.extend(
        [
            "",
            "Note: these odds lock the known Semi Final 1 result, then simulate Semi Final 2 and the Final. The featured final section is one candidate final row until the other finalist is confirmed.",
            "",
            "## 3. Model Confidence",
        "",
        ]
    )

    for _, row in model_probs.iterrows():
        lines.append(f"- {row['model']}: {row['team_a_win_probability'] * 100:.1f}% {TEAM_A}, {row['team_b_win_probability'] * 100:.1f}% {TEAM_B}")

    lines.extend(["", "Stacking meta-learner weights:", ""])
    for _, row in meta_weights.iterrows():
        sign = "+" if row["meta_weight"] >= 0 else ""
        lines.append(f"- {row['base_model']}: {sign}{row['meta_weight']:.4f}")

    if (team_a_prob < 0.5 and (base_probs > 0.5).all()) or (team_a_prob > 0.5 and (base_probs < 0.5).all()):
        lines.extend(
            [
                "",
                "Model agreement note: the base tree models favor the opposite side from the stacked output. The stacked meta-learner is therefore treated as a low-confidence calibration layer, not a strong cricket signal.",
            ]
        )

    lines.extend(["", "## 4. SHAP Key Deciding Factors", ""])
    if (team_a_prob < 0.5 and bundle["final_base_probs"]["XGBoost"] > 0.5) or (team_a_prob > 0.5 and bundle["final_base_probs"]["XGBoost"] < 0.5):
        lines.extend(
            [
                "Important: the SHAP explanation below is for the XGBoost tree model, as required for tree SHAP. In this run XGBoost favors a different side than the final stacked probability, so these factors explain the tree-model push, while the final stacked winner remains the calibrated ensemble output.",
                "",
            ]
        )
    for _, row in local_shap.head(7).iterrows():
        sign = "+" if row["approx_probability_points"] >= 0 else ""
        lines.append(
            f"- {row['label']}: raw value {row['raw_value']:.3f}; supports {row['supports']}; "
            f"approx impact {sign}{row['approx_probability_points']:.2f} percentage points"
        )

    lines.extend(["", "Top global historical factors from XGBoost SHAP:", ""])
    for _, row in global_shap.head(5).iterrows():
        lines.append(f"- {row['label']}: mean |SHAP| {row['mean_abs_shap']:.4f}")

    lines.extend(["", "## 5. Player Of The Match Ranking", ""])
    lines.append(f"Most likely POTM: **{top_player['Player_Name']} ({top_player['Team']})** - {top_player['POTM_Probability']:.1f}%")
    lines.append(f"Reason: {player_reason(top_player)}.")
    lines.append("")
    for rank, (_, row) in enumerate(players.head(5).iterrows(), start=1):
        lines.append(
            f"{rank}. {row['Player_Name']} ({row['Team']}) - {row['POTM_Probability']:.1f}% "
            f"[bat win {row['Batting_Match_Winner_Probability'] * 100:.1f}%, bowl win {row['Bowling_Match_Winner_Probability'] * 100:.1f}%]"
        )

    featured_forecast = player_forecast[player_forecast["Player_Name"].eq(top_player["Player_Name"])]
    lines.extend(["", f"## 6. {top_player['Player_Name']} Performance Forecast", ""])
    if not featured_forecast.empty:
        row = featured_forecast.iloc[0]
        lines.extend(
            [
                f"Expected batting runs: **{row['expected_runs']:.1f}**",
                f"Likely runs band: **{row['likely_runs_low']:.0f} to {row['likely_runs_high']:.0f}**",
                f"Expected balls faced: **{row['expected_balls_faced']:.1f}**",
                f"30+ probability: **{row['probability_30_plus'] * 100:.1f}%**",
                f"50+ probability: **{row['probability_50_plus'] * 100:.1f}%**",
                f"Dismissal probability: **{row['dismissal_probability'] * 100:.1f}%**",
                f"Not-out probability: **{row['not_out_probability'] * 100:.1f}%**",
                f"Expected wickets if he bowls: **{row['expected_wickets']:.2f}**",
                f"Probability of taking at least one wicket: **{row['probability_takes_wicket'] * 100:.1f}%**",
                "",
                f"Interpretation: {top_player['Player_Name']} is the current top POTM candidate for this featured final matchup.",
            ]
        )
    else:
        lines.append(f"{top_player['Player_Name']} was not found in the player forecast table.")

    lines.extend(["", "## 7. Scenario Sensitivity", ""])
    for _, row in scenarios.iterrows():
        sign = "+" if row["team_a_probability_shift_points"] >= 0 else ""
        lines.append(
            f"- {row['scenario']}: {TEAM_A} {row['team_a_win_probability'] * 100:.1f}%, "
            f"{TEAM_B} {row['team_b_win_probability'] * 100:.1f}% ({sign}{row['team_a_probability_shift_points']:.2f} pts)"
        )

    lines.extend(["", "## 8. Uncertainty Factors", ""])
    for _, row in sensitivity.head(5).iterrows():
        lines.append(f"- {row['label']}: probability range {row['probability_range_points']:.2f} pts under feature shock")

    lines.extend(
        [
            "",
            "## 9. XAI Method",
            "",
            f"The stacked ensemble is used for final probability. SHAP is applied to the XGBoost tree model, not directly to the stacked logistic meta-learner, because tree SHAP gives stable feature-level explanations for the base model. The local SHAP rows explain which inputs push the {TEAM_A}-vs-{TEAM_B} candidate final prediction toward either side. The feature matrix includes playoff live-weather inputs, right/left matchup inputs, and the known Semi Final 1 result when those CSV rows are available.",
            "",
            "Generated files:",
            "- reports/shap_global_bar.png",
            "- reports/shap_global_beeswarm.png",
            "- reports/shap_local_waterfall.png",
            "- reports/monte_carlo_distribution.png",
            "- reports/shap_local_explanation.csv",
            "- reports/player_of_match_ranking.csv",
            "- reports/player_performance_forecast.csv",
            "- reports/playoff_cup_probabilities.csv",
            "- reports/final_pair_probabilities.csv",
            "- reports/playoff_fixture_probabilities.csv",
            "- reports/scenario_sensitivity.csv",
            "- reports/stacking_meta_weights.csv",
        ]
    )

    report_path = REPORTS_DIR / "ipl_2026_final_prediction_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()
    ensure_feature_files()

    print("Preparing training and final prediction matrices...")
    data = prepare_data()
    print(f"Train shape: {data['x_train'].shape}; final test shape: {data['x_test'].shape}")

    print("Evaluating base models with recency weighting...")
    metrics = evaluate_base_models(data["x_train"], data["y_train"], data["sample_weight"])
    print(metrics.to_string(index=False))

    print("Training stacked ensemble...")
    bundle = train_stacked_models(data["x_train"], data["y_train"], data["sample_weight"], data["x_test"])
    print(f"Stacked {TEAM_A} win probability: {bundle['stacked_prob'] * 100:.2f}%")
    print(f"Stacked {TEAM_B} win probability: {(1.0 - bundle['stacked_prob']) * 100:.2f}%")
    save_model_artifacts(bundle, data)

    print("Running Monte Carlo simulation...")
    sim_df, sensitivity = run_monte_carlo(data, bundle, n_simulations=10000)

    print("Building SHAP explanations...")
    global_shap, local_shap = build_shap_outputs(data, bundle)

    print("Ranking Player of the Match candidates...")
    players = build_player_of_match_ranking(bundle["stacked_prob"])
    scenarios = build_scenarios(data, bundle, players.iloc[0])

    print("Simulating four-team playoff cup probabilities...")
    cup_probs, final_pairs, fixture_probs = build_playoff_cup_probabilities(data, bundle)
    print(cup_probs.to_string(index=False))

    print("Forecasting player match performance...")
    player_forecast = build_player_performance_forecast(bundle["stacked_prob"])

    report_path = write_final_report(
        data,
        bundle,
        metrics,
        sim_df,
        sensitivity,
        global_shap,
        local_shap,
        players,
        scenarios,
        cup_probs,
        final_pairs,
        player_forecast,
    )
    print(f"Report saved: {report_path}")
    print(f"Plots and CSV outputs saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
