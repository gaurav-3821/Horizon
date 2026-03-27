from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from track_c_data_loader import load_combined

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "lightgbm is required for track_c_model.py. Install it with: pip install lightgbm"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FORECAST_TARGET = "daily_new"
CLASSIFY_TARGET = "risk_label"

LAG_FEATURES = ["lag_1d", "lag_3d", "lag_7d", "lag_14d"]
ROLLING_FEATURES = ["rolling_7d", "rolling_14d", "rolling_28d"]
BASE_FEATURES = [
    "days_since_first",
    "growth_rate",
    "doubling_time",
    "Global_Rolling_Infection_Rate",
    "population_density",
    "median_age",
    "gdp_per_capita",
    "human_development_index",
    "stringency_index",
    "vaccination_rate",
    "testing_rate",
]
SPATIAL_FEATURES = ["region"]


def load_base_dataframe():
    """Load merged JHU+OWID using relative paths when available."""
    jhu_rel = DATA_DIR / "time_series_covid19_confirmed_global.csv"
    owid_rel = DATA_DIR / "owid-covid-data.csv"
    if jhu_rel.exists():
        return load_combined(str(jhu_rel), str(owid_rel))
    return load_combined()


def add_owid_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create explicit vaccination/testing features and handle merge NaNs."""
    out = df.copy()
    out["vaccination_rate"] = out.get("total_vaccinations_per_hundred", np.nan)
    out["testing_rate"] = out.get("total_tests_per_thousand", np.nan)
    out["stringency_index"] = out.get("stringency_index", np.nan)

    out = out.sort_values(["Country/Region", "date"]).reset_index(drop=True)
    out["vaccination_rate"] = out.groupby("Country/Region")["vaccination_rate"].ffill().bfill()
    out["testing_rate"] = out.groupby("Country/Region")["testing_rate"].ffill().bfill()
    out["stringency_index"] = out.groupby("Country/Region")["stringency_index"].ffill().bfill()
    return out


def engineer_spatiotemporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal lag/rolling features and a spatial categorical feature."""
    out = df.copy()
    out = out.sort_values(["Country/Region", "date"]).reset_index(drop=True)
    grouped = out.groupby("Country/Region", group_keys=False)

    for lag in [1, 3, 7, 14]:
        out[f"lag_{lag}d"] = grouped[FORECAST_TARGET].shift(lag)

    for win in [7, 14, 28]:
        out[f"rolling_{win}d"] = grouped[FORECAST_TARGET].transform(
            lambda s: s.shift(1).rolling(win, min_periods=1).mean()
        )

    if "Global_Rolling_Infection_Rate" not in out.columns:
        global_daily = out.groupby("date", as_index=False)[FORECAST_TARGET].sum().sort_values("date")
        global_daily["Global_Rolling_Infection_Rate"] = global_daily[FORECAST_TARGET].rolling(
            7, min_periods=1
        ).mean()
        out = out.merge(global_daily[["date", "Global_Rolling_Infection_Rate"]], on="date", how="left")

    out["region"] = out["Country/Region"].astype(str)
    return out


def temporal_split(df: pd.DataFrame, holdout_days: int = 45):
    """Strict chronological split: train on past, test on future."""
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] > cutoff].copy()

    if train_df.empty or test_df.empty:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def make_feature_list(df: pd.DataFrame):
    features = []
    for col in (LAG_FEATURES + ROLLING_FEATURES + BASE_FEATURES + SPATIAL_FEATURES):
        if col in df.columns:
            features.append(col)
    return features


def build_pipeline(feature_cols):
    cat_cols = [c for c in feature_cols if c in SPATIAL_FEATURES]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return preprocessor


def train_forecaster(df: pd.DataFrame):
    print("\n--- Training forecaster (LightGBM regressor, strict temporal split) ---")
    feature_cols = make_feature_list(df)
    usable = df[df[FORECAST_TARGET].notna()].copy()

    train_df, test_df = temporal_split(usable, holdout_days=45)
    X_train, y_train = train_df[feature_cols], train_df[FORECAST_TARGET]
    X_test, y_test = test_df[feature_cols], test_df[FORECAST_TARGET]

    preprocessor = build_pipeline(feature_cols)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print(f"Temporal split window: train <= {train_df['date'].max().date()}, test >= {test_df['date'].min().date()}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")
    if r2 < 0.90:
        print("Warning: R2 below target 0.90. Consider richer geo features and model tuning.")

    results = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    return pipe, results, (X_test, y_test, y_pred)


def train_classifier(df: pd.DataFrame):
    print("\n--- Training risk classifier (LightGBM classifier, strict temporal split) ---")
    feature_cols = make_feature_list(df)
    usable = df[df[CLASSIFY_TARGET].notna()].copy()
    usable = usable[usable[CLASSIFY_TARGET].astype(str).str.lower() != "nan"]

    train_df, test_df = temporal_split(usable, holdout_days=45)
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[CLASSIFY_TARGET].astype(str))
    y_test = le.transform(test_df[CLASSIFY_TARGET].astype(str))

    preprocessor = build_pipeline(feature_cols)
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results = {
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "classes": list(le.classes_),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    return pipe, le, results


def save_artifacts(forecaster, classifier, le, f_results, c_results):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    forecaster_path = ARTIFACTS_DIR / "track_c_forecaster.pkl"
    classifier_path = ARTIFACTS_DIR / "track_c_classifier.pkl"
    results_path = ARTIFACTS_DIR / "track_c_results.json"

    joblib.dump({"model": forecaster}, forecaster_path)
    joblib.dump({"model": classifier, "label_encoder": le}, classifier_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"forecaster": f_results, "classifier": c_results}, f, indent=2)

    # Backward-compatible project-root copies for existing app loaders.
    joblib.dump({"model": forecaster}, PROJECT_ROOT / "track_c_forecaster.pkl")
    joblib.dump({"model": classifier, "label_encoder": le}, PROJECT_ROOT / "track_c_classifier.pkl")
    with open(PROJECT_ROOT / "track_c_results.json", "w", encoding="utf-8") as f:
        json.dump({"forecaster": f_results, "classifier": c_results}, f, indent=2)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR.resolve()}")
    print("Saved compatibility copies in project root.")


if __name__ == "__main__":
    df = load_base_dataframe()
    df = add_owid_rate_features(df)
    df = engineer_spatiotemporal_features(df)

    forecaster, f_results, _ = train_forecaster(df)
    classifier, le, c_results = train_classifier(df)

    save_artifacts(forecaster, classifier, le, f_results, c_results)
    print("\nAll models saved.")
