from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "tox21_processed.pkl"
MODELS_PATH = ARTIFACTS_DIR / "tabular_models.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "tabular_features.json"
METRICS_PATH = ARTIFACTS_DIR / "tabular_metrics.json"

SCALAR_FEATURES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "NumSaturatedRings",
    "NumAliphaticRings",
]

TARGET_COLS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def load_processed_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}")
    return pd.read_pickle(DATA_PATH)


def flatten_morgan_fp(fp_value, n_bits: int | None = None) -> np.ndarray:
    arr = np.asarray(fp_value, dtype=np.float32).reshape(-1)
    if n_bits is None:
        return arr
    if arr.shape[0] < n_bits:
        arr = np.pad(arr, (0, n_bits - arr.shape[0]), mode="constant")
    elif arr.shape[0] > n_bits:
        arr = arr[:n_bits]
    return arr


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    scalar_df = df[SCALAR_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    fp_lengths = df["morgan_fp"].apply(lambda x: len(np.asarray(x).reshape(-1)))
    fp_bits = int(fp_lengths.mode().iloc[0])
    fp_matrix = np.vstack(df["morgan_fp"].apply(lambda x: flatten_morgan_fp(x, fp_bits)))
    fp_cols = [f"morgan_fp_{i}" for i in range(fp_bits)]
    fp_df = pd.DataFrame(fp_matrix, index=df.index, columns=fp_cols, dtype=np.float32)
    features = pd.concat([scalar_df, fp_df], axis=1)
    return features, list(features.columns), fp_bits


def build_model(y_train: pd.Series) -> XGBClassifier:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )


def train_per_endpoint(features: pd.DataFrame, df: pd.DataFrame) -> tuple[dict, dict]:
    models: dict[str, XGBClassifier] = {}
    metrics: dict[str, dict] = {}

    for target in TARGET_COLS:
        valid_mask = df[target].notna()
        X = features.loc[valid_mask].reset_index(drop=True)
        y = df.loc[valid_mask, target].astype(int).reset_index(drop=True)

        if y.nunique() < 2:
            metrics[target] = {"fold_auc": [], "mean_auc": None, "n_samples": int(len(y))}
            continue

        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for train_idx, val_idx in splitter.split(X, y):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = build_model(y_train)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_val, probs)))

        final_model = build_model(y)
        final_model.fit(X, y)
        models[target] = final_model
        metrics[target] = {
            "fold_auc": fold_aucs,
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "n_samples": int(len(y)),
            "positive_rate": float(y.mean()),
        }

    return models, metrics


def save_outputs(models: dict, feature_names: list[str], metrics: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODELS_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    df = load_processed_dataframe()
    features, feature_names, _ = build_feature_matrix(df)
    models, metrics = train_per_endpoint(features, df)
    save_outputs(models, feature_names, metrics)

    print("Saved models:", MODELS_PATH)
    print("Saved feature names:", FEATURES_PATH)
    print("Saved metrics:", METRICS_PATH)
    for target, values in metrics.items():
        if values["mean_auc"] is None:
            print(f"{target}: skipped (single class or insufficient labels)")
        else:
            print(f"{target}: mean AUC={values['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
