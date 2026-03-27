import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ANTIBIOTIC_TARGETS = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
LABEL_MAP = {"S": 0, "I": 1, "R": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

BREAKPOINTS = {
    "IMIPENEM": {"R": 13, "S": 16},
    "CEFTAZIDIME": {"R": 14, "S": 18},
    "GENTAMICIN": {"R": 12, "S": 15},
    "AUGMENTIN": {"R": 13, "S": 18},
    "CIPROFLOXACIN": {"R": 15, "S": 21},
}

COL_MAP = {
    "IPM": "IMIPENEM",
    "CZ": "CEFTAZIDIME",
    "GEN": "GENTAMICIN",
    "AMC": "AUGMENTIN",
    "CIP": "CIPROFLOXACIN",
}


def normalize_label(val):
    if pd.isna(val) or str(val).strip() in ["?", "missing", ""]:
        return np.nan
    val = str(val).strip().upper()
    if val == "INTERMEDIATE":
        return "I"
    if val in ["R", "S", "I"]:
        return val
    return np.nan


def mic_to_label(value, breakpoints):
    if pd.isna(value):
        return np.nan
    if value <= breakpoints["R"]:
        return "R"
    if value <= breakpoints["S"]:
        return "I"
    return "S"


def load_dataset(path):
    p = Path(path)
    if not p.is_absolute():
        candidates = [SCRIPT_DIR / p, PROJECT_ROOT / p, DATA_DIR / p]
        p = next((c for c in candidates if c.exists()), p)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    return df


def standardize_dataframe(df):
    out = df.copy()
    out.rename(columns=COL_MAP, inplace=True)

    # Normalize likely target columns if present.
    for col in ANTIBIOTIC_TARGETS:
        if col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].apply(lambda x: mic_to_label(x, BREAKPOINTS[col]))
            else:
                out[col] = out[col].apply(normalize_label)

    # Lightweight cleanup for obvious non-feature columns.
    out.drop(columns=["ID", "Name", "Email", "Address", "Notes", "Collection_Date"], inplace=True, errors="ignore")
    return out


def detect_high_cardinality_columns(X):
    preferred = []
    for col in X.columns:
        name = col.lower()
        if "location" in name or "species" in name or "souches" in name:
            preferred.append(col)

    if preferred:
        return preferred

    high_card = []
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            nunique = X[col].nunique(dropna=True)
            if nunique >= 15:
                high_card.append(col)
    return high_card


def fit_target_encoding(train_df, y_train, cols):
    mappings = {}
    global_mean = float(np.mean(y_train))
    for col in cols:
        tmp = pd.DataFrame({"cat": train_df[col].astype(str).fillna("unknown"), "y": y_train})
        mappings[col] = tmp.groupby("cat")["y"].mean().to_dict()
    return mappings, global_mean


def apply_target_encoding(df, cols, mappings, global_mean):
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype(str).fillna("unknown").map(mappings[col]).fillna(global_mean)
    return out


def build_fold_features(X_train, X_val, y_train):
    X_train = X_train.copy()
    X_val = X_val.copy()

    high_card_cols = detect_high_cardinality_columns(X_train)
    mappings, global_mean = fit_target_encoding(X_train, y_train, high_card_cols)
    X_train = apply_target_encoding(X_train, high_card_cols, mappings, global_mean)
    X_val = apply_target_encoding(X_val, high_card_cols, mappings, global_mean)

    cat_cols = [c for c in X_train.columns if (X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("category"))]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Numeric impute + scale
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(num_imputer.fit_transform(X_train[num_cols])) if num_cols else np.empty((len(X_train), 0))
    X_val_num = scaler.transform(num_imputer.transform(X_val[num_cols])) if num_cols else np.empty((len(X_val), 0))

    # Categorical impute + ordinal encode
    cat_imputer = SimpleImputer(strategy="most_frequent")
    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_cat = ord_encoder.fit_transform(cat_imputer.fit_transform(X_train[cat_cols])) if cat_cols else np.empty((len(X_train), 0))
    X_val_cat = ord_encoder.transform(cat_imputer.transform(X_val[cat_cols])) if cat_cols else np.empty((len(X_val), 0))

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_val_final = np.hstack([X_val_num, X_val_cat])

    encoder_artifact = {
        "high_cardinality_columns": high_card_cols,
        "target_encoding_mappings": mappings,
        "target_encoding_global_mean": global_mean,
        "categorical_columns": cat_cols,
        "categorical_imputer": cat_imputer,
        "ordinal_encoder": ord_encoder,
        "numeric_columns": num_cols,
        "numeric_imputer": num_imputer,
    }
    return X_train_final, X_val_final, encoder_artifact, scaler


def build_full_features(X, y):
    X = X.copy()
    high_card_cols = detect_high_cardinality_columns(X)
    mappings, global_mean = fit_target_encoding(X, y, high_card_cols)
    X = apply_target_encoding(X, high_card_cols, mappings, global_mean)

    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("category"))]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num = scaler.fit_transform(num_imputer.fit_transform(X[num_cols])) if num_cols else np.empty((len(X), 0))

    cat_imputer = SimpleImputer(strategy="most_frequent")
    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = ord_encoder.fit_transform(cat_imputer.fit_transform(X[cat_cols])) if cat_cols else np.empty((len(X), 0))

    X_final = np.hstack([X_num, X_cat])
    encoder_artifact = {
        "high_cardinality_columns": high_card_cols,
        "target_encoding_mappings": mappings,
        "target_encoding_global_mean": global_mean,
        "categorical_columns": cat_cols,
        "categorical_imputer": cat_imputer,
        "ordinal_encoder": ord_encoder,
        "numeric_columns": num_cols,
        "numeric_imputer": num_imputer,
        "all_feature_columns": list(X.columns),
    }
    return X_final, encoder_artifact, scaler


def make_xgb_classifier(num_classes):
    if num_classes == 2:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def compute_auc(y_true, y_proba, classes):
    if len(classes) == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def apply_smotenc_on_fold(X_train, y_train, encoder_artifact, target_name, fold_id):
    cat_cols = encoder_artifact.get("categorical_columns", [])
    num_cols = encoder_artifact.get("numeric_columns", [])
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    if not cat_indices:
        print(f"  [{target_name}][fold {fold_id}] SMOTENC skipped: no categorical features after encoding.")
        return X_train, y_train

    class_counts = pd.Series(y_train).value_counts().sort_index()
    min_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_count <= 1:
        print(f"  [{target_name}][fold {fold_id}] SMOTENC skipped: a class has <=1 sample.")
        return X_train, y_train

    k_neighbors = min(5, min_count - 1)
    if k_neighbors < 1:
        print(f"  [{target_name}][fold {fold_id}] SMOTENC skipped: insufficient neighbors.")
        return X_train, y_train

    sampler = SMOTENC(
        categorical_features=cat_indices,
        k_neighbors=k_neighbors,
        random_state=42,
    )
    X_res, y_res = sampler.fit_resample(X_train, y_train)

    before_dist = class_counts.to_dict()
    after_dist = pd.Series(y_res).value_counts().sort_index().to_dict()
    print(f"  [{target_name}][fold {fold_id}] class dist before={before_dist} after={after_dist}")
    return X_res, y_res


def train_per_antibiotic(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in ANTIBIOTIC_TARGETS]
    summary = {}

    for target in ANTIBIOTIC_TARGETS:
        if target not in df.columns:
            print(f"Skipping {target}: column missing.")
            continue

        target_series = df[target].apply(normalize_label)
        valid_mask = target_series.notna()
        X = df.loc[valid_mask, feature_cols].copy()
        y = target_series.loc[valid_mask].map(LABEL_MAP).astype(int).to_numpy()

        if len(y) < 30:
            print(f"Skipping {target}: too few samples ({len(y)}).")
            continue

        class_counts = pd.Series(y).value_counts()
        if len(class_counts) < 2:
            print(f"Skipping {target}: only one class present.")
            continue

        # Requested: 10-fold Stratified K-Fold (fallback only if mathematically impossible).
        n_splits = 10 if class_counts.min() >= 10 else int(class_counts.min())
        if n_splits < 2:
            print(f"Skipping {target}: insufficient per-class samples for stratified CV.")
            continue
        if n_splits < 10:
            print(f"Warning for {target}: using {n_splits}-fold (insufficient minority class for 10-fold).")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc_scores = []
        auc_scores = []
        classes = np.unique(y)

        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            X_train_f, X_val_f, encoder_artifact, _ = build_fold_features(X_train, X_val, y_train)
            X_train_f, y_train_bal = apply_smotenc_on_fold(
                X_train_f,
                y_train,
                encoder_artifact,
                target_name=target,
                fold_id=fold_id,
            )
            model = make_xgb_classifier(num_classes=len(classes))
            model.fit(X_train_f, y_train_bal)

            y_pred = model.predict(X_val_f)
            y_proba = model.predict_proba(X_val_f)

            acc_scores.append(accuracy_score(y_val, y_pred))
            try:
                auc_scores.append(compute_auc(y_val, y_proba, classes))
            except ValueError:
                auc_scores.append(np.nan)

        acc_mean = float(np.nanmean(acc_scores))
        acc_std = float(np.nanstd(acc_scores))
        auc_mean = float(np.nanmean(auc_scores))
        auc_std = float(np.nanstd(auc_scores))

        print(
            f"{target}: Accuracy={acc_mean:.4f} ± {acc_std:.4f} | ROC-AUC={auc_mean:.4f} ± {auc_std:.4f}"
        )

        # Fit final model on full target data and export artifacts.
        X_full_f, encoder_artifact, scaler = build_full_features(X, y)
        final_model = make_xgb_classifier(num_classes=len(classes))
        final_model.fit(X_full_f, y)

        safe_target = target.lower()
        model_path = output_dir / f"{safe_target}_xgb_model.pkl"
        scaler_path = output_dir / f"{safe_target}_scaler.pkl"
        encoder_path = output_dir / f"{safe_target}_encoder.pkl"

        joblib.dump(final_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder_artifact, encoder_path)

        summary[target] = {
            "n_samples": int(len(y)),
            "classes_present": [REVERSE_LABEL_MAP.get(int(c), str(c)) for c in classes],
            "cv_folds": int(n_splits),
            "accuracy_mean": round(acc_mean, 4),
            "accuracy_std": round(acc_std, 4),
            "roc_auc_mean": round(auc_mean, 4),
            "roc_auc_std": round(auc_std, 4),
            "artifacts": {
                "model": str(model_path),
                "scaler": str(scaler_path),
                "encoder": str(encoder_path),
            },
        }

    summary_path = output_dir / "track_b_cv_summary.pkl"
    joblib.dump(summary, summary_path)
    print(f"\nSaved summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train 5 separate XGBoost antibiotic resistance models with leakage-safe 10-fold CV."
    )
    parser.add_argument("--data", required=True, help="Path to dataset file (.csv/.xlsx/.zip CSV).")
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACTS_DIR),
        help="Directory to save model/scaler/encoder artifacts.",
    )
    args = parser.parse_args()

    df_raw = load_dataset(args.data)
    df = standardize_dataframe(df_raw)

    # Keep only columns that have at least some signal.
    df = df.dropna(axis=1, how="all")
    print(f"Loaded dataset shape: {df.shape}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    train_per_antibiotic(df, output_dir)


if __name__ == "__main__":
    main()
