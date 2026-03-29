import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_DATA_PATH = ARTIFACTS_DIR / "unified_dataset_final.csv"
MODEL_OUTPUT_PATH = ARTIFACTS_DIR / "stacked_model.pkl"
SUMMARY_OUTPUT_PATH = ARTIFACTS_DIR / "stacked_metrics.json"
DIAGNOSTIC_OUTPUT_PATH = ARTIFACTS_DIR / "xgb_feature_diagnostics.json"
HYBRID_DIAGNOSTIC_OUTPUT_PATH = ARTIFACTS_DIR / "xgb_hybrid_diagnostic.json"
MENDELEY_ONLY_METRICS_PATH = ARTIFACTS_DIR / "mendeley_only_metrics.json"
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.json"
STACKED_TUNED_MODEL_PATH = ARTIFACTS_DIR / "stacked_model_tuned.pkl"
PSEUDO_MODEL_PATH = ARTIFACTS_DIR / "stacked_model_pseudo.pkl"
PSEUDO_METRICS_PATH = ARTIFACTS_DIR / "pseudo_metrics.json"
SHAP_GLOBAL_PATH = ARTIFACTS_DIR / "shap_global.png"
SHAP_FEATURES_PATH = ARTIFACTS_DIR / "shap_features.json"

TARGET_COLUMN = "resistance_label"
POSITIVE_LABEL = "R"
IDENTIFIER_COLUMNS = {"source_row_id"}
OUTER_FOLDS = 5
STACKING_INNER_FOLDS = 5
LEAKAGE_COLUMNS = {
    "aro_gene_candidates",
    "aro_gene_name",
    "card_gene_name",
    "card_aro_accession",
    "card_gene_candidates",
    "card_associated_antibiotics",
    "card_resistance_mechanism",
    "card_amr_gene_family",
    "card_match_count",
    "antibiotic_name_norm",
    "resolved_gene_name",
    "resolved_gene_name_norm",
    "source_row_id",
    "source_dataset",
}
ARO_COLUMNS = {
    "aro_antibiotic_class",
    "aro_match_count",
}
CARD_COLUMNS = {
    "fasta_sequence_found",
}
TOP_KMER_COUNT = 20
MENDELEY_WEIGHT = 5.0
KAGGLE_WEIGHT = 1.0
PSEUDO_KAGGLE_WEIGHT = 2.0
PSEUDO_POSITIVE_THRESHOLD = 0.65
PSEUDO_NEGATIVE_THRESHOLD = 0.15


def load_unified_dataset(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value)
    if not path.is_absolute():
        candidates = [SCRIPT_DIR / path, ARTIFACTS_DIR / path]
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.exists():
        raise FileNotFoundError(f"Unified dataset not found: {path}")
    return pd.read_csv(path)


def merge_intermediate_into_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' in unified dataset.")

    out = df.copy()
    out = out[out[TARGET_COLUMN].notna()].copy()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(str).str.strip().str.upper()
    out = out[out[TARGET_COLUMN].isin(["S", "I", "R"])].copy()
    out["binary_target"] = (out[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    return out, out["binary_target"]


def report_binary_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().sort_index()
    return {
        "susceptible_0": int(counts.get(0, 0)),
        "resistant_1": int(counts.get(1, 0)),
    }


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COLUMN, "binary_target"} | IDENTIFIER_COLUMNS | LEAKAGE_COLUMNS
    return [column for column in df.columns if column not in excluded]


def identify_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_categorical_dtype(df[column])
        or pd.api.types.is_bool_dtype(df[column])
    ]


def get_transformed_feature_names(preprocessor: dict) -> list[str]:
    return preprocessor["categorical_columns"] + preprocessor["numeric_columns"]


def fit_preprocessor(X: pd.DataFrame) -> dict:
    categorical_columns = identify_categorical_columns(X)
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    num_imputer = SimpleImputer(strategy="median")

    preprocessor = {
        "feature_columns": list(X.columns),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "cat_imputer": None,
        "cat_encoder": None,
        "num_imputer": None,
    }

    if categorical_columns:
        cat_values = cat_imputer.fit_transform(X[categorical_columns])
        cat_encoder.fit(cat_values)
        preprocessor["cat_imputer"] = cat_imputer
        preprocessor["cat_encoder"] = cat_encoder

    if numeric_columns:
        num_imputer.fit(X[numeric_columns])
        preprocessor["num_imputer"] = num_imputer

    return preprocessor


def transform_features(X: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    feature_columns = preprocessor["feature_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    numeric_columns = preprocessor["numeric_columns"]

    X = X.copy()
    X = X.reindex(columns=feature_columns)

    transformed_parts = []

    if categorical_columns:
        cat_values = preprocessor["cat_imputer"].transform(X[categorical_columns])
        cat_encoded = preprocessor["cat_encoder"].transform(cat_values)
        transformed_parts.append(cat_encoded)

    if numeric_columns:
        num_values = preprocessor["num_imputer"].transform(X[numeric_columns])
        transformed_parts.append(num_values)

    if not transformed_parts:
        raise ValueError("No feature columns available after preprocessing.")

    return np.hstack(transformed_parts).astype(np.float32)


def get_smotenc_indices(preprocessor: dict) -> list[int]:
    return list(range(len(preprocessor["categorical_columns"])))


def apply_smotenc(X_train: np.ndarray, y_train: np.ndarray, categorical_indices: list[int], fold_id: int):
    class_counts = pd.Series(y_train).value_counts().sort_index()
    if not categorical_indices:
        print(f"[fold {fold_id}] SMOTENC skipped: no categorical columns detected.")
        return X_train, y_train

    if class_counts.min() <= 1:
        print(f"[fold {fold_id}] SMOTENC skipped: minority class has <= 1 sample.")
        return X_train, y_train

    k_neighbors = min(5, int(class_counts.min()) - 1)
    if k_neighbors < 1:
        print(f"[fold {fold_id}] SMOTENC skipped: insufficient minority neighbors.")
        return X_train, y_train

    sampler = SMOTENC(
        categorical_features=categorical_indices,
        k_neighbors=k_neighbors,
        random_state=42,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    before_dist = class_counts.to_dict()
    after_dist = pd.Series(y_resampled).value_counts().sort_index().to_dict()
    print(f"[fold {fold_id}] class distribution before={before_dist} after={after_dist}")
    return X_resampled, y_resampled


def apply_smotenc_with_source(
    X_train: np.ndarray,
    y_train: np.ndarray,
    categorical_indices: list[int],
    source_series: pd.Series,
    fold_id: int,
):
    return apply_smotenc_with_group_weights(
        X_train=X_train,
        y_train=y_train,
        categorical_indices=categorical_indices,
        group_series=source_series,
        weight_map={"mendeley": MENDELEY_WEIGHT, "kaggle": KAGGLE_WEIGHT},
        fold_id=fold_id,
    )


def apply_smotenc_with_group_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    categorical_indices: list[int],
    group_series: pd.Series,
    weight_map: dict[str, float],
    fold_id: int,
):
    normalized_groups = group_series.astype(str).str.lower()
    known_groups = list(weight_map.keys())
    group_to_code = {group_name: idx for idx, group_name in enumerate(known_groups)}
    fallback_group = known_groups[0]
    fallback_code = group_to_code[fallback_group]
    group_codes = normalized_groups.map(group_to_code).fillna(fallback_code).astype(np.float32).to_numpy().reshape(-1, 1)

    X_with_group = np.hstack([group_codes, X_train])
    categorical_indices_with_group = [0] + [index + 1 for index in categorical_indices]
    class_counts = pd.Series(y_train).value_counts().sort_index()

    def build_weights_from_codes(code_array: np.ndarray) -> np.ndarray:
        code_to_weight = {group_to_code[group_name]: weight for group_name, weight in weight_map.items()}
        return np.array([code_to_weight.get(int(code), weight_map[fallback_group]) for code in code_array], dtype=np.float32)

    if class_counts.min() <= 1:
        print(f"[fold {fold_id}] SMOTENC skipped: minority class has <= 1 sample.")
        return X_train, y_train, build_weights_from_codes(group_codes.ravel())

    k_neighbors = min(5, int(class_counts.min()) - 1)
    if k_neighbors < 1:
        print(f"[fold {fold_id}] SMOTENC skipped: insufficient minority neighbors.")
        return X_train, y_train, build_weights_from_codes(group_codes.ravel())

    sampler = SMOTENC(
        categorical_features=categorical_indices_with_group,
        k_neighbors=k_neighbors,
        random_state=42,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_with_group, y_train)
    group_codes_resampled = np.rint(X_resampled[:, 0]).astype(int)
    sample_weight = build_weights_from_codes(group_codes_resampled)
    model_matrix = X_resampled[:, 1:].astype(np.float32)

    before_dist = class_counts.to_dict()
    after_dist = pd.Series(y_resampled).value_counts().sort_index().to_dict()
    print(f"[fold {fold_id}] class distribution before={before_dist} after={after_dist}")
    return model_matrix, y_resampled, sample_weight


def make_stacking_classifier(xgb_params: dict | None = None, lgbm_params: dict | None = None) -> StackingClassifier:
    if CatBoostClassifier is None:
        raise ImportError(
            "catboost is not installed. Install it with "
            "\"python -m pip install catboost\" before running Track B training."
        )

    estimators = [
        ("xgb", make_xgb_classifier(xgb_params=xgb_params)),
        ("lgbm", make_lgbm_classifier(lgbm_params=lgbm_params)),
        ("catboost", make_catboost_classifier()),
    ]

    meta_learner = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=STACKING_INNER_FOLDS, shuffle=True, random_state=42),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def make_xgb_classifier(xgb_params: dict | None = None) -> XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if xgb_params:
        params.update(xgb_params)
    return XGBClassifier(**params)


def make_lgbm_classifier(lgbm_params: dict | None = None) -> LGBMClassifier:
    params = {
        "objective": "binary",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if lgbm_params:
        params.update(lgbm_params)
    return LGBMClassifier(**params)


def make_catboost_classifier() -> CatBoostClassifier:
    if CatBoostClassifier is None:
        raise ImportError(
            "catboost is not installed. Install it with "
            "\"python -m pip install catboost\" before running Track B training."
        )
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=250,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def assign_sample_weights(df: pd.DataFrame) -> np.ndarray:
    source_values = df.get("source_dataset", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    return np.where(source_values.eq("mendeley"), MENDELEY_WEIGHT, KAGGLE_WEIGHT).astype(np.float32)


def safe_fill_categorical(series: pd.Series, fallback: str) -> pd.Series:
    return series.astype(object).where(series.notna(), fallback).astype(str).replace({"nan": fallback, "": fallback})


def add_fold_engineered_features(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_out = train_df.copy()
    val_out = val_df.copy()

    train_out["species_clean"] = safe_fill_categorical(train_out.get("species"), "unknown_species")
    val_out["species_clean"] = safe_fill_categorical(val_out.get("species"), "unknown_species")
    train_out["antibiotic_class_clean"] = safe_fill_categorical(train_out.get("antibiotic_class"), "unknown_class")
    val_out["antibiotic_class_clean"] = safe_fill_categorical(val_out.get("antibiotic_class"), "unknown_class")

    train_out["antibiotic_species_interaction"] = (
        train_out["antibiotic_class_clean"] + "__" + train_out["species_clean"]
    )
    val_out["antibiotic_species_interaction"] = (
        val_out["antibiotic_class_clean"] + "__" + val_out["species_clean"]
    )

    global_rate = float(train_out["binary_target"].mean())
    class_rate = train_out.groupby("antibiotic_class_clean")["binary_target"].mean().to_dict()
    species_rate = train_out.groupby("species_clean")["binary_target"].mean().to_dict()

    train_out["antibiotic_class_resistance_rate"] = train_out["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    val_out["antibiotic_class_resistance_rate"] = val_out["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    train_out["species_resistance_rate"] = train_out["species_clean"].map(species_rate).fillna(global_rate)
    val_out["species_resistance_rate"] = val_out["species_clean"].map(species_rate).fillna(global_rate)

    base_columns = [column for column in select_feature_columns(train_out) if not column.startswith("kmer4_")]
    kmer_columns = [column for column in select_feature_columns(train_out) if column.startswith("kmer4_")]
    if kmer_columns:
        train_kmer = train_out[kmer_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        top_kmers = (
            train_kmer.var(axis=0)
            .sort_values(ascending=False)
            .head(TOP_KMER_COUNT)
            .index.tolist()
        )
    else:
        top_kmers = []

    feature_columns = base_columns + top_kmers
    metadata = {
        "selected_kmer_columns": top_kmers,
        "feature_columns": feature_columns,
        "global_rate": global_rate,
        "antibiotic_class_rate_map": class_rate,
        "species_rate_map": species_rate,
    }
    return train_out[feature_columns].copy(), val_out[feature_columns].copy(), metadata


def compute_binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def evaluate_pipeline(df: pd.DataFrame) -> dict:
    y = df["binary_target"].astype(int).to_numpy()
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    metrics = {
        "stack": {"folds": [], "aggregate": {}},
        "base_models": {
            "xgb": {"folds": [], "aggregate": {}},
            "lgbm": {"folds": [], "aggregate": {}},
            "catboost": {"folds": [], "aggregate": {}},
        },
    }
    training_start = time.perf_counter()

    for fold_id, (train_idx, val_idx) in enumerate(outer_cv.split(df, y), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train, X_val, _ = add_fold_engineered_features(train_df, val_df)
        preprocessor = fit_preprocessor(X_train)
        categorical_indices = get_smotenc_indices(preprocessor)
        X_train_processed = transform_features(X_train, preprocessor)
        X_val_processed = transform_features(X_val, preprocessor)
        X_train_balanced, y_train_balanced, sample_weight_balanced = apply_smotenc_with_source(
            X_train_processed,
            y_train,
            categorical_indices,
            train_df["source_dataset"],
            fold_id,
        )

        base_models = {
            "xgb": make_xgb_classifier(),
            "lgbm": make_lgbm_classifier(),
            "catboost": make_catboost_classifier(),
        }

        for model_name, model in base_models.items():
            model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weight_balanced)
            probabilities = model.predict_proba(X_val_processed)[:, 1]
            fold_result = {"fold": fold_id, **compute_binary_metrics(y_val, probabilities)}
            metrics["base_models"][model_name]["folds"].append(fold_result)
            print(
                f"[{model_name}] Fold {fold_id}: "
                f"AUC={fold_result['auc_roc']:.4f}, "
                f"F1={fold_result['f1']:.4f}, "
                f"Precision={fold_result['precision']:.4f}, "
                f"Recall={fold_result['recall']:.4f}"
            )

        stack_model = make_stacking_classifier()
        stack_model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weight_balanced)
        stack_probabilities = stack_model.predict_proba(X_val_processed)[:, 1]
        stack_result = {"fold": fold_id, **compute_binary_metrics(y_val, stack_probabilities)}
        metrics["stack"]["folds"].append(stack_result)
        print(
            f"[stack] Fold {fold_id}: "
            f"AUC={stack_result['auc_roc']:.4f}, "
            f"F1={stack_result['f1']:.4f}, "
            f"Precision={stack_result['precision']:.4f}, "
            f"Recall={stack_result['recall']:.4f}"
        )

    metric_names = ["auc_roc", "f1", "precision", "recall"]
    for model_name in ["stack", "xgb", "lgbm", "catboost"]:
        section = metrics["stack"] if model_name == "stack" else metrics["base_models"][model_name]
        auc_values = [fold["auc_roc"] for fold in section["folds"]]
        print(f"[{model_name}] Per-fold AUC: {[round(value, 4) for value in auc_values]}")
        for metric_name in metric_names:
            metric_values = [fold[metric_name] for fold in section["folds"]]
            section["aggregate"][metric_name] = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
            }
            print(
                f"[{model_name}] {metric_name.upper()}: "
                f"{section['aggregate'][metric_name]['mean']:.4f} ± "
                f"{section['aggregate'][metric_name]['std']:.4f}"
            )

    metrics["training_time_seconds"] = float(time.perf_counter() - training_start)
    print(f"Total training time (s): {metrics['training_time_seconds']:.2f}")
    return metrics
def get_original_tabular_columns(df: pd.DataFrame) -> list[str]:
    base_columns = []
    for column in select_feature_columns(df):
        if column in ARO_COLUMNS:
            continue
        if column in CARD_COLUMNS:
            continue
        if column.startswith("kmer4_"):
            continue
        base_columns.append(column)
    return base_columns


def get_tabular_plus_aro_columns(df: pd.DataFrame) -> list[str]:
    allowed = set(get_original_tabular_columns(df)) | ARO_COLUMNS
    return [column for column in select_feature_columns(df) if column in allowed]


def get_all_feature_columns(df: pd.DataFrame) -> list[str]:
    return select_feature_columns(df)


def evaluate_xgb_feature_set(df: pd.DataFrame, feature_columns: list[str], label: str) -> dict:
    X = df[feature_columns].copy()
    y = df["binary_target"].astype(int).to_numpy()
    class_counts = pd.Series(y).value_counts()
    n_splits = min(OUTER_FOLDS, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(f"{label}: not enough samples per class for stratified CV.")
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []
    for fold_id, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        preprocessor = fit_preprocessor(X_train)
        categorical_indices = get_smotenc_indices(preprocessor)
        X_train_processed = transform_features(X_train, preprocessor)
        X_val_processed = transform_features(X_val, preprocessor)
        X_train_balanced, y_train_balanced = apply_smotenc(
            X_train_processed,
            y_train,
            categorical_indices,
            fold_id,
        )

        model = make_xgb_classifier()
        model.fit(X_train_balanced, y_train_balanced)
        val_probabilities = model.predict_proba(X_val_processed)[:, 1]
        fold_auc = float(roc_auc_score(y_val, val_probabilities))
        auc_scores.append(fold_auc)
        print(f"[{label}] Fold {fold_id} AUC={fold_auc:.4f}")

    result = {
        "feature_count": int(len(feature_columns)),
        "auc_per_fold": auc_scores,
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "n_splits": int(n_splits),
        "n_rows": int(len(df)),
    }
    print(
        f"[{label}] AUC mean Â± std = "
        f"{result['auc_mean']:.4f} Â± {result['auc_std']:.4f}"
    )
    return result


def get_top_xgb_feature_importance(df: pd.DataFrame, feature_columns: list[str], top_n: int = 20) -> list[dict]:
    X = df[feature_columns].copy()
    y = df["binary_target"].astype(int).to_numpy()
    preprocessor = fit_preprocessor(X)
    X_processed = transform_features(X, preprocessor)
    model = make_xgb_classifier()
    model.fit(X_processed, y)

    feature_names = get_transformed_feature_names(preprocessor)
    importances = model.feature_importances_
    ranking = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    return [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in ranking
    ]


def run_feature_group_diagnostics(df: pd.DataFrame, output_dir: Path) -> dict:
    original_columns = get_original_tabular_columns(df)
    aro_columns = get_tabular_plus_aro_columns(df)
    all_columns = get_all_feature_columns(df)

    print(f"Original tabular feature count: {len(original_columns)}")
    print(f"Tabular + ARO feature count: {len(aro_columns)}")
    print(f"All feature count: {len(all_columns)}")

    diagnostics = {
        "original_tabular_only": evaluate_xgb_feature_set(df, original_columns, "original_tabular_only"),
        "tabular_plus_aro": evaluate_xgb_feature_set(df, aro_columns, "tabular_plus_aro"),
        "all_285_features": evaluate_xgb_feature_set(df, all_columns, "all_285_features"),
        "top_20_feature_importance_all_features": get_top_xgb_feature_importance(df, all_columns, top_n=20),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / DIAGNOSTIC_OUTPUT_PATH.name
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(f"Saved diagnostic summary to: {diagnostics_path}")
    return diagnostics


def run_source_comparison_diagnostics(df: pd.DataFrame, output_dir: Path) -> dict:
    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column for source-wise diagnostics.")

    diagnostics = {}
    feature_columns = select_feature_columns(df)

    subsets = {
        "mendeley_only": df[df["source_dataset"] == "mendeley"].copy(),
        "kaggle_only": df[df["source_dataset"] == "kaggle"].copy(),
        "combined": df.copy(),
    }

    for label, subset in subsets.items():
        print(f"{label} rows: {len(subset)}")
        diagnostics[label] = evaluate_xgb_feature_set(subset, feature_columns, label)

    diagnostics["top_20_feature_importance_combined"] = get_top_xgb_feature_importance(
        subsets["combined"], feature_columns, top_n=20
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / DIAGNOSTIC_OUTPUT_PATH.name
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(f"Saved diagnostic summary to: {diagnostics_path}")
    return diagnostics


def evaluate_weighted_hybrid_xgb(df: pd.DataFrame, output_dir: Path) -> dict:
    y = df["binary_target"].astype(int).to_numpy()
    sample_weights = assign_sample_weights(df)
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []
    selected_kmers_by_fold = []
    start_time = time.perf_counter()

    for fold_id, (train_idx, val_idx) in enumerate(outer_cv.split(df, y), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        weight_train = sample_weights[train_idx]

        X_train, X_val, engineered_meta = add_fold_engineered_features(train_df, val_df)
        preprocessor = fit_preprocessor(X_train)
        X_train_processed = transform_features(X_train, preprocessor)
        X_val_processed = transform_features(X_val, preprocessor)

        model = make_xgb_classifier()
        model.fit(X_train_processed, y_train, sample_weight=weight_train)
        val_probabilities = model.predict_proba(X_val_processed)[:, 1]
        fold_auc = float(roc_auc_score(y_val, val_probabilities))
        fold_metrics.append(fold_auc)
        selected_kmers_by_fold.append(engineered_meta["selected_kmer_columns"])
        print(f"[weighted_hybrid_xgb] Fold {fold_id} AUC={fold_auc:.4f}")

    total_time = float(time.perf_counter() - start_time)

    full_df = df.copy()
    full_df["species_clean"] = safe_fill_categorical(full_df.get("species"), "unknown_species")
    full_df["antibiotic_class_clean"] = safe_fill_categorical(full_df.get("antibiotic_class"), "unknown_class")
    full_df["antibiotic_species_interaction"] = (
        full_df["antibiotic_class_clean"] + "__" + full_df["species_clean"]
    )
    global_rate = float(full_df["binary_target"].mean())
    class_rate = full_df.groupby("antibiotic_class_clean")["binary_target"].mean().to_dict()
    species_rate = full_df.groupby("species_clean")["binary_target"].mean().to_dict()
    full_df["antibiotic_class_resistance_rate"] = full_df["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    full_df["species_resistance_rate"] = full_df["species_clean"].map(species_rate).fillna(global_rate)

    base_columns = [column for column in select_feature_columns(full_df) if not column.startswith("kmer4_")]
    kmer_columns = [column for column in select_feature_columns(full_df) if column.startswith("kmer4_")]
    if kmer_columns:
        full_kmer = full_df[kmer_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        selected_kmers = (
            full_kmer.var(axis=0).sort_values(ascending=False).head(TOP_KMER_COUNT).index.tolist()
        )
    else:
        selected_kmers = []

    feature_columns = base_columns + selected_kmers
    X_full = full_df[feature_columns].copy()
    preprocessor = fit_preprocessor(X_full)
    X_full_processed = transform_features(X_full, preprocessor)
    model = make_xgb_classifier()
    model.fit(X_full_processed, y, sample_weight=sample_weights)

    feature_names = get_transformed_feature_names(preprocessor)
    ranking = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    diagnostics = {
        "auc_per_fold": fold_metrics,
        "auc_mean": float(np.mean(fold_metrics)),
        "auc_std": float(np.std(fold_metrics)),
        "training_time_seconds": total_time,
        "selected_kmers_final": selected_kmers,
        "selected_kmers_by_fold": selected_kmers_by_fold,
        "top_20_feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in ranking
        ],
        "sample_weighting": {"mendeley": MENDELEY_WEIGHT, "kaggle": KAGGLE_WEIGHT},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / HYBRID_DIAGNOSTIC_OUTPUT_PATH.name
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(f"Per-fold AUC: {[round(value, 4) for value in fold_metrics]}")
    print(f"AUC mean Â± std = {diagnostics['auc_mean']:.4f} Â± {diagnostics['auc_std']:.4f}")
    print(f"Training time (s): {diagnostics['training_time_seconds']:.2f}")
    print(f"Saved hybrid diagnostic summary to: {diagnostics_path}")
    return diagnostics


def fit_final_model(df: pd.DataFrame) -> dict:
    full_df = df.copy()
    y = full_df["binary_target"].astype(int).to_numpy()

    full_df["species_clean"] = safe_fill_categorical(full_df.get("species"), "unknown_species")
    full_df["antibiotic_class_clean"] = safe_fill_categorical(full_df.get("antibiotic_class"), "unknown_class")
    full_df["antibiotic_species_interaction"] = (
        full_df["antibiotic_class_clean"] + "__" + full_df["species_clean"]
    )
    global_rate = float(full_df["binary_target"].mean())
    class_rate = full_df.groupby("antibiotic_class_clean")["binary_target"].mean().to_dict()
    species_rate = full_df.groupby("species_clean")["binary_target"].mean().to_dict()
    full_df["antibiotic_class_resistance_rate"] = full_df["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    full_df["species_resistance_rate"] = full_df["species_clean"].map(species_rate).fillna(global_rate)

    base_columns = [column for column in select_feature_columns(full_df) if not column.startswith("kmer4_")]
    kmer_columns = [column for column in select_feature_columns(full_df) if column.startswith("kmer4_")]
    if kmer_columns:
        full_kmer = full_df[kmer_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        selected_kmers = (
            full_kmer.var(axis=0).sort_values(ascending=False).head(TOP_KMER_COUNT).index.tolist()
        )
    else:
        selected_kmers = []

    feature_columns = base_columns + selected_kmers
    X_full = full_df[feature_columns].copy()
    preprocessor = fit_preprocessor(X_full)
    categorical_indices = get_smotenc_indices(preprocessor)
    X_full_processed = transform_features(X_full, preprocessor)
    X_balanced, y_balanced, sample_weight_balanced = apply_smotenc_with_source(
        X_full_processed,
        y,
        categorical_indices,
        full_df["source_dataset"],
        fold_id=0,
    )

    model = make_stacking_classifier()
    model.fit(X_balanced, y_balanced, sample_weight=sample_weight_balanced)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "binary_mapping": {"S": 0, "I": 0, "R": 1},
        "selected_kmer_columns": selected_kmers,
        "feature_engineering": {
            "global_rate": global_rate,
            "antibiotic_class_rate_map": class_rate,
            "species_rate_map": species_rate,
        },
        "sample_weighting": {"mendeley": MENDELEY_WEIGHT, "kaggle": KAGGLE_WEIGHT},
    }
def save_artifacts(model_bundle: dict, metrics: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_OUTPUT_PATH.name
    metrics_path = output_dir / SUMMARY_OUTPUT_PATH.name

    joblib.dump(model_bundle, model_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved stacked model to: {model_path}")
    print(f"Saved training metrics to: {metrics_path}")


def run_mendeley_only_stack(data_path: str | Path = DEFAULT_DATA_PATH, output_dir: str | Path = ARTIFACTS_DIR) -> dict:
    df = load_unified_dataset(data_path)
    df = df.dropna(axis=1, how="all")
    df, y_binary = merge_intermediate_into_binary(df)

    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column for Mendeley-only training.")

    df_mendeley = df[df["source_dataset"].astype(str).str.lower() == "mendeley"].copy()
    if df_mendeley.empty:
        raise ValueError("No Mendeley rows found in unified dataset.")

    distribution = report_binary_distribution(df_mendeley["binary_target"])
    print("Mendeley-only binary class distribution:", distribution)

    metrics = evaluate_pipeline(df_mendeley)
    metrics["binary_class_distribution"] = distribution
    metrics["dataset_shape"] = {"rows": int(df_mendeley.shape[0]), "columns": int(df_mendeley.shape[1])}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / MENDELEY_ONLY_METRICS_PATH.name
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved Mendeley-only metrics to: {metrics_path}")
    return metrics


def get_mendeley_only_dataframe(data_path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    df = load_unified_dataset(data_path)
    df = df.dropna(axis=1, how="all")
    df, _ = merge_intermediate_into_binary(df)
    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column for Mendeley-only tuning.")
    df_mendeley = df[df["source_dataset"].astype(str).str.lower() == "mendeley"].copy()
    if df_mendeley.empty:
        raise ValueError("No Mendeley rows found in unified dataset.")
    distribution = report_binary_distribution(df_mendeley["binary_target"])
    return df_mendeley, distribution


def build_optuna_search_space(trial) -> tuple[dict, dict]:
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("xgb_min_child_samples", 10, 100),
    }
    lgbm_params = {
        "device": "gpu",
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 12),
        "num_leaves": trial.suggest_int("lgbm_num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 10, 100),
        "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0),
    }
    return xgb_params, lgbm_params


def evaluate_mendeley_stack_with_params(df_mendeley: pd.DataFrame, xgb_params: dict, lgbm_params: dict) -> dict:
    y = df_mendeley["binary_target"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(df_mendeley, y), start=1):
        train_df = df_mendeley.iloc[train_idx].copy()
        val_df = df_mendeley.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train, X_val, _ = add_fold_engineered_features(train_df, val_df)
        preprocessor = fit_preprocessor(X_train)
        categorical_indices = get_smotenc_indices(preprocessor)
        X_train_processed = transform_features(X_train, preprocessor)
        X_val_processed = transform_features(X_val, preprocessor)
        X_train_balanced, y_train_balanced, sample_weight_balanced = apply_smotenc_with_source(
            X_train_processed,
            y_train,
            categorical_indices,
            train_df["source_dataset"],
            fold_id,
        )

        stack_model = make_stacking_classifier(xgb_params=xgb_params, lgbm_params=lgbm_params)
        stack_model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weight_balanced)
        probabilities = stack_model.predict_proba(X_val_processed)[:, 1]
        fold_results.append({"fold": fold_id, **compute_binary_metrics(y_val, probabilities)})

    auc_values = [fold["auc_roc"] for fold in fold_results]
    aggregate = {}
    for metric_name in ["auc_roc", "f1", "precision", "recall"]:
        metric_values = [fold[metric_name] for fold in fold_results]
        aggregate[metric_name] = {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
        }
    return {"folds": fold_results, "aggregate": aggregate, "auc_values": auc_values}


def fit_tuned_mendeley_stack(df_mendeley: pd.DataFrame, xgb_params: dict, lgbm_params: dict) -> dict:
    full_df = df_mendeley.copy()
    y = full_df["binary_target"].astype(int).to_numpy()

    full_df["species_clean"] = safe_fill_categorical(full_df.get("species"), "unknown_species")
    full_df["antibiotic_class_clean"] = safe_fill_categorical(full_df.get("antibiotic_class"), "unknown_class")
    full_df["antibiotic_species_interaction"] = (
        full_df["antibiotic_class_clean"] + "__" + full_df["species_clean"]
    )
    global_rate = float(full_df["binary_target"].mean())
    class_rate = full_df.groupby("antibiotic_class_clean")["binary_target"].mean().to_dict()
    species_rate = full_df.groupby("species_clean")["binary_target"].mean().to_dict()
    full_df["antibiotic_class_resistance_rate"] = full_df["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    full_df["species_resistance_rate"] = full_df["species_clean"].map(species_rate).fillna(global_rate)

    base_columns = [column for column in select_feature_columns(full_df) if not column.startswith("kmer4_")]
    kmer_columns = [column for column in select_feature_columns(full_df) if column.startswith("kmer4_")]
    if kmer_columns:
        full_kmer = full_df[kmer_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        selected_kmers = (
            full_kmer.var(axis=0).sort_values(ascending=False).head(TOP_KMER_COUNT).index.tolist()
        )
    else:
        selected_kmers = []

    feature_columns = base_columns + selected_kmers
    X_full = full_df[feature_columns].copy()
    preprocessor = fit_preprocessor(X_full)
    categorical_indices = get_smotenc_indices(preprocessor)
    X_full_processed = transform_features(X_full, preprocessor)
    X_balanced, y_balanced, sample_weight_balanced = apply_smotenc_with_source(
        X_full_processed,
        y,
        categorical_indices,
        full_df["source_dataset"],
        fold_id=0,
    )
    model = make_stacking_classifier(xgb_params=xgb_params, lgbm_params=lgbm_params)
    model.fit(X_balanced, y_balanced, sample_weight=sample_weight_balanced)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "binary_mapping": {"S": 0, "I": 0, "R": 1},
        "selected_kmer_columns": selected_kmers,
        "feature_engineering": {
            "global_rate": global_rate,
            "antibiotic_class_rate_map": class_rate,
            "species_rate_map": species_rate,
        },
        "sample_weighting": {"mendeley": MENDELEY_WEIGHT, "kaggle": KAGGLE_WEIGHT},
        "best_params": {"xgb": xgb_params, "lgbm": lgbm_params},
    }


def load_flat_best_params(path: str | Path = BEST_PARAMS_PATH) -> tuple[dict, dict]:
    params_path = Path(path)
    with open(params_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "xgb" in payload and "lgbm" in payload:
        return payload["xgb"], payload["lgbm"]

    xgb_params = {
        "n_estimators": payload["xgb_n_estimators"],
        "learning_rate": payload["xgb_learning_rate"],
        "max_depth": payload["xgb_max_depth"],
        "subsample": payload["xgb_subsample"],
        "colsample_bytree": payload["xgb_colsample_bytree"],
        "min_child_weight": payload["xgb_min_child_samples"],
    }
    lgbm_params = {
        "n_estimators": payload["lgbm_n_estimators"],
        "learning_rate": payload["lgbm_learning_rate"],
        "max_depth": payload["lgbm_max_depth"],
        "num_leaves": payload["lgbm_num_leaves"],
        "min_child_samples": payload["lgbm_min_child_samples"],
        "subsample": payload["lgbm_subsample"],
        "colsample_bytree": payload["lgbm_colsample_bytree"],
    }
    return xgb_params, lgbm_params


def prepare_bundle_features(df: pd.DataFrame, model_bundle: dict) -> np.ndarray:
    frame = df.copy()
    frame["species_clean"] = safe_fill_categorical(frame.get("species"), "unknown_species")
    frame["antibiotic_class_clean"] = safe_fill_categorical(frame.get("antibiotic_class"), "unknown_class")
    frame["antibiotic_species_interaction"] = (
        frame["antibiotic_class_clean"] + "__" + frame["species_clean"]
    )

    feature_engineering = model_bundle.get("feature_engineering", {})
    global_rate = float(feature_engineering.get("global_rate", 0.0))
    class_rate = feature_engineering.get("antibiotic_class_rate_map", {})
    species_rate = feature_engineering.get("species_rate_map", {})

    frame["antibiotic_class_resistance_rate"] = frame["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    frame["species_resistance_rate"] = frame["species_clean"].map(species_rate).fillna(global_rate)

    X_frame = frame[model_bundle["feature_columns"]].copy()
    return transform_features(X_frame, model_bundle["preprocessor"])


def fit_tuned_stack_on_weighted_dataframe(df_input: pd.DataFrame, xgb_params: dict, lgbm_params: dict) -> dict:
    full_df = df_input.copy()
    y = full_df["binary_target"].astype(int).to_numpy()

    full_df["species_clean"] = safe_fill_categorical(full_df.get("species"), "unknown_species")
    full_df["antibiotic_class_clean"] = safe_fill_categorical(full_df.get("antibiotic_class"), "unknown_class")
    full_df["antibiotic_species_interaction"] = (
        full_df["antibiotic_class_clean"] + "__" + full_df["species_clean"]
    )
    global_rate = float(full_df["binary_target"].mean())
    class_rate = full_df.groupby("antibiotic_class_clean")["binary_target"].mean().to_dict()
    species_rate = full_df.groupby("species_clean")["binary_target"].mean().to_dict()
    full_df["antibiotic_class_resistance_rate"] = full_df["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    full_df["species_resistance_rate"] = full_df["species_clean"].map(species_rate).fillna(global_rate)

    base_columns = [column for column in select_feature_columns(full_df) if not column.startswith("kmer4_")]
    kmer_columns = [column for column in select_feature_columns(full_df) if column.startswith("kmer4_")]
    if kmer_columns:
        full_kmer = full_df[kmer_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        selected_kmers = (
            full_kmer.var(axis=0).sort_values(ascending=False).head(TOP_KMER_COUNT).index.tolist()
        )
    else:
        selected_kmers = []

    feature_columns = base_columns + selected_kmers
    X_full = full_df[feature_columns].copy()
    preprocessor = fit_preprocessor(X_full)
    categorical_indices = get_smotenc_indices(preprocessor)
    X_full_processed = transform_features(X_full, preprocessor)
    X_balanced, y_balanced, sample_weight_balanced = apply_smotenc_with_group_weights(
        X_train=X_full_processed,
        y_train=y,
        categorical_indices=categorical_indices,
        group_series=full_df["source_dataset"],
        weight_map={"mendeley": MENDELEY_WEIGHT, "pseudo_kaggle": PSEUDO_KAGGLE_WEIGHT},
        fold_id=0,
    )
    model = make_stacking_classifier(xgb_params=xgb_params, lgbm_params=lgbm_params)
    model.fit(X_balanced, y_balanced, sample_weight=sample_weight_balanced)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "binary_mapping": {"S": 0, "I": 0, "R": 1},
        "selected_kmer_columns": selected_kmers,
        "feature_engineering": {
            "global_rate": global_rate,
            "antibiotic_class_rate_map": class_rate,
            "species_rate_map": species_rate,
        },
        "sample_weighting": {"mendeley": MENDELEY_WEIGHT, "pseudo_kaggle": PSEUDO_KAGGLE_WEIGHT},
        "best_params": {"xgb": xgb_params, "lgbm": lgbm_params},
    }


def run_pseudo_label_stack(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = ARTIFACTS_DIR,
    tuned_model_path: str | Path = STACKED_TUNED_MODEL_PATH,
    best_params_path: str | Path = BEST_PARAMS_PATH,
) -> dict:
    df = load_unified_dataset(data_path)
    df = df.dropna(axis=1, how="all")
    df, _ = merge_intermediate_into_binary(df)
    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column for pseudo-labeling.")

    df_mendeley = df[df["source_dataset"].astype(str).str.lower() == "mendeley"].copy()
    df_kaggle = df[df["source_dataset"].astype(str).str.lower() == "kaggle"].copy()
    if df_mendeley.empty or df_kaggle.empty:
        raise ValueError("Pseudo-label branch requires both Mendeley and Kaggle rows.")

    tuned_bundle = joblib.load(tuned_model_path)
    kaggle_matrix = prepare_bundle_features(df_kaggle, tuned_bundle)
    kaggle_probabilities = tuned_bundle["model"].predict_proba(kaggle_matrix)[:, 1]

    positive_mask = kaggle_probabilities > PSEUDO_POSITIVE_THRESHOLD
    negative_mask = kaggle_probabilities < PSEUDO_NEGATIVE_THRESHOLD
    keep_mask = positive_mask | negative_mask
    df_pseudo = df_kaggle.loc[keep_mask].copy()
    if df_pseudo.empty:
        raise ValueError("No high-confidence Kaggle rows passed the pseudo-label thresholds.")

    df_pseudo["binary_target"] = np.where(
        kaggle_probabilities[keep_mask] > PSEUDO_POSITIVE_THRESHOLD,
        1,
        0,
    ).astype(int)
    df_pseudo[TARGET_COLUMN] = np.where(df_pseudo["binary_target"] == 1, "R", "S")
    df_pseudo["source_dataset"] = "pseudo_kaggle"

    combined_df = pd.concat([df_mendeley, df_pseudo], ignore_index=True)
    y = combined_df["binary_target"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    training_start = time.perf_counter()
    xgb_params, lgbm_params = load_flat_best_params(best_params_path)

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(combined_df, y), start=1):
        train_df = combined_df.iloc[train_idx].copy()
        val_df = combined_df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train, X_val, _ = add_fold_engineered_features(train_df, val_df)
        preprocessor = fit_preprocessor(X_train)
        categorical_indices = get_smotenc_indices(preprocessor)
        X_train_processed = transform_features(X_train, preprocessor)
        X_val_processed = transform_features(X_val, preprocessor)
        X_train_balanced, y_train_balanced, sample_weight_balanced = apply_smotenc_with_group_weights(
            X_train=X_train_processed,
            y_train=y_train,
            categorical_indices=categorical_indices,
            group_series=train_df["source_dataset"],
            weight_map={"mendeley": MENDELEY_WEIGHT, "pseudo_kaggle": PSEUDO_KAGGLE_WEIGHT},
            fold_id=fold_id,
        )

        stack_model = make_stacking_classifier(xgb_params=xgb_params, lgbm_params=lgbm_params)
        stack_model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weight_balanced)
        probabilities = stack_model.predict_proba(X_val_processed)[:, 1]
        fold_result = {"fold": fold_id, **compute_binary_metrics(y_val, probabilities)}
        fold_results.append(fold_result)
        print(
            f"[pseudo_stack] Fold {fold_id}: "
            f"AUC={fold_result['auc_roc']:.4f}, "
            f"F1={fold_result['f1']:.4f}, "
            f"Precision={fold_result['precision']:.4f}, "
            f"Recall={fold_result['recall']:.4f}"
        )

    aggregate = {}
    for metric_name in ["auc_roc", "f1", "precision", "recall"]:
        metric_values = [fold[metric_name] for fold in fold_results]
        aggregate[metric_name] = {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
        }

    final_model_bundle = fit_tuned_stack_on_weighted_dataframe(combined_df, xgb_params, lgbm_params)

    metrics_payload = {
        "folds": fold_results,
        "aggregate": aggregate,
        "auc_values": [fold["auc_roc"] for fold in fold_results],
        "training_time_seconds": float(time.perf_counter() - training_start),
        "pseudo_labeling": {
            "positive_threshold": PSEUDO_POSITIVE_THRESHOLD,
            "negative_threshold": PSEUDO_NEGATIVE_THRESHOLD,
            "selected_pseudo_rows": int(len(df_pseudo)),
            "selected_positive_rows": int(positive_mask.sum()),
            "selected_negative_rows": int(negative_mask.sum()),
        },
        "dataset_shape": {
            "mendeley_rows": int(len(df_mendeley)),
            "kaggle_rows": int(len(df_kaggle)),
            "combined_rows": int(len(combined_df)),
        },
        "sample_weighting": {"mendeley": MENDELEY_WEIGHT, "pseudo_kaggle": PSEUDO_KAGGLE_WEIGHT},
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model_bundle, output_dir / PSEUDO_MODEL_PATH.name)
    with open(output_dir / PSEUDO_METRICS_PATH.name, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(f"[pseudo_stack] Per-fold AUC: {[round(value, 4) for value in metrics_payload['auc_values']]}")
    print(
        f"[pseudo_stack] AUC mean ± std: "
        f"{aggregate['auc_roc']['mean']:.4f} ± {aggregate['auc_roc']['std']:.4f}"
    )
    print(f"Saved pseudo-labeled model to: {output_dir / PSEUDO_MODEL_PATH.name}")
    print(f"Saved pseudo-labeled metrics to: {output_dir / PSEUDO_METRICS_PATH.name}")
    return metrics_payload


def run_shap_explainability(
    data_path: str | Path = DEFAULT_DATA_PATH,
    tuned_model_path: str | Path = STACKED_TUNED_MODEL_PATH,
    output_dir: str | Path = ARTIFACTS_DIR,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = joblib.load(tuned_model_path)
    model = model_bundle["model"]
    if not hasattr(model, "named_estimators_") or "xgb" not in model.named_estimators_:
        raise ValueError("The tuned stacked model does not expose a fitted XGBoost base model.")

    df = load_unified_dataset(data_path)
    df = df.dropna(axis=1, how="all")
    df, _ = merge_intermediate_into_binary(df)
    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column for SHAP explainability.")

    df_mendeley = df[df["source_dataset"].astype(str).str.lower() == "mendeley"].copy()
    if df_mendeley.empty:
        raise ValueError("No Mendeley rows found for SHAP explainability.")

    X_matrix = prepare_bundle_features(df_mendeley, model_bundle)
    feature_names = get_transformed_feature_names(model_bundle["preprocessor"])
    xgb_model = model.named_estimators_["xgb"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    top_features = [
        {"feature": feature_names[idx], "mean_abs_shap": float(mean_abs_shap[idx])}
        for idx in top_indices
    ]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_matrix,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_dir / SHAP_GLOBAL_PATH.name, dpi=200, bbox_inches="tight")
    plt.close()

    with open(output_dir / SHAP_FEATURES_PATH.name, "w", encoding="utf-8") as handle:
        json.dump(top_features, handle, indent=2)

    print(f"Saved SHAP global summary to: {output_dir / SHAP_GLOBAL_PATH.name}")
    print(f"Saved SHAP top features to: {output_dir / SHAP_FEATURES_PATH.name}")
    return {
        "rows_explained": int(len(df_mendeley)),
        "feature_count": int(len(feature_names)),
        "top_features": top_features,
    }


def run_mendeley_only_optuna_tuning(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = ARTIFACTS_DIR,
    n_trials: int = 100,
) -> dict:
    import optuna

    df_mendeley, distribution = get_mendeley_only_dataframe(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Mendeley-only binary class distribution:", distribution)

    def objective(trial):
        xgb_params, lgbm_params = build_optuna_search_space(trial)
        results = evaluate_mendeley_stack_with_params(df_mendeley, xgb_params, lgbm_params)
        return results["aggregate"]["auc_roc"]["mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_xgb_params, best_lgbm_params = build_optuna_search_space(study.best_trial)
    final_results = evaluate_mendeley_stack_with_params(df_mendeley, best_xgb_params, best_lgbm_params)
    tuned_model_bundle = fit_tuned_mendeley_stack(df_mendeley, best_xgb_params, best_lgbm_params)

    best_params_payload = {
        "study_best_value_auc": float(study.best_value),
        "study_best_trial": int(study.best_trial.number),
        "xgb": best_xgb_params,
        "lgbm": best_lgbm_params,
    }
    with open(output_dir / BEST_PARAMS_PATH.name, "w", encoding="utf-8") as handle:
        json.dump(best_params_payload, handle, indent=2)

    joblib.dump(tuned_model_bundle, output_dir / STACKED_TUNED_MODEL_PATH.name)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best study AUC: {study.best_value:.4f}")
    print(f"Final tuned stack per-fold AUC: {[round(v, 4) for v in final_results['auc_values']]}")
    print(
        "Final tuned stack AUC mean ± std: "
        f"{final_results['aggregate']['auc_roc']['mean']:.4f} ± "
        f"{final_results['aggregate']['auc_roc']['std']:.4f}"
    )
    print(f"Saved best params to: {output_dir / BEST_PARAMS_PATH.name}")
    print(f"Saved tuned model to: {output_dir / STACKED_TUNED_MODEL_PATH.name}")

    return {
        "best_params": best_params_payload,
        "final_cv": final_results,
        "binary_class_distribution": distribution,
        "dataset_shape": {"rows": int(df_mendeley.shape[0]), "columns": int(df_mendeley.shape[1])},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Track B stacked ensemble on the unified binary resistance dataset."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to unified_dataset_final.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACTS_DIR),
        help="Directory to save Track B training artifacts.",
    )
    parser.add_argument(
        "--diagnostic-xgb-ablation",
        action="store_true",
        help="Run XGBoost-only feature-group diagnostics instead of stacked ensemble training.",
    )
    parser.add_argument(
        "--diagnostic-source-comparison",
        action="store_true",
        help="Run XGBoost source-wise diagnostics on mendeley, kaggle, and combined data.",
    )
    parser.add_argument(
        "--diagnostic-weighted-hybrid-xgb",
        action="store_true",
        help="Run weighted XGBoost diagnostic with fold-safe engineered features and top-variance k-mers.",
    )
    args = parser.parse_args()

    df = load_unified_dataset(args.data)
    df = df.dropna(axis=1, how="all")
    df, y_binary = merge_intermediate_into_binary(df)

    distribution = report_binary_distribution(y_binary)
    print("Binary class distribution after merging I into S:", distribution)

    output_dir = Path(args.output_dir)

    if args.diagnostic_xgb_ablation:
        diagnostics = run_feature_group_diagnostics(df, output_dir)
        diagnostics["binary_class_distribution"] = distribution
        diagnostics["dataset_shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        return

    if args.diagnostic_source_comparison:
        diagnostics = run_source_comparison_diagnostics(df, output_dir)
        diagnostics["binary_class_distribution"] = distribution
        diagnostics["dataset_shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        return

    if args.diagnostic_weighted_hybrid_xgb:
        diagnostics = evaluate_weighted_hybrid_xgb(df, output_dir)
        diagnostics["binary_class_distribution"] = distribution
        diagnostics["dataset_shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        return

    metrics = evaluate_pipeline(df)
    metrics["binary_class_distribution"] = distribution
    metrics["dataset_shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    final_model = fit_final_model(df)
    save_artifacts(final_model, metrics, output_dir)


if __name__ == "__main__":
    OPTUNA_TUNE = False
    MENDELEY_ONLY = False
    PSEUDO_LABEL = False
    RUN_SHAP = True
    if OPTUNA_TUNE:
        run_mendeley_only_optuna_tuning()
    elif MENDELEY_ONLY:
        run_mendeley_only_stack()
    elif PSEUDO_LABEL:
        run_pseudo_label_stack()
    elif RUN_SHAP:
        run_shap_explainability()
    else:
        main()


