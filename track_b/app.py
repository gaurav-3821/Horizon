from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

LABEL_MAP = {0: "Susceptible", 1: "Intermediate", 2: "Resistant"}
LABEL_COLOR = {"Susceptible": "green", "Intermediate": "orange", "Resistant": "red"}
ANTIBIOTICS = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]

st.set_page_config(page_title="Track B - Antibiotic Resistance", layout="wide")
st.title("Antibiotic Resistance Prediction")


def artifact_path(name: str) -> Path:
    return ARTIFACTS_DIR / name


def apply_target_encoding(df, cols, mappings, global_mean):
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = global_mean
        else:
            out[col] = out[col].astype(str).fillna("unknown").map(mappings[col]).fillna(global_mean)
    return out


def transform_input(input_df, encoder_artifact, scaler):
    df = input_df.copy()
    high_card_cols = encoder_artifact.get("high_cardinality_columns", [])
    mappings = encoder_artifact.get("target_encoding_mappings", {})
    global_mean = encoder_artifact.get("target_encoding_global_mean", 0.0)
    df = apply_target_encoding(df, high_card_cols, mappings, global_mean)

    num_cols = encoder_artifact.get("numeric_columns", [])
    cat_cols = encoder_artifact.get("categorical_columns", [])
    num_imputer = encoder_artifact.get("numeric_imputer")
    cat_imputer = encoder_artifact.get("categorical_imputer")
    ord_encoder = encoder_artifact.get("ordinal_encoder")

    X_num = scaler.transform(num_imputer.transform(df[num_cols])) if num_cols else np.empty((len(df), 0))
    X_cat = ord_encoder.transform(cat_imputer.transform(df[cat_cols])) if cat_cols else np.empty((len(df), 0))
    return np.hstack([X_num, X_cat])


@st.cache_resource
def load_bundles():
    bundles = {}
    for antibiotic in ANTIBIOTICS:
        safe_name = antibiotic.lower()
        model_file = artifact_path(f"{safe_name}_xgb_model.pkl")
        scaler_file = artifact_path(f"{safe_name}_scaler.pkl")
        encoder_file = artifact_path(f"{safe_name}_encoder.pkl")
        if model_file.exists() and scaler_file.exists() and encoder_file.exists():
            bundles[antibiotic] = {
                "model": joblib.load(model_file),
                "scaler": joblib.load(scaler_file),
                "encoder": joblib.load(encoder_file),
            }
    return bundles


@st.cache_data
def load_summary():
    summary_file = artifact_path("track_b_cv_summary.pkl")
    if not summary_file.exists():
        return None
    return joblib.load(summary_file)


bundles = load_bundles()
summary = load_summary()

tab1, tab2 = st.tabs(["Predict Resistance", "Model Performance"])

with tab1:
    st.subheader("Enter Patient and Sample Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 35)
        gender = st.selectbox("Gender", ["M", "F"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No", "Unknown"])
    with col2:
        hypertension = st.selectbox("Hypertension", ["Yes", "No", "Unknown"])
        hospital_before = st.selectbox("Prior hospitalisation", ["Yes", "No", "Unknown"])
        infection_freq = st.selectbox("Infection frequency", ["First", "Recurrent", "Unknown"])
    with col3:
        site = st.text_input("Sample site (e.g. IFE)", "IFE")
        sample_type = st.text_input("Sample type (e.g. T)", "T")

    if st.button("Predict Resistance Profile") and bundles:
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "gender": gender,
                    "Diabetes": diabetes,
                    "Hypertension": hypertension,
                    "Hospital_before": hospital_before,
                    "Infection_Freq": infection_freq,
                    "site": site,
                    "sample_type": sample_type,
                    "Souches": "unknown",
                    "source": "dataset2",
                }
            ]
        )

        st.subheader("Resistance Prediction")
        cols = st.columns(len(ANTIBIOTICS))
        for i, antibiotic in enumerate(ANTIBIOTICS):
            if antibiotic not in bundles:
                cols[i].warning(f"{antibiotic}: model missing")
                continue
            try:
                bundle = bundles[antibiotic]
                X = transform_input(input_df, bundle["encoder"], bundle["scaler"])
                pred = bundle["model"].predict(X)[0]
                label = LABEL_MAP.get(int(pred), "Unknown")
                color = LABEL_COLOR.get(label, "gray")
                cols[i].markdown(f"**{antibiotic}**")
                cols[i].markdown(f":{color}[{label}]")
            except Exception:
                cols[i].warning(f"{antibiotic}: error")
    elif not bundles:
        st.warning("No Track B model artifacts found in track_b/artifacts.")

with tab2:
    if summary:
        st.subheader("Per-antibiotic Model Performance")
        rows = []
        for antibiotic, metrics in summary.items():
            rows.append(
                {
                    "Antibiotic": antibiotic,
                    "Accuracy": round(metrics.get("accuracy_mean", 0.0), 3),
                    "ROC-AUC": round(metrics.get("roc_auc_mean", 0.0), 3),
                    "Samples": metrics.get("n_samples", 0),
                    "CV Folds": metrics.get("cv_folds", 0),
                }
            )
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df, width="stretch")

        fig, ax = plt.subplots(figsize=(8, 4))
        perf_df.set_index("Antibiotic")[["Accuracy", "ROC-AUC"]].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Model performance by antibiotic")
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        st.info("No Track B summary artifact found in track_b/artifacts.")
