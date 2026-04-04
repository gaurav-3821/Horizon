<div align="center">

# 🧬 Horizon — Health Intelligence Platform

**A unified multi-track ML dashboard for drug toxicity, antibiotic resistance, and epidemic forecasting**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)
[![Hackathon](https://img.shields.io/badge/CodeCure_AI-Hackathon-purple?style=flat-square)]()

</div>

---

## ✨ Overview

Horizon is a Streamlit-based health intelligence dashboard that bundles three independent machine learning tracks into one deployable repository. Each track is fully self-contained — its own model, dataset, artifacts, and UI — all unified under a single navigation hub.

Built for the **CodeCure AI Hackathon**.

---

## 🧪 Tracks

### Track A — Drug Toxicity Prediction
Molecular toxicity classification across 12 Tox21 endpoints using a tabular XGBoost pipeline with SHAP explainability.
- **Model:** XGBoost ensemble
- **Mean AUC:** 0.8514 across 12 endpoints
- **Explainability:** SHAP feature importance

### Track B — Antibiotic Resistance Prediction
Predicts antibiotic resistance from clinical microbiology data using a stacked ensemble with SMOTE-NC balancing.
- **Model:** Stacked ensemble (XGBoost + LightGBM + CatBoost + Logistic Regression meta-learner)
- **AUC:** 0.8540 (Optuna-tuned)
- **Features:** Species resistance rate, antibiotic class resistance rate, interaction terms
- **Explainability:** SHAP + AI Clinical Advisor (LLaMA 3.3 70B via OpenRouter)

### Track C — Epidemic Spread Forecasting
Time-series forecasting and classification of epidemic spread using JHU and OWID data.
- **Model:** Forecaster + Classifier pipeline
- **Data:** JHU CSSE + Our World in Data merge

---

## 🏗️ Repository Structure
```
Horizon/
├── horizon.py              # Unified Streamlit navigation hub (main entrypoint)
├── requirements.txt        # Repository-wide dependencies
├── runtime.txt             # Streamlit Cloud runtime pin (python-3.12)
├── Dockerfile              # Optional container entrypoint
├── PROJECT_STRUCTURE.md    # Quick file map
├── README.md
├── LICENSE
│
├── track_a/                # Drug Toxicity
│   ├── app.py
│   ├── track_a_pipeline.py
│   ├── data/
│   └── artifacts/
│
├── track_b/                # Antibiotic Resistance
│   ├── app.py
│   ├── track_b_model.py
│   ├── track_b_data_loader.py
│   ├── data/
│   └── artifacts/
│
├── track_c/                # Epidemic Forecasting
│   ├── app.py
│   ├── track_c_model.py
│   ├── track_c_data_loader.py
│   ├── data/
│   └── artifacts/
│
└── docs/                   # Handoff reports and presentation assets
```

---

## 🚀 Quick Start

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the unified dashboard**
```bash
streamlit run horizon.py
```

**Run a single track directly**
```bash
streamlit run track_a/app.py
streamlit run track_b/app.py
streamlit run track_c/app.py
```

---

## ☁️ Streamlit Cloud Deployment

| Setting | Value |
|--------|-------|
| Repository | `gaurav-3821/Horizon` |
| Branch | `main` |
| Main file path | `horizon.py` |
| Python runtime | `python-3.12` (from `runtime.txt`) |

---

## 🐳 Docker (Optional)
```bash
docker build -t horizon .
docker run -p 8501:8501 horizon
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit, custom neobrutalist CSS |
| ML Models | XGBoost, LightGBM, CatBoost, scikit-learn |
| Explainability | SHAP |
| Balancing | SMOTE-NC |
| Tuning | Optuna |
| AI Advisor | LLaMA 3.3 70B via OpenRouter |
| Data | Tox21, Mendeley AMR, JHU CSSE, OWID |

---

## 📏 Design Rules

- One authoritative entrypoint: `horizon.py`
- One authoritative dependency file: `requirements.txt`
- Track-local assets stay inside their matching track folder
- No duplicate master-app files
- All paths resolved with `pathlib` relative to each app's directory

---

## 👤 Author

**Gaurav** · [@gaurav-3821](https://github.com/gaurav-3821)

---

<div align="center">
⭐ Star this repo if you find it useful!
</div>
