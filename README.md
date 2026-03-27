# Flux

## Multimodal Health Intelligence Platform

**Flux** is a multi-modal, cloud-deployed health intelligence platform built for hackathon-scale decision support across molecular safety, antimicrobial resistance, and epidemic forecasting. It unifies three independent machine learning systems into a single Streamlit dashboard so judges, clinicians, and reviewers can evaluate model behavior, inference speed, and deployment readiness from one interface.

---

## Technical Overview

Flux combines three specialized tracks:

- **Track A:** Molecular toxicity prediction from chemical structure
- **Track B:** Antibiotic resistance classification from small clinical data
- **Track C:** Epidemic spread forecasting from temporal and public-health signals

Each track was designed with a different modeling paradigm, but the overall system shares the same engineering priorities:

- **Mathematically honest validation**
- **Deployment-aware architecture**
- **Clear separation of training, inference, and dashboard layers**

---

## Track A: Toxicity Prediction

### GNN + Tabular Hybrid

Track A models molecular toxicity using a **hybrid architecture that combines Graph Neural Networks with tabular molecular descriptors**. The graph branch captures structural relationships directly from SMILES-derived molecular graphs, while the tabular branch encodes physicochemical descriptors and fingerprint-based structural summaries for late fusion classification across 12 Tox21 endpoints.

### V2 Upgrades

- **MACCS Keys Swap:** We upgraded the structural fingerprint pipeline to **MACCS Keys**, giving the model a more stable and deployment-friendly structural representation.
- **Knowledge Distillation:** We introduced **Teacher-Student distillation**, where the heavier hybrid model teaches a lightweight student model for **high-speed inference** in deployment scenarios.
- **Focal Loss with Masked Labels:** Because Tox21 contains extreme imbalance and many missing labels, loss is computed only where labels exist, preventing the model from learning false negatives from NaN targets.

### Key Results

- **Overall Validation AUC:** **0.8416**
- **NR-AR AUC:** **0.8357**
- Strong performance was retained even on the **highly imbalanced NR-AR target**, supported by **Focal Loss** and masked supervision.

---

## Track B: Antibiotic Resistance

### XGBoost with Leakage-Safe Validation

Track B predicts antibiotic resistance on a **very small clinical dataset (274 rows)**, where standard train/test reporting can easily inflate performance. Instead of chasing vanity metrics, we designed the pipeline around **strict 10-Fold Stratified Cross-Validation**, ensuring every performance estimate is evaluated under realistic sample scarcity.

### Methodological Strength

- **10-Fold Stratified CV:** **Mathematical honesty over vanity metrics**. This was the central design choice for a low-sample clinical dataset.
- **Per-antibiotic modeling:** Separate XGBoost models were trained for each antibiotic rather than forcing a single monolithic classifier.
- **Target encoding inside validation workflow:** High-cardinality clinical categories were encoded in a leakage-aware way.

### V2 Upgrade

- **Fold-Isolated SMOTE-NC:** We implemented **SMOTE-NC strictly inside the training fold of each CV split**, never before splitting. This lets us synthetically balance minority classes such as Ciprofloxacin resistance **without contaminating validation folds**.

This makes Track B one of the strongest methodological components of Flux: small-data aware, clinically cautious, and statistically defensible.

---

## Track C: Epidemic Forecaster

### Spatiotemporal LightGBM

Track C forecasts epidemic dynamics with a **spatiotemporal LightGBM pipeline** that predicts future case behavior from lagged case signals, rolling windows, spatial proxies, and public-health context features.

### Modeling Strategy

- **Strict Chronological / Temporal Split:** The model predicts the **actual future**, not randomly shuffled holdout rows.
- **External OWID Integration:** We merged **Our World in Data (OWID)** signals such as vaccination, testing, and stringency-related public-health indicators.
- **Behavioral + Global Context Features:** The feature set includes temporal lags, rolling means, and global epidemic context to improve forecasting realism.

### Key Results

- **R2 Score:** **0.8751**
- **MAE:** **389.19**

This track demonstrates a strong forecasting baseline with disciplined temporal evaluation and meaningful external-signal integration.

---

## Unified Dashboard

Flux is deployed as a **single master Streamlit application** that routes across all three tracks from one interface. This makes it easy for judges to:

- inspect model loading behavior
- compare inference workflows across tracks
- validate cloud deployment readiness
- evaluate a full-stack ML product rather than isolated notebooks

---

## Why Flux Stands Out

- **Multi-modal system design:** chemistry, clinical resistance, and epidemiology in one platform
- **Architectural depth:** **GNN + tabular fusion**, **Knowledge Distillation**, **XGBoost with fold-safe SMOTE-NC**, and **spatiotemporal LightGBM**
- **Validation rigor:** **masked focal loss**, **10-Fold CV**, and **strict temporal splits**
- **Deployment awareness:** unified Streamlit hub, path-safe code organization, and cloud-ready packaging

---

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the unified dashboard:

```bash
streamlit run flux.py
```

---

## Project Identity

**Flux** is not a single-model demo. It is a **unified medical and epidemiological AI dashboard** built to show how rigorous modeling, honest validation, and deployment-aware engineering can be combined into one coherent health intelligence platform.
