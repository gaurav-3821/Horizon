# Horizon — Full Gemini Context

This file is a handoff context document for Gemini or any other external assistant that needs an accurate view of the current Horizon repository, architecture, feature set, deployment state, and known issues.

## Project Identity

- Project name: `Horizon`
- Repository: `gaurav-3821/Horizon`
- Primary stack: `Streamlit`
- Root app entrypoint: `flux.py`
- Structure: multi-track health AI platform

Horizon is a unified health intelligence prototype built for the CodeCure AI Hackathon. It combines three tracks under one deployable repository:

1. Track A — Drug Toxicity Prediction
2. Track B — Antibiotic Resistance Prediction
3. Track C — Epidemic Spread Forecasting

The repo is organized so that each track has its own:

- app
- model/training code
- data loader
- `data/` directory
- `artifacts/` directory

The top-level `flux.py` acts as the launcher/navigation hub.

## Current Product State

The strongest and most polished part of the product is **Track B**.

Current relative maturity:

- Track B: most production-ready, most iterated UI, best end-to-end experience
- Track C: functional dashboard with rebuilt UI matching the design system
- Track A: currently transitioned away from dependency-heavy cloud runtime toward a tabular-only deployment-safe path

## Repository Layout

```text
HorizonRepo/
|-- flux.py
|-- README.md
|-- requirements.txt
|-- runtime.txt
|-- Dockerfile
|-- PROJECT_STRUCTURE.md
|-- GEMINI_TRACK_B_CONTEXT.md
|-- GEMINI_FULL_CONTEXT.md
|-- docs/
|-- track_a/
|-- track_b/
`-- track_c/
```

## Root-Level App Behavior

### `flux.py`

Purpose:
- unified Streamlit launcher for all tracks

Behavior:
- default entry path is Track B
- routes to individual track apps through runtime dispatch
- Track A cloud fallback was previously handled gracefully because of missing scientific dependencies
- Track C needed local path fixes so it could load correctly via the root launcher

Important:
- Track apps should be able to run from within `flux.py` even when executed from the repo root
- `sys.path` insertion was added to track apps where needed for local sibling imports

## Design System Used In The Apps

Track B established the main UI system, and later Track C and Track A were aligned to it.

Shared design language:

- light off-white background: `#f5f5f0`
- cards: white
- black hard borders
- black box-shadows
- accent blue: `#0066cc`
- font: `Inter`
- neobrutalism-inspired styling
- amber disclaimer banners
- hard-edged metric cards

## Track B — Antibiotic Resistance Prediction

### Purpose

Track B predicts whether a case is likely:

- `Susceptible`
- `Resistant`

based on structured patient/sample features and a tuned ensemble model.

### Data Sources Integrated

Track B combines multiple AMR-related sources:

1. Mendeley dataset
2. Kaggle multi-resistance dataset
3. `aro.tsv`
4. `card.json`
5. protein FASTA homolog data

### Data Loader

Main file:
- `track_b/track_b_data_loader.py`

Responsibilities:
- integrates the data sources
- creates unified dataset
- attaches annotation-derived features
- exports unified CSV to `track_b/artifacts/unified_dataset_final.csv`

### Modeling

Main file:
- `track_b/track_b_model.py`

Track B evolved through many phases:

1. early per-antibiotic XGBoost path
2. CV + SMOTE-NC path
3. leakage cleanup
4. stacked ensemble
5. Mendeley-only tuned stack for strongest inference path

Final high-value training path:
- tuned stacked ensemble
- base learners:
  - XGBoost
  - LightGBM
  - CatBoost
- meta learner:
  - Logistic Regression

### Key modeling decisions

- leakage-prone lookup features were removed
- Mendeley-only path showed stronger quality than combined data path
- SHAP explanations were added
- PCA projection used for visual cluster positioning

### Track B metric

Important validated metric:
- Mendeley-only tuned stacked ensemble AUC: `0.8540`

### Track B artifacts

Important artifacts in `track_b/artifacts/` include:

- `stacked_model_tuned.pkl`
- `stacked_model.pkl`
- `stacked_model_pseudo.pkl`
- `tabular`-style XGB artifacts from earlier stages
- `shap_features.json`
- `shap_global.png`
- `best_params.json`
- `mendeley_only_metrics.json`
- `stacked_metrics.json`
- `pseudo_metrics.json`
- `unified_dataset_final.csv`

### Track B app

Main file:
- `track_b/app.py`

Current UI characteristics:

- title: `Horizon`
- subtitle: `Track B — Antibiotic Resistance Prediction`
- metric ribbon at top
- left column:
  - Patient Input Command Center
- center column:
  - 3D PCA plot
  - prediction result
  - confidence meter
  - local SHAP contributions
  - Horizon Intelligence box
- right column:
  - Top Global Drivers

### PCA visualization notes

The PCA visualization was tuned repeatedly.

Current intent:
- visually dense cluster
- no legend
- no plot title inside the Plotly chart itself
- separate section header above the plot:
  - `Proximity to Historical Data Clusters`
- current patient shown as yellow diamond with blue outline

### Track B UI/input issues that were fixed

Recent fixes included:

- species dropdown sourcing
- hypertension/diabetes options normalized
- dropdown styling so options are visible on light background
- graph spacing and overlap issues
- header rename from `AI Clinical Advisor` to `Horizon`

### Important Track B current state

Track B is the flagship demo path and should be treated as the primary user-facing workflow.

## Track C — Epidemic Spread Forecasting

### Purpose

Track C forecasts epidemic spread and displays country-level spread/risk information.

### Main files

- `track_c/app.py`
- `track_c/track_c_model.py`
- `track_c/track_c_data_loader.py`

### Data

Track C uses epidemic time-series plus merged external signals.

Known relevant data sources:
- JHU-style time-series confirmed cases
- OWID-style public health context

### Track C artifacts

In `track_c/artifacts/`:

- `track_c_forecaster.pkl`
- `track_c_classifier.pkl`
- `track_c_results.json`

### Track C metrics

From `track_c_results.json`:

- Forecaster:
  - `mae = 389.19`
  - `rmse = 1781.47`
  - `r2 = 0.8751`
- Classifier:
  - `accuracy = 0.9713`
  - `macro_f1 = 0.9701`

### Track C app status

Track C was rebuilt to match the Track B visual system:

- same background
- same neobrutalism metric cards
- same Inter font
- same amber disclaimer pattern

Key note:
- `sys.path` insertion was added at the top of `track_c/app.py` so it loads correctly when launched through `flux.py`

## Track A — Drug Toxicity Prediction

### Original architecture

Track A originally used a hybrid molecular pipeline:

- graph neural network
- tabular molecular descriptors
- scientific dependencies:
  - `rdkit`
  - `torch`
  - `torch-geometric`

This original Track A pipeline is still present in source files such as:

- `track_a/track_a_pipeline.py`
- `track_a/tox_data_loader.py`
- `track_a/hybrid_tox_predictor.py`

### Important artifact state

The saved Track A processed artifact:
- `track_a/artifacts/tox21_processed.pkl`

contains:

- toxicity target columns
- scalar molecular descriptors
- `morgan_fp`
- `graph`
- `mask_vector`
- `label_vector`

This means the processed artifact already includes tabular features that can be used without rdkit at runtime.

### Track A tabular-only pivot

A new deployment-safe Track A path was created:

- no `rdkit`
- no `torch`
- no `torch-geometric`

New file:
- `track_a/track_a_tabular.py`

This file is intended to:

- load `tox21_processed.pkl`
- extract scalar molecular descriptors
- flatten `morgan_fp`
- train one XGBoost classifier per endpoint
- run 5-fold CV
- save:
  - `tabular_models.pkl`
  - `tabular_features.json`
  - `tabular_metrics.json`

### Current Track A app

`track_a/app.py` was rebuilt into a dependency-light Streamlit page that:

- uses the same Track B design system
- loads tabular-only artifacts
- takes numeric descriptor inputs
- predicts all 12 endpoints
- displays SHAP and training images already present in artifacts

### Important current Track A limitation

The tabular training script was created and syntax-checked, but when it was run it did not complete successfully within the shell timeout.

Observed outcome:
- no `tabular_models.pkl` yet
- no `tabular_features.json` yet
- no `tabular_metrics.json` yet

So Track A’s new tabular app exists, but it currently depends on artifacts that still need to be generated.

This is the main unresolved Track A issue.

## Known Metrics Across Prototype

### Track B
- AUC = `0.8540` on Mendeley-only tuned stacked ensemble

### Track C
- Forecaster R² = `0.8751`
- Forecaster MAE = `389.19`
- Classifier Accuracy = `0.9713`
- Classifier Macro F1 = `0.9701`

### Track A
- current tabular retraining metrics are not yet available because the new XGBoost training script has not finished and artifacts have not been produced

## Deployment Notes

### Dependencies

Root `requirements.txt` was previously simplified for Streamlit Cloud compatibility and now focuses on lighter runtime dependencies.

Track A’s original dependency-heavy path is not cloud-safe due to:
- `rdkit`
- `torch`
- `torch-geometric`

That is why Track A was redirected toward a tabular-only path.

### Streamlit Cloud notes

Important repo entrypoint:
- `flux.py`

Track-specific issues handled over time:
- Track B import path fixes
- Track C import path fixes
- Track A fallback messaging for cloud-incompatible scientific deps

## Files Worth Knowing

### Root
- `flux.py`
- `README.md`
- `requirements.txt`
- `runtime.txt`

### Track A
- `track_a/app.py`
- `track_a/track_a_tabular.py`
- `track_a/track_a_pipeline.py`
- `track_a/tox_data_loader.py`
- `track_a/hybrid_tox_predictor.py`

### Track B
- `track_b/app.py`
- `track_b/track_b_model.py`
- `track_b/track_b_data_loader.py`

### Track C
- `track_c/app.py`
- `track_c/track_c_model.py`
- `track_c/track_c_data_loader.py`

## What Gemini Should Assume Right Now

1. Horizon is a multi-track Streamlit repository
2. Track B is the strongest and should be treated as the flagship
3. Track C is functional and styled consistently
4. Track A is mid-transition from dependency-heavy GNN path to tabular-only deployment-safe path
5. Track A tabular training script exists but has not completed successfully yet
6. If improving or discussing the product, center Track B first
7. If helping with deployment, remember Track A original scientific stack is not cloud-friendly

## Best Short Description

Horizon is a multi-track health intelligence prototype that combines antibiotic resistance prediction, molecular toxicity analysis, and epidemic forecasting into one deployable Streamlit platform, with Track B currently serving as the flagship explainable clinical workflow.
