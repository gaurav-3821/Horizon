# Horizon

Horizon is a Streamlit Cloud-ready health intelligence dashboard that bundles three independent machine learning tracks into one deployable repository. The repo is organized by track so each model, dataset, artifact, and app can be updated manually without digging through mixed notebooks or duplicate entrypoints.

## Overview

The repository is split into three self-contained application areas:

- `track_a/`: toxicity prediction
- `track_b/`: antibiotic resistance prediction
- `track_c/`: epidemic spread forecasting

The single supported top-level app entrypoint is `flux.py`. That file routes into each track-specific Streamlit app through native `st.Page()` and `st.navigation()`.

## Streamlit Cloud Deployment

Use these settings when deploying on Streamlit Cloud:

- Repository: `gaurav-3821/Horizon`
- Branch: `main`
- Main file path: `flux.py`
- Python runtime: `python-3.12` from `runtime.txt`

The deployment is designed around:

- one root `requirements.txt`
- one root app entrypoint
- track-local `data/` and `artifacts/` directories
- `pathlib`-based path resolution from each app's own directory

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the unified dashboard:

```bash
streamlit run flux.py
```

Run a single track directly when you only need one app:

```bash
streamlit run track_a/app.py
streamlit run track_b/app.py
streamlit run track_c/app.py
```

## Repository Layout

### Root

- `flux.py`: unified Streamlit navigation hub
- `requirements.txt`: repository-wide Python dependencies
- `runtime.txt`: Streamlit Cloud runtime pin
- `Dockerfile`: optional container entrypoint for the same root app
- `PROJECT_STRUCTURE.md`: quick file map
- `README.md`: deployment and structure guide

### Track A

Track A contains the molecular toxicity system.

- `track_a/app.py`: Streamlit inference UI
- `track_a/track_a_pipeline.py`: preprocessing, training, distillation, and evaluation
- `track_a/data/`: Tox21 and ZINC source data
- `track_a/artifacts/`: trained models, SHAP outputs, and processed data

### Track B

Track B contains the antibiotic resistance system.

- `track_b/app.py`: Streamlit inference UI
- `track_b/track_b_model.py`: model training and validation
- `track_b/track_b_data_loader.py`: dataset cleaning and unification
- `track_b/data/`: Track B source datasets
- `track_b/artifacts/`: saved models, encoders, scalers, and CV summaries

### Track C

Track C contains the epidemic forecasting system.

- `track_c/app.py`: Streamlit analytics and prediction UI
- `track_c/track_c_model.py`: forecasting and classification pipeline
- `track_c/track_c_data_loader.py`: JHU and OWID merge logic plus feature engineering
- `track_c/data/`: epidemic time-series files
- `track_c/artifacts/`: saved forecaster, classifier, and evaluation results

### Docs

- `docs/`: handoff reports and presentation assets

## Design Rules In This Repo

- one authoritative Streamlit app entrypoint: `flux.py`
- one authoritative dependency file: `requirements.txt`
- track-local assets stay with their matching app
- no duplicate master-app files
- deployment settings should match the root repo layout, not old nested layouts

## Notes For Manual Editing

- If a saved artifact changes shape, update the app in the same track folder.
- Keep new models, pickles, and plots in that track's `artifacts/` folder.
- Keep raw or source datasets inside the matching track's `data/` folder.
- If deployment behavior changes, update `flux.py`, `requirements.txt`, and `runtime.txt` together.

## Current Deployment Target

This repository is prepared primarily for Streamlit Cloud. Docker is still available for local container use, but the intended deployment path is:

```bash
streamlit run flux.py
```
