# Horizon

Horizon is a multi-track Streamlit application built for the CodeCure AI Hackathon. It combines three health AI workflows in one repository:

- Track B: antibiotic resistance prediction
- Track A: drug toxicity prediction
- Track C: epidemic spread forecasting

The repository is structured so each track can be edited, trained, and deployed independently, while `flux.py` provides a single entrypoint for the full app.

## Current Status

The primary polished experience in this repo is **Track B**.

Track B currently includes:

- unified AMR dataset pipeline across multiple data sources
- tuned stacked ensemble artifacts for Mendeley-focused inference
- local SHAP explainability
- Streamlit clinical dashboard with PCA cluster view and prediction UI

## Repository Layout

```text
HorizonRepo/
|-- flux.py
|-- requirements.txt
|-- runtime.txt
|-- Dockerfile
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- docs/
|-- track_a/
|-- track_b/
`-- track_c/
```

### Root files

- [flux.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/flux.py): unified Streamlit launcher
- [requirements.txt](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/requirements.txt): deployment dependencies
- [runtime.txt](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/runtime.txt): Python runtime pin for Streamlit Cloud
- [Dockerfile](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/Dockerfile): optional container entrypoint

### Track A

- [app.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_a/app.py): Streamlit UI
- [track_a_pipeline.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_a/track_a_pipeline.py): training and evaluation pipeline

### Track B

- [app.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/app.py): Streamlit clinical dashboard
- [track_b_data_loader.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/track_b_data_loader.py): dataset integration and cleaning
- [track_b_model.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/track_b_model.py): feature engineering, diagnostics, and training
- `track_b/data/`: raw Track B datasets and CARD/ARO assets
- `track_b/artifacts/`: tuned models, SHAP outputs, metrics, diagnostics, and unified dataset

### Track C

- [app.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_c/app.py): Streamlit UI
- [track_c_model.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_c/track_c_model.py): forecasting and classification pipeline
- [track_c_data_loader.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_c/track_c_data_loader.py): epidemic data merge and feature prep

## Track B Summary

Track B is the most production-ready path in this repo.

### Data sources integrated

- Mendeley AMR dataset
- Kaggle multi-resistance dataset
- `aro.tsv`
- `card.json`
- protein FASTA homolog model data

### Training outputs available

Key Track B artifacts already present in:

- [track_b/artifacts/unified_dataset_final.csv](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/unified_dataset_final.csv)
- [track_b/artifacts/stacked_model_tuned.pkl](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/stacked_model_tuned.pkl)
- [track_b/artifacts/shap_features.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/shap_features.json)
- [track_b/artifacts/shap_global.png](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/shap_global.png)
- [track_b/artifacts/mendeley_only_metrics.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/mendeley_only_metrics.json)
- [track_b/artifacts/best_params.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/best_params.json)

### Modeling notes

Track B work in this repo includes:

- leakage cleanup for identifier-like columns
- Mendeley-only tuned stack for higher-quality clinical inference
- SHAP-based local and global explainability
- PCA projection view for positioning current input against historical cases

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The App

Run the unified app:

```bash
streamlit run flux.py
```

Run Track B directly:

```bash
streamlit run track_b/app.py
```

Run Track A directly:

```bash
streamlit run track_a/app.py
```

Run Track C directly:

```bash
streamlit run track_c/app.py
```

## Streamlit Cloud

Recommended deployment settings:

- Repository: `gaurav-3821/Horizon`
- Branch: `main`
- Main file path: `flux.py`
- Python version: from [runtime.txt](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/runtime.txt)

## Design Principles In This Repo

- track-local code stays inside its own folder
- models and generated outputs stay in the matching `artifacts/` folder
- raw data stays in the matching `data/` folder
- root app orchestration stays in `flux.py`
- deployment dependencies are controlled from the root `requirements.txt`

## Notes

- Track A may fail in environments where `rdkit` is not available.
- Track B is the current priority path and default user-facing workflow.
- Large model/data artifacts are included under Track B and may require Git LFS-aware workflows.

## License

This repository includes a [LICENSE](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/LICENSE) file at the root.
