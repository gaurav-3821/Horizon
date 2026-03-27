# Horizon Repository Structure

This repository is organized by track so each part of the project is easy to find and edit manually.

## Root

- `flux.py`: unified Streamlit hub
- `requirements.txt`: repository-wide Python dependencies
- `runtime.txt`: Streamlit Cloud Python version
- `Dockerfile`: optional container entrypoint for `flux.py`
- `README.md`: project overview

## Track A

- `track_a/app.py`: toxicity dashboard
- `track_a/track_a_pipeline.py`: main training pipeline
- `track_a/data/`: toxicity datasets
- `track_a/artifacts/`: trained model and explainability outputs

## Track B

- `track_b/app.py`: antibiotic resistance dashboard
- `track_b/track_b_model.py`: training and evaluation
- `track_b/track_b_data_loader.py`: data cleaning and loading
- `track_b/data/`: Track B datasets
- `track_b/artifacts/`: per-antibiotic models and CV summary

## Track C

- `track_c/app.py`: epidemic forecasting dashboard
- `track_c/track_c_model.py`: training and forecasting pipeline
- `track_c/track_c_data_loader.py`: merged epidemic data loader
- `track_c/data/`: JHU and OWID data
- `track_c/artifacts/`: trained forecaster, classifier, and metrics

## Docs

- `docs/`: technical report and presentation files
