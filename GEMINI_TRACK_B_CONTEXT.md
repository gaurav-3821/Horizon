# Gemini Context: Horizon Track B

## Project Scope
This context is only for **Track B: Antibiotic Resistance Prediction** in the private repo `gaurav-3821/Horizon`.

Do not assume Track A or Track C are relevant here. Current work is intentionally focused on Track B only.

## Repo And Active Files
- Repo root: `C:\Users\Gaurav\Documents\CodeCure\Roses G1\HorizonRepo`
- Main Track B training script: [track_b/track_b_model.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/track_b_model.py)
- Main Track B app: [track_b/app.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/app.py)
- Main Track B loader: [track_b/track_b_data_loader.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/track_b_data_loader.py)
- Runtime artifacts dir: `track_b/artifacts/`

## Track B Goal
Predict antibiotic resistance from clinical/sample metadata and curated AMR-related features, then expose this through a clinician-facing Streamlit app with explainability and an Anthropic-powered advisory section.

## Data Sources Integrated
The Track B unified dataset pipeline merged multiple sources:
1. Mendeley dataset
2. Kaggle multi-resistance dataset
3. `aro.tsv`
4. `card.json`
5. `protein_fasta_protein_homolog_model.fasta`

These were unified into:
- [track_b/artifacts/unified_dataset_final.csv](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/unified_dataset_final.csv)

Important:
- This CSV is large and is **not** required at runtime anymore for the app.
- A small runtime metadata file is now used instead:
  - [track_b/artifacts/app_metadata.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/app_metadata.json)

## Modeling Decisions Already Implemented

### Label Setup
- Track B was converted to **binary classification**
- `I` merged into `S`
- Final mapping:
  - `R = 1`
  - `S/I = 0`

### Leakage Cleanup
Several lookup-style columns were removed from training because they encoded the answer too directly.

Examples removed:
- `aro_gene_candidates`
- `aro_gene_name`
- `card_gene_name`
- `card_aro_accession`
- `card_gene_candidates`
- `card_associated_antibiotics`
- `card_resistance_mechanism`
- `card_amr_gene_family`
- `card_match_count`
- `antibiotic_name_norm`
- `resolved_gene_name`
- `resolved_gene_name_norm`
- `source_row_id`
- `source_dataset` as a training feature

Note:
- `source_dataset` is still retained in the data for weighting logic and source-aware analysis, but not used as a predictor.

### Feature Strategy In Final Cleaned Pipeline
The cleaned training/inference path uses:
- Fold-safe engineered features:
  - `antibiotic_class_resistance_rate`
  - `species_resistance_rate`
  - `antibiotic_species_interaction`
- Kept clean AMR features:
  - `aro_match_count`
  - `aro_antibiotic_class`
- Core clinical/sample fields:
  - `species`
  - `species_clean`
  - `antibiotic_name`
  - `antibiotic_class`
  - `age`
  - `gender`
  - `site`
  - `sample_type`
  - `Hospital_before`
  - `Hypertension`
  - `Diabetes`
  - `Infection_Freq`
- Top 20 k-mers by variance only

### Imbalance Handling
- `SMOTE-NC` is applied **inside each fold only**
- It is not applied globally before CV
- This was done specifically to avoid leakage

### Weighting
- Mendeley rows are trusted more heavily than Kaggle rows
- Weighting used:
  - `mendeley = 5.0`
  - `kaggle = 1.0`

There is also a pseudo-label branch that uses:
- `pseudo_kaggle = 2.0`

## Final Model Paths And Results

### Full stacked model on combined data
Artifact:
- [track_b/artifacts/stacked_model.pkl](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/stacked_model.pkl)

Model:
- XGBoost + LightGBM + CatBoost base models
- LogisticRegression meta-learner via `StackingClassifier`

Combined-data CV result:
- AUC mean: `0.7738`

Interpretation:
- Combined performance is dragged down by Kaggle

### Mendeley-only tuned model
Primary deployment artifact:
- [track_b/artifacts/stacked_model_tuned.pkl](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/stacked_model_tuned.pkl)

Best params file:
- [track_b/artifacts/best_params.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/best_params.json)

Mendeley-only tuned CV result:
- Fold AUCs: `0.8588, 0.8329, 0.8332, 0.8736, 0.8713`
- Mean AUC: `0.8540`
- Std: `0.0178`

This is the model currently used by the Streamlit app.

## Why Mendeley-Only Was Chosen
Source-wise diagnostics showed:
- Mendeley-only AUC stayed around `0.85`
- Kaggle-only AUC stayed around `0.77`
- Combined stayed near Kaggle, proving Kaggle is the main drag

Conclusion:
- The app now uses the stronger **Mendeley-only tuned stack**

## Pseudo-Label Branch Status
A separate pseudo-labeling branch was added in `track_b_model.py`.

Logic:
- Load tuned Mendeley-only stack
- Score Kaggle rows
- Keep only confident pseudo labels
- Combine with Mendeley rows
- Retrain stack

Observed result:
- With thresholds `>0.85` positive and `<0.15` negative:
  - selected pseudo rows were all negative
- Even after lowering positive threshold to `0.65`:
  - still `0` positive pseudo labels
  - only negative pseudo labels were selected

Pseudo artifact paths:
- [track_b/artifacts/stacked_model_pseudo.pkl](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/stacked_model_pseudo.pkl)
- [track_b/artifacts/pseudo_metrics.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/pseudo_metrics.json)

Important conclusion:
- The pseudo-label branch currently reinforces negatives only
- Its AUC is not trustworthy as real improvement
- It should not be treated as the primary model

## Explainability Status

SHAP was added for the tuned Mendeley-only model.

Artifacts:
- [track_b/artifacts/shap_global.png](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/shap_global.png)
- [track_b/artifacts/shap_features.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/shap_features.json)

Top SHAP global features currently:
1. `species`
2. `antibiotic_name`
3. `site`
4. `aro_antibiotic_class`
5. `sample_type`
6. `gender`
7. `antibiotic_species_interaction`
8. `Diabetes`
9. `age`
10. `kmer4_RALV`

The app also computes **local SHAP** for the current patient/sample prediction using the XGBoost base model inside the tuned stack.

## Current Track B App
File:
- [track_b/app.py](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/app.py)

What it does:
- Loads once via `@st.cache_resource`
- Uses:
  - tuned model bundle
  - SHAP global image
  - SHAP feature ranking JSON
  - lightweight runtime metadata JSON
- Does not load FASTA/CARD raw files at runtime
- Does not depend on the 110 MB unified dataset CSV at runtime

### UI Features
- Dark medical theme
- Prediction card:
  - Resistant / Susceptible
  - confidence meter
- Local explainability:
  - top 5 SHAP features as HTML/CSS bars
- Global explainability:
  - SHAP global bar image
  - top feature ranking table
- Anthropic-powered advisor section

## Anthropic API Integration
The app calls:
- `claude-sonnet-4-20250514`

The key is read from:
- `st.secrets["ANTHROPIC_API_KEY"]`

Local secrets path:
- `C:\Users\Gaurav\Documents\CodeCure\Roses G1\HorizonRepo\.streamlit\secrets.toml`

Expected content:
```toml
ANTHROPIC_API_KEY = "your-real-api-key-here"
```

This secrets file is ignored by git via:
- [\.gitignore](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/.gitignore)

## Important Runtime Note
There was confusion earlier because the app initially depended on:
- `track_b/artifacts/unified_dataset_final.csv`

That large file was not appropriate as a required runtime dependency for the pushed app.

This has now been fixed by switching the app to:
- [track_b/artifacts/app_metadata.json](C:/Users/Gaurav/Documents/CodeCure/Roses%20G1/HorizonRepo/track_b/artifacts/app_metadata.json)

So the pushed Track B app is now much more deployable.

## Local Run
From:
- `C:\Users\Gaurav\Documents\CodeCure\Roses G1\HorizonRepo\track_b`

Run:
```bash
streamlit run app.py
```

Observed local startup:
- App successfully starts on `http://localhost:8501`

## Current Known Issues
1. `use_container_width` deprecation warnings still exist in the app output.
2. Pseudo-label branch is not currently trustworthy because it selects only negative Kaggle pseudo labels.
3. The app currently relies on the tuned Mendeley-only model, which is scientifically more defensible than the combined-data model, but narrower in source coverage.

## What Has Been Pushed
The latest pushed Track B delivery includes:
- updated `track_b/app.py`
- tuned model artifact
- SHAP artifacts
- app metadata artifact
- requirements update

GitHub commit referenced during this phase:
- `405f29b` `Add Track B clinical advisor app with Anthropic secrets support`

## What Gemini Should Understand
If continuing this project, Gemini should assume:
1. Track B is the active scope
2. The deployed/inference model is the tuned **Mendeley-only stacked ensemble**
3. Combined-data performance is worse because Kaggle data drags AUC down
4. Pseudo-labeling has been explored but is currently not reliable
5. SHAP is available globally and locally
6. Anthropic integration is already implemented in the app
7. The next highest-value work is likely one of:
   - better Kaggle harmonization / denoising
   - improved calibrated confidence thresholds
   - replacing pseudo-labeling with more principled domain adaptation
   - production cleanup of Streamlit warnings and UI polish
