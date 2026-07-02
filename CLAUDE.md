# Spokane AVM — Automated Valuation Model

ML portfolio project: predict residential sale prices for Spokane, WA properties.
Part of a 6-month real estate AI portfolio. Single-family residential, Spokane city limits first.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Launch Streamlit app
uv run streamlit run app/main.py

# Run a notebook (from repo root)
uv run jupyter lab
```

## Stack

- **Python** via uv (see pyproject.toml for deps)
- **IDE**: PyCharm
- **Modeling**: scikit-learn, XGBoost, LightGBM, PyTorch (tabular MLP)
- **Experiment tracking**: none yet — Weights & Biases coming in Month 3
- **App**: Streamlit + Folium (comparable sales map)
- **Data**: Redfin bulk CSV, Zillow Research CSV, Walk Score API, GreatSchools/NCES

## Project Structure

```
data/
  raw/          # downloaded CSVs, never modified
  processed/    # cleaned, feature-engineered outputs
notebooks/      # EDA and experimentation (not production code)
src/
  features.py   # sklearn Pipeline + ColumnTransformer feature engineering
  train.py      # model training and Optuna tuning
  evaluate.py   # metrics, SHAP analysis, error analysis
  predict.py    # inference utilities
app/
  main.py       # Streamlit UI
models/         # saved model artifacts (.pkl, .pt)
```

## Conventions

- Feature engineering lives in `src/features.py` as a reusable sklearn Pipeline — not inline in notebooks
- All models trained in `src/train.py`; notebooks are for exploration only, not model definitions
- Saved models go in `models/` with a version suffix: `xgb_v1.pkl`, `mlp_v1.pt`
- Data in `data/raw/` is read-only — never write to it; write outputs to `data/processed/`
- Keep a `model_comparison.md` at repo root updated with RMSE/MAE/R² for each model run

## What I Own (Do Not Generate)

The following must be written by the developer, not generated:
- All feature engineering logic in `src/features.py`
- Model architecture and training loop in `src/train.py`
- Optuna objective functions and hyperparameter search spaces
- SHAP analysis and plots in `src/evaluate.py`
- EDA decisions in notebooks

Generate freely: data download scripts, API wrappers, Streamlit UI boilerplate,
Dockerfiles, CI config, README drafts, utility/helper functions.

## Anti-Patterns to Avoid

- Don't mix feature engineering logic into notebooks — it belongs in `src/features.py`
- Don't one-hot encode zip codes directly; use target encoding (median price per zip) as primary
- Don't train on the full dataset before confirming the pipeline works on a 1,000-row sample first
- Don't skip the linear regression baseline — it's the benchmark everything else is measured against
