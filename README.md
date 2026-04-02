# equity-predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)

This is my Numerai pipeline. Numerai is basically a hedge fund that crowdsources ML predictions on obfuscated stock data — you download a dataset of anonymized features, predict targets, submit weekly, and get scored against other models worldwide. I've been competing for a while and this repo is where I keep all the actual research.

Stack: XGBoost + LightGBM stacked ensemble, era-aware cross-validation, feature neutralization. Still tweaking it.

## How it works

The pipeline has three stages:

1. **Feature Engineering** — Raw Numerai features enriched with rolling window statistics, cross-sectional rank normalization, era-neutral transformations, and pairwise interaction terms.
2. **Base Models** — XGBoost and LightGBM trained independently using 5-fold CV grouped by era. Out-of-fold predictions become the meta-learner's training signal.
3. **Meta-Learner** — Ridge regression that blends OOF predictions from both base models to produce the final output.

```
equity-predictor/
├── src/
│   ├── features.py        # Feature engineering pipeline
│   ├── models.py          # XGBoost, LightGBM, StackedEnsemble
│   ├── train.py           # Training entrypoint with CV
│   └── predict.py         # Inference on new data
├── analysis/
│   └── feature_analysis.py  # Feature importance & stability plots
├── notebooks/
│   └── model_development.ipynb
├── models/
├── requirements.txt
└── README.md
```

## Key technical decisions

- **Era-structured CV** — folds respect temporal era boundaries so there's no lookahead bias
- **Cross-sectional rank normalization** — features and targets ranked within each era to remove distributional shift
- **Era-neutral features** — values projected orthogonal to era means to isolate stock-specific signal
- **MI feature selection** — top-N features chosen via mutual information + Pearson filter
- **Sharpe-based evaluation** — weekly grouped correlations give a Sharpe ratio that mirrors the Numerai leaderboard metric

## Models

| Model | Library | Key Params |
|---|---|---|
| XGBoost | `xgboost` | `n_estimators=500`, `max_depth=5`, `lr=0.01`, `subsample=0.8` |
| LightGBM | `lightgbm` | `n_estimators=500`, `num_leaves=31`, `lr=0.01`, `feature_fraction=0.8` |
| Ridge Meta-Learner | `scikit-learn` | `alpha=1.0`, inputs = OOF from XGB + LGBM |

Competitive Numerai submissions target validation Spearman correlation of ~0.025–0.045 with Sharpe > 1.0.

## Install

```bash
git clone https://github.com/Bruh-Gang/equity-predictor.git
cd equity-predictor
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Download Numerai data:

```python
import numerapi
napi = numerapi.NumerAPI()
napi.download_dataset("v4.3/train.parquet", "data/train.parquet")
napi.download_dataset("v4.3/validation.parquet", "data/validation.parquet")
napi.download_dataset("v4.3/live.parquet", "data/live.parquet")
```

Train:

```bash
python src/train.py \
  --data_path data/train.parquet \
  --output_dir models/ \
  --n_folds 5
```

Predict:

```bash
python src/predict.py \
  --model_path models/ensemble.pkl \
  --data_path data/live.parquet \
  --output_path predictions/live_predictions.csv
```

Feature analysis:

```bash
python analysis/feature_analysis.py \
  --model_path models/ \
  --data_path data/train.parquet \
  --output_dir analysis/plots/
```

MIT License
