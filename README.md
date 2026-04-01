# Equity Return Predictor — Numerai Tournament Research

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green?logo=lightgbm&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Ensemble machine learning system for predicting equity returns using Numerai's obfuscated, era-structured financial dataset. Combines gradient boosting, feature engineering, and stacked meta-learners to compete in a live global prediction tournament evaluated against professional quantitative researchers.

---

## Architecture

```
equity-predictor/
├── src/
│   ├── features.py        # Feature engineering pipeline
│   ├── models.py          # XGBoost, LightGBM, StackedEnsemble
│   ├── train.py           # Training entrypoint with cross-validation
│   └── predict.py         # Inference on new data
├── analysis/
│   └── feature_analysis.py  # Feature importance & stability plots
├── notebooks/
│   └── model_development.ipynb  # Research walkthrough
├── models/                # Serialized trained models
├── requirements.txt
└── README.md
```

The pipeline follows a three-stage design:

1. **Feature Engineering** — Raw Numerai features are enriched with rolling window statistics, cross-sectional rank normalization, era-neutral transformations, and pairwise interaction terms.
2. **Base Models** — XGBoost and LightGBM are trained independently using 5-fold cross-validation grouped by era. Out-of-fold (OOF) predictions serve as the meta-learner's training signal.
3. **Meta-Learner (Stacked Ensemble)** — A Ridge regression meta-learner blends OOF predictions from both base models to produce the final output, reducing variance while preserving signal.

---

## Features

- **Era-structured cross-validation** — folds respect temporal era boundaries to prevent lookahead bias
- **Rolling statistics** — mean, std, and skew computed over 5, 10, and 20 period windows
- **Cross-sectional rank normalization** — features and targets ranked within each era to remove distributional shift
- **Era-neutral features** — feature values projected orthogonal to era means to isolate stock-specific signal
- **Interaction terms** — pairwise products of high-MI feature pairs
- **Mutual information feature selection** — top-N features chosen via MI + Pearson correlation filter
- **Sharpe-based evaluation** — weekly grouped correlations produce a Sharpe ratio analogous to the Numerai leaderboard metric
- **Full model serialization** — all models saved with `joblib` for reproducible inference

---

## Models Used

| Model | Library | Key Hyperparameters |
|---|---|---|
| XGBoost | `xgboost` | `n_estimators=500`, `max_depth=5`, `learning_rate=0.01`, `subsample=0.8` |
| LightGBM | `lightgbm` | `n_estimators=500`, `num_leaves=31`, `learning_rate=0.01`, `feature_fraction=0.8` |
| Ridge Meta-Learner | `scikit-learn` | `alpha=1.0`, inputs = OOF predictions from XGB + LGBM |

---

## Performance Metrics

The evaluation suite reports the following metrics on held-out validation data:

| Metric | Description |
|---|---|
| **Pearson Correlation** | Linear correlation between predicted ranks and realized returns |
| **Spearman Correlation** | Rank-based correlation (primary Numerai metric) |
| **Sharpe Ratio** | Mean era-wise correlation divided by std — measures signal consistency |
| **Max Drawdown** | Worst peak-to-trough sequence of negative era correlations |

Competitive Numerai submissions typically target a mean validation Spearman correlation of **0.025–0.045** with a Sharpe > 1.0.

---

## Installation

```bash
git clone https://github.com/Bruh-Gang/equity-predictor.git
cd equity-predictor
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Download Numerai Data

```python
import numerapi
napi = numerapi.NumerAPI()
napi.download_dataset("v4.3/train.parquet", "data/train.parquet")
napi.download_dataset("v4.3/validation.parquet", "data/validation.parquet")
napi.download_dataset("v4.3/live.parquet", "data/live.parquet")
```

### Train

```bash
python src/train.py \
  --data_path data/train.parquet \
  --output_dir models/ \
  --n_folds 5
```

### Predict

```bash
python src/predict.py \
  --model_path models/ensemble.pkl \
  --data_path data/live.parquet \
  --output_path predictions/live_predictions.csv
```

### Feature Analysis

```bash
python analysis/feature_analysis.py \
  --model_path models/ \
  --data_path data/train.parquet \
  --output_dir analysis/plots/
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

Built as research tooling for the [Numerai](https://numer.ai) tournament — a weekly, stock-market-neutral ML competition using obfuscated financial features.
