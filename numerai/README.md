# Numerai Tournament — Competition Pipeline

A production-quality, competition-winning framework for the [Numerai Tournament](https://numer.ai).

## What Is Numerai?

Numerai is a crowdsourced hedge fund. Every week you get encrypted financial data (~2300 features representing global stocks) and must predict which stocks will outperform over the next 20 days. Your predictions are combined with thousands of other participants to form the Meta Model, which controls real capital. You earn NMR cryptocurrency proportional to your CORR + MMC scores.

The trick: the data is obfuscated (you can't know what the features represent). This levels the playing field — pure ML skill wins.

---

## Architecture

```
numerai/
├── train.py                  # Main training pipeline (run this!)
├── predict.py                # Weekly live prediction submission
├── requirements.txt
├── configs/
│   └── config.py             # All configuration in one place
├── models/
│   ├── lgbm_model.py         # LightGBM (fastest, strongest)
│   ├── xgb_model.py          # XGBoost (good regularization)
│   ├── catboost_model.py     # CatBoost (symmetric trees, diverse)
│   ├── neural_net.py         # MLP with BN + SWA (orthogonal signal)
│   └── ensemble.py           # Weighted ensemble + neutralization
├── utils/
│   ├── data_loader.py        # Numerai data download & loading
│   ├── metrics.py            # CORR, Sharpe, feature neutralization
│   ├── cross_validation.py   # ERA-aware CV (critical for Numerai)
│   └── submission.py         # API submission utilities
└── notebooks/
    └── explore.py            # Diagnostics and visualization
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your Numerai API keys
- Create an account at [numer.ai](https://numer.ai)
- Go to Account Settings → API Keys
- Set environment variables:
```bash
export NUMERAI_PUBLIC_ID="your_public_id"
export NUMERAI_SECRET_KEY="your_secret_key"
```

### 3. Create a model slot on Numerai
- Go to numer.ai → Models → Create Model
- Note the model name exactly

---

## Training

### Full pipeline (recommended — takes 2-6 hours with 50GB RAM)
```bash
python train.py --feature-set medium
```

### Use all 2300 features (maximum signal, ~10-15 hours)
```bash
python train.py --feature-set all
```

### Fast mode (LightGBM only, no neural net, ~30 min)
```bash
python train.py --no-xgb --no-catboost --no-nn --feature-set small
```

### With hyperparameter optimization (adds ~1-2 hours)
```bash
python train.py --tune
```

### Auto-submit after training
```bash
python train.py --submit --model-name your_model_name
```

---

## Weekly Submission (every week!)

Numerai requires weekly submissions. After initial training, just run:
```bash
python predict.py --model-name your_model_name --submit
```

---

## Strategy & Scoring

### Metrics That Matter

| Metric | Description | Paid? |
|--------|-------------|-------|
| **CORR** | Your predictions' correlation to target (raw alpha) | ✅ Yes |
| **MMC** | Your contribution to the Meta Model (originality) | ✅ Yes |
| **FNC** | Feature-neutral correlation (after neutralization) | ❌ Info only |
| **CWMM** | Correlation with Meta Model (lower = more unique) | ❌ Info only |

### What Wins
1. **High Sharpe** (consistent CORR across many eras) beats occasional high CORR spikes
2. **Low correlation with Meta Model** = higher MMC payout (be different!)
3. **Feature neutralization** dramatically improves Sharpe by reducing feature exposure risk
4. **Multi-target ensembling** (training on 5+ targets, averaging) smooths out noise
5. **Era-balanced training** (weight each era equally, not each stock)

### The Neutralization Trade-off
- More neutralization → lower raw CORR, higher Sharpe, more MMC
- Less neutralization → higher raw CORR, more drawdown risk
- `0.5` is the community-validated sweet spot

---

## Configuration

Edit `configs/config.py` to change:
- Which targets to train on (use `target_cyrus_v4_20` as primary)
- Model hyperparameters
- Neutralization proportion
- Feature set size

### Recommended Targets (v4.2 dataset)
```python
# Primary
"target_cyrus_v4_20"     # Main 20-day target

# Auxiliary (train models on these, average predictions)
"target_victor_v4_20"
"target_ralph_v4_20"
"target_waldo_v4_20"
"target_jerome_v4_20"
"target_evelyn_v4_20"
```

---

## Advanced Tips

### GPU Acceleration
In `configs/config.py`, set:
```python
# LightGBM
"device": "gpu"
# XGBoost
"device": "cuda"
# CatBoost
"task_type": "GPU"
# Neural Net
device = "cuda"
```

### Different Feature Sets
- `small` (~50 features): Fast experiments, debugging
- `medium` (~700 features): Best speed/accuracy balance ✅
- `all` (~2300 features): Maximum signal, requires 50GB+ RAM

### Understanding the Neutralization Math
Feature neutralization removes the linear component of your predictions that's explained by the features themselves. It forces your model to capture non-linear alpha that isn't already priced into the obvious features — exactly what the hedge fund wants.

```
pred_neutralized = pred - proportion * X(X'X)^{-1}X' pred
```

---

## Troubleshooting

**"Round not open"** — Submissions accepted Sat 18:00 UTC → Mon 14:30 UTC

**Memory errors** — Switch to `feature_set="small"` or `feature_set="medium"`, and set `int8=True`

**Low validation scores** — Try increasing `neutralization_proportion` to 0.7 or adding more targets

**Negative MMC** — Your model is too correlated with the benchmark. Try different architectures or feature groups.

---

## Resources

- [Numerai Docs](https://docs.numer.ai)
- [Numerai Forum](https://forum.numer.ai) — best community resource
- [NumerAPI Docs](https://numerapi.readthedocs.io)
- [Example Scripts (official)](https://github.com/numerai/example-scripts)
- [Scoring Tools](https://github.com/numerai/numerai-tools)
