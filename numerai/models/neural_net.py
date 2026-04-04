"""
Neural Network for Numerai
--------------------------
Multi-layer MLP with:
- Batch normalization (critical for obfuscated features)
- Dropout for regularization
- SWA (Stochastic Weight Averaging) for better generalization
- Label smoothing / soft targets
- Era-balanced sampling via WeightedRandomSampler

Neural nets provide orthogonal signal to GBDTs, improving ensemble diversity.
Best used with feature neutralization to reduce feature exposure.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NumeraiMLP:
    """
    PyTorch-based MLP optimized for the Numerai tournament.

    Architecture: Input → BN → [Linear→BN→GELU→Dropout] × N → Linear

    Key design choices:
    - GELU activation (smoother than ReLU, works well with financial data)
    - Batch normalization after each linear layer (helps with obfuscated features)
    - Low dropout (0.1-0.2) to not destroy signal in noisy financial data
    - Adam + CosineAnnealingLR schedule
    - SWA for final model (community-validated improvement)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [512, 256, 128, 64],
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 8192,
        epochs: int = 100,
        swa_start: int = 50,
        swa_lr: float = 1e-4,
        targets: Optional[List[str]] = None,
        era_col: str = "era",
        device: str = "auto",
        seed: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.targets = targets or ["target"]
        self.era_col = era_col
        self.seed = seed
        self.models: Dict[str, Any] = {}
        self.feature_cols: Optional[List[str]] = None
        self._scalers: Dict[str, Any] = {}

        # Device selection
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _build_model(self, input_size: int):
        """Build the MLP architecture."""
        import torch
        import torch.nn as nn

        layers = []
        in_size = input_size
        layers.append(nn.BatchNorm1d(in_size))

        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ])
            in_size = hidden_size

        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)

    def _era_weights_tensor(self, df: pd.DataFrame):
        """Era-balanced sample weights as tensor."""
        import torch
        era_sizes = df.groupby(self.era_col)[self.era_col].transform("count")
        weights = (1.0 / era_sizes).values.astype(np.float32)
        return torch.tensor(weights)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        val_df: Optional[pd.DataFrame] = None,
        patience: int = 20,
    ) -> "NumeraiMLP":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
            from torch.optim.swa_utils import AveragedModel, SWALR
        except ImportError:
            raise ImportError("Run: pip install torch")

        from scipy.stats import rankdata

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.feature_cols = feature_cols
        n_features = len(feature_cols)

        for target in self.targets:
            if target not in train_df.columns:
                logger.warning(f"Target '{target}' not found, skipping.")
                continue

            logger.info(f"Training MLP on target: {target} (device={self.device})")

            # Prepare tensors
            X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
            # Rank-normalize targets (helps neural nets)
            y_raw = train_df[target].values
            y_ranked = rankdata(y_raw) / len(y_raw) - 0.5   # center around 0
            y_train = torch.tensor(y_ranked, dtype=torch.float32).unsqueeze(1)

            era_weights = self._era_weights_tensor(train_df)
            sampler = WeightedRandomSampler(era_weights, num_samples=len(era_weights), replacement=True)

            dataset = TensorDataset(X_train, y_train)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                                num_workers=0, pin_memory=(self.device == "cuda"))

            # Validation
            if val_df is not None and target in val_df.columns:
                X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
                y_val_raw = val_df[target].values
                y_val_ranked = rankdata(y_val_raw) / len(y_val_raw) - 0.5
                y_val = torch.tensor(y_val_ranked, dtype=torch.float32).unsqueeze(1)
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 4,
                                        shuffle=False, num_workers=0)
            else:
                val_loader = None

            # Model, optimizer, scheduler
            model = self._build_model(n_features).to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
            criterion = nn.MSELoss()

            # SWA model
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)

            best_val_loss = float("inf")
            best_state = None
            patience_count = 0

            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                if epoch >= self.swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

                # Validation
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            pred = model(X_batch)
                            val_loss += criterion(pred, y_batch).item()
                    val_loss /= len(val_loader)

                    if epoch % 10 == 0:
                        logger.info(f"  Epoch {epoch:3d} | train_loss={total_loss/len(loader):.5f} | val_loss={val_loss:.5f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= patience:
                            logger.info(f"  Early stopping at epoch {epoch}.")
                            break
                elif epoch % 10 == 0:
                    logger.info(f"  Epoch {epoch:3d} | train_loss={total_loss/len(loader):.5f}")

            # Update SWA BN stats
            torch.optim.swa_utils.update_bn(loader, swa_model, device=self.device)

            # Use SWA model if trained long enough, otherwise best checkpoint
            if epoch >= self.swa_start and best_state is None:
                self.models[target] = swa_model.cpu()
            elif best_state is not None:
                model.load_state_dict(best_state)
                self.models[target] = model.cpu()
            else:
                self.models[target] = model.cpu()

            logger.info(f"  MLP training complete for target: {target}")

        return self

    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        try:
            import torch
        except ImportError:
            raise ImportError("Run: pip install torch")

        from scipy.stats import rankdata

        fc = feature_cols or self.feature_cols
        X = torch.tensor(df[fc].values, dtype=torch.float32)

        preds_list = []
        for target, model in self.models.items():
            model.eval()
            with torch.no_grad():
                p = model(X).squeeze().numpy()
            p_ranked = rankdata(p) / len(p)
            preds_list.append(p_ranked)

        return np.mean(preds_list, axis=0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> "NumeraiMLP":
        import torch
        return torch.load(path, map_location="cpu")
