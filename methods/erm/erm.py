from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import build_mlp


@dataclass
class ERMConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    weight_decay: float = 0.0
    patience: int = 20
    min_delta: float = 0.0
    optimizer: str = "adam"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility-first settings.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_xy(X: torch.Tensor, y: torch.Tensor) -> None:
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X shape (N, 4), got {tuple(X.shape)}")
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"Expected y shape (N, 1), got {tuple(y.shape)}")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not torch.isfinite(X).all() or not torch.isfinite(y).all():
        raise ValueError("X or y contains NaN/Inf.")


def build_optimizer(model: nn.Module, cfg: ERMConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


@torch.no_grad()
def evaluate_mse(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    _validate_xy(X, y)

    model.eval()
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    squared_error_sum = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        squared_error_sum += torch.sum((pred - yb) ** 2).item()
        n += yb.numel()

    return squared_error_sum / n


def train_erm(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: ERMConfig,
    device: torch.device,
    checkpoint_path: Optional[str | Path] = None,
    normalization: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    """
    Train the common 4->64->64->1 MLP with standard ERM (mean MSE).

    Validation MSE selects the best epoch.
    The best model, not the final epoch, is returned.
    """
    _validate_xy(X_train, y_train)
    _validate_xy(X_val, y_val)
    set_seed(cfg.seed)

    model = build_mlp().to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = build_optimizer(model, cfg)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    history: List[Dict[str, float | int]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()

        train_squared_error_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_squared_error_sum += torch.sum((pred.detach() - yb) ** 2).item()
            train_count += yb.numel()

        train_mse = train_squared_error_sum / train_count
        val_mse = evaluate_mse(
            model,
            X_val,
            y_val,
            device=device,
            batch_size=max(cfg.batch_size, 1024),
        )

        history.append({
            "epoch": epoch,
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
        })

        improved = val_mse < (best_val_mse - cfg.min_delta)

        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                payload = {
                    "model_state_dict": best_state,
                    "seed": cfg.seed,
                    "best_epoch": best_epoch,
                    "best_val_mse": float(best_val_mse),
                    "config": asdict(cfg),
                }

                if normalization is not None:
                    payload["normalization"] = {
                        key: value.detach().cpu()
                        for key, value in normalization.items()
                    }

                torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if verbose and (
            epoch == 1
            or epoch % 10 == 0
            or improved and epoch <= 5
        ):
            print(
                f"[ERM][seed={cfg.seed}] "
                f"epoch={epoch:03d} "
                f"train_mse={train_mse:.6f} "
                f"val_mse={val_mse:.6f} "
                f"best={best_val_mse:.6f}@{best_epoch}"
            )

        if epochs_without_improvement >= cfg.patience:
            if verbose:
                print(
                    f"[ERM][seed={cfg.seed}] early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}, best val MSE={best_val_mse:.6f}"
                )
            break

    model.load_state_dict(best_state)

    result = {
        "best_epoch": best_epoch,
        "best_val_mse": float(best_val_mse),
        "history": history,
        "config": asdict(cfg),
    }
    return model, result


def load_erm_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    """
    Load a seed-specific ERM checkpoint.
    GAS-DRO can use the returned model_state_dict / model as its predictor initialization.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_mlp().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
