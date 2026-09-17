"""
Empirical Risk Minimization (ERM)
=================================

Common regression baseline.

Predictor:
    Professor-provided common MLP
    4 -> 64 -> 64 -> 1

Objective:
    Minimize empirical mean squared error on nominal training data.

Important:
- ERM does NOT use SCM information.
- ERM does NOT use OOD environments during training.
- All training hyperparameters must be supplied externally so that
  the final experiment can use the common benchmark protocol.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from models.mlp import MLP


# ============================================================
# Configuration
# ============================================================

@dataclass
class ERMConfig:

    batch_size: int
    learning_rate: float
    epochs: int

    optimizer: str = "adam"

    weight_decay: float = 0.0

    seed: int = 42

    verbose: bool = True


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Input validation
# ============================================================

def validate_xy(
    X: torch.Tensor,
    y: torch.Tensor,
) -> None:

    if X.ndim != 2 or X.shape[1] != 4:

        raise ValueError(
            f"Expected X shape [N, 4], "
            f"got {tuple(X.shape)}"
        )

    if y.ndim == 1:

        y = y.unsqueeze(1)

    if y.ndim != 2 or y.shape[1] != 1:

        raise ValueError(
            f"Expected y shape [N, 1], "
            f"got {tuple(y.shape)}"
        )

    if X.shape[0] != y.shape[0]:

        raise ValueError(
            "X and y must contain the same "
            "number of samples."
        )

    if not torch.isfinite(X).all():

        raise ValueError(
            "X contains NaN or Inf."
        )

    if not torch.isfinite(y).all():

        raise ValueError(
            "y contains NaN or Inf."
        )


# ============================================================
# ERM
# ============================================================

class ERM:

    def __init__(
        self,
        predictor: MLP,
        device: str | torch.device,
        config: ERMConfig,
    ):

        self.device = torch.device(
            device
        )

        self.config = config

        self.predictor = predictor.to(
            self.device
        )

        self.criterion = nn.MSELoss()

        self.history: Dict[
            str,
            List[float],
        ] = {
            "train_mse": [],
        }


    # ========================================================
    # Optimizer
    # ========================================================

    def _build_optimizer(
        self,
    ) -> torch.optim.Optimizer:

        name = (
            self.config.optimizer
            .lower()
        )

        if name == "adam":

            return torch.optim.Adam(
                self.predictor.parameters(),
                lr=
                    self.config.learning_rate,
                weight_decay=
                    self.config.weight_decay,
            )

        if name == "adamw":

            return torch.optim.AdamW(
                self.predictor.parameters(),
                lr=
                    self.config.learning_rate,
                weight_decay=
                    self.config.weight_decay,
            )

        if name == "sgd":

            return torch.optim.SGD(
                self.predictor.parameters(),
                lr=
                    self.config.learning_rate,
                weight_decay=
                    self.config.weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer: "
            f"{self.config.optimizer}"
        )


    # ========================================================
    # Training
    # ========================================================

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ):

        set_seed(
            self.config.seed
        )

        X = (
            X.detach()
            .cpu()
            .float()
        )

        y = (
            y.detach()
            .cpu()
            .float()
        )

        if y.ndim == 1:

            y = y.unsqueeze(1)

        validate_xy(
            X,
            y,
        )

        dataset = TensorDataset(
            X,
            y,
        )

        generator = torch.Generator()

        generator.manual_seed(
            self.config.seed
        )

        loader = DataLoader(
            dataset,
            batch_size=
                self.config.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )

        optimizer = (
            self._build_optimizer()
        )

        self.predictor.train()

        if self.config.verbose:

            print(
                "\n=========================================="
            )

            print(
                "ERM TRAINING"
            )

            print(
                "=========================================="
            )

            print(
                f"Samples       : {len(dataset)}"
            )

            print(
                f"Batch size    : "
                f"{self.config.batch_size}"
            )

            print(
                f"Epochs        : "
                f"{self.config.epochs}"
            )

            print(
                f"Optimizer     : "
                f"{self.config.optimizer}"
            )

            print(
                f"Learning rate : "
                f"{self.config.learning_rate}"
            )

            print(
                f"Weight decay  : "
                f"{self.config.weight_decay}"
            )

            print(
                f"Seed          : "
                f"{self.config.seed}"
            )

            print(
                f"Device        : "
                f"{self.device}"
            )

            print(
                "==========================================\n"
            )

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):

            self.predictor.train()

            total_squared_error = 0.0

            total_elements = 0


            for batch_X, batch_y in loader:

                batch_X = batch_X.to(
                    self.device
                )

                batch_y = batch_y.to(
                    self.device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = (
                    self.predictor(
                        batch_X
                    )
                )

                loss = self.criterion(
                    prediction,
                    batch_y,
                )

                if not torch.isfinite(
                    loss
                ):

                    raise RuntimeError(
                        "Non-finite ERM loss "
                        f"at epoch {epoch}."
                    )

                loss.backward()

                optimizer.step()

                squared_error = (
                    (
                        prediction.detach()
                        -
                        batch_y
                    )
                    ** 2
                )

                total_squared_error += (
                    squared_error.sum().item()
                )

                total_elements += (
                    squared_error.numel()
                )


            epoch_mse = (
                total_squared_error
                /
                max(
                    total_elements,
                    1,
                )
            )

            self.history[
                "train_mse"
            ].append(
                epoch_mse
            )


            if (
                self.config.verbose
                and
                (
                    epoch == 1
                    or
                    epoch == self.config.epochs
                    or
                    epoch % 10 == 0
                )
            ):

                print(
                    f"Epoch "
                    f"{epoch:04d}/"
                    f"{self.config.epochs} "
                    f"| MSE="
                    f"{epoch_mse:.6f}"
                )


        if self.config.verbose:

            print(
                "\nERM Training Complete."
            )


        return {
            "predictor":
                self.predictor,

            "history":
                self.history,
        }


    # ========================================================
    # Prediction
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:

        X = X.float()

        if (
            X.ndim != 2
            or
            X.shape[1] != 4
        ):

            raise ValueError(
                f"Expected X shape [N, 4], "
                f"got {tuple(X.shape)}"
            )

        self.predictor.eval()

        prediction = (
            self.predictor(
                X.to(
                    self.device
                )
            )
        )

        return (
            prediction
            .detach()
            .cpu()
        )


    # ========================================================
    # Evaluation
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:

        X = X.float()

        y = y.float()

        if y.ndim == 1:

            y = y.unsqueeze(1)

        validate_xy(
            X,
            y,
        )

        prediction = self.predict(
            X
        )

        mse = (
            (
                prediction
                -
                y.cpu()
            )
            ** 2
        ).mean()

        return float(
            mse.item()
        )


    # ========================================================
    # State
    # ========================================================

    def state_dict(
        self,
    ):

        return copy.deepcopy(
            self.predictor.state_dict()
        )