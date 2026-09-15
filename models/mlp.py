"""
Common regression MLP for all methods.

Input:
    X = [X1, X2, X3, X4]

Output:
    Y_hat

Architecture:
    4 -> 64 -> 64 -> 1
    ReLU activation

This predictor must be shared by:
- ERM
- W-DRO
- KL-DRO
- SCOT
- GAS-DRO
- Ours

The purpose is to ensure that performance differences come from
the ambiguity-set / robust-training method, not from different
prediction architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Finite check
# ============================================================

def check_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Stop immediately if NaN or Inf appears.
    """

    if not torch.isfinite(tensor).all():

        nan_count = torch.isnan(
            tensor
        ).sum().item()

        inf_count = torch.isinf(
            tensor
        ).sum().item()

        raise RuntimeError(
            f"[Non-Finite Tensor] {name}\n"
            f"shape={tuple(tensor.shape)}\n"
            f"nan_count={nan_count}\n"
            f"inf_count={inf_count}"
        )


# ============================================================
# Common MLP
# ============================================================

class RegressionMLP(nn.Module):
    """
    Shared predictor used by all experimental methods.

    Architecture:

        4
        ↓
        Linear(4, 64)
        ↓
        ReLU
        ↓
        Linear(64, 64)
        ↓
        ReLU
        ↓
        Linear(64, 1)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output = self.net(x)

        return output


    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        path: str | Path,
        extra: Optional[dict] = None,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {

            "model_state_dict":
                self.state_dict(),

            "config": {
                "input_dim":
                    self.input_dim,

                "hidden_dim":
                    self.hidden_dim,

                "output_dim":
                    self.output_dim,
            },

            "extra":
                extra or {},
        }

        torch.save(
            checkpoint,
            path,
        )


    # ========================================================
    # Load
    # ========================================================

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "RegressionMLP":

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        config = checkpoint[
            "config"
        ]

        model = cls(
            input_dim=
                config["input_dim"],

            hidden_dim=
                config["hidden_dim"],

            output_dim=
                config["output_dim"],
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.to(device)

        return model


# ============================================================
# Dataset helper
# ============================================================

def split_joint_xy(
    joint: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Convert joint GAS-DRO samples

        [X1, X2, X3, X4, Y]

    into

        X = [X1, X2, X3, X4]
        y = [Y]
    """

    if joint.ndim != 2:

        raise ValueError(
            "joint must have shape [N, 5]"
        )

    if joint.shape[1] != 5:

        raise ValueError(
            f"Expected 5 columns, "
            f"got {joint.shape[1]}"
        )

    check_finite(
        joint,
        "joint_data",
    )

    x = joint[:, :4]

    y = joint[:, 4:5]

    return x, y


# ============================================================
# Training
# ============================================================

def train_regression_mlp(
    model: RegressionMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str | torch.device,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 5.0,
    verbose_every: int = 10,
) -> List[float]:

    """
    Standard ERM-style MSE training.

    GAS-DRO will later reuse this predictor-training logic
    on adversarially generated samples.
    """

    device = torch.device(
        device
    )

    x = x.float()

    y = y.float()


    if x.ndim != 2:

        raise ValueError(
            "x must have shape [N, 4]"
        )


    if x.shape[1] != 4:

        raise ValueError(
            f"Expected X dimension 4, "
            f"got {x.shape[1]}"
        )


    if y.ndim == 1:

        y = y.unsqueeze(1)


    if y.shape[1] != 1:

        raise ValueError(
            f"Expected Y dimension 1, "
            f"got {y.shape[1]}"
        )


    check_finite(
        x,
        "training_x",
    )

    check_finite(
        y,
        "training_y",
    )


    dataset = TensorDataset(
        x,
        y,
    )


    loader = DataLoader(

        dataset,

        batch_size=
            batch_size,

        shuffle=
            True,

        drop_last=
            False,
    )


    model = model.to(
        device
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=
            lr,

        weight_decay=
            weight_decay,
    )


    criterion = nn.MSELoss()


    history: List[float] = []


    print("\n==========================================")
    print("Regression MLP Training")
    print("==========================================")

    print(
        f"Samples    : {len(dataset)}"
    )

    print(
        f"Input dim  : {x.shape[1]}"
    )

    print(
        f"Epochs     : {epochs}"
    )

    print(
        f"Batch size : {batch_size}"
    )

    print(
        f"Device     : {device}"
    )

    print("==========================================\n")


    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        epoch_loss = 0.0

        sample_count = 0


        for (
            batch_x,
            batch_y
        ) in loader:


            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )


            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            prediction = model(
                batch_x
            )


            check_finite(
                prediction,
                "mlp_prediction",
            )


            loss = criterion(
                prediction,
                batch_y,
            )


            check_finite(
                loss,
                "mlp_loss",
            )


            loss.backward()


            if grad_clip is not None:

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(

                        model.parameters(),

                        max_norm=
                            grad_clip,
                    )
                )


                if not torch.isfinite(
                    grad_norm
                ):

                    raise RuntimeError(
                        "Non-finite MLP "
                        "gradient norm."
                    )


            optimizer.step()


            actual_batch = (
                batch_x.shape[0]
            )


            epoch_loss += (
                loss.item()
                * actual_batch
            )


            sample_count += (
                actual_batch
            )


        avg_loss = (
            epoch_loss
            /
            max(
                sample_count,
                1
            )
        )


        history.append(
            avg_loss
        )


        if (
            epoch == 1
            or
            epoch % verbose_every == 0
            or
            epoch == epochs
        ):

            print(

                f"Epoch "
                f"{epoch:04d}/{epochs} | "
                f"MSE: {avg_loss:.6f}"
            )


    print(
        "\nRegression MLP Training Complete."
    )

    return history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_regression_mlp(
    model: RegressionMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str | torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:

    """
    Return regression MSE.

    Later the common evaluation pipeline will use this for:

        ID MSE
        Mild OOD MSE
        Moderate OOD MSE
        Strong OOD MSE
        Worst OOD MSE
    """

    device = torch.device(
        device
    )


    x = x.float()

    y = y.float()


    if y.ndim == 1:

        y = y.unsqueeze(1)


    dataset = TensorDataset(
        x,
        y,
    )


    loader = DataLoader(

        dataset,

        batch_size=
            batch_size,

        shuffle=
            False,

        drop_last=
            False,
    )


    model.eval()

    model.to(
        device
    )


    squared_error_sum = 0.0

    sample_count = 0


    for (
        batch_x,
        batch_y
    ) in loader:


        batch_x = batch_x.to(
            device
        )


        batch_y = batch_y.to(
            device
        )


        prediction = model(
            batch_x
        )


        check_finite(
            prediction,
            "evaluation_prediction",
        )


        squared_error = (
            prediction
            - batch_y
        ) ** 2


        squared_error_sum += (
            squared_error
            .sum()
            .item()
        )


        sample_count += (
            batch_y.numel()
        )


    mse = (
        squared_error_sum
        /
        max(
            sample_count,
            1
        )
    )


    return {
        "mse": mse
    }


# ============================================================
# Per-sample loss
#
# GAS-DRO needs this later as the reward / adversarial signal.
# ============================================================

def per_sample_mse(
    model: RegressionMLP,
    joint: torch.Tensor,
) -> torch.Tensor:

    """
    Compute one MSE value per joint sample.

    joint:
        [X1, X2, X3, X4, Y]

    returns:
        [batch]

    This will later be used by GAS-DRO as the predictor-loss
    signal for adversarial generator optimization.
    """

    x, y = split_joint_xy(
        joint
    )


    prediction = model(
        x
    )


    check_finite(
        prediction,
        "gas_dro_prediction",
    )


    loss = (
        prediction
        - y
    ) ** 2


    loss = loss.squeeze(1)


    check_finite(
        loss,
        "gas_dro_per_sample_mse",
    )


    return loss