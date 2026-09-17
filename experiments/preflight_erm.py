"""
ERM Preflight Test
==================

Purpose
-------
Check that:

1. Professor-provided common MLP loads correctly.
2. ERM receives X [N, 4], y [N, 1].
3. Training loss is finite.
4. Training actually reduces MSE.
5. predict() works.
6. evaluate() works.

IMPORTANT
---------
This is NOT the final ERM experiment.

The hyperparameters below are temporary smoke-test values only.
Final optimizer / learning rate / batch size / epochs must use
the common benchmark protocol when it is provided.
"""

from __future__ import annotations

import time

import torch

from models.mlp import build_mlp

from methods.erm.erm import (
    ERM,
    ERMConfig,
    set_seed,
)


# ============================================================
# Temporary smoke configuration
#
# NOT FINAL EXPERIMENT HYPERPARAMETERS
# ============================================================

SEED = 42

SMOKE_CONFIG = ERMConfig(
    batch_size=64,
    learning_rate=1e-3,
    epochs=30,
    optimizer="adam",
    weight_decay=0.0,
    seed=SEED,
    verbose=True,
)


# ============================================================
# Temporary regression data
#
# X: [N, 4]
# y: [N, 1]
#
# This is only for implementation verification.
# ============================================================

def make_smoke_data(
    n: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    X = torch.randn(
        n,
        4,
    )

    noise = (
        0.1
        *
        torch.randn(
            n,
            1,
        )
    )

    y = (
        0.8 * X[:, 0:1]
        - 0.5 * X[:, 1:2]
        + 0.3 * X[:, 2:3]
        + 0.2 * X[:, 3:4]
        + noise
    )

    return X, y


# ============================================================
# Main
# ============================================================

def main():

    start_time = time.time()

    set_seed(
        SEED
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=========================================="
    )
    print(
        "ERM PREFLIGHT"
    )
    print(
        "=========================================="
    )

    print(
        "Device:",
        device,
    )

    if device == "cpu":

        print(
            "NOTE: CPU is fine for this ERM preflight."
        )

    # --------------------------------------------------------
    # Temporary train / test
    # --------------------------------------------------------

    X_train, y_train = (
        make_smoke_data(
            n=2000
        )
    )

    X_test, y_test = (
        make_smoke_data(
            n=1000
        )
    )

    print(
        "\n[1] DATA CHECK"
    )

    print(
        "X_train:",
        X_train.shape,
    )

    print(
        "y_train:",
        y_train.shape,
    )

    print(
        "Train finite:",
        torch.isfinite(
            X_train
        ).all().item()
        and
        torch.isfinite(
            y_train
        ).all().item(),
    )

    assert (
        X_train.shape
        ==
        (2000, 4)
    )

    assert (
        y_train.shape
        ==
        (2000, 1)
    )

    # --------------------------------------------------------
    # Professor-provided common MLP
    # --------------------------------------------------------

    print(
        "\n[2] COMMON MLP CHECK"
    )

    predictor = (
        build_mlp()
    )

    print(
        predictor
    )

    parameter_count = sum(
        p.numel()
        for p in predictor.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    assert (
        parameter_count
        ==
        4545
    )

    # --------------------------------------------------------
    # Build ERM
    # --------------------------------------------------------

    print(
        "\n[3] BUILD ERM"
    )

    erm = ERM(
        predictor=
            predictor,

        device=
            device,

        config=
            SMOKE_CONFIG,
    )

    # --------------------------------------------------------
    # MSE before training
    # --------------------------------------------------------

    mse_before = (
        erm.evaluate(
            X_test,
            y_test,
        )
    )

    print(
        "Test MSE before training:",
        mse_before,
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    print(
        "\n[4] ERM TRAINING"
    )

    result = erm.fit(
        X_train,
        y_train,
    )

    train_history = (
        result[
            "history"
        ][
            "train_mse"
        ]
    )

    assert (
        len(
            train_history
        )
        ==
        SMOKE_CONFIG.epochs
    )

    assert all(
        torch.isfinite(
            torch.tensor(
                value
            )
        ).item()
        for value in train_history
    )

    print(
        "\nTraining history finite: True"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\n[5] EVALUATION"
    )

    mse_after = (
        erm.evaluate(
            X_test,
            y_test,
        )
    )

    print(
        "Test MSE before:",
        mse_before,
    )

    print(
        "Test MSE after :",
        mse_after,
    )

    improvement = (
        mse_before
        -
        mse_after
    )

    print(
        "MSE reduction  :",
        improvement,
    )

    if not (
        mse_after
        <
        mse_before
    ):

        raise RuntimeError(
            "ERM training did not reduce test MSE."
        )

    print(
        "[PASS] ERM training reduced MSE."
    )

    # --------------------------------------------------------
    # Prediction interface
    # --------------------------------------------------------

    print(
        "\n[6] PREDICT CHECK"
    )

    prediction = (
        erm.predict(
            X_test[:8]
        )
    )

    print(
        "Prediction shape:",
        prediction.shape,
    )

    print(
        "Prediction finite:",
        torch.isfinite(
            prediction
        ).all().item(),
    )

    assert (
        prediction.shape
        ==
        (8, 1)
    )

    assert (
        torch.isfinite(
            prediction
        ).all()
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    runtime = (
        time.time()
        -
        start_time
    )

    print(
        "\n=========================================="
    )

    print(
        "ERM PREFLIGHT COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        "Professor MLP loaded      : YES"
    )

    print(
        "Input/output shapes valid : YES"
    )

    print(
        "ERM training finite       : YES"
    )

    print(
        "MSE reduced               : YES"
    )

    print(
        "Prediction finite         : YES"
    )

    print(
        "Runtime:",
        round(
            runtime,
            2,
        ),
        "sec",
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Smoke hyperparameters are NOT final."
    )


if __name__ == "__main__":
    main()