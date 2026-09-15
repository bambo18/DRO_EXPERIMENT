"""
Pilot OOD experiment: ERM vs GAS-DRO.

IMPORTANT:
This is NOT the final paper experiment.

Purpose:
- Verify whether the current GAS-DRO implementation can improve
  robustness under causally-compatible exogenous distribution shifts.
- Use the currently agreed synthetic SCM.
- Compare the same predictor before and after GAS-DRO training.

Final experiment will later replace this temporary data generator
with the team's common synthetic_scm.py and OOD environment code.
"""

from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from models.mlp import (
    RegressionMLP,
    train_regression_mlp,
    evaluate_regression_mlp,
    split_joint_xy,
)

from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion,
    check_finite,
)

from methods.gas_dro.gas_dro import (
    VectorGasDRO,
    GasDROConfig,
)


# ============================================================
# Settings
# ============================================================

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TRAIN_SAMPLES = 2000
ID_TEST_SAMPLES = 2000

OOD_ENVS_PER_LEVEL = 3
OOD_SAMPLES_PER_ENV = 2000


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SCM
#
# X1 = U1
# X2 = 0.8 X1 + U2
# X3 = tanh(X2) + 0.5 X1 + U3
# X4 = 0.5 X1 - 0.3 X2 + U4
# Y  = X3 + 0.7 X4 + 0.5 sin(X2) + Uy
#
# All mechanisms remain FIXED.
# Only exogenous distributions change under OOD.
# ============================================================

def generate_joint_data(
    n_samples: int,
    seed: int,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:

    generator = torch.Generator()
    generator.manual_seed(seed)

    if mu is None:
        mu = torch.zeros(5)

    if sigma is None:
        sigma = torch.ones(5)

    mu = torch.as_tensor(
        mu,
        dtype=torch.float32,
    )

    sigma = torch.as_tensor(
        sigma,
        dtype=torch.float32,
    )

    if mu.shape != (5,):
        raise ValueError(
            f"mu must have shape (5,), got {mu.shape}"
        )

    if sigma.shape != (5,):
        raise ValueError(
            f"sigma must have shape (5,), got {sigma.shape}"
        )

    if torch.any(sigma <= 0):
        raise ValueError(
            "All sigma values must be positive."
        )

    eps = torch.randn(
        n_samples,
        5,
        generator=generator,
    )

    u = (
        mu.unsqueeze(0)
        +
        sigma.unsqueeze(0)
        * eps
    )

    u1 = u[:, 0]
    u2 = u[:, 1]
    u3 = u[:, 2]
    u4 = u[:, 3]
    uy = u[:, 4]

    x1 = u1

    x2 = (
        0.8 * x1
        + u2
    )

    x3 = (
        torch.tanh(x2)
        + 0.5 * x1
        + u3
    )

    x4 = (
        0.5 * x1
        - 0.3 * x2
        + u4
    )

    y = (
        x3
        + 0.7 * x4
        + 0.5 * torch.sin(x2)
        + uy
    )

    joint = torch.stack(
        [
            x1,
            x2,
            x3,
            x4,
            y,
        ],
        dim=1,
    ).float()

    check_finite(
        joint,
        "generated_joint",
    )

    return joint


# ============================================================
# OOD environment generation
# ============================================================

SHIFT_CONFIG = {

    "Mild": {
        "mu_low": -0.5,
        "mu_high": 0.5,
        "sigma_low": 0.8,
        "sigma_high": 1.2,
    },

    "Moderate": {
        "mu_low": -1.0,
        "mu_high": 1.0,
        "sigma_low": 0.6,
        "sigma_high": 1.4,
    },

    "Strong": {
        "mu_low": -1.5,
        "mu_high": 1.5,
        "sigma_low": 0.5,
        "sigma_high": 1.5,
    },
}


def sample_environment_parameters(
    level: str,
    seed: int,
):

    config = SHIFT_CONFIG[level]

    generator = torch.Generator()
    generator.manual_seed(seed)

    mu = (
        config["mu_low"]
        +
        (
            config["mu_high"]
            -
            config["mu_low"]
        )
        *
        torch.rand(
            5,
            generator=generator,
        )
    )

    sigma = (
        config["sigma_low"]
        +
        (
            config["sigma_high"]
            -
            config["sigma_low"]
        )
        *
        torch.rand(
            5,
            generator=generator,
        )
    )

    return mu, sigma


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_joint(
    model: RegressionMLP,
    joint: torch.Tensor,
) -> float:

    x, y = split_joint_xy(
        joint
    )

    result = evaluate_regression_mlp(

        model=model,

        x=x,

        y=y,

        device=DEVICE,
    )

    return result["mse"]


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.time()

    print("\n==========================================")
    print("PILOT OOD EXPERIMENT")
    print("ERM vs GAS-DRO")
    print("==========================================")

    print(
        f"Device : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU    :",
            torch.cuda.get_device_name(0),
        )

    print(
        f"Seed   : {SEED}"
    )

    print("==========================================")


    # ========================================================
    # 1. Nominal training data
    # ========================================================

    print("\n[1] Generating nominal train data...")

    train_joint = generate_joint_data(

        n_samples=TRAIN_SAMPLES,

        seed=SEED,
    )

    train_x, train_y = split_joint_xy(
        train_joint
    )


    # ========================================================
    # 2. Common nominal predictor
    #
    # Both ERM and GAS-DRO start from EXACTLY the same
    # pretrained predictor.
    # ========================================================

    print("\n[2] Training common nominal predictor...")


    base_predictor = RegressionMLP(

        input_dim=4,

        hidden_dim=64,

        output_dim=1,
    )


    train_regression_mlp(

        model=base_predictor,

        x=train_x,

        y=train_y,

        device=DEVICE,

        epochs=20,

        batch_size=128,

        lr=1e-3,

        grad_clip=5.0,

        verbose_every=5,
    )


    # --------------------------------------------------------
    # ERM freezes here.
    # GAS-DRO continues from the SAME predictor.
    # --------------------------------------------------------

    erm_predictor = copy.deepcopy(
        base_predictor
    ).to(DEVICE)

    gas_predictor = copy.deepcopy(
        base_predictor
    ).to(DEVICE)


    # ========================================================
    # 3. Nominal diffusion theta_0
    # ========================================================

    print("\n[3] Training nominal diffusion...")


    nominal_diffusion = VectorDiffusion(

        data_dim=5,

        timesteps=30,

        beta_start=1e-4,

        beta_end=2e-2,

        hidden_dim=128,

        time_dim=64,
    )


    _, standardizer = train_vector_diffusion(

        model=nominal_diffusion,

        data=train_joint,

        device=DEVICE,

        epochs=10,

        batch_size=128,

        lr=1e-3,

        grad_clip=1.0,

        verbose_every=2,
    )


    # ========================================================
    # 4. GAS-DRO
    # ========================================================

    print("\n[4] Training GAS-DRO...")


    gas_config = GasDROConfig(

        outer_epochs=2,

        generator_inner_epochs=2,

        generator_batches_per_epoch=4,

        predictor_inner_epochs=2,

        batch_size=128,

        generator_lr=1e-4,

        predictor_lr=1e-4,

        ppo_clip=0.4,

        mu=1.0,

        eta=0.1,

        budget=0.8,

        adjust_timesteps=5,

        step_size=2,

        discount_factor=0.05,

        grad_clip=1.0,

        verbose=False,
    )


    gas_dro = VectorGasDRO(

        nominal_diffusion=
            nominal_diffusion,

        standardizer=
            standardizer,

        predictor=
            gas_predictor,

        device=
            DEVICE,

        config=
            gas_config,
    )


    gas_result = gas_dro.fit(
        real_joint=train_joint
    )


    robust_predictor = gas_result[
        "predictor"
    ]


    # ========================================================
    # 5. ID evaluation
    # ========================================================

    print("\n[5] Evaluating ID...")


    id_joint = generate_joint_data(

        n_samples=ID_TEST_SAMPLES,

        seed=1000,
    )


    erm_id = evaluate_joint(
        erm_predictor,
        id_joint,
    )


    gas_id = evaluate_joint(
        robust_predictor,
        id_joint,
    )


    # ========================================================
    # 6. OOD evaluation
    # ========================================================

    print("\n[6] Evaluating OOD environments...")


    rows = []

    summary = {}


    for level_index, level in enumerate(
        [
            "Mild",
            "Moderate",
            "Strong",
        ]
    ):

        erm_losses = []
        gas_losses = []


        print(
            f"\n----- {level} -----"
        )


        for env_idx in range(
            OOD_ENVS_PER_LEVEL
        ):

            env_seed = (
                10000
                +
                level_index * 1000
                +
                env_idx
            )


            mu, sigma = (
                sample_environment_parameters(

                    level=level,

                    seed=env_seed,
                )
            )


            data_seed = (
                20000
                +
                level_index * 1000
                +
                env_idx
            )


            ood_joint = generate_joint_data(

                n_samples=
                    OOD_SAMPLES_PER_ENV,

                seed=
                    data_seed,

                mu=
                    mu,

                sigma=
                    sigma,
            )


            erm_mse = evaluate_joint(

                erm_predictor,

                ood_joint,
            )


            gas_mse = evaluate_joint(

                robust_predictor,

                ood_joint,
            )


            erm_losses.append(
                erm_mse
            )

            gas_losses.append(
                gas_mse
            )


            improvement = (

                (
                    erm_mse
                    -
                    gas_mse
                )

                /
                erm_mse

                *
                100.0
            )


            print(

                f"Env {env_idx + 1} | "

                f"ERM={erm_mse:.6f} | "

                f"GAS-DRO={gas_mse:.6f} | "

                f"Improvement="
                f"{improvement:+.2f}%"
            )


            rows.append({

                "level":
                    level,

                "environment":
                    env_idx + 1,

                "erm_mse":
                    erm_mse,

                "gas_dro_mse":
                    gas_mse,

                "improvement_percent":
                    improvement,

                "mu":
                    mu.tolist(),

                "sigma":
                    sigma.tolist(),
            })


        erm_avg = (
            sum(erm_losses)
            /
            len(erm_losses)
        )


        gas_avg = (
            sum(gas_losses)
            /
            len(gas_losses)
        )


        summary[level] = {

            "erm_avg":
                erm_avg,

            "gas_avg":
                gas_avg,

            "erm_worst":
                max(erm_losses),

            "gas_worst":
                max(gas_losses),
        }


    # ========================================================
    # 7. Overall metrics
    # ========================================================

    all_erm_ood = [
        row["erm_mse"]
        for row in rows
    ]

    all_gas_ood = [
        row["gas_dro_mse"]
        for row in rows
    ]


    erm_avg_ood = (
        sum(all_erm_ood)
        /
        len(all_erm_ood)
    )


    gas_avg_ood = (
        sum(all_gas_ood)
        /
        len(all_gas_ood)
    )


    erm_worst_ood = max(
        all_erm_ood
    )


    gas_worst_ood = max(
        all_gas_ood
    )


    # ========================================================
    # 8. Save CSV
    # ========================================================

    output_path = (

        PROJECT_ROOT
        /
        "results"
        /
        "raw"
        /
        "pilot_ood_gas_dro_seed42.csv"
    )


    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,
    )


    with open(

        output_path,

        "w",

        newline="",

        encoding="utf-8",

    ) as f:

        writer = csv.writer(f)


        writer.writerow([

            "level",

            "environment",

            "erm_mse",

            "gas_dro_mse",

            "improvement_percent",

            "mu",

            "sigma",
        ])


        for row in rows:

            writer.writerow([

                row["level"],

                row["environment"],

                row["erm_mse"],

                row["gas_dro_mse"],

                row[
                    "improvement_percent"
                ],

                row["mu"],

                row["sigma"],
            ])


    # ========================================================
    # 9. Final summary
    # ========================================================

    total_time = (
        time.time()
        -
        total_start
    )


    print("\n\n==========================================")
    print("PILOT OOD RESULT")
    print("==========================================")


    print(
        f"ID          | "
        f"ERM={erm_id:.6f} | "
        f"GAS-DRO={gas_id:.6f}"
    )


    for level in [
        "Mild",
        "Moderate",
        "Strong",
    ]:

        s = summary[level]

        print(

            f"{level:8s}    | "

            f"ERM avg="
            f"{s['erm_avg']:.6f} | "

            f"GAS avg="
            f"{s['gas_avg']:.6f}"
        )


    print("------------------------------------------")


    print(

        f"Average OOD | "

        f"ERM="
        f"{erm_avg_ood:.6f} | "

        f"GAS-DRO="
        f"{gas_avg_ood:.6f}"
    )


    print(

        f"Worst OOD   | "

        f"ERM="
        f"{erm_worst_ood:.6f} | "

        f"GAS-DRO="
        f"{gas_worst_ood:.6f}"
    )


    avg_improvement = (

        (
            erm_avg_ood
            -
            gas_avg_ood
        )

        /
        erm_avg_ood

        *
        100.0
    )


    worst_improvement = (

        (
            erm_worst_ood
            -
            gas_worst_ood
        )

        /
        erm_worst_ood

        *
        100.0
    )


    print("------------------------------------------")


    print(

        f"Average OOD improvement : "
        f"{avg_improvement:+.2f}%"
    )


    print(

        f"Worst OOD improvement   : "
        f"{worst_improvement:+.2f}%"
    )


    print(

        f"Final GAS-DRO mu         : "
        f"{gas_result['mu']:.6f}"
    )


    print(

        f"Runtime                  : "
        f"{total_time:.2f} sec"
    )


    print(
        "=========================================="
    )


    print(
        "\nCSV saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()