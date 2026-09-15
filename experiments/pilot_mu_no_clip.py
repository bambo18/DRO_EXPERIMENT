"""
Pilot sensitivity experiment for fixed GAS-DRO mu.

NOT a final paper experiment.

Purpose
-------
Diagnose how strongly the JSM / diffusion constraint affects
the adversarial generator.

We fix mu during training:

    objective = -PPO + mu * JSM

and compare:

    mu = 0.0
    mu = 0.1
    mu = 1.0
    mu = 5.0
    mu = 10.0

eta = 0.0

Therefore mu does NOT change during training.

All configurations use:
- identical nominal train data
- identical predictor initialization
- identical nominal diffusion
- identical OOD environments
"""

from __future__ import annotations

import copy
import csv
import math
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
)

from methods.gas_dro.gas_dro import (
    VectorGasDRO,
    GasDROConfig,
)

from experiments.pilot_ood_gas_dro import (
    generate_joint_data,
    sample_environment_parameters,
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

MU_VALUES = [
    0.0,
    1.0,
    10.0,
    40.0,
    50.0,
]

TRAIN_SAMPLES = 2000

ID_TEST_SAMPLES = 2000

OOD_ENVS_PER_LEVEL = 3

OOD_SAMPLES_PER_ENV = 2000


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Evaluation
# ============================================================

def evaluate_joint(
    model,
    joint,
):

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
# Fixed evaluation environments
# ============================================================

def build_evaluation_data():

    id_joint = generate_joint_data(

        n_samples=
            ID_TEST_SAMPLES,

        seed=
            1000,
    )


    ood_sets = {}


    for level_index, level in enumerate(
        [
            "Mild",
            "Moderate",
            "Strong",
        ]
    ):

        ood_sets[level] = []


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

                    level=
                        level,

                    seed=
                        env_seed,
                )
            )


            data_seed = (
                20000
                +
                level_index * 1000
                +
                env_idx
            )


            joint = generate_joint_data(

                n_samples=
                    OOD_SAMPLES_PER_ENV,

                seed=
                    data_seed,

                mu=
                    mu,

                sigma=
                    sigma,
            )


            ood_sets[
                level
            ].append(
                joint
            )


    return (
        id_joint,
        ood_sets,
    )


# ============================================================
# Parameter distance
#
# Measures how far adversarial theta moved from theta_0.
# ============================================================

@torch.no_grad()
def parameter_l2_distance(
    model_a,
    model_b,
):

    total = 0.0


    for pa, pb in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):

        diff = (
            pa.detach()
            -
            pb.detach()
        )

        total += (
            diff.pow(2)
            .sum()
            .item()
        )


    return math.sqrt(
        total
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.time()


    print("\n==========================================")
    print("GAS-DRO FIXED MU SENSITIVITY PILOT")
    print("==========================================")

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"mu     : {MU_VALUES}"
    )

    print(
        "eta    : 0.0 "
        "(mu remains fixed)"
    )

    print("==========================================")


    # ========================================================
    # 1. Common nominal data
    # ========================================================

    print(
        "\n[1] Generating nominal data..."
    )


    train_joint = generate_joint_data(

        n_samples=
            TRAIN_SAMPLES,

        seed=
            SEED,
    )


    train_x, train_y = (
        split_joint_xy(
            train_joint
        )
    )


    # ========================================================
    # 2. Common base predictor
    # ========================================================

    print(
        "\n[2] Training ONE base predictor..."
    )


    base_predictor = RegressionMLP(

        input_dim=4,

        hidden_dim=64,

        output_dim=1,
    )


    train_regression_mlp(

        model=
            base_predictor,

        x=
            train_x,

        y=
            train_y,

        device=
            DEVICE,

        epochs=
            20,

        batch_size=
            128,

        lr=
            1e-3,

        grad_clip=
            5.0,

        verbose_every=
            5,
    )


    # ========================================================
    # 3. Common nominal diffusion
    # ========================================================

    print(
        "\n[3] Training ONE nominal diffusion..."
    )


    nominal_diffusion = VectorDiffusion(

        data_dim=5,

        timesteps=30,

        beta_start=1e-4,

        beta_end=2e-2,

        hidden_dim=128,

        time_dim=64,
    )


    _, standardizer = (
        train_vector_diffusion(

            model=
                nominal_diffusion,

            data=
                train_joint,

            device=
                DEVICE,

            epochs=
                10,

            batch_size=
                128,

            lr=
                1e-3,

            grad_clip=
                1.0,

            verbose_every=
                2,
        )
    )


    # ========================================================
    # 4. Fixed evaluation sets
    # ========================================================

    print(
        "\n[4] Building fixed ID / OOD sets..."
    )


    (
        id_joint,
        ood_sets,
    ) = build_evaluation_data()


    # ========================================================
    # 5. ERM reference
    # ========================================================

    erm_model = copy.deepcopy(
        base_predictor
    ).to(DEVICE)


    erm_id = evaluate_joint(

        erm_model,

        id_joint,
    )


    erm_level_results = {}

    erm_all_ood = []


    for level in [
        "Mild",
        "Moderate",
        "Strong",
    ]:

        losses = []


        for joint in ood_sets[level]:

            mse = evaluate_joint(

                erm_model,

                joint,
            )

            losses.append(
                mse
            )

            erm_all_ood.append(
                mse
            )


        erm_level_results[
            level
        ] = (
            sum(losses)
            /
            len(losses)
        )


    erm_avg_ood = (
        sum(erm_all_ood)
        /
        len(erm_all_ood)
    )


    erm_worst_ood = max(
        erm_all_ood
    )


    print("\n==========================================")
    print("ERM REFERENCE")
    print("==========================================")

    print(
        f"ID          : {erm_id:.6f}"
    )

    print(
        f"Mild        : "
        f"{erm_level_results['Mild']:.6f}"
    )

    print(
        f"Moderate    : "
        f"{erm_level_results['Moderate']:.6f}"
    )

    print(
        f"Strong      : "
        f"{erm_level_results['Strong']:.6f}"
    )

    print(
        f"Average OOD : {erm_avg_ood:.6f}"
    )

    print(
        f"Worst OOD   : {erm_worst_ood:.6f}"
    )


    # ========================================================
    # 6. Fixed-mu sweep
    # ========================================================

    results = []


    for fixed_mu in MU_VALUES:

        print("\n\n##########################################")

        print(
            f"# FIXED MU = {fixed_mu}"
        )

        print("##########################################")


        # ----------------------------------------------------
        # Reset RNG so each condition starts comparably.
        # ----------------------------------------------------

        torch.manual_seed(
            SEED
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                SEED
            )


        predictor = copy.deepcopy(
            base_predictor
        ).to(DEVICE)


        diffusion = copy.deepcopy(
            nominal_diffusion
        ).to(DEVICE)


        config = GasDROConfig(

            outer_epochs=2,

            generator_inner_epochs=2,

            generator_batches_per_epoch=4,

            predictor_inner_epochs=2,

            batch_size=128,

            generator_lr=1e-4,

            predictor_lr=1e-4,

            ppo_clip=0.4,

            # ----------------------------------------
            # Main variable in this experiment
            # ----------------------------------------
            mu=fixed_mu,

            # ----------------------------------------
            # IMPORTANT:
            # Keep mu fixed.
            # ----------------------------------------
            eta=0.0,

            # Irrelevant when eta=0,
            # but retained for API compatibility.
            budget=0.8,

            adjust_timesteps=5,

            step_size=2,

            discount_factor=0.05,

            grad_clip=None,

            verbose=False,
        )


        gas_dro = VectorGasDRO(

            nominal_diffusion=
                diffusion,

            standardizer=
                standardizer,

            predictor=
                predictor,

            device=
                DEVICE,

            config=
                config,
        )


        gas_result = gas_dro.fit(

            real_joint=
                train_joint
        )


        model = gas_result[
            "predictor"
        ]


        adversarial_diffusion = (
            gas_result[
                "adversarial_diffusion"
            ]
        )


        # ----------------------------------------------------
        # How far did theta move from theta_0?
        # ----------------------------------------------------

        theta_distance = (
            parameter_l2_distance(

                nominal_diffusion,

                adversarial_diffusion,
            )
        )


        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        id_mse = evaluate_joint(

            model,

            id_joint,
        )


        # ----------------------------------------------------
        # OOD
        # ----------------------------------------------------

        level_results = {}

        all_ood = []


        for level in [
            "Mild",
            "Moderate",
            "Strong",
        ]:

            losses = []


            for joint in ood_sets[level]:

                mse = evaluate_joint(

                    model,

                    joint,
                )

                losses.append(
                    mse
                )

                all_ood.append(
                    mse
                )


            level_results[
                level
            ] = (
                sum(losses)
                /
                len(losses)
            )


        avg_ood = (
            sum(all_ood)
            /
            len(all_ood)
        )


        worst_ood = max(
            all_ood
        )


        avg_change = (

            (
                erm_avg_ood
                -
                avg_ood
            )

            /
            erm_avg_ood

            *
            100.0
        )


        worst_change = (

            (
                erm_worst_ood
                -
                worst_ood
            )

            /
            erm_worst_ood

            *
            100.0
        )


        history = gas_result[
            "history"
        ]


        last_s_theta_mse = (

            history[
                "generated_mean_mse"
            ][-1]

            if len(
                history[
                    "generated_mean_mse"
                ]
            ) > 0

            else float("nan")
        )


        last_ratio_mean = (

            history[
                "ratio_mean"
            ][-1]

            if len(
                history[
                    "ratio_mean"
                ]
            ) > 0

            else float("nan")
        )


        results.append({

            "mu":
                fixed_mu,

            "id":
                id_mse,

            "mild":
                level_results["Mild"],

            "moderate":
                level_results["Moderate"],

            "strong":
                level_results["Strong"],

            "avg_ood":
                avg_ood,

            "worst_ood":
                worst_ood,

            "avg_change":
                avg_change,

            "worst_change":
                worst_change,

            "theta_distance":
                theta_distance,

            "last_s_theta_mse":
                last_s_theta_mse,

            "last_ratio_mean":
                last_ratio_mean,

            "final_mu":
                gas_result["mu"],
        })


        print("\nRESULT")

        print(
            f"ID              : "
            f"{id_mse:.6f}"
        )

        print(
            f"Mild            : "
            f"{level_results['Mild']:.6f}"
        )

        print(
            f"Moderate        : "
            f"{level_results['Moderate']:.6f}"
        )

        print(
            f"Strong          : "
            f"{level_results['Strong']:.6f}"
        )

        print(
            f"Average OOD     : "
            f"{avg_ood:.6f}"
        )

        print(
            f"Worst OOD       : "
            f"{worst_ood:.6f}"
        )

        print(
            f"Theta distance   : "
            f"{theta_distance:.6f}"
        )

        print(
            f"Last S_theta MSE: "
            f"{last_s_theta_mse:.6f}"
        )

        print(
            f"Last ratio mean : "
            f"{last_ratio_mean:.6f}"
        )

        print(
            f"Final mu        : "
            f"{gas_result['mu']:.6f}"
        )


    # ========================================================
    # 7. Summary
    # ========================================================

    print("\n\n==========================================")
    print("FIXED MU SWEEP SUMMARY")
    print("==========================================")

    print(
        "mu | ID | Mild | Moderate | Strong | "
        "Avg OOD | Worst OOD | Avg Δ | Worst Δ | "
        "ThetaDist | S_theta MSE"
    )

    print(
        "-" * 125
    )


    for r in results:

        print(

            f"{r['mu']:>4.1f} | "

            f"{r['id']:.4f} | "

            f"{r['mild']:.4f} | "

            f"{r['moderate']:.4f} | "

            f"{r['strong']:.4f} | "

            f"{r['avg_ood']:.4f} | "

            f"{r['worst_ood']:.4f} | "

            f"{r['avg_change']:+.2f}% | "

            f"{r['worst_change']:+.2f}% | "

            f"{r['theta_distance']:.5f} | "

            f"{r['last_s_theta_mse']:.4f}"
        )


    # ========================================================
    # Best values
    # ========================================================

    best_avg = min(

        results,

        key=lambda r:
            r["avg_ood"],
    )


    best_worst = min(

        results,

        key=lambda r:
            r["worst_ood"],
    )


    print("\n------------------------------------------")

    print(
        f"Best Average OOD mu : "
        f"{best_avg['mu']} "
        f"(MSE={best_avg['avg_ood']:.6f})"
    )

    print(
        f"Best Worst OOD mu   : "
        f"{best_worst['mu']} "
        f"(MSE={best_worst['worst_ood']:.6f})"
    )


    # ========================================================
    # 8. Save
    # ========================================================

    output_path = (

        PROJECT_ROOT
        /
        "results"
        /
        "raw"
        /
        "pilot_mu_sweep_seed42.csv"
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

        writer = csv.DictWriter(

            f,

            fieldnames=[
                "mu",
                "id",
                "mild",
                "moderate",
                "strong",
                "avg_ood",
                "worst_ood",
                "avg_change",
                "worst_change",
                "theta_distance",
                "last_s_theta_mse",
                "last_ratio_mean",
                "final_mu",
            ],
        )


        writer.writeheader()

        writer.writerows(
            results
        )


    runtime = (
        time.time()
        -
        total_start
    )


    print(
        f"\nRuntime: {runtime:.2f} sec"
    )

    print(
        "\nCSV saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()