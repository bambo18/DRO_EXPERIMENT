"""
End-to-end smoke test for the Vector GAS-DRO baseline.

Pipeline:

Synthetic SCM
    ↓
Nominal predictor pre-training
    ↓
Nominal 5D diffusion pre-training
    ↓
VectorGasDRO
    ↓
Adversarial generator update
    ↓
S_theta generation
    ↓
Predictor update
    ↓
Final ID / adversarial loss check

This is still a SMOKE TEST.

It is NOT the final paper experiment and does not yet include:
- Mild / Moderate / Strong OOD environments
- hyperparameter validation
- 5 random seeds
- final result table
"""

from __future__ import annotations

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

from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion,
    check_finite,
)

from methods.gas_dro.gas_dro import (
    VectorGasDRO,
    GasDROConfig,
)

from models.mlp import (
    RegressionMLP,
    train_regression_mlp,
    evaluate_regression_mlp,
    split_joint_xy,
    per_sample_mse,
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

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Temporary SCM
#
# Later this will be replaced by data/synthetic_scm.py.
# ============================================================

def generate_scm_data(
    n_samples: int,
    seed: int,
) -> torch.Tensor:

    generator = torch.Generator()
    generator.manual_seed(seed)

    u1 = torch.randn(
        n_samples,
        generator=generator,
    )

    u2 = torch.randn(
        n_samples,
        generator=generator,
    )

    u3 = torch.randn(
        n_samples,
        generator=generator,
    )

    u4 = torch.randn(
        n_samples,
        generator=generator,
    )

    uy = torch.randn(
        n_samples,
        generator=generator,
    )


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
        "scm_joint",
    )

    return joint


# ============================================================
# Simple statistics
# ============================================================

def print_joint_stats(
    name: str,
    joint: torch.Tensor,
):

    x = joint.detach().cpu()

    print(
        f"\n========== {name} =========="
    )

    print(
        "Shape:",
        tuple(x.shape),
    )

    print(
        "Mean:",
        [
            round(v, 4)
            for v in x.mean(dim=0).tolist()
        ],
    )

    print(
        "Std :",
        [
            round(v, 4)
            for v in x.std(
                dim=0,
                unbiased=False,
            ).tolist()
        ],
    )

    print(
        "Finite:",
        bool(torch.isfinite(x).all()),
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.time()

    print("\n==========================================")
    print("VECTOR GAS-DRO END-TO-END SMOKE TEST")
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
            "VRAM   : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"Seed   : {SEED}"
    )

    print("==========================================")


    # ========================================================
    # STEP 0 — Data
    # ========================================================

    train_joint = generate_scm_data(
        n_samples=2000,
        seed=SEED,
    )

    id_test_joint = generate_scm_data(
        n_samples=1000,
        seed=100,
    )


    print_joint_stats(
        "TRAIN DATA",
        train_joint,
    )


    train_x, train_y = split_joint_xy(
        train_joint
    )

    id_x, id_y = split_joint_xy(
        id_test_joint
    )


    # ========================================================
    # STEP 1 — Nominal predictor
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 1 — NOMINAL PREDICTOR")
    print("##########################################")


    predictor = RegressionMLP(
        input_dim=4,
        hidden_dim=64,
        output_dim=1,
    )


    predictor_start = time.time()


    train_regression_mlp(

        model=predictor,

        x=train_x,

        y=train_y,

        device=DEVICE,

        epochs=20,

        batch_size=128,

        lr=1e-3,

        grad_clip=5.0,

        verbose_every=5,
    )


    predictor_time = (
        time.time()
        - predictor_start
    )


    before_id_result = evaluate_regression_mlp(

        model=predictor,

        x=id_x,

        y=id_y,

        device=DEVICE,
    )


    print(
        "\nNominal predictor ID MSE BEFORE GAS-DRO: "
        f"{before_id_result['mse']:.6f}"
    )

    print(
        "[TIMER] Predictor pretrain: "
        f"{predictor_time:.2f} sec"
    )


    # ========================================================
    # STEP 2 — Nominal diffusion theta_0
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 2 — NOMINAL DIFFUSION theta_0")
    print("##########################################")


    nominal_diffusion = VectorDiffusion(

        data_dim=5,

        timesteps=30,

        beta_start=1e-4,

        beta_end=2e-2,

        hidden_dim=128,

        time_dim=64,
    )


    diffusion_start = time.time()


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


    diffusion_time = (
        time.time()
        - diffusion_start
    )


    print(
        "\n[TIMER] Nominal diffusion pretrain: "
        f"{diffusion_time:.2f} sec"
    )


    # ========================================================
    # STEP 3 — Check nominal generator
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 3 — NOMINAL GENERATOR CHECK")
    print("##########################################")


    nominal_diffusion = nominal_diffusion.to(
        DEVICE
    )


    nominal_norm = nominal_diffusion.sample(

        num_samples=1000,

        device=DEVICE,

        return_trajectory=False,
    )


    nominal_generated = (
        standardizer.inverse_transform(
            nominal_norm
        )
    )


    check_finite(
        nominal_generated,
        "nominal_generated",
    )


    predictor = predictor.to(
        DEVICE
    )


    with torch.no_grad():

        nominal_gen_loss = per_sample_mse(

            model=predictor,

            joint=nominal_generated,
        )


    check_finite(
        nominal_gen_loss,
        "nominal_generated_loss",
    )


    print_joint_stats(
        "NOMINAL GENERATED DATA",
        nominal_generated,
    )


    print(
        "\nPredictor loss on nominal generated data:"
    )

    print(
        f"  Mean MSE : "
        f"{nominal_gen_loss.mean().item():.6f}"
    )

    print(
        f"  Max MSE  : "
        f"{nominal_gen_loss.max().item():.6f}"
    )


    nominal_generated_mean_mse = (
        nominal_gen_loss.mean().item()
    )


    # ========================================================
    # STEP 4 — GAS-DRO Configuration
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 4 — BUILD GAS-DRO")
    print("##########################################")


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

        verbose=True,
    )


    print(
        gas_config
    )


    gas_dro = VectorGasDRO(

        nominal_diffusion=
            nominal_diffusion,

        standardizer=
            standardizer,

        predictor=
            predictor,

        device=
            DEVICE,

        config=
            gas_config,
    )


    # ========================================================
    # STEP 5 — GAS-DRO Training
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 5 — GAS-DRO TRAINING")
    print("##########################################")


    gas_start = time.time()


    gas_result = gas_dro.fit(
        real_joint=train_joint
    )


    gas_time = (
        time.time()
        - gas_start
    )


    print(
        "\n[TIMER] GAS-DRO training: "
        f"{gas_time:.2f} sec"
    )


    robust_predictor = gas_result[
        "predictor"
    ]


    adversarial_diffusion = gas_result[
        "adversarial_diffusion"
    ]


    final_mu = gas_result[
        "mu"
    ]


    history = gas_result[
        "history"
    ]


    # ========================================================
    # STEP 6 — ID evaluation after GAS-DRO
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 6 — ID EVALUATION")
    print("##########################################")


    after_id_result = evaluate_regression_mlp(

        model=robust_predictor,

        x=id_x,

        y=id_y,

        device=DEVICE,
    )


    print(
        "ID MSE BEFORE GAS-DRO : "
        f"{before_id_result['mse']:.6f}"
    )

    print(
        "ID MSE AFTER GAS-DRO  : "
        f"{after_id_result['mse']:.6f}"
    )


    # ========================================================
    # STEP 7 — Final adversarial sample check
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 7 — FINAL ADVERSARIAL DISTRIBUTION")
    print("##########################################")


    final_adversarial_joint = (
        gas_dro.generate_adversarial_samples(
            num_samples=1000
        )
    )


    check_finite(
        final_adversarial_joint,
        "final_adversarial_joint",
    )


    print_joint_stats(
        "FINAL S_THETA",
        final_adversarial_joint,
    )


    robust_predictor.eval()


    with torch.no_grad():

        final_adv_loss = per_sample_mse(

            model=robust_predictor,

            joint=
                final_adversarial_joint,
        )


    check_finite(
        final_adv_loss,
        "final_adversarial_loss",
    )


    print(
        "\nRobust predictor on final S_theta:"
    )

    print(
        f"  Mean MSE : "
        f"{final_adv_loss.mean().item():.6f}"
    )

    print(
        f"  Max MSE  : "
        f"{final_adv_loss.max().item():.6f}"
    )


    # ========================================================
    # STEP 8 — Save smoke checkpoints
    # ========================================================

    predictor_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "gas_dro"
        / "gas_dro_predictor_smoke.pt"
    )


    diffusion_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "gas_dro"
        / "gas_dro_adversarial_diffusion_smoke.pt"
    )


    robust_predictor.save(

        predictor_path,

        extra={
            "seed": SEED,
            "type": "gas_dro_smoke",
            "final_mu": final_mu,
        },
    )


    adversarial_diffusion.save(

        diffusion_path,

        standardizer=standardizer,

        extra={
            "seed": SEED,
            "type": "gas_dro_smoke",
            "final_mu": final_mu,
        },
    )


    # ========================================================
    # Summary
    # ========================================================

    total_time = (
        time.time()
        - total_start
    )


    print("\n\n==========================================")
    print("GAS-DRO SMOKE TEST SUMMARY")
    print("==========================================")


    print(
        f"ID MSE before GAS-DRO       : "
        f"{before_id_result['mse']:.6f}"
    )

    print(
        f"ID MSE after GAS-DRO        : "
        f"{after_id_result['mse']:.6f}"
    )


    print(
        f"Nominal generated mean MSE  : "
        f"{nominal_generated_mean_mse:.6f}"
    )


    if len(
        history["generated_mean_mse"]
    ) > 0:

        print(
            f"Last S_theta mean MSE        : "
            f"{history['generated_mean_mse'][-1]:.6f}"
        )


    if len(
        history["generated_max_mse"]
    ) > 0:

        print(
            f"Last S_theta max MSE         : "
            f"{history['generated_max_mse'][-1]:.6f}"
        )


    print(
        f"Final robust S_theta MSE     : "
        f"{final_adv_loss.mean().item():.6f}"
    )


    print(
        f"Final mu                     : "
        f"{final_mu:.6f}"
    )


    print(
        f"Predictor pretrain runtime   : "
        f"{predictor_time:.2f} sec"
    )

    print(
        f"Diffusion pretrain runtime   : "
        f"{diffusion_time:.2f} sec"
    )

    print(
        f"GAS-DRO runtime              : "
        f"{gas_time:.2f} sec"
    )

    print(
        f"TOTAL runtime                : "
        f"{total_time:.2f} sec"
    )


    print(
        "All final samples finite     :",
        bool(
            torch.isfinite(
                final_adversarial_joint
            ).all()
        ),
    )


    print(
        "All final losses finite      :",
        bool(
            torch.isfinite(
                final_adv_loss
            ).all()
        ),
    )


    print("==========================================")


    print(
        "\nPredictor checkpoint:"
    )

    print(
        predictor_path
    )

    print(
        "\nAdversarial diffusion checkpoint:"
    )

    print(
        diffusion_path
    )


if __name__ == "__main__":
    main()