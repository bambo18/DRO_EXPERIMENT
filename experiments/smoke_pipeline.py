"""
Integration smoke test:

Synthetic joint data
        ↓
Nominal predictor (MLP)
        ↓
Vector diffusion
        ↓
Generated joint samples
        ↓
Generated X/Y split
        ↓
Predictor evaluation

This is NOT GAS-DRO adversarial training yet.
It only verifies that the components can communicate correctly.
"""

import sys
import time
from pathlib import Path

import torch


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion,
    check_finite,
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
# Later this will be replaced with data/synthetic_scm.py
# from the data-generation team.
# ============================================================

def generate_scm_data(
    n_samples: int,
    seed: int,
):

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
        "generated_scm_data",
    )


    return joint


# ============================================================
# Statistics
# ============================================================

def print_stats(
    title: str,
    data: torch.Tensor,
):

    data = (
        data
        .detach()
        .cpu()
    )


    print(
        f"\n========== {title} =========="
    )

    print(
        "Shape:",
        tuple(data.shape),
    )

    print(
        "Mean :",
        [
            round(v, 4)
            for v in data.mean(dim=0).tolist()
        ],
    )

    print(
        "Std  :",
        [
            round(v, 4)
            for v in data.std(
                dim=0,
                unbiased=False,
            ).tolist()
        ],
    )

    print(
        "Finite:",
        bool(
            torch.isfinite(data).all()
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.time()


    print("\n==========================================")
    print("GAS-DRO COMPONENT PIPELINE SMOKE TEST")
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
    # 1. Nominal Train / ID Test
    # ========================================================

    train_joint = generate_scm_data(
        n_samples=2000,
        seed=42,
    )

    test_joint = generate_scm_data(
        n_samples=1000,
        seed=100,
    )


    print_stats(
        "TRAIN JOINT DATA",
        train_joint,
    )


    train_x, train_y = split_joint_xy(
        train_joint
    )

    test_x, test_y = split_joint_xy(
        test_joint
    )


    # ========================================================
    # 2. Nominal Predictor
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 1 — NOMINAL MLP")
    print("##########################################")


    predictor = RegressionMLP(
        input_dim=4,
        hidden_dim=64,
        output_dim=1,
    )


    ml_start = time.time()


    ml_history = train_regression_mlp(

        model=predictor,

        x=train_x,

        y=train_y,

        device=DEVICE,

        epochs=30,

        batch_size=128,

        lr=1e-3,

        verbose_every=5,
    )


    ml_time = (
        time.time()
        - ml_start
    )


    id_result = evaluate_regression_mlp(

        model=predictor,

        x=test_x,

        y=test_y,

        device=DEVICE,
    )


    print(
        f"\nNominal ID MSE: "
        f"{id_result['mse']:.6f}"
    )

    print(
        f"[TIMER] MLP Training: "
        f"{ml_time:.2f} sec"
    )


    # ========================================================
    # 3. Nominal Vector Diffusion
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 2 — VECTOR DIFFUSION")
    print("##########################################")


    diffusion = VectorDiffusion(

        data_dim=5,

        # Integration smoke test is intentionally smaller.
        timesteps=30,

        beta_start=1e-4,

        beta_end=2e-2,

        hidden_dim=128,

        time_dim=64,
    )


    diffusion_start = time.time()


    diffusion_history, standardizer = (
        train_vector_diffusion(

            model=diffusion,

            data=train_joint,

            device=DEVICE,

            # Smaller than previous 30-epoch diffusion smoke test.
            epochs=10,

            batch_size=128,

            lr=1e-3,

            grad_clip=1.0,

            verbose_every=2,
        )
    )


    diffusion_time = (
        time.time()
        - diffusion_start
    )


    print(
        f"\n[TIMER] Diffusion Training: "
        f"{diffusion_time:.2f} sec"
    )


    # ========================================================
    # 4. Generate Joint Samples
    # ========================================================

    print("\n\n##########################################")
    print("# STEP 3 — GENERATE JOINT SAMPLES")
    print("##########################################")


    diffusion = diffusion.to(
        DEVICE
    )


    sampling_start = time.time()


    generated_normalized = diffusion.sample(

        num_samples=1000,

        device=DEVICE,

        return_trajectory=False,
    )


    generated_joint = (
        standardizer.inverse_transform(
            generated_normalized
        )
    )


    sampling_time = (
        time.time()
        - sampling_start
    )


    check_finite(
        generated_joint,
        "generated_joint",
    )


    print_stats(
        "GENERATED JOINT DATA",
        generated_joint,
    )


    print(
        f"\n[TIMER] Sampling: "
        f"{sampling_time:.2f} sec"
    )


    # ========================================================
    # 5. Split generated samples
    # ========================================================

    generated_x, generated_y = (
        split_joint_xy(
            generated_joint
        )
    )


    print(
        "\nGenerated X shape:",
        tuple(generated_x.shape),
    )

    print(
        "Generated Y shape:",
        tuple(generated_y.shape),
    )


    # ========================================================
    # 6. Current Predictor Loss on Generated Samples
    #
    # This is important because GAS-DRO will later use these
    # per-sample losses as the adversarial signal.
    # ========================================================

    predictor = predictor.to(
        DEVICE
    )

    generated_joint_device = (
        generated_joint.to(
            DEVICE
        )
    )


    with torch.no_grad():

        generated_losses = (
            per_sample_mse(

                model=predictor,

                joint=
                    generated_joint_device,
            )
        )


    check_finite(
        generated_losses,
        "generated_predictor_losses",
    )


    print("\n==========================================")
    print("PREDICTOR LOSS ON GENERATED DATA")
    print("==========================================")

    print(
        f"Mean loss : "
        f"{generated_losses.mean().item():.6f}"
    )

    print(
        f"Max loss  : "
        f"{generated_losses.max().item():.6f}"
    )

    print(
        f"Min loss  : "
        f"{generated_losses.min().item():.6f}"
    )


    # ========================================================
    # 7. Sanity Checks
    # ========================================================

    generated_cpu = (
        generated_joint
        .detach()
        .cpu()
    )


    real_mean = (
        train_joint.mean(dim=0)
    )

    fake_mean = (
        generated_cpu.mean(dim=0)
    )


    real_std = train_joint.std(
        dim=0,
        unbiased=False,
    )

    fake_std = generated_cpu.std(
        dim=0,
        unbiased=False,
    )


    mean_error = (
        torch.abs(
            real_mean
            - fake_mean
        )
        .mean()
        .item()
    )


    std_error = (
        torch.abs(
            real_std
            - fake_std
        )
        .mean()
        .item()
    )


    # ========================================================
    # 8. Save Predictor / Diffusion
    # ========================================================

    predictor_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "gas_dro"
        / "pipeline_predictor_smoke.pt"
    )


    diffusion_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "gas_dro"
        / "pipeline_diffusion_smoke.pt"
    )


    predictor.save(

        predictor_path,

        extra={
            "seed": SEED,
            "type": "pipeline_smoke",
        },
    )


    diffusion.save(

        diffusion_path,

        standardizer=standardizer,

        extra={
            "seed": SEED,
            "type": "pipeline_smoke",
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
    print("PIPELINE SMOKE TEST SUMMARY")
    print("==========================================")

    print(
        f"ID MSE                  : "
        f"{id_result['mse']:.6f}"
    )

    print(
        f"Generated Mean Error    : "
        f"{mean_error:.6f}"
    )

    print(
        f"Generated Std Error     : "
        f"{std_error:.6f}"
    )

    print(
        f"Diffusion Initial Loss  : "
        f"{diffusion_history[0]:.6f}"
    )

    print(
        f"Diffusion Final Loss    : "
        f"{diffusion_history[-1]:.6f}"
    )

    print(
        f"Generated Mean MSE      : "
        f"{generated_losses.mean().item():.6f}"
    )

    print(
        f"Generated Worst MSE     : "
        f"{generated_losses.max().item():.6f}"
    )

    print(
        f"MLP Runtime             : "
        f"{ml_time:.2f} sec"
    )

    print(
        f"Diffusion Runtime       : "
        f"{diffusion_time:.2f} sec"
    )

    print(
        f"Total Runtime           : "
        f"{total_time:.2f} sec"
    )

    print(
        "Generated Shape         :",
        tuple(generated_joint.shape),
    )

    print(
        "All Generated Finite    :",
        bool(
            torch.isfinite(
                generated_joint
            ).all()
        ),
    )

    print(
        "All Losses Finite       :",
        bool(
            torch.isfinite(
                generated_losses
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
        "\nDiffusion checkpoint:"
    )

    print(
        diffusion_path
    )


if __name__ == "__main__":
    main()