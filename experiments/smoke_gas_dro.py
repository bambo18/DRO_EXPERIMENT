"""
Smoke test for the 5D vector diffusion used by GAS-DRO.

Purpose:
1. Create a small dummy 5D joint distribution.
2. Train VectorDiffusion.
3. Generate new 5D samples.
4. Check NaN / Inf.
5. Compare simple statistics.
6. Save a checkpoint.

This is NOT the final GAS-DRO experiment.
It only verifies the generative component.
"""

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


from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion,
    check_finite,
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Dummy 5D joint data
#
# Z = [X1, X2, X3, X4, Y]
#
# This is only for smoke testing.
# Final experiment will use data/synthetic_scm.py.
# ============================================================

def make_dummy_joint_data(
    n_samples: int = 2000,
) -> torch.Tensor:

    u1 = torch.randn(n_samples)
    u2 = torch.randn(n_samples)
    u3 = torch.randn(n_samples)
    u4 = torch.randn(n_samples)
    uy = torch.randn(n_samples)

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
    )

    return joint.float()


# ============================================================
# Statistics
# ============================================================

def print_statistics(
    name: str,
    data: torch.Tensor,
):

    data = data.detach().cpu()

    print(
        f"\n========== {name} =========="
    )

    print(
        "Shape :",
        tuple(data.shape),
    )

    print(
        "Mean  :",
        data.mean(dim=0).tolist(),
    )

    print(
        "Std   :",
        data.std(
            dim=0,
            unbiased=False,
        ).tolist(),
    )

    print(
        "Min   :",
        data.min(dim=0).values.tolist(),
    )

    print(
        "Max   :",
        data.max(dim=0).values.tolist(),
    )

    print(
        "Finite:",
        bool(torch.isfinite(data).all()),
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.time()

    print("\n==========================================")
    print("GAS-DRO Vector Diffusion Smoke Test")
    print("==========================================")

    print(
        f"Device : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU    :",
            torch.cuda.get_device_name(0),
        )

        memory_gb = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM   : {memory_gb:.2f} GB"
        )

    print(
        f"Seed   : {SEED}"
    )

    print("==========================================")


    # --------------------------------------------------------
    # 1. Generate dummy joint data
    # --------------------------------------------------------

    data = make_dummy_joint_data(
        n_samples=2000
    )

    check_finite(
        data,
        "dummy_joint_data",
    )

    print_statistics(
        "REAL DATA",
        data,
    )


    # --------------------------------------------------------
    # 2. Build diffusion model
    # --------------------------------------------------------

    model = VectorDiffusion(

        data_dim=5,

        timesteps=50,

        beta_start=1e-4,

        beta_end=2e-2,

        hidden_dim=128,

        time_dim=64,
    )


    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{parameter_count:,}"
    )


    # --------------------------------------------------------
    # 3. Train nominal vector diffusion
    # --------------------------------------------------------

    train_start = time.time()

    history, standardizer = (
        train_vector_diffusion(

            model=model,

            data=data,

            device=DEVICE,

            epochs=30,

            batch_size=128,

            lr=1e-3,

            grad_clip=1.0,

            verbose_every=5,
        )
    )

    train_time = (
        time.time()
        - train_start
    )

    print(
        f"\n[TIMER] Diffusion training: "
        f"{train_time:.2f} sec"
    )


    # --------------------------------------------------------
    # 4. Generate samples
    #
    # The diffusion was trained in standardized space.
    # Therefore:
    #
    # generated_norm
    #       ↓ inverse transform
    # generated_data
    # --------------------------------------------------------

    sample_start = time.time()

    model = model.to(DEVICE)

    generated_norm = model.sample(

        num_samples=1000,

        device=DEVICE,

        return_trajectory=False,
    )


    check_finite(
        generated_norm,
        "generated_normalized_samples",
    )


    generated_data = (
        standardizer.inverse_transform(
            generated_norm
        )
    )


    check_finite(
        generated_data,
        "generated_original_scale",
    )


    sample_time = (
        time.time()
        - sample_start
    )


    print(
        f"\n[TIMER] Sampling: "
        f"{sample_time:.2f} sec"
    )


    # --------------------------------------------------------
    # 5. Compare statistics
    # --------------------------------------------------------

    print_statistics(
        "GENERATED DATA",
        generated_data,
    )


    real_mean = data.mean(dim=0)

    generated_mean = (
        generated_data
        .detach()
        .cpu()
        .mean(dim=0)
    )


    real_std = data.std(
        dim=0,
        unbiased=False,
    )

    generated_std = (
        generated_data
        .detach()
        .cpu()
        .std(
            dim=0,
            unbiased=False,
        )
    )


    mean_error = (
        torch.abs(
            real_mean
            - generated_mean
        )
        .mean()
        .item()
    )


    std_error = (
        torch.abs(
            real_std
            - generated_std
        )
        .mean()
        .item()
    )


    print("\n==========================================")
    print("Simple Distribution Check")
    print("==========================================")

    print(
        f"Mean absolute error : "
        f"{mean_error:.6f}"
    )

    print(
        f"Std absolute error  : "
        f"{std_error:.6f}"
    )

    print(
        f"Initial loss        : "
        f"{history[0]:.6f}"
    )

    print(
        f"Final loss          : "
        f"{history[-1]:.6f}"
    )


    # --------------------------------------------------------
    # 6. Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "gas_dro"
        / "vector_diffusion_smoke.pt"
    )


    model.save(

        path=checkpoint_path,

        standardizer=standardizer,

        extra={
            "seed": SEED,
            "type": "smoke_test",
        },
    )


    print(
        f"\nCheckpoint saved:"
        f"\n{checkpoint_path}"
    )


    # --------------------------------------------------------
    # 7. GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            f"\nPeak CUDA memory: "
            f"{peak_memory:.2f} MB"
        )


    total_time = (
        time.time()
        - total_start
    )


    print("\n==========================================")
    print("SMOKE TEST COMPLETE")
    print("==========================================")

    print(
        f"Total runtime: "
        f"{total_time:.2f} sec"
    )

    print(
        "Generated shape:",
        tuple(generated_data.shape),
    )

    print(
        "All finite:",
        bool(
            torch.isfinite(
                generated_data
            ).all()
        ),
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()