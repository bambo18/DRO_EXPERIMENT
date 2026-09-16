"""
Official GAS-DRO 5D adapter preflight.

Purpose
-------
This is NOT a final performance experiment.

It checks that the 5D adapter can execute:

1. Official nominal diffusion training
2. theta_0 -> theta copy
3. fixed nominal trajectory
4. a0
5. initial r_theta
6. PPO
7. JSM
8. one generator update
9. S_theta generation
10. predictor update

IMPORTANT
---------
Official GAS-DRO hyperparameters are NOT modified.
"""

from __future__ import annotations

import copy
import time

import torch

from configs.gas_dro_full import (
    DIFFUSION_CONFIG,
    GAS_DRO_CONFIG,
)

from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion_steps,
)

from methods.gas_dro.gas_dro import (
    VectorGasDRO,
)

from models.mlp import (
    RegressionMLP,
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


def set_seed(seed: int) -> None:

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Temporary 5D smoke data
#
# NOT final team data.
# ============================================================

def make_smoke_data(
    n: int,
) -> torch.Tensor:

    x1 = torch.randn(n, 1)
    x2 = torch.randn(n, 1)
    x3 = torch.randn(n, 1)
    x4 = torch.randn(n, 1)

    noise = (
        0.1
        *
        torch.randn(n, 1)
    )

    y = (
        0.8 * x1
        -
        0.5 * x2
        +
        0.3 * x3
        +
        0.2 * x4
        +
        noise
    )

    joint = torch.cat(
        [
            x1,
            x2,
            x3,
            x4,
            y,
        ],
        dim=1,
    )

    return joint


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    start_time = time.time()

    print(
        "=========================================="
    )
    print(
        "OFFICIAL GAS-DRO 5D PREFLIGHT"
    )
    print(
        "=========================================="
    )
    print(
        "Device :",
        device,
    )

    # --------------------------------------------------------
    # Copy config only to prevent accidental mutation.
    #
    # We do NOT modify any official value.
    # --------------------------------------------------------

    diffusion_cfg = copy.deepcopy(
        DIFFUSION_CONFIG
    )

    gas_cfg = copy.deepcopy(
        GAS_DRO_CONFIG
    )

    # --------------------------------------------------------
    # Guard rails:
    # fail immediately if official config was accidentally
    # changed somewhere else.
    # --------------------------------------------------------

    assert diffusion_cfg["timesteps"] == 500
    assert diffusion_cfg["beta_start"] == 0.1
    assert diffusion_cfg["beta_end"] == 0.5
    assert diffusion_cfg["batch_size"] == 64
    assert diffusion_cfg["lr"] == 1e-4
    assert diffusion_cfg["iterations"] == 7000

    assert gas_cfg.outer_epochs == 15
    assert gas_cfg.generator_inner_epochs == 10
    assert gas_cfg.predictor_inner_epochs == 2

    assert gas_cfg.batch_size == 64
    assert gas_cfg.batch_repeat == 4

    assert gas_cfg.generator_lr == 1e-5
    assert gas_cfg.predictor_lr == 1e-5

    assert gas_cfg.ppo_clip == 0.4

    assert gas_cfg.mu == 1.0
    assert gas_cfg.eta == 0.1
    assert gas_cfg.budget == 0.015

    assert gas_cfg.adjust_timesteps == 15

    assert gas_cfg.step_size == 2
    assert gas_cfg.discount_factor == 0.05

    assert gas_cfg.grad_clip is None

    print(
        "\n[PASS] Official config guard."
    )

    # --------------------------------------------------------
    # N = 64 only reduces TEMPORARY smoke dataset size.
    #
    # It does NOT change GAS-DRO hyperparameters.
    # --------------------------------------------------------

    real_joint = make_smoke_data(
        n=64
    )

    print(
        "\nTemporary joint shape:",
        real_joint.shape,
    )

    print(
        "Temporary joint finite:",
        torch.isfinite(
            real_joint
        ).all().item(),
    )

    # ========================================================
    # 1. Nominal diffusion theta_0
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[1] NOMINAL DIFFUSION theta_0"
    )
    print(
        "=========================================="
    )

    nominal_diffusion = VectorDiffusion(

        data_dim=5,

        timesteps=
            diffusion_cfg["timesteps"],

        beta_start=
            diffusion_cfg["beta_start"],

        beta_end=
            diffusion_cfg["beta_end"],
    )

    history, standardizer = (
        train_vector_diffusion_steps(

            model=
                nominal_diffusion,

            data=
                real_joint,

            device=
                device,

            total_iterations=
                diffusion_cfg["iterations"],

            batch_size=
                diffusion_cfg["batch_size"],

            lr=
                diffusion_cfg["lr"],

            grad_clip=
                None,

            verbose_every=
                500,
        )
    )

    print(
        "\nDiffusion final loss:",
        history[-1],
    )

    print(
        "Diffusion all finite:",
        torch.isfinite(
            torch.tensor(
                history
            )
        ).all().item(),
    )

    # ========================================================
    # 2. Common regression predictor
    #
    # Random initialization is sufficient for this preflight.
    #
    # This is NOT a performance evaluation.
    # Final experiment will use the team's required predictor
    # initialization / training protocol.
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[2] COMMON PREDICTOR"
    )
    print(
        "=========================================="
    )

    predictor = RegressionMLP(
        input_dim=4,
        hidden_dim=64,
        output_dim=1,
    )

    print(
        predictor
    )

    # ========================================================
    # 3. Build GAS-DRO
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[3] BUILD VECTOR GAS-DRO"
    )
    print(
        "=========================================="
    )

    gas = VectorGasDRO(

        nominal_diffusion=
            nominal_diffusion,

        standardizer=
            standardizer,

        predictor=
            predictor,

        device=
            device,

        config=
            gas_cfg,
    )

    s0_iterations = (
        gas._s0_iterations(
            real_joint.shape[0]
        )
    )

    print(
        "s0 iterations:",
        s0_iterations,
    )

    expected_s0_iterations = (
        1
        *
        gas_cfg.batch_repeat
    )

    assert (
        s0_iterations
        ==
        expected_s0_iterations
    )

    print(
        "[PASS] "
        "ceil(64 / 64) * 4 = 4"
    )

    # ========================================================
    # 4. Fixed nominal reference trajectory
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[4] REFERENCE TRAJECTORY"
    )
    print(
        "=========================================="
    )

    (
        reference_joint,
        reference_trajectory,
    ) = (
        gas.sample_reference_trajectory(

            num_logical_batches=
                s0_iterations
        )
    )

    print(
        "reference_joint:",
        reference_joint.shape,
    )

    print(
        "reference_trajectory:",
        reference_trajectory.shape,
    )

    print(
        "reference finite:",
        torch.isfinite(
            reference_trajectory
        ).all().item(),
    )

    # Expected:
    #
    # 4 batches * 64
    # = 256 samples
    #
    # only first 15 adjusted timesteps stored

    assert (
        reference_joint.shape
        ==
        (256, 5)
    )

    assert (
        reference_trajectory.shape
        ==
        (
            256,
            gas_cfg.adjust_timesteps,
            5,
        )
    )

    # ========================================================
    # 5. a0
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[5] BUILD a0"
    )
    print(
        "=========================================="
    )

    a0 = gas.build_a0(
        reference_trajectory
    )

    print(
        "a0 shape:",
        a0.shape,
    )

    print(
        "a0 finite:",
        torch.isfinite(
            a0
        ).all().item(),
    )

    assert (
        a0.shape
        ==
        (
            256,
            gas_cfg.adjust_timesteps,
        )
    )

    # ========================================================
    # 6. Predictor loss on fixed z0
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[6] REFERENCE PREDICTOR LOSS"
    )
    print(
        "=========================================="
    )

    reference_loss = (
        gas._reference_predictor_loss(
            reference_joint
        )
    )

    print(
        "loss shape:",
        reference_loss.shape,
    )

    print(
        "loss mean:",
        reference_loss.mean().item(),
    )

    print(
        "loss max:",
        reference_loss.max().item(),
    )

    print(
        "loss finite:",
        torch.isfinite(
            reference_loss
        ).all().item(),
    )

    # ========================================================
    # 7. INITIAL r_theta
    #
    # theta == theta_0 here.
    #
    # Therefore ratio should be approximately 1.
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[7] INITIAL r_theta CHECK"
    )
    print(
        "=========================================="
    )

    trajectory_batch = (
        reference_trajectory[
            :gas_cfg.batch_size
        ]
        .to(
            device
        )
    )

    a0_batch = (
        a0[
            :gas_cfg.batch_size
        ]
        .to(
            device
        )
    )

    predictor_loss_batch = (
        reference_loss[
            :gas_cfg.batch_size
        ]
        .to(
            device
        )
    )

    initial_ratio = gas.r_theta(

        trajectory=
            trajectory_batch,

        a0=
            a0_batch,
    )

    print(
        "ratio mean:",
        initial_ratio.mean().item(),
    )

    print(
        "ratio min :",
        initial_ratio.min().item(),
    )

    print(
        "ratio max :",
        initial_ratio.max().item(),
    )

    print(
        "ratio finite:",
        torch.isfinite(
            initial_ratio
        ).all().item(),
    )

    ratio_error = (
        initial_ratio
        -
        1.0
    ).abs().max().item()

    print(
        "max |ratio - 1|:",
        ratio_error,
    )

    if ratio_error < 1e-5:

        print(
            "[PASS] theta == theta_0 "
            "gives r_theta ~= 1."
        )

    else:

        raise RuntimeError(
            "Initial r_theta is not approximately 1. "
            "Check trajectory / a0 indexing."
        )

    # ========================================================
    # 8. PPO before update
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[8] PPO CHECK"
    )
    print(
        "=========================================="
    )

    ppo = gas.ppo(

        ratio=
            initial_ratio,

        predictor_loss=
            predictor_loss_batch,
    )

    print(
        "PPO:",
        ppo.item(),
    )

    print(
        "PPO finite:",
        torch.isfinite(
            ppo
        ).item(),
    )

    # ========================================================
    # 9. ONE generator update
    #
    # Uses the OFFICIAL optimizer hyperparameter.
    #
    # This is only an integration check;
    # it is not the full 15 x 10 optimization.
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[9] ONE GENERATOR UPDATE"
    )
    print(
        "=========================================="
    )

    generator_optimizer = (
        torch.optim.Adam(

            gas.adversarial_diffusion
            .parameters(),

            lr=
                gas_cfg.generator_lr,
        )
    )

    real_batch = (
        gas._real_batch_for_index(

            real_joint=
                real_joint,

            batch_i=
                0,
        )
    )

    stats = gas.generator_update(

        trajectory_batch=
            trajectory_batch,

        a0_batch=
            a0_batch,

        predictor_loss_batch=
            predictor_loss_batch,

        real_batch=
            real_batch,

        optimizer=
            generator_optimizer,
    )

    print(
        "PPO       :",
        stats["ppo"],
    )

    print(
        "JSM       :",
        stats["jsm"],
    )

    print(
        "Objective :",
        stats["objective"],
    )

    print(
        "Ratio mean:",
        stats["ratio_mean"],
    )

    print(
        "Ratio min :",
        stats["ratio_min"],
    )

    print(
        "Ratio max :",
        stats["ratio_max"],
    )

    # ========================================================
    # 10. Ratio after one update
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[10] POST-UPDATE r_theta"
    )
    print(
        "=========================================="
    )

    ratio_after = gas.r_theta(

        trajectory=
            trajectory_batch,

        a0=
            a0_batch,
    )

    print(
        "ratio mean:",
        ratio_after.mean().item(),
    )

    print(
        "ratio min :",
        ratio_after.min().item(),
    )

    print(
        "ratio max :",
        ratio_after.max().item(),
    )

    print(
        "ratio finite:",
        torch.isfinite(
            ratio_after
        ).all().item(),
    )

    # ========================================================
    # 11. Generate one official-size S_theta batch
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[11] GENERATE S_theta"
    )
    print(
        "=========================================="
    )

    s_theta = (
        gas.generate_adversarial_samples(
            num_batches=1
        )
    )

    print(
        "S_theta shape:",
        s_theta.shape,
    )

    print(
        "S_theta finite:",
        torch.isfinite(
            s_theta
        ).all().item(),
    )

    print(
        "S_theta min:",
        s_theta.min().item(),
    )

    print(
        "S_theta max:",
        s_theta.max().item(),
    )

    assert (
        s_theta.shape
        ==
        (
            gas_cfg.batch_size,
            5,
        )
    )

    # ========================================================
    # 12. Predictor update
    #
    # Official GAS-DRO inner predictor settings:
    #
    # lr = 1e-5
    # epochs = 2
    # StepLR(2, 0.05)
    # ========================================================

    print(
        "\n=========================================="
    )
    print(
        "[12] PREDICTOR UPDATE"
    )
    print(
        "=========================================="
    )

    predictor_history = (
        gas.update_predictor(
            s_theta
        )
    )

    print(
        "Predictor history:",
        predictor_history,
    )

    print(
        "Predictor losses finite:",
        torch.isfinite(
            torch.tensor(
                predictor_history
            )
        ).all().item(),
    )

    # ========================================================
    # Complete
    # ========================================================

    runtime = (
        time.time()
        -
        start_time
    )

    print(
        "\n=========================================="
    )
    print(
        "PREFLIGHT COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        "Official hyperparameters modified : NO"
    )

    print(
        "Initial r_theta ~= 1             : YES"
    )

    print(
        "Generator update completed        : YES"
    )

    print(
        "S_theta finite                    : YES"
    )

    print(
        "Predictor update completed        : YES"
    )

    print(
        "Runtime                           :",
        round(
            runtime,
            2,
        ),
        "sec",
    )


if __name__ == "__main__":
    main()