"""
Gradient diagnostic for the current Vector GAS-DRO implementation.

Purpose
-------
Measure how much PPO and JSM actually contribute to the
adversarial generator update.

Generator objective:

    L(theta) = - PPO(theta) + mu * JSM(theta)

We measure:

1. || grad PPO ||
2. || grad JSM ||
3. cosine similarity between the two gradients
4. mu * ||grad JSM|| / ||grad PPO||
5. || -grad PPO + mu * grad JSM ||
6. effective norm after gradient clipping

This is NOT a performance experiment.
It is a debugging / optimization diagnostic.
"""

from __future__ import annotations

import copy
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
    split_joint_xy,
    per_sample_mse,
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

from experiments.pilot_ood_gas_dro import (
    generate_joint_data,
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
    0.1,
    1.0,
    5.0,
    10.0,
]

GRAD_CLIP = 1.0

BATCH_SIZE = 128


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Gradient helpers
# ============================================================

def flatten_gradients(
    gradients,
    parameters,
):
    """
    Convert parameter-wise gradients into one flat vector.

    If a parameter has no gradient, insert zeros with the
    same shape so vectors remain aligned.
    """

    flat = []

    for grad, parameter in zip(
        gradients,
        parameters,
    ):

        if grad is None:

            flat.append(
                torch.zeros_like(
                    parameter
                ).reshape(-1)
            )

        else:

            flat.append(
                grad.reshape(-1)
            )

    return torch.cat(
        flat
    )


def vector_norm(
    vector: torch.Tensor,
) -> float:

    return float(
        torch.linalg.vector_norm(
            vector
        ).detach().cpu()
    )


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> float:

    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)

    denominator = (
        a_norm
        *
        b_norm
    )

    if denominator.item() < eps:
        return float("nan")

    cosine = (
        torch.dot(a, b)
        /
        denominator
    )

    return float(
        cosine.detach().cpu()
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = time.time()


    print("\n==========================================")
    print("GAS-DRO GRADIENT DIAGNOSTIC")
    print("==========================================")

    print(
        f"Device    : {DEVICE}"
    )

    print(
        f"Batch     : {BATCH_SIZE}"
    )

    print(
        f"Grad clip : {GRAD_CLIP}"
    )

    print(
        f"mu values : {MU_VALUES}"
    )

    print("==========================================")


    # ========================================================
    # 1. Nominal data
    # ========================================================

    print(
        "\n[1] Generating nominal data..."
    )


    train_joint = generate_joint_data(

        n_samples=2000,

        seed=SEED,
    )


    train_x, train_y = split_joint_xy(
        train_joint
    )


    # ========================================================
    # 2. Predictor
    # ========================================================

    print(
        "\n[2] Training predictor..."
    )


    predictor = RegressionMLP(

        input_dim=4,

        hidden_dim=64,

        output_dim=1,
    )


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


    # ========================================================
    # 3. Nominal diffusion
    # ========================================================

    print(
        "\n[3] Training nominal diffusion..."
    )


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
    # 4. GAS-DRO object
    #
    # mu here is irrelevant because we directly inspect
    # PPO and JSM separately.
    # ========================================================

    print(
        "\n[4] Building GAS-DRO diagnostic object..."
    )


    config = GasDROConfig(

        outer_epochs=1,

        generator_inner_epochs=1,

        generator_batches_per_epoch=1,

        predictor_inner_epochs=1,

        batch_size=BATCH_SIZE,

        generator_lr=1e-4,

        predictor_lr=1e-4,

        ppo_clip=0.4,

        mu=1.0,

        eta=0.0,

        budget=0.8,

        adjust_timesteps=5,

        step_size=2,

        discount_factor=0.05,

        grad_clip=GRAD_CLIP,

        verbose=False,
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
            config,
    )


    # ========================================================
    # 5. Fixed nominal reference trajectory
    # ========================================================

    print(
        "\n[5] Generating reference trajectory..."
    )


    (
        reference_joint,
        reference_trajectory,
    ) = gas_dro.sample_reference_trajectory(

        num_samples=BATCH_SIZE
    )


    a0 = gas_dro.build_a0(
        reference_trajectory
    )


    predictor = predictor.to(
        DEVICE
    )

    predictor.eval()


    with torch.no_grad():

        reference_loss = per_sample_mse(

            model=predictor,

            joint=reference_joint,
        )


    check_finite(
        reference_loss,
        "reference_loss",
    )


    print(
        f"Reference predictor loss mean : "
        f"{reference_loss.mean().item():.6f}"
    )

    print(
        f"Reference predictor loss max  : "
        f"{reference_loss.max().item():.6f}"
    )


    # ========================================================
    # 6. Real batch for JSM
    # ========================================================

    real_batch = (
        train_joint[
            :BATCH_SIZE
        ]
        .float()
        .to(DEVICE)
    )


    normalized_real = (
        standardizer.transform(
            real_batch
        )
    )


    # ========================================================
    # 7. Parameters
    # ========================================================

    parameters = [

        p

        for p in
        gas_dro.adversarial_diffusion.parameters()

        if p.requires_grad
    ]


    # ========================================================
    # 8. PPO gradient
    # ========================================================

    print(
        "\n[6] Computing PPO gradient..."
    )


    ratio = gas_dro.r_theta(

        trajectory=
            reference_trajectory,

        a0=
            a0,
    )


    ppo_loss = gas_dro.ppo(

        ratio=
            ratio,

        predictor_loss=
            reference_loss,
    )


    ppo_grads = torch.autograd.grad(

        outputs=
            ppo_loss,

        inputs=
            parameters,

        retain_graph=
            False,

        create_graph=
            False,

        allow_unused=
            True,
    )


    g_ppo = flatten_gradients(

        ppo_grads,

        parameters,
    ).detach()


    check_finite(
        g_ppo,
        "g_ppo",
    )


    # ========================================================
    # 9. JSM gradient
    #
    # Reset RNG so JSM comparison is reproducible.
    # ========================================================

    print(
        "\n[7] Computing JSM gradient..."
    )


    torch.manual_seed(
        SEED + 100
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED + 100
        )


    jsm_loss = gas_dro.jsm_loss(
        normalized_real
    )


    jsm_grads = torch.autograd.grad(

        outputs=
            jsm_loss,

        inputs=
            parameters,

        retain_graph=
            False,

        create_graph=
            False,

        allow_unused=
            True,
    )


    g_jsm = flatten_gradients(

        jsm_grads,

        parameters,
    ).detach()


    check_finite(
        g_jsm,
        "g_jsm",
    )


    # ========================================================
    # 10. Basic gradient diagnostics
    # ========================================================

    ppo_norm = vector_norm(
        g_ppo
    )

    jsm_norm = vector_norm(
        g_jsm
    )


    raw_cosine = cosine_similarity(

        g_ppo,

        g_jsm,
    )


    # Generator minimizes:
    #
    #     - PPO + mu * JSM
    #
    # So the optimization directions are:
    #
    #     -g_ppo
    #     +mu*g_jsm
    #

    objective_cosine = cosine_similarity(

        -g_ppo,

        g_jsm,
    )


    print("\n==========================================")
    print("RAW GRADIENT DIAGNOSTICS")
    print("==========================================")

    print(
        f"PPO loss               : "
        f"{ppo_loss.item():.6f}"
    )

    print(
        f"JSM loss               : "
        f"{jsm_loss.item():.6f}"
    )

    print(
        f"||grad PPO||           : "
        f"{ppo_norm:.8f}"
    )

    print(
        f"||grad JSM||           : "
        f"{jsm_norm:.8f}"
    )

    print(
        f"JSM/PPO norm ratio     : "
        f"{jsm_norm / max(ppo_norm, 1e-12):.8f}"
    )

    print(
        f"cos(grad PPO, grad JSM): "
        f"{raw_cosine:.6f}"
    )

    print(
        f"cos(-grad PPO, grad JSM): "
        f"{objective_cosine:.6f}"
    )

    print("==========================================")


    # ========================================================
    # 11. Combined objective gradient by mu
    # ========================================================

    print("\n==========================================")
    print("MU CONTRIBUTION DIAGNOSTIC")
    print("==========================================")

    print(
        "mu | mu*JSM/PPO | combined_norm | "
        "clip_scale | clipped_norm | "
        "cos(combined,-PPO) | cos(combined,JSM)"
    )

    print(
        "-" * 110
    )


    for mu in MU_VALUES:

        # ----------------------------------------
        # Gradient of generator minimization loss:
        #
        #   -PPO + mu * JSM
        # ----------------------------------------

        combined = (

            -g_ppo

            +

            mu
            *
            g_jsm
        )


        combined_norm = vector_norm(
            combined
        )


        # ----------------------------------------
        # clip_grad_norm equivalent scale
        # ----------------------------------------

        if combined_norm > GRAD_CLIP:

            clip_scale = (
                GRAD_CLIP
                /
                combined_norm
            )

        else:

            clip_scale = 1.0


        clipped = (
            combined
            *
            clip_scale
        )


        clipped_norm = vector_norm(
            clipped
        )


        contribution_ratio = (

            mu
            *
            jsm_norm

            /
            max(
                ppo_norm,
                1e-12
            )
        )


        cos_to_ppo_direction = (
            cosine_similarity(

                combined,

                -g_ppo,
            )
        )


        cos_to_jsm = (
            cosine_similarity(

                combined,

                g_jsm,
            )
        )


        print(

            f"{mu:>4.1f} | "

            f"{contribution_ratio:>12.6f} | "

            f"{combined_norm:>13.6f} | "

            f"{clip_scale:>10.6f} | "

            f"{clipped_norm:>12.6f} | "

            f"{cos_to_ppo_direction:>18.6f} | "

            f"{cos_to_jsm:>17.6f}"
        )


    # ========================================================
    # 12. Interpretation hints
    # ========================================================

    print("\n==========================================")
    print("INTERPRETATION GUIDE")
    print("==========================================")

    print(
        """
Case A:
    ||grad JSM|| << ||grad PPO||

Then JSM barely affects the generator unless mu is very large.

Case B:
    ||grad JSM|| is comparable to ||grad PPO||
    but all combined norms exceed grad_clip=1

Then clipping equalizes UPDATE MAGNITUDES,
although mu can still change the gradient direction.

Case C:
    cos(-grad PPO, grad JSM) < 0

Then PPO and JSM partially oppose each other,
which is expected for a constraint term.

Case D:
    cos(-grad PPO, grad JSM) > 0

Then the two terms push the generator in a similar direction,
so increasing mu does not strongly restrain the adversary.

Case E:
    mu*JSM/PPO changes greatly but combined direction barely moves

Then the two gradient vectors are highly aligned.
"""
    )


    runtime = (
        time.time()
        -
        total_start
    )


    print(
        f"Runtime: {runtime:.2f} sec"
    )


if __name__ == "__main__":
    main()