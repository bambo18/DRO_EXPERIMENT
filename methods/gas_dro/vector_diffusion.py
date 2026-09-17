"""
Vector Diffusion for GAS-DRO baseline.

This file adapts the official GAS-DRO diffusion component to our
5-dimensional synthetic regression experiment.

Joint endogenous vector:

    z = [X1, X2, X3, X4, Y]

Important
---------
- This is NOT the causal / exogenous model used by Ours.
- GAS-DRO operates directly on the joint endogenous space.
- No SCM constraint is imposed here.

Official GAS-DRO diffusion settings can be used directly:

    T = 500
    beta_1 = 0.1
    beta_T = 0.5

The reverse diffusion equation follows the official GAS-DRO
DDPM implementation directly, avoiding reconstruction through

    1 / sqrt(alpha_bar_t)

which can overflow when alpha_bar_t underflows to zero in float32.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


# ============================================================
# Utility
# ============================================================

def check_finite(
    tensor: torch.Tensor,
    name: str,
    raise_error: bool = True,
) -> bool:
    """
    Check whether a tensor contains NaN or Inf.

    We fail immediately instead of silently skipping invalid
    values. This makes numerical problems easy to locate.
    """

    finite = torch.isfinite(
        tensor
    ).all().item()

    if not finite:

        msg = (
            f"[Non-Finite Tensor] {name}\n"
            f"shape={tuple(tensor.shape)}\n"
        )

        finite_values = tensor[
            torch.isfinite(tensor)
        ]

        if finite_values.numel() > 0:

            msg += (
                f"finite min="
                f"{finite_values.min().item():.6f}, "
                f"max="
                f"{finite_values.max().item():.6f}\n"
            )

        msg += (
            f"nan_count="
            f"{torch.isnan(tensor).sum().item()}, "
            f"inf_count="
            f"{torch.isinf(tensor).sum().item()}"
        )

        if raise_error:

            raise RuntimeError(
                msg
            )

        print(
            msg
        )

    return finite


# ============================================================
# Standardization
#
# Task-specific adapter.
#
# GAS-DRO operates on our 5D joint endogenous vector.
# Standardization is performed using nominal training statistics.
# ============================================================

class VectorStandardizer:

    def __init__(
        self,
        eps: float = 1e-6,
    ):

        self.eps = eps

        self.mean: Optional[
            torch.Tensor
        ] = None

        self.std: Optional[
            torch.Tensor
        ] = None


    def fit(
        self,
        x: torch.Tensor,
    ) -> "VectorStandardizer":

        if x.ndim != 2:

            raise ValueError(
                f"Expected 2D tensor [N, D], "
                f"got {tuple(x.shape)}"
            )

        self.mean = x.mean(
            dim=0,
            keepdim=True,
        )

        self.std = x.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        )

        self.std = torch.clamp(
            self.std,
            min=self.eps,
        )

        check_finite(
            self.mean,
            "standardizer.mean",
        )

        check_finite(
            self.std,
            "standardizer.std",
        )

        return self


    def transform(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if (
            self.mean is None
            or
            self.std is None
        ):

            raise RuntimeError(
                "Standardizer must be fitted "
                "before transform()."
            )

        mean = self.mean.to(
            x.device
        )

        std = self.std.to(
            x.device
        )

        z = (
            x - mean
        ) / std

        check_finite(
            z,
            "standardized_data",
        )

        return z


    def inverse_transform(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if (
            self.mean is None
            or
            self.std is None
        ):

            raise RuntimeError(
                "Standardizer must be fitted "
                "before inverse_transform()."
            )

        mean = self.mean.to(
            z.device
        )

        std = self.std.to(
            z.device
        )

        x = (
            z * std
            +
            mean
        )

        check_finite(
            x,
            "inverse_standardized_data",
        )

        return x


    def state_dict(
        self,
    ) -> Dict[str, torch.Tensor]:

        if (
            self.mean is None
            or
            self.std is None
        ):

            raise RuntimeError(
                "Standardizer has not been fitted."
            )

        return {

            "mean":
                self.mean
                .detach()
                .cpu(),

            "std":
                self.std
                .detach()
                .cpu(),
        }


    def load_state_dict(
        self,
        state: Dict[str, torch.Tensor],
    ) -> None:

        self.mean = (
            state["mean"]
            .clone()
        )

        self.std = (
            state["std"]
            .clone()
        )


# ============================================================
# Timestep Embedding
# ============================================================

class SinusoidalTimeEmbedding(
    nn.Module
):

    """
    Sinusoidal timestep embedding.

    Input:
        t: [batch]

    Output:
        embedding:
        [batch, time_dim]
    """

    def __init__(
        self,
        time_dim: int = 64,
    ):

        super().__init__()

        if time_dim % 2 != 0:

            raise ValueError(
                "time_dim must be even."
            )

        self.time_dim = (
            time_dim
        )


    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:

        half_dim = (
            self.time_dim
            //
            2
        )

        device = (
            t.device
        )

        exponent = (

            -math.log(
                10000.0
            )

            *

            torch.arange(

                half_dim,

                device=device,

                dtype=torch.float32,
            )

            /

            max(
                half_dim - 1,
                1,
            )
        )

        frequencies = torch.exp(
            exponent
        )

        args = (

            t.float()
            .unsqueeze(1)

            *

            frequencies
            .unsqueeze(0)
        )

        embedding = torch.cat(

            [
                torch.sin(args),
                torch.cos(args),
            ],

            dim=1,
        )

        return embedding


# ============================================================
# 5D Noise Predictor
# ============================================================

class VectorDenoiser(
    nn.Module
):

    """
    Small MLP that predicts epsilon.

    Input
    -----
    x_t:
        [X1, X2, X3, X4, Y]
        at diffusion timestep t.

    Output
    ------
    epsilon_theta:
        predicted noise,
        same dimension as x_t.
    """

    def __init__(
        self,
        data_dim: int = 5,
        hidden_dim: int = 128,
        time_dim: int = 64,
    ):

        super().__init__()

        self.data_dim = (
            data_dim
        )

        self.time_embedding = (
            SinusoidalTimeEmbedding(
                time_dim=time_dim
            )
        )

        self.net = nn.Sequential(

            nn.Linear(
                data_dim
                +
                time_dim,

                hidden_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                data_dim,
            ),
        )


    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        time_emb = (
            self.time_embedding(
                t
            )
        )

        h = torch.cat(

            [
                x_t,
                time_emb,
            ],

            dim=1,
        )

        predicted_noise = (
            self.net(
                h
            )
        )

        return (
            predicted_noise
        )


# ============================================================
# Vector DDPM
# ============================================================

class VectorDiffusion(
    nn.Module
):

    """
    GAS-DRO-compatible DDPM for low-dimensional vectors.

    Default joint variable:

        Z = [X1, X2, X3, X4, Y]

    Official GAS-DRO settings can be used:

        T = 500
        beta_1 = 0.1
        beta_T = 0.5

    Important
    ---------
    alpha_bar can underflow to zero under the official schedule.

    This is NOT itself an error.

    Therefore we do NOT precompute or use:

        sqrt(1 / alpha_bar)

    in reverse sampling.

    Reverse sampling uses the same epsilon-based mean equation
    as the official GAS-DRO DDPM implementation.
    """

    def __init__(
        self,
        data_dim: int = 5,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 128,
        time_dim: int = 64,
    ):

        super().__init__()


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if timesteps < 2:

            raise ValueError(
                "timesteps must be >= 2"
            )

        if not (
            0.0
            <
            beta_start
            <
            1.0
        ):

            raise ValueError(
                "beta_start must be between 0 and 1."
            )

        if not (
            0.0
            <
            beta_end
            <
            1.0
        ):

            raise ValueError(
                "beta_end must be between 0 and 1."
            )


        # ----------------------------------------------------
        # Config
        # ----------------------------------------------------

        self.data_dim = (
            data_dim
        )

        self.timesteps = (
            timesteps
        )

        self.beta_start = (
            beta_start
        )

        self.beta_end = (
            beta_end
        )

        self.hidden_dim = (
            hidden_dim
        )

        self.time_dim = (
            time_dim
        )


        # ----------------------------------------------------
        # Denoising network
        # ----------------------------------------------------

        self.denoiser = (
            VectorDenoiser(

                data_dim=
                    data_dim,

                hidden_dim=
                    hidden_dim,

                time_dim=
                    time_dim,
            )
        )


        # ----------------------------------------------------
        # Diffusion schedule
        #
        # Official GAS-DRO:
        #
        # beta_t = linear(beta_1, beta_T)
        #
        # alpha_t = 1 - beta_t
        #
        # alpha_bar_t =
        #     product_{s <= t} alpha_s
        # ----------------------------------------------------

        betas = torch.linspace(

            start=
                beta_start,

            end=
                beta_end,

            steps=
                timesteps,

            dtype=
                torch.float32,
        )


        alphas = (
            1.0
            -
            betas
        )


        alpha_bars = torch.cumprod(

            alphas,

            dim=0,
        )


        alpha_bars_prev = torch.cat(

            [
                torch.ones(
                    1,
                    dtype=torch.float32,
                ),

                alpha_bars[:-1],
            ],

            dim=0,
        )


        # ----------------------------------------------------
        # Forward diffusion coefficients
        #
        # alpha_bar == 0 is numerically acceptable:
        #
        # sqrt(alpha_bar) = 0
        # sqrt(1-alpha_bar) = 1
        # ----------------------------------------------------

        sqrt_alpha_bars = (
            torch.sqrt(
                alpha_bars
            )
        )


        sqrt_one_minus_alpha_bars = (
            torch.sqrt(

                torch.clamp(

                    1.0
                    -
                    alpha_bars,

                    min=0.0,
                )
            )
        )


        # ----------------------------------------------------
        # Official reverse variance:
        #
        # beta_tilde_t =
        #
        # beta_t *
        # (1-alpha_bar_{t-1})
        # -------------------------
        # (1-alpha_bar_t)
        #
        # Same expression as official GAS-DRO DiffusionProcess.
        # ----------------------------------------------------

        denominator = torch.clamp(

            1.0
            -
            alpha_bars,

            min=1e-20,
        )


        posterior_variance = (

            betas

            *

            (
                1.0
                -
                alpha_bars_prev
            )

            /

            denominator
        )


        # ----------------------------------------------------
        # Buffers
        # ----------------------------------------------------

        self.register_buffer(
            "betas",
            betas,
        )


        self.register_buffer(
            "alphas",
            alphas,
        )


        self.register_buffer(
            "alpha_bars",
            alpha_bars,
        )


        self.register_buffer(
            "alpha_bars_prev",
            alpha_bars_prev,
        )


        self.register_buffer(
            "sqrt_alpha_bars",
            sqrt_alpha_bars,
        )


        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            sqrt_one_minus_alpha_bars,
        )


        self.register_buffer(
            "posterior_variance",
            posterior_variance,
        )


    # ========================================================
    # Forward diffusion
    #
    # q(x_t | x_0)
    # ========================================================

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        if noise is None:

            noise = torch.randn_like(
                x_0
            )


        sqrt_alpha_bar_t = (

            self.sqrt_alpha_bars[
                t
            ]

            .unsqueeze(1)
        )


        sqrt_one_minus_alpha_bar_t = (

            self.sqrt_one_minus_alpha_bars[
                t
            ]

            .unsqueeze(1)
        )


        x_t = (

            sqrt_alpha_bar_t
            *
            x_0

            +

            sqrt_one_minus_alpha_bar_t
            *
            noise
        )


        check_finite(
            x_t,
            "q_sample.x_t",
        )


        return (
            x_t,
            noise,
        )


    # ========================================================
    # Diffusion denoising loss
    #
    # Official GAS-DRO:
    #
    # MSE(
    #     epsilon_theta(x_t,t),
    #     epsilon
    # )
    # ========================================================

    def loss(
        self,
        x_0: torch.Tensor,
        t: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        batch_size = (
            x_0.shape[0]
        )


        if t is None:

            t = torch.randint(

                low=0,

                high=
                    self.timesteps,

                size=(
                    batch_size,
                ),

                device=
                    x_0.device,
            )


        x_t, true_noise = (
            self.q_sample(

                x_0=
                    x_0,

                t=
                    t,
            )
        )


        predicted_noise = (
            self.denoiser(

                x_t,

                t,
            )
        )


        check_finite(
            predicted_noise,
            "predicted_noise",
        )


        loss = F.mse_loss(

            predicted_noise,

            true_noise,
        )


        check_finite(
            loss,
            "diffusion_loss",
        )


        return loss


    # ========================================================
    # Official GAS-DRO reverse mean
    #
    # mu_theta(x_t,t)
    #
    # =
    #
    # sqrt(1 / alpha_t)
    #
    # *
    #
    # [
    #   x_t
    #
    #   -
    #
    #   beta_t
    #   ------------------------
    #   sqrt(1-alpha_bar_t)
    #
    #   epsilon_theta(x_t,t)
    # ]
    #
    # IMPORTANT
    # ---------
    # We intentionally do NOT reconstruct x_0 through
    # 1 / sqrt(alpha_bar_t).
    # ========================================================

    def reverse_mean(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        predicted_noise = (
            self.denoiser(

                x_t,

                t,
            )
        )


        check_finite(
            predicted_noise,
            "reverse_predicted_noise",
        )


        alpha_t = (

            self.alphas[
                t
            ]

            .unsqueeze(1)
        )


        beta_t = (

            self.betas[
                t
            ]

            .unsqueeze(1)
        )


        alpha_bar_t = (

            self.alpha_bars[
                t
            ]

            .unsqueeze(1)
        )


        denominator = torch.sqrt(

            torch.clamp(

                1.0
                -
                alpha_bar_t,

                min=1e-20,
            )
        )


        mean = (

            torch.sqrt(
                1.0
                /
                alpha_t
            )

            *

            (
                x_t

                -

                (
                    beta_t
                    /
                    denominator
                )

                *

                predicted_noise
            )
        )


        check_finite(
            mean,
            "reverse_mean",
        )


        return (
            mean,
            predicted_noise,
        )


    # ========================================================
    # p_theta(x_{t-1} | x_t)
    # ========================================================

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        mean, predicted_noise = (
            self.reverse_mean(

                x_t=
                    x_t,

                t=
                    t,
            )
        )


        variance = (

            self.posterior_variance[
                t
            ]

            .unsqueeze(1)
        )


        # t=0 has zero posterior variance.
        # Sampling at t=0 is deterministic anyway.
        #
        # A tiny clamp keeps log-prob computations finite.
        variance = torch.clamp(

            variance,

            min=1e-20,
        )


        check_finite(
            variance,
            "reverse_variance",
        )


        return (

            mean,

            variance,

            predicted_noise,
        )


    # ========================================================
    # One reverse diffusion step
    # ========================================================

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t_scalar: int,
    ) -> torch.Tensor:

        batch_size = (
            x_t.shape[0]
        )


        t = torch.full(

            (
                batch_size,
            ),

            t_scalar,

            device=
                x_t.device,

            dtype=
                torch.long,
        )


        mean, variance, _ = (
            self.p_mean_variance(

                x_t=
                    x_t,

                t=
                    t,
            )
        )


        # ----------------------------------------------------
        # Official GAS-DRO:
        #
        # noise = 0 if t == 0
        # else random Gaussian noise
        # ----------------------------------------------------

        if t_scalar == 0:

            x_prev = (
                mean
            )

        else:

            noise = torch.randn_like(
                x_t
            )

            x_prev = (

                mean

                +

                torch.sqrt(
                    variance
                )

                *

                noise
            )


        check_finite(

            x_prev,

            f"reverse_sample_t"
            f"{t_scalar}",
        )


        return x_prev


    # ========================================================
    # Full reverse sampling
    # ========================================================

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | str,
        return_trajectory: bool = False,
    ):

        device = torch.device(
            device
        )

        self.eval()


        # x_T ~ N(0, I)

        x_t = torch.randn(

            num_samples,

            self.data_dim,

            device=device,
        )


        trajectory: List[
            torch.Tensor
        ] = []


        # ----------------------------------------------------
        # Existing adapter API:
        #
        # trajectory[:, 0] = initial x_T
        #
        # This is kept for backward compatibility with
        # earlier smoke scripts.
        #
        # The GAS-DRO official-style reference trajectory is
        # built separately inside gas_dro.py.
        # ----------------------------------------------------

        if return_trajectory:

            trajectory.append(

                x_t
                .detach()
                .cpu()
            )


        for t in reversed(
            range(
                self.timesteps
            )
        ):

            x_t = self.p_sample(

                x_t=
                    x_t,

                t_scalar=
                    t,
            )


            if return_trajectory:

                trajectory.append(

                    x_t
                    .detach()
                    .cpu()
                )


        if return_trajectory:

            trajectory_tensor = (
                torch.stack(

                    trajectory,

                    dim=1,
                )
            )


            return (

                x_t,

                trajectory_tensor,
            )


        return x_t


    # ========================================================
    # Reverse transition log probability
    #
    # Retained for diagnostics / backwards compatibility.
    #
    # Official-style GAS-DRO r_theta is implemented directly
    # in methods/gas_dro/gas_dro.py.
    # ========================================================

    def transition_log_prob(
        self,
        x_prev: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        mean, variance, _ = (
            self.p_mean_variance(

                x_t=
                    x_t,

                t=
                    t,
            )
        )


        log_variance = torch.log(
            variance
        )


        log_prob_per_dim = (

            -0.5

            *

            (
                math.log(
                    2.0
                    *
                    math.pi
                )

                +

                log_variance

                +

                (
                    x_prev
                    -
                    mean
                )
                ** 2

                /

                variance
            )
        )


        log_prob = (
            log_prob_per_dim
            .sum(
                dim=1
            )
        )


        check_finite(
            log_prob,
            "transition_log_prob",
        )


        return log_prob


    # ========================================================
    # Save checkpoint
    # ========================================================

    def save(
        self,
        path: str | Path,
        standardizer: Optional[
            VectorStandardizer
        ] = None,
        extra: Optional[
            dict
        ] = None,
    ) -> None:

        path = Path(
            path
        )


        path.parent.mkdir(

            parents=True,

            exist_ok=True,
        )


        checkpoint = {

            "model_state_dict":
                self.state_dict(),

            "config": {

                "data_dim":
                    self.data_dim,

                "timesteps":
                    self.timesteps,

                "beta_start":
                    self.beta_start,

                "beta_end":
                    self.beta_end,

                "hidden_dim":
                    self.hidden_dim,

                "time_dim":
                    self.time_dim,
            },

            "extra":
                extra
                or {},
        }


        if standardizer is not None:

            checkpoint[
                "standardizer"
            ] = (
                standardizer
                .state_dict()
            )


        torch.save(

            checkpoint,

            path,
        )


# ============================================================
# Official iteration-based diffusion training
#
# Matches the training semantics of the official GAS-DRO repo:
#
#     Adam
#     total_iteration updates
#     restart DataLoader iterator when exhausted
#
# Final official setting:
#
#     total_iterations = 7000
#     batch_size       = 64
#     lr               = 1e-4
#
# No gradient clipping in official implementation.
# ============================================================

def train_vector_diffusion_steps(
    model: VectorDiffusion,
    data: torch.Tensor,
    device: torch.device | str,
    total_iterations: int,
    batch_size: int,
    lr: float,
    standardizer: Optional[
        VectorStandardizer
    ] = None,
    grad_clip: Optional[
        float
    ] = None,
    verbose_every: int = 100,
) -> Tuple[
    List[float],
    VectorStandardizer,
]:

    """
    Train nominal diffusion using a fixed number of optimizer
    updates, matching the official GAS-DRO training protocol.
    """

    device = torch.device(
        device
    )


    if total_iterations <= 0:

        raise ValueError(
            "total_iterations must be positive."
        )


    if data.ndim != 2:

        raise ValueError(
            "data must have shape [N, D]."
        )


    if (
        data.shape[1]
        !=
        model.data_dim
    ):

        raise ValueError(

            f"Expected D="
            f"{model.data_dim}, "

            f"got D="
            f"{data.shape[1]}."
        )


    data = data.float()


    check_finite(
        data,
        "raw_training_data_steps",
    )


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    if standardizer is None:

        standardizer = (
            VectorStandardizer()
            .fit(
                data
            )
        )


    normalized_data = (
        standardizer
        .transform(
            data
        )
    )


    # --------------------------------------------------------
    # Official GAS-DRO-style dataset expansion
    #
    # The official implementation first repeats the nominal
    # data IN ORDER until it contains exactly:
    #
    #     batch_size * total_iterations
    #
    # samples, then uses:
    #
    #     shuffle=False
    #     drop_last=True
    #
    # In our 5D adapter, one row [X1, X2, X3, X4, Y]
    # corresponds to one complete diffusion sample.
    # --------------------------------------------------------

    total_samples = (
        batch_size
        *
        total_iterations
    )


    num_original_samples = (
        normalized_data.shape[0]
    )


    if num_original_samples == 0:

        raise ValueError(
            "Training dataset is empty."
        )


    repeat_times = (
        total_samples
        //
        num_original_samples
    ) + 1


    expanded_data = (
        normalized_data
        .repeat(
            repeat_times,
            1,
        )
        [:total_samples]
        .contiguous()
    )


    check_finite(
        expanded_data,
        "expanded_diffusion_training_data",
    )


    dataset = TensorDataset(
        expanded_data
    )


    loader = DataLoader(

        dataset,

        batch_size=
            batch_size,

        shuffle=
            False,

        drop_last=
            True,
    )


    if len(loader) != total_iterations:

        raise RuntimeError(
            f"Expected {total_iterations} batches, "
            f"got {len(loader)}."
        )


    data_iterator = iter(
        loader
    )


    model = model.to(
        device
    )


    # --------------------------------------------------------
    # Official diffusion optimizer:
    #
    # torch.optim.Adam(...)
    #
    # No AdamW / weight decay.
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=
            lr,
    )


    history: List[
        float
    ] = []


    print(
        "\n=========================================="
    )

    print(
        "Official-Style Vector Diffusion Training"
    )

    print(
        "=========================================="
    )


    print(
        f"Original samples   : "
        f"{num_original_samples}"
    )


    print(
        f"Expanded samples   : "
        f"{len(dataset)}"
    )


    print(
        f"DataLoader batches : "
        f"{len(loader)}"
    )


    print(
        f"Shuffle            : False"
    )


    print(
        f"Drop last          : True"
    )


    print(
        f"Dimensions         : "
        f"{model.data_dim}"
    )


    print(
        f"Timesteps          : "
        f"{model.timesteps}"
    )


    print(
        f"Batch size         : "
        f"{batch_size}"
    )


    print(
        f"Total iterations   : "
        f"{total_iterations}"
    )


    print(
        f"Learning rate      : "
        f"{lr}"
    )


    print(
        f"Gradient clipping  : "
        f"{grad_clip}"
    )


    print(
        f"Device             : "
        f"{device}"
    )


    print(
        "==========================================\n"
    )


    for iteration in range(

        1,

        total_iterations + 1,
    ):

        model.train()


        # ----------------------------------------------------
        # The expanded dataset contains exactly
        # total_iterations batches, so no iterator restart
        # is needed in the official-style path.
        # ----------------------------------------------------

        try:

            (batch,) = next(
                data_iterator
            )

        except StopIteration as exc:

            raise RuntimeError(
                "Diffusion DataLoader ended before "
                "total_iterations. "
                "Dataset expansion is misaligned."
            ) from exc


        batch = batch.to(

            device,

            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss = model.loss(
            batch
        )


        check_finite(
            loss,
            "official_diffusion_loss",
        )


        loss.backward()


        # Official code does not clip gradients.
        # This option exists only for diagnostics and should
        # remain None in the official final configuration.
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

                    "Non-finite gradient norm "
                    f"at iteration {iteration}."
                )


        optimizer.step()


        loss_value = (
            float(
                loss.detach()
                .cpu()
            )
        )


        history.append(
            loss_value
        )


        if (

            iteration == 1

            or

            iteration
            %
            verbose_every
            == 0

            or

            iteration
            ==
            total_iterations
        ):

            recent_window = (

                history[
                    -min(
                        verbose_every,
                        len(history),
                    ):
                ]
            )


            recent_avg = (

                sum(
                    recent_window
                )

                /

                len(
                    recent_window
                )
            )


            print(

                f"Iteration "
                f"{iteration:05d}/"
                f"{total_iterations} | "

                f"Loss: "
                f"{loss_value:.6f} | "

                f"Recent avg: "
                f"{recent_avg:.6f}"
            )


    print(
        "\nOfficial-Style Vector Diffusion "
        "Training Complete."
    )


    return (

        history,

        standardizer,
    )