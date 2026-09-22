"""
GAS-DRO baseline adapted to 5D synthetic regression.

Joint endogenous variable:
    Z = [X1, X2, X3, X4, Y]

Official GAS-DRO structure preserved:
- fixed nominal diffusion theta_0
- adversarial diffusion theta initialized from theta_0
- fixed nominal z0 / trajectory
- a0 and r_theta
- PPO clipping
- JSM constraint
- dual update for mu
- Adam + StepLR
- alternating generator / predictor updates

Task-specific adaptations only:
- Carbon 28x28 blocks -> one 5D joint vector per diffusion sample
- DeepLSTM -> common regression MLP
- one predictor loss scalar per 5D joint sample
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    VectorStandardizer,
    check_finite,
)
from models.mlp import MLP

def split_joint_xy(
    joint: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split joint endogenous vector

        Z = [X1, X2, X3, X4, Y]

    into:

        X: [N, 4]
        y: [N, 1]
    """

    if joint.ndim != 2 or joint.shape[1] != 5:
        raise ValueError(
            f"Expected joint shape [N, 5], got {tuple(joint.shape)}"
        )

    x = joint[:, :4]
    y = joint[:, 4:5]

    return x, y

@dataclass
class GasDROConfig:
    # Official GAS-DRO settings
    outer_epochs: int = 15
    generator_inner_epochs: int = 10
    predictor_inner_epochs: int = 2

    batch_size: int = 64
    batch_repeat: int = 4

    generator_lr: float = 1e-5
    predictor_lr: float = 1e-5

    ppo_clip: float = 0.4

    mu: float = 1.0
    eta: float = 0.1
    budget: float = 0.015

    adjust_timesteps: int = 15

    step_size: int = 2
    discount_factor: float = 0.05

    # Official config contains P_S0=0.
    # The released GAS-DRO class stores it but does not use it.
    p_s0: float = 0.0

    # Official generator training does not use gradient clipping.
    grad_clip: Optional[float] = None

    verbose: bool = True


class VectorGasDRO:
    def __init__(
        self,
        nominal_diffusion: VectorDiffusion,
        standardizer: VectorStandardizer,
        predictor: MLP,
        device: str | torch.device,
        config: Optional[GasDROConfig] = None,
        predictor_x_mean: Optional[torch.Tensor] = None,
        predictor_x_std: Optional[torch.Tensor] = None,
    ):
        self.device = torch.device(device)
        self.config = config if config is not None else GasDROConfig()

        # theta_0: fixed nominal diffusion
        self.nominal_diffusion = copy.deepcopy(
            nominal_diffusion
        ).to(self.device)

        self.nominal_diffusion.eval()

        for p in self.nominal_diffusion.parameters():
            p.requires_grad_(False)

        # theta: adversarial diffusion initialized from theta_0
        self.adversarial_diffusion = copy.deepcopy(
            nominal_diffusion
        ).to(self.device)

        self.predictor = predictor.to(self.device)
        self.standardizer = standardizer

        # Predictor normalization is separate from the diffusion
        # standardizer. The diffusion models the raw joint Z=[X,Y],
        # while the common MLP must receive X normalized exactly as
        # in the seed-matched ERM checkpoint.
        self.predictor_x_mean = None
        self.predictor_x_std = None

        if predictor_x_mean is not None or predictor_x_std is not None:
            if predictor_x_mean is None or predictor_x_std is None:
                raise ValueError(
                    "predictor_x_mean and predictor_x_std must be provided together."
                )

            predictor_x_mean = predictor_x_mean.detach().float().reshape(1, 4)
            predictor_x_std = predictor_x_std.detach().float().reshape(1, 4)

            if not torch.isfinite(predictor_x_mean).all():
                raise ValueError("predictor_x_mean contains NaN/Inf.")
            if not torch.isfinite(predictor_x_std).all():
                raise ValueError("predictor_x_std contains NaN/Inf.")
            if torch.any(predictor_x_std <= 0):
                raise ValueError("predictor_x_std must be strictly positive.")

            self.predictor_x_mean = predictor_x_mean.cpu()
            self.predictor_x_std = predictor_x_std.cpu()

        self.mu = float(self.config.mu)

        self.history: Dict[str, List[float]] = {
            "ppo": [],
            "jsm": [],
            "generator_objective": [],
            "mu": [],
            "generated_mean_mse": [],
            "generated_max_mse": [],
            "ratio_mean": [],
            "ratio_min": [],
            "ratio_max": [],
        }

    @property
    def selected_timesteps(self) -> List[int]:
        """
        Official main.py passes [0, 1, ..., T-1]
        and GAS-DRO keeps the first ADJUST_TIMESTEPS.
        """

        k = min(
            self.config.adjust_timesteps,
            self.nominal_diffusion.timesteps,
        )

        return list(range(k))

    def _normalize_predictor_x(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the TRAIN-only normalization stored in the ERM checkpoint."""

        if self.predictor_x_mean is None or self.predictor_x_std is None:
            return x

        mean = self.predictor_x_mean.to(x.device)
        std = self.predictor_x_std.to(x.device)

        x_norm = (x - mean) / std

        check_finite(
            x_norm,
            "predictor_normalized_x",
        )

        return x_norm

    def _s0_iterations(
        self,
        num_real_samples: int,
    ) -> int:
        """
        Official Carbon code:

            ceil(
                len(s0)
                /
                (PIC_SIZE^2 * BATCH_SIZE)
            )
            *
            BATCH_REPEAT

        Carbon packs PIC_SIZE^2 scalar observations into
        one diffusion image.

        In our experiment, one row is already one complete
        5D diffusion sample.

        Therefore the task-adapted equivalent is:

            ceil(N / BATCH_SIZE)
            *
            BATCH_REPEAT
        """

        if num_real_samples <= 0:
            raise ValueError(
                "real_joint must contain at least one sample."
            )

        return (
            math.ceil(
                num_real_samples
                /
                self.config.batch_size
            )
            *
            self.config.batch_repeat
        )

    # ========================================================
    # Official reverse mean
    # ========================================================

    def mu_t(
        self,
        diffusion: VectorDiffusion,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:

        t_tensor = torch.full(
            (x_t.shape[0],),
            t,
            dtype=torch.long,
            device=x_t.device,
        )

        epsilon = diffusion.denoiser(
            x_t,
            t_tensor,
        )

        alpha_t = diffusion.alphas[t]
        beta_t = diffusion.betas[t]
        alpha_bar_t = diffusion.alpha_bars[t]

        mu = (
            torch.sqrt(
                1.0 / alpha_t
            )
            *
            (
                x_t
                -
                beta_t
                /
                torch.sqrt(
                    1.0 - alpha_bar_t
                )
                *
                epsilon
            )
        )

        check_finite(
            mu,
            f"mu_t_{t}",
        )

        return mu

    # ========================================================
    # Fixed nominal trajectory
    # ========================================================

    @torch.no_grad()
    def sample_reference_trajectory(
        self,
        num_logical_batches: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generate the fixed nominal z0 and trajectories.

        Official GAS-DRO creates one generated trajectory
        batch per s0 inner iteration.

        We retain only:
            x_0 ... x_(K-1)

        where:
            K = ADJUST_TIMESTEPS

        The official implementation stores all T states, but
        the GAS-DRO objective only reads these first K states
        after flipping the trajectory.

        This reduces memory only. It does not change the
        GAS-DRO objective.
        """

        if num_logical_batches <= 0:
            raise ValueError(
                "num_logical_batches must be positive."
            )

        diffusion = self.nominal_diffusion
        diffusion.eval()

        selected = self.selected_timesteps
        selected_set = set(selected)

        all_z0: List[torch.Tensor] = []
        all_trajectories: List[torch.Tensor] = []

        for _ in range(
            num_logical_batches
        ):

            x = torch.randn(
                self.config.batch_size,
                diffusion.data_dim,
                device=self.device,
            )

            saved_by_t: Dict[
                int,
                torch.Tensor,
            ] = {}

            # Official sampling:
            #
            # T-1 -> ... -> 0
            #
            # After reverse step t,
            # x corresponds to x_t.
            for t in reversed(
                range(
                    diffusion.timesteps
                )
            ):

                x = diffusion.p_sample(
                    x_t=x,
                    t_scalar=t,
                )

                if t in selected_set:

                    saved_by_t[t] = (
                        x.detach().clone()
                    )

            selected_trajectory = torch.stack(
                [
                    saved_by_t[t]
                    for t in selected
                ],
                dim=1,
            )

            normalized_z0 = (
                selected_trajectory[
                    :,
                    0,
                    :
                ]
            )

            z0 = (
                self.standardizer
                .inverse_transform(
                    normalized_z0
                )
            )

            check_finite(
                selected_trajectory,
                "reference_trajectory_batch",
            )

            check_finite(
                z0,
                "reference_z0_batch",
            )

            # Keep reference data on CPU between updates.
            all_trajectories.append(
                selected_trajectory.cpu()
            )

            all_z0.append(
                z0.cpu()
            )

        reference_joint = torch.cat(
            all_z0,
            dim=0,
        )

        reference_trajectory = torch.cat(
            all_trajectories,
            dim=0,
        )

        check_finite(
            reference_joint,
            "reference_joint",
        )

        check_finite(
            reference_trajectory,
            "reference_trajectory",
        )

        return (
            reference_joint,
            reference_trajectory,
        )

    # ========================================================
    # a0
    # ========================================================

    @torch.no_grad()
    def build_a0(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official:

            a0_t =
                (z_{t-1} - mu_theta0(z_t,t))^2

        Carbon later reduces a spatial dimension to align
        with its sequence predictor losses.

        In our pointwise 5D task one predictor loss belongs
        to one entire joint vector, so all five dimensions
        are summed.
        """

        all_chunks: List[
            torch.Tensor
        ] = []

        for start in range(
            0,
            trajectory.shape[0],
            self.config.batch_size,
        ):

            end = min(
                start
                +
                self.config.batch_size,

                trajectory.shape[0],
            )

            traj_batch = (
                trajectory[
                    start:end
                ]
                .to(
                    self.device
                )
            )

            batch_a0: List[
                torch.Tensor
            ] = []

            for idx, t in enumerate(
                self.selected_timesteps
            ):

                z_t = (
                    traj_batch[
                        :,
                        idx,
                        :
                    ]
                )

                if t == 0:

                    z_t_minus_1 = z_t

                else:

                    z_t_minus_1 = (
                        traj_batch[
                            :,
                            idx - 1,
                            :
                        ]
                    )

                mu_nominal = self.mu_t(
                    diffusion=
                        self.nominal_diffusion,

                    x_t=
                        z_t,

                    t=
                        t,
                )

                a0_t = (
                    (
                        z_t_minus_1
                        -
                        mu_nominal
                    )
                    ** 2
                ).sum(
                    dim=1
                )

                batch_a0.append(
                    a0_t
                )

            all_chunks.append(
                torch.stack(
                    batch_a0,
                    dim=1,
                )
                .detach()
                .cpu()
            )

        a0_tensor = torch.cat(
            all_chunks,
            dim=0,
        )

        check_finite(
            a0_tensor,
            "a0_tensor",
        )

        return a0_tensor

    # ========================================================
    # r_theta
    # ========================================================

    def r_theta(
        self,
        trajectory: torch.Tensor,
        a0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official:

            a_diff +=
                (a_theta - a0)
                /
                (2 * sigma_t^2)

            r_theta =
                exp(-a_diff)

        idx == 0 is skipped exactly as in the
        released GAS-DRO implementation.
        """

        a_diff = torch.zeros(
            trajectory.shape[0],
            device=self.device,
        )

        for idx, t in enumerate(
            self.selected_timesteps
        ):

            if idx == 0:
                continue

            z_t = (
                trajectory[
                    :,
                    idx,
                    :
                ]
            )

            z_t_minus_1 = (
                trajectory[
                    :,
                    idx - 1,
                    :
                ]
            )

            mu_theta = self.mu_t(
                diffusion=
                    self.adversarial_diffusion,

                x_t=
                    z_t,

                t=
                    t,
            )

            a = (
                (
                    z_t_minus_1
                    -
                    mu_theta
                )
                ** 2
            ).sum(
                dim=1
            )

            a0_t = (
                a0[
                    :,
                    idx
                ]
            )

            sigma_sq = (
                self.adversarial_diffusion
                .posterior_variance[t]
            )

            sigma_value = float(
                sigma_sq
                .detach()
                .cpu()
            )

            if sigma_value <= 0.0:

                raise RuntimeError(
                    f"Invalid sigma^2 "
                    f"at timestep {t}: "
                    f"{sigma_value}"
                )

            a_diff = (
                a_diff
                +
                (
                    a
                    -
                    a0_t.detach()
                )
                /
                (
                    2.0
                    *
                    sigma_sq
                )
            )

        check_finite(
            a_diff,
            "a_diff",
        )

        ratio = torch.exp(
            -a_diff
        )

        check_finite(
            ratio,
            "r_theta",
        )

        return ratio

    # ========================================================
    # PPO
    # ========================================================

    def ppo(
        self,
        ratio: torch.Tensor,
        predictor_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official PPO objective.
        Predictor loss h_w is detached.
        """

        loss_hw = (
            predictor_loss.detach()
        )

        unclipped = (
            ratio
            *
            loss_hw
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0
            -
            self.config.ppo_clip,

            1.0
            +
            self.config.ppo_clip,
        )

        clipped = (
            clipped_ratio
            *
            loss_hw
        )

        ppo_loss = torch.min(
            unclipped,
            clipped,
        ).mean()

        check_finite(
            ppo_loss,
            "ppo_loss",
        )

        return ppo_loss

    # ========================================================
    # JSM
    # ========================================================

    def jsm_loss(
        self,
        normalized_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official:

            loss_fn(
                x=s0,
                T_prime=ADJUST_TIMESTEPS
            )

        This samples:
            t ~ Uniform{0,...,T'-1}
        """

        t = torch.randint(
            low=0,
            high=len(
                self.selected_timesteps
            ),
            size=(
                normalized_real.shape[0],
            ),
            device=self.device,
        )

        loss = (
            self.adversarial_diffusion
            .loss(
                x_0=
                    normalized_real,

                t=
                    t,
            )
        )

        check_finite(
            loss,
            "jsm_loss",
        )

        return loss

    # ========================================================
    # One adversarial generator update
    # ========================================================

    def generator_update(
        self,
        trajectory_batch: torch.Tensor,
        a0_batch: torch.Tensor,
        predictor_loss_batch: torch.Tensor,
        real_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:

        ratio = self.r_theta(
            trajectory=
                trajectory_batch,

            a0=
                a0_batch,
        )

        ppo_loss = self.ppo(
            ratio=
                ratio,

            predictor_loss=
                predictor_loss_batch,
        )

        normalized_real = (
            self.standardizer
            .transform(
                real_batch
            )
        )

        jsm = self.jsm_loss(
            normalized_real
        )

        # Official GAS-DRO:
        #
        # dro_inner_loss =
        #     -(PPO - MU * JSM)

        dro_inner_loss = -(
            ppo_loss
            -
            self.mu
            *
            jsm
        )

        check_finite(
            dro_inner_loss,
            "dro_inner_loss",
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        dro_inner_loss.backward()

        # Official released code has NO generator
        # gradient clipping.
        if self.config.grad_clip is not None:

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    self.adversarial_diffusion
                    .parameters(),

                    max_norm=
                        self.config.grad_clip,
                )
            )

            if not torch.isfinite(
                grad_norm
            ):

                raise RuntimeError(
                    "Non-finite generator gradient."
                )

            grad_value = float(
                grad_norm
                .detach()
                .cpu()
            )

        else:

            grad_value = float(
                "nan"
            )

        optimizer.step()

        return {
            "ppo":
                float(
                    ppo_loss
                    .detach()
                    .cpu()
                ),

            "jsm":
                float(
                    jsm
                    .detach()
                    .cpu()
                ),

            "objective":
                float(
                    dro_inner_loss
                    .detach()
                    .cpu()
                ),

            "ratio_mean":
                float(
                    ratio.mean()
                    .detach()
                    .cpu()
                ),

            "ratio_min":
                float(
                    ratio.min()
                    .detach()
                    .cpu()
                ),

            "ratio_max":
                float(
                    ratio.max()
                    .detach()
                    .cpu()
                ),

            "grad_norm":
                grad_value,
        }

    # ========================================================
    # Predictor loss on fixed z0
    # ========================================================

    def _reference_predictor_loss(
        self,
        reference_joint: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official GAS-DRO recomputes predictor loss on
        the SAME fixed z0 every outer epoch.

        Official z0 loader is sequential:
            shuffle=False
        """

        x, y = split_joint_xy(
            reference_joint
        )

        losses: List[
            torch.Tensor
        ] = []

        self.predictor.eval()

        with torch.no_grad():

            for start in range(
                0,
                x.shape[0],
                self.config.batch_size,
            ):

                end = (
                    start
                    +
                    self.config.batch_size
                )

                batch_x = (
                    x[
                        start:end
                    ]
                    .to(
                        self.device
                    )
                )

                batch_y = (
                    y[
                        start:end
                    ]
                    .to(
                        self.device
                    )
                )

                batch_x = self._normalize_predictor_x(
                    batch_x
                )

                prediction = (
                    self.predictor(
                        batch_x
                    )
                )

                per_sample = (
                    (
                        prediction
                        -
                        batch_y
                    )
                    ** 2
                ).reshape(
                    batch_x.shape[0],
                    -1,
                ).mean(
                    dim=1
                )

                losses.append(
                    per_sample.cpu()
                )

        reference_loss = torch.cat(
            losses,
            dim=0,
        )

        check_finite(
            reference_loss,
            "reference_predictor_loss",
        )

        return reference_loss

    # ========================================================
    # s0 data batch
    # ========================================================

    def _real_batch_for_index(
        self,
        real_joint: torch.Tensor,
        batch_i: int,
    ) -> torch.Tensor:
        """
        Official dfu_dataset repeats s0 to exactly:

            BATCH_SIZE * s0_iteration

        samples and iterates without shuffle.

        We reproduce that with cyclic indexing instead
        of physically duplicating the whole dataset.
        """

        n = real_joint.shape[0]

        start = (
            batch_i
            *
            self.config.batch_size
        )

        indices = (
            torch.arange(
                start,
                start
                +
                self.config.batch_size,
                dtype=torch.long,
            )
            %
            n
        )

        return (
            real_joint[
                indices
            ]
            .to(
                self.device
            )
        )

    # ========================================================
    # Generate S_theta
    # ========================================================

    @torch.no_grad()
    def generate_adversarial_samples(
        self,
        num_batches: int = 1,
    ) -> torch.Tensor:
        """
        Official outer loop calls:

            gen_s_theta(
                gen_iterations=1
            )

        Therefore fit() keeps:
            num_batches = 1
        """

        if num_batches <= 0:
            raise ValueError(
                "num_batches must be positive."
            )

        self.adversarial_diffusion.eval()

        generated: List[
            torch.Tensor
        ] = []

        for _ in range(
            num_batches
        ):

            normalized_samples = (
                self.adversarial_diffusion
                .sample(
                    num_samples=
                        self.config.batch_size,

                    device=
                        self.device,

                    return_trajectory=
                        False,
                )
            )

            samples = (
                self.standardizer
                .inverse_transform(
                    normalized_samples
                )
            )

            check_finite(
                samples,
                "s_theta_batch",
            )

            generated.append(
                samples.cpu()
            )

        s_theta = torch.cat(
            generated,
            dim=0,
        )

        check_finite(
            s_theta,
            "s_theta",
        )

        return s_theta

    # ========================================================
    # Per-sample regression loss
    # ========================================================

    def _joint_mse(
        self,
        joint: torch.Tensor,
    ) -> torch.Tensor:

        x, y = split_joint_xy(
            joint
        )

        x = x.to(
            self.device
        )

        y = y.to(
            self.device
        )

        x = self._normalize_predictor_x(
            x
        )

        self.predictor.eval()

        with torch.no_grad():

            prediction = (
                self.predictor(
                    x
                )
            )

            losses = (
                (
                    prediction
                    -
                    y
                )
                ** 2
            ).reshape(
                x.shape[0],
                -1,
            ).mean(
                dim=1
            )

        check_finite(
            losses,
            "joint_predictor_loss",
        )

        return losses.cpu()

    # ========================================================
    # Predictor update
    # ========================================================

    def update_predictor(
        self,
        adversarial_joint: torch.Tensor,
    ) -> List[float]:
        """
        Official GAS-DRO train_ml():

        - recreates Adam each call
        - recreates StepLR each call
        - shuffle=False
        - drop_last=True
        """

        x, y = split_joint_xy(
            adversarial_joint
            .detach()
            .cpu()
        )

        loader = DataLoader(
            TensorDataset(
                x,
                y,
            ),
            batch_size=
                self.config.batch_size,

            shuffle=
                False,

            drop_last=
                True,
        )

        if len(loader) == 0:

            raise RuntimeError(
                "Generated S_theta is smaller "
                "than one GAS-DRO batch."
            )

        optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=
                self.config.predictor_lr,
        )

        scheduler = StepLR(
            optimizer,
            step_size=
                self.config.step_size,

            gamma=
                self.config.discount_factor,
        )

        criterion = nn.MSELoss()

        history: List[
            float
        ] = []

        self.predictor.train()

        for epoch in range(
            1,
            self.config.predictor_inner_epochs
            +
            1,
        ):

            total_loss = 0.0
            count = 0

            for batch_x, batch_y in loader:

                batch_x = batch_x.to(
                    self.device
                )

                batch_y = batch_y.to(
                    self.device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                batch_x = self._normalize_predictor_x(
                    batch_x
                )

                prediction = (
                    self.predictor(
                        batch_x
                    )
                )

                loss = criterion(
                    prediction,
                    batch_y,
                )

                check_finite(
                    loss,
                    "predictor_inner_loss",
                )

                loss.backward()

                optimizer.step()

                n = (
                    batch_x.shape[0]
                )

                total_loss += (
                    loss.item()
                    *
                    n
                )

                count += n

            scheduler.step()

            avg_loss = (
                total_loss
                /
                max(
                    count,
                    1,
                )
            )

            history.append(
                avg_loss
            )

            if self.config.verbose:

                print(
                    f"Predictor "
                    f"{epoch}/"
                    f"{self.config.predictor_inner_epochs} "
                    f"| MSE="
                    f"{avg_loss:.6f}"
                )

        return history

    # ========================================================
    # Full GAS-DRO
    # ========================================================

    def fit(
        self,
        real_joint: torch.Tensor,
        outer_epoch_callback: Optional[
            Callable[[int, "VectorGasDRO"], None]
        ] = None,
    ):

        real_joint = (
            real_joint
            .detach()
            .cpu()
            .float()
        )

        check_finite(
            real_joint,
            "real_joint",
        )

        # ----------------------------------------------------
        # Official-style s0_iteration
        # ----------------------------------------------------

        s0_iterations = (
            self._s0_iterations(
                real_joint.shape[0]
            )
        )

        reference_count = (
            s0_iterations
            *
            self.config.batch_size
        )

        print(
            "\nGenerating fixed nominal "
            "reference trajectory..."
        )

        print(
            f"Official-style s0 iterations : "
            f"{s0_iterations} "
            f"(ceil("
            f"{real_joint.shape[0]}/"
            f"{self.config.batch_size}) "
            f"* BATCH_REPEAT "
            f"{self.config.batch_repeat})"
        )

        (
            reference_joint,
            reference_trajectory,
        ) = (
            self.sample_reference_trajectory(
                num_logical_batches=
                    s0_iterations
            )
        )

        a0 = self.build_a0(
            reference_trajectory
        )

        print(
            "Reference samples    :",
            reference_joint.shape,
        )

        print(
            "Reference trajectory :",
            reference_trajectory.shape,
        )

        print(
            "a0                   :",
            a0.shape,
        )

        if (
            reference_joint.shape[0]
            !=
            reference_count
        ):

            raise RuntimeError(
                "Reference sample count does not "
                "match logical batch count."
            )

        # ----------------------------------------------------
        # Generator optimizer
        # ----------------------------------------------------

        generator_optimizer = (
            torch.optim.Adam(
                self.adversarial_diffusion
                .parameters(),

                lr=
                    self.config.generator_lr,
            )
        )

        generator_scheduler = StepLR(
            generator_optimizer,
            step_size=
                self.config.step_size,

            gamma=
                self.config.discount_factor,
        )

        print(
            "\n=========================================="
        )

        print(
            "VECTOR GAS-DRO TRAINING"
        )

        print(
            "=========================================="
        )

        print(
            f"Outer epochs           : "
            f"{self.config.outer_epochs}"
        )

        print(
            f"Generator inner epochs : "
            f"{self.config.generator_inner_epochs}"
        )

        print(
            f"s0 iterations / inner  : "
            f"{s0_iterations}"
        )

        print(
            f"BATCH_REPEAT           : "
            f"{self.config.batch_repeat}"
        )

        print(
            f"Predictor inner epochs : "
            f"{self.config.predictor_inner_epochs}"
        )

        print(
            f"Batch size             : "
            f"{self.config.batch_size}"
        )

        print(
            f"Selected timesteps     : "
            f"{self.selected_timesteps}"
        )

        print(
            f"Initial mu             : "
            f"{self.mu:.6f}"
        )

        print(
            f"Budget                 : "
            f"{self.config.budget:.6f}"
        )

        print(
            f"Generator LR           : "
            f"{self.config.generator_lr}"
        )

        print(
            f"Gradient clipping      : "
            f"{self.config.grad_clip}"
        )

        print(
            "==========================================\n"
        )

        # ----------------------------------------------------
        # Outer loop
        # ----------------------------------------------------

        for outer in range(
            1,
            self.config.outer_epochs
            +
            1,
        ):

            print(
                "\n------------------------------------------"
            )

            print(
                f"GAS-DRO Outer "
                f"{outer}/"
                f"{self.config.outer_epochs}"
            )

            print(
                "------------------------------------------"
            )

            # Official:
            # recompute loss on SAME fixed z0.
            reference_loss = (
                self._reference_predictor_loss(
                    reference_joint
                )
            )

            print(
                "Reference predictor loss | "
                f"mean="
                f"{reference_loss.mean().item():.6f} | "
                f"max="
                f"{reference_loss.max().item():.6f}"
            )

            self.adversarial_diffusion.train()

            # ------------------------------------------------
            # Generator inner epochs
            # ------------------------------------------------

            for inner_epoch in range(
                1,
                self.config.generator_inner_epochs
                +
                1,
            ):

                jsm_total = 0.0

                # Official:
                # exactly s0_iteration updates.
                for batch_i in range(
                    s0_iterations
                ):

                    start = (
                        batch_i
                        *
                        self.config.batch_size
                    )

                    end = (
                        start
                        +
                        self.config.batch_size
                    )

                    trajectory_batch = (
                        reference_trajectory[
                            start:end
                        ]
                        .to(
                            self.device
                        )
                    )

                    a0_batch = (
                        a0[
                            start:end
                        ]
                        .to(
                            self.device
                        )
                    )

                    loss_batch = (
                        reference_loss[
                            start:end
                        ]
                        .to(
                            self.device
                        )
                    )

                    real_batch = (
                        self._real_batch_for_index(
                            real_joint=
                                real_joint,

                            batch_i=
                                batch_i,
                        )
                    )

                    stats = (
                        self.generator_update(
                            trajectory_batch=
                                trajectory_batch,

                            a0_batch=
                                a0_batch,

                            predictor_loss_batch=
                                loss_batch,

                            real_batch=
                                real_batch,

                            optimizer=
                                generator_optimizer,
                        )
                    )

                    jsm_total += (
                        stats["jsm"]
                    )

                    self.history[
                        "ppo"
                    ].append(
                        stats["ppo"]
                    )

                    self.history[
                        "jsm"
                    ].append(
                        stats["jsm"]
                    )

                    self.history[
                        "generator_objective"
                    ].append(
                        stats[
                            "objective"
                        ]
                    )

                    self.history[
                        "ratio_mean"
                    ].append(
                        stats[
                            "ratio_mean"
                        ]
                    )

                    self.history[
                        "ratio_min"
                    ].append(
                        stats[
                            "ratio_min"
                        ]
                    )

                    self.history[
                        "ratio_max"
                    ].append(
                        stats[
                            "ratio_max"
                        ]
                    )

                    if self.config.verbose:

                        print(
                            f"[Inner "
                            f"{inner_epoch}/"
                            f"{self.config.generator_inner_epochs} "
                            f"| Batch "
                            f"{batch_i + 1}/"
                            f"{s0_iterations}] "
                            f"PPO="
                            f"{stats['ppo']:.6f} | "
                            f"JSM="
                            f"{stats['jsm']:.6f} | "
                            f"Obj="
                            f"{stats['objective']:.6f} | "
                            f"Ratio="
                            f"{stats['ratio_mean']:.4f} "
                            f"["
                            f"{stats['ratio_min']:.4f}, "
                            f"{stats['ratio_max']:.4f}"
                            f"]"
                        )

                # --------------------------------------------
                # Official dual update
                # --------------------------------------------

                jsm_avg = (
                    jsm_total
                    /
                    s0_iterations
                )

                self.mu = (
                    self.mu
                    +
                    self.config.eta
                    *
                    (
                        jsm_avg
                        -
                        self.config.budget
                    )
                )

                self.history[
                    "mu"
                ].append(
                    self.mu
                )

                # Official scheduler step is here.
                generator_scheduler.step()

                current_lr = (
                    generator_scheduler
                    .get_last_lr()[0]
                )

                print(
                    f"Inner epoch "
                    f"{inner_epoch} complete | "
                    f"JSM avg="
                    f"{jsm_avg:.6f} | "
                    f"mu="
                    f"{self.mu:.6f} | "
                    f"generator_lr="
                    f"{current_lr:.8f}"
                )

            # ------------------------------------------------
            # S_theta
            #
            # Official released code:
            # gen_iterations = 1
            # ------------------------------------------------

            s_theta = (
                self.generate_adversarial_samples(
                    num_batches=1
                )
            )

            s_theta_loss = (
                self._joint_mse(
                    s_theta
                )
            )

            mean_mse = (
                s_theta_loss
                .mean()
                .item()
            )

            max_mse = (
                s_theta_loss
                .max()
                .item()
            )

            self.history[
                "generated_mean_mse"
            ].append(
                mean_mse
            )

            self.history[
                "generated_max_mse"
            ].append(
                max_mse
            )

            print(
                "\nS_theta before predictor update:"
            )

            print(
                f"  Samples  = "
                f"{s_theta.shape[0]}"
            )

            print(
                f"  Mean MSE = "
                f"{mean_mse:.6f}"
            )

            print(
                f"  Max MSE  = "
                f"{max_mse:.6f}"
            )

            print(
                "\nUpdating predictor on S_theta..."
            )

            self.update_predictor(
                s_theta
            )

            # ------------------------------------------------
            # Optional model-selection callback.
            # This is intentionally called only AFTER one full
            # GAS-DRO outer epoch (generator + predictor update).
            # The callback may evaluate OOD validation data and
            # save the best checkpoint, but it must not backprop
            # through the validation set.
            # ------------------------------------------------
            if outer_epoch_callback is not None:
                outer_epoch_callback(
                    outer,
                    self,
                )

        print(
            "\n=========================================="
        )

        print(
            "VECTOR GAS-DRO TRAINING COMPLETE"
        )

        print(
            "==========================================\n"
        )

        return {
            "predictor":
                self.predictor,

            "adversarial_diffusion":
                self.adversarial_diffusion,

            "mu":
                self.mu,

            "history":
                self.history,
        }