from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mlp import build_mlp
from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion_steps,
)
from methods.gas_dro.gas_dro import (
    GasDROConfig,
    VectorGasDRO,
)


FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_xy(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    missing = [
        col
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]
        if col not in data.dtype.names
    ]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    X = np.column_stack(
        [data[col] for col in FEATURE_COLUMNS]
    ).astype(np.float32)

    y = np.asarray(
        data[TARGET_COLUMN],
        dtype=np.float32,
    ).reshape(-1, 1)

    return torch.from_numpy(X), torch.from_numpy(y)


def build_joint(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X [N,4], got {tuple(X.shape)}")
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"Expected y [N,1], got {tuple(y.shape)}")
    if len(X) != len(y):
        raise ValueError("X and y length mismatch.")
    return torch.cat([X, y], dim=1).float()


@torch.no_grad()
def evaluate_predictor(
    model: torch.nn.Module,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    model.eval()

    x_mean = x_mean.to(device)
    x_std = x_std.to(device)

    total_sq = 0.0
    count = 0

    for start in range(0, len(X_raw), batch_size):
        end = min(start + batch_size, len(X_raw))

        xb = X_raw[start:end].to(device)
        yb = y[start:end].to(device)

        xb = (xb - x_mean) / x_std

        pred = model(xb)
        total_sq += torch.sum((pred - yb) ** 2).item()
        count += yb.numel()

    return total_sq / count


def load_erm_seed_checkpoint(
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ERM checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    ckpt_seed = int(checkpoint["seed"])
    if ckpt_seed != seed:
        raise ValueError(
            f"Seed mismatch: requested seed={seed}, "
            f"checkpoint seed={ckpt_seed}"
        )

    if "normalization" not in checkpoint:
        raise KeyError(
            "ERM checkpoint has no normalization statistics."
        )

    norm = checkpoint["normalization"]

    x_mean = norm["x_mean"].detach().float().reshape(1, 4)
    x_std = norm["x_std"].detach().float().reshape(1, 4)

    if not torch.isfinite(x_mean).all():
        raise ValueError("Checkpoint x_mean contains NaN/Inf.")
    if not torch.isfinite(x_std).all() or torch.any(x_std <= 0):
        raise ValueError("Checkpoint x_std is invalid.")

    predictor = build_mlp().to(device)
    predictor.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return predictor, checkpoint, x_mean, x_std


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing train.csv and test_id.csv",
    )

    parser.add_argument(
        "--erm-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "raw",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data_dir = args.data_dir.resolve()
    erm_checkpoint_path = (
        args.erm_checkpoint_dir.resolve()
        / f"erm_seed{args.seed}_best.pt"
    )

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test_id.csv"

    if not train_path.exists():
        parser.error(f"Missing: {train_path}")
    if not test_path.exists():
        parser.error(f"Missing: {test_path}")

    X_train, y_train = read_xy(train_path)
    X_test, y_test = read_xy(test_path)

    real_joint = build_joint(X_train, y_train)

    (
        predictor,
        erm_checkpoint,
        x_mean,
        x_std,
    ) = load_erm_seed_checkpoint(
        erm_checkpoint_path,
        seed=args.seed,
        device=device,
    )

    initial_id_mse = evaluate_predictor(
        predictor,
        X_test,
        y_test,
        x_mean,
        x_std,
        device=device,
    )

    print("\n==========================================")
    print("GAS-DRO / ERM CHECKPOINT WIRING")
    print("==========================================")
    print(f"Seed                  : {args.seed}")
    print(f"Device                : {device}")
    print(f"Train samples         : {len(X_train)}")
    print(f"ERM checkpoint        : {erm_checkpoint_path}")
    print(f"ERM best epoch        : {erm_checkpoint['best_epoch']}")
    print(f"ERM best val MSE      : {erm_checkpoint['best_val_mse']:.9f}")
    print(f"ERM ID test MSE       : {initial_id_mse:.9f}")
    print(f"X normalization mean  : {x_mean.flatten().tolist()}")
    print(f"X normalization std   : {x_std.flatten().tolist()}")
    print("Predictor input        : TRAIN-normalized X")
    print("Diffusion joint        : raw [X1,X2,X3,X4,Y]")
    print("==========================================\n")

    if device.type != "cuda":
        print(
            "[WARNING] GAS-DRO is running on CPU. "
            "The smoke/full modes are expected to be slow."
        )

    # --------------------------------------------------------
    # Smoke mode is ONLY an end-to-end wiring/numerical test.
    # It does not change the final experiment configuration.
    #
    # We intentionally keep the official diffusion schedule
    # T=500, beta=[0.1, 0.5] so the numerical path that caused
    # the alpha_bar underflow issue is still exercised.
    # --------------------------------------------------------
    is_smoke = True

    if is_smoke:
        smoke_n = 64
        training_joint = real_joint[:smoke_n].clone()
        diffusion_iterations = 200
        outer_epochs = 1
        generator_inner_epochs = 1
        predictor_inner_epochs = 1
        verbose_every = 50

        print("\n==========================================")
        print("CUDA GAS-DRO SMOKE TEST")
        print("==========================================")
        print("Purpose               : end-to-end test only")
        print(f"TRAIN subset           : first {smoke_n} samples")
        print(f"Diffusion iterations   : {diffusion_iterations}")
        print("Diffusion T             : 500 (official schedule retained)")
        print("Outer epochs            : 1")
        print("Generator inner epochs  : 1")
        print("Predictor inner epochs  : 1")
        print("Final experiment config : NOT changed")
        print("==========================================\n")
    else:
        training_joint = real_joint
        diffusion_iterations = 7000
        outer_epochs = 15
        generator_inner_epochs = 10
        predictor_inner_epochs = 2
        verbose_every = 100

    # --------------------------------------------------------
    # Official nominal diffusion architecture/schedule
    # --------------------------------------------------------
    nominal_diffusion = VectorDiffusion(
        data_dim=5,
        timesteps=500,
        beta_start=0.1,
        beta_end=0.5,
        hidden_dim=128,
        time_dim=64,
    ).to(device)

    print("\nTraining nominal diffusion...")
    start_time = time.time()

    diffusion_history, diffusion_standardizer = (
        train_vector_diffusion_steps(
            model=nominal_diffusion,
            data=training_joint,
            device=device,
            total_iterations=diffusion_iterations,
            batch_size=64,
            lr=1e-4,
            standardizer=None,
            grad_clip=None,
            verbose_every=verbose_every,
        )
    )

    diffusion_seconds = time.time() - start_time

    if is_smoke:
        nominal_filename = f"gas_dro_smoke_seed{args.seed}_nominal_diffusion.pt"
    else:
        nominal_filename = f"gas_dro_seed{args.seed}_nominal_diffusion.pt"

    nominal_path = (
        args.checkpoint_dir.resolve()
        / nominal_filename
    )

    nominal_diffusion.save(
        nominal_path,
        standardizer=diffusion_standardizer,
        extra={
            "seed": args.seed,
            "mode": "smoke",
            "iterations": diffusion_iterations,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "train_samples": int(training_joint.shape[0]),
        },
    )

    # --------------------------------------------------------
    # GAS-DRO configuration.
    # All algorithmic hyperparameters remain official.
    # Smoke mode only shortens loop counts.
    # --------------------------------------------------------
    config = GasDROConfig(
        outer_epochs=outer_epochs,
        generator_inner_epochs=generator_inner_epochs,
        predictor_inner_epochs=predictor_inner_epochs,
        batch_size=64,
        batch_repeat=4,
        generator_lr=1e-5,
        predictor_lr=1e-5,
        ppo_clip=0.4,
        mu=1.0,
        eta=0.1,
        budget=0.015,
        adjust_timesteps=15,
        step_size=2,
        discount_factor=0.05,
        p_s0=0.0,
        grad_clip=None,
        verbose=args.verbose,
    )

    gas_dro = VectorGasDRO(
        nominal_diffusion=nominal_diffusion,
        standardizer=diffusion_standardizer,
        predictor=predictor,
        device=device,
        config=config,
        predictor_x_mean=x_mean,
        predictor_x_std=x_std,
    )

    print("\nStarting GAS-DRO from seed-matched ERM predictor...")
    gas_start = time.time()

    output = gas_dro.fit(
        real_joint=training_joint,
    )

    gas_seconds = time.time() - gas_start

    final_predictor = output["predictor"]

    final_id_mse = evaluate_predictor(
        final_predictor,
        X_test,
        y_test,
        x_mean,
        x_std,
        device=device,
    )

    final_name = (
        f"gas_dro_smoke_seed{args.seed}_final.pt"
        if is_smoke
        else f"gas_dro_seed{args.seed}_final.pt"
    )

    final_checkpoint_path = (
        args.checkpoint_dir.resolve()
        / final_name
    )
    final_checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "seed": args.seed,
            "mode": "smoke",
            "erm_checkpoint": str(erm_checkpoint_path),
            "predictor_state_dict": final_predictor.state_dict(),
            "adversarial_diffusion_state_dict":
                output["adversarial_diffusion"].state_dict(),
            "nominal_diffusion_checkpoint": str(nominal_path),
            "predictor_normalization": {
                "x_mean": x_mean.cpu(),
                "x_std": x_std.cpu(),
            },
            "gas_dro_config": vars(config),
            "final_mu": float(output["mu"]),
            "history": output["history"],
            "initial_erm_id_mse": float(initial_id_mse),
            "final_gas_dro_id_mse": float(final_id_mse),
        },
        final_checkpoint_path,
    )

    summary = {
        "seed": args.seed,
        "mode": "smoke",
        "device": str(device),
        "erm_checkpoint": str(erm_checkpoint_path),
        "initial_erm_id_mse": float(initial_id_mse),
        "final_gas_dro_id_mse": float(final_id_mse),
        "nominal_diffusion_seconds": float(diffusion_seconds),
        "gas_dro_seconds": float(gas_seconds),
        "final_mu": float(output["mu"]),
        "gas_dro_checkpoint": str(final_checkpoint_path),
        "ood_evaluation": "pending",
    }

    result_filename = (
        f"gas_dro_smoke_seed{args.seed}.json"
        if is_smoke
        else f"gas_dro_seed{args.seed}.json"
    )

    result_path = (
        args.results_dir.resolve()
        / result_filename
    )
    save_json(result_path, summary)

    print("\n==========================================")
    print("GAS-DRO COMPLETE")
    print("==========================================")
    print(f"Initial ERM ID MSE     : {initial_id_mse:.9f}")
    print(f"Final GAS-DRO ID MSE   : {final_id_mse:.9f}")
    print(f"Diffusion time (sec)   : {diffusion_seconds:.2f}")
    print(f"GAS-DRO time (sec)     : {gas_seconds:.2f}")
    print(f"Checkpoint             : {final_checkpoint_path}")
    print(f"Result JSON            : {result_path}")
    print("OOD evaluation          : pending")
    print("==========================================\n")


if __name__ == "__main__":
    main()
