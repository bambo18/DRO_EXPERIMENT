from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mlp import build_mlp
from methods.gas_dro.gas_dro import GasDROConfig, VectorGasDRO
from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    VectorStandardizer,
    train_vector_diffusion_steps,
)

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def check_params(name: str, fn, required: list[str]) -> None:
    sig = inspect.signature(fn)
    params = set(sig.parameters)
    missing = [p for p in required if p not in params]
    if missing:
        fail(
            f"{name} interface mismatch.\n"
            f"Missing parameters: {missing}\n"
            f"Current signature: {sig}"
        )
    print(f"[PASS] {name}: {sig}")


def read_xy(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.dtype.names is None:
        fail(f"{path}: CSV header not found.")

    missing = [
        c for c in FEATURE_COLUMNS + [TARGET_COLUMN]
        if c not in data.dtype.names
    ]
    if missing:
        fail(f"{path}: missing columns {missing}")

    X = np.column_stack([data[c] for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.asarray(data[TARGET_COLUMN], dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(X), torch.from_numpy(y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-training compatibility check for the OOD-validation "
            "ERM -> GAS-DRO pipeline."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPO_ROOT / "data" / "nominal" / "train.csv",
    )
    parser.add_argument(
        "--erm-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help=(
            "For a pre-OOD check, point to existing preliminary ERM checkpoints. "
            "After OOD selection, point to checkpoints/final_oodval."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("\n==========================================")
    print("OOD-VAL PIPELINE PREFLIGHT")
    print("==========================================")
    print("This script performs NO training.")
    print("This script loads NO OOD test data.")
    print("==========================================\n")

    # 1) GAS-DRO config fields
    required_config_fields = {
        "outer_epochs",
        "generator_inner_epochs",
        "predictor_inner_epochs",
        "batch_size",
        "batch_repeat",
        "generator_lr",
        "predictor_lr",
        "ppo_clip",
        "mu",
        "eta",
        "budget",
        "adjust_timesteps",
        "step_size",
        "discount_factor",
        "p_s0",
        "grad_clip",
        "verbose",
    }
    config_fields = set(GasDROConfig.__dataclass_fields__.keys())
    missing_config = sorted(required_config_fields - config_fields)
    if missing_config:
        fail(
            "GasDROConfig mismatch.\n"
            f"Missing fields: {missing_config}\n"
            f"Current fields: {sorted(config_fields)}"
        )
    print("[PASS] GasDROConfig fields match the tuning runner.")

    # 2) Constructor / fit signatures actually needed by run_gas_dro_tunable.py
    check_params(
        "VectorGasDRO.__init__",
        VectorGasDRO.__init__,
        [
            "nominal_diffusion",
            "standardizer",
            "predictor",
            "device",
            "config",
            "predictor_x_mean",
            "predictor_x_std",
        ],
    )

    check_params(
        "VectorGasDRO.fit",
        VectorGasDRO.fit,
        ["real_joint"],
    )

    check_params(
        "train_vector_diffusion_steps",
        train_vector_diffusion_steps,
        [
            "model",
            "data",
            "device",
            "total_iterations",
            "batch_size",
            "lr",
            "standardizer",
            "grad_clip",
            "verbose_every",
        ],
    )

    check_params(
        "VectorDiffusion.save",
        VectorDiffusion.save,
        ["path", "standardizer", "extra"],
    )

    # 3) Train CSV
    train_path = args.train_csv.resolve()
    if not train_path.exists():
        fail(f"Train CSV not found: {train_path}")

    X_train, y_train = read_xy(train_path)
    if X_train.shape[1] != 4 or y_train.shape[1] != 1:
        fail(
            f"Unexpected train shapes: X={tuple(X_train.shape)}, "
            f"y={tuple(y_train.shape)}"
        )
    print(
        f"[PASS] train.csv: X={tuple(X_train.shape)}, "
        f"y={tuple(y_train.shape)}"
    )

    # 4) Seed-matched ERM checkpoint
    ckpt_path = (
        args.erm_checkpoint_dir.resolve()
        / f"erm_seed{args.seed}_best.pt"
    )
    if not ckpt_path.exists():
        fail(f"ERM checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    for key in ("seed", "model_state_dict", "normalization"):
        if key not in ckpt:
            fail(f"{ckpt_path}: missing checkpoint key '{key}'")

    if int(ckpt["seed"]) != args.seed:
        fail(
            f"Seed mismatch: requested seed={args.seed}, "
            f"checkpoint seed={ckpt['seed']}"
        )

    norm = ckpt["normalization"]
    for key in ("x_mean", "x_std"):
        if key not in norm:
            fail(f"{ckpt_path}: normalization missing '{key}'")

    x_mean = torch.as_tensor(norm["x_mean"]).float().reshape(1, 4)
    x_std = torch.as_tensor(norm["x_std"]).float().reshape(1, 4)

    if not torch.isfinite(x_mean).all():
        fail("x_mean contains NaN/Inf.")
    if not torch.isfinite(x_std).all() or torch.any(x_std <= 0):
        fail("x_std is invalid.")

    predictor = build_mlp()
    predictor.load_state_dict(ckpt["model_state_dict"])

    print(
        f"[PASS] seed-matched ERM checkpoint: seed={args.seed}\n"
        f"       {ckpt_path}"
    )
    print("[PASS] ERM TRAIN normalization keys/shapes are compatible.")

    # 5) Cheap construction test. No diffusion training / GAS-DRO fit.
    real_joint = torch.cat([X_train, y_train], dim=1).float()

    standardizer = VectorStandardizer().fit(real_joint)

    nominal = VectorDiffusion(
        data_dim=5,
        timesteps=500,
        beta_start=0.1,
        beta_end=0.5,
        hidden_dim=128,
        time_dim=64,
    )

    config = GasDROConfig(
        outer_epochs=15,
        generator_inner_epochs=10,
        predictor_inner_epochs=2,
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
        verbose=False,
    )

    _ = VectorGasDRO(
        nominal_diffusion=nominal,
        standardizer=standardizer,
        predictor=predictor,
        device="cpu",
        config=config,
        predictor_x_mean=x_mean,
        predictor_x_std=x_std,
    )

    print("[PASS] VectorGasDRO object construction succeeds.")
    print("[PASS] Same-seed ERM -> GAS-DRO wiring is compatible.")
    print("[PASS] Predictor normalization is supplied from ERM TRAIN stats.")
    print("[PASS] No OOD validation/test data was used in this check.")

    print("\n==========================================")
    print("PREFLIGHT RESULT: PASS")
    print("==========================================\n")


if __name__ == "__main__":
    main()
