from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Tuple

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
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.dtype.names is None:
        raise ValueError(f"{path}: CSV header not found.")
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in data.dtype.names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    X = np.column_stack([data[c] for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.asarray(data[TARGET_COLUMN], dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(X), torch.from_numpy(y)


def build_joint(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X [N,4], got {tuple(X.shape)}")
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"Expected y [N,1], got {tuple(y.shape)}")
    return torch.cat([X, y], dim=1).float()


def load_erm_seed_checkpoint(path: Path, seed: int, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(
            f"Seed-matched final ERM checkpoint not found: {path}"
        )

    checkpoint = torch.load(path, map_location=device)

    if int(checkpoint["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch: requested {seed}, checkpoint has {checkpoint['seed']}"
        )

    norm = checkpoint["normalization"]
    x_mean = torch.as_tensor(norm["x_mean"], dtype=torch.float32).reshape(1, 4)
    x_std = torch.as_tensor(norm["x_std"], dtype=torch.float32).reshape(1, 4)

    predictor = build_mlp().to(device)
    predictor.load_state_dict(checkpoint["model_state_dict"])

    return predictor, checkpoint, x_mean, x_std


def safe_tag(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ONE GAS-DRO OOD-validation tuning candidate. "
            "Uses train.csv only for optimization and a seed-matched final ERM checkpoint. "
            "OOD validation/test are not loaded here."
        )
    )

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument(
        "--erm-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "final_oodval",
        help="Must contain erm_seed{seed}_best.pt selected by OOD validation.",
    )
    parser.add_argument("--run-tag", required=True)

    # Official defaults. Supply alternate values explicitly during the sweep.
    parser.add_argument("--generator-lr", type=float, default=1e-5)
    parser.add_argument("--predictor-lr", type=float, default=1e-5)
    parser.add_argument("--ppo-clip", type=float, default=0.4)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--budget", type=float, default=0.015)

    parser.add_argument("--outer-epochs", type=int, default=15)
    parser.add_argument("--generator-inner-epochs", type=int, default=10)
    parser.add_argument("--predictor-inner-epochs", type=int, default=2)

    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "oodval_candidates" / "gas_dro",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "oodval_candidates" / "gas_dro",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_path = args.train_csv.resolve()
    if not train_path.exists():
        parser.error(f"Missing train CSV: {train_path}")

    # No validation or test CSV is accepted here by design.
    X_train, y_train = read_xy(train_path)
    real_joint = build_joint(X_train, y_train)

    erm_path = (
        args.erm_checkpoint_dir.resolve()
        / f"erm_seed{args.seed}_best.pt"
    )

    predictor, erm_checkpoint, x_mean, x_std = load_erm_seed_checkpoint(
        erm_path,
        seed=args.seed,
        device=device,
    )

    print("\n==========================================")
    print("GAS-DRO OOD-VAL CANDIDATE TRAINING")
    print("==========================================")
    print(f"Seed              : {args.seed}")
    print(f"Run tag           : {args.run_tag}")
    print(f"Train samples     : {len(X_train)}")
    print(f"ERM checkpoint    : {erm_path}")
    print(f"ERM seed          : {erm_checkpoint['seed']}")
    print("Normalization     : restored from seed-matched ERM (TRAIN only)")
    print("OOD validation    : NOT used for gradients")
    print("OOD test          : NOT loaded")
    print("==========================================\n")

    # Official nominal diffusion settings.
    nominal = VectorDiffusion(
        data_dim=5,
        timesteps=500,
        beta_start=0.1,
        beta_end=0.5,
        hidden_dim=128,
        time_dim=64,
    ).to(device)

    t0 = time.time()
    diffusion_history, standardizer = train_vector_diffusion_steps(
        model=nominal,
        data=real_joint,
        device=device,
        total_iterations=7000,
        batch_size=64,
        lr=1e-4,
        standardizer=None,
        grad_clip=None,
        verbose_every=100,
    )
    diffusion_seconds = time.time() - t0

    config = GasDROConfig(
        outer_epochs=args.outer_epochs,
        generator_inner_epochs=args.generator_inner_epochs,
        predictor_inner_epochs=args.predictor_inner_epochs,
        batch_size=64,
        batch_repeat=4,
        generator_lr=args.generator_lr,
        predictor_lr=args.predictor_lr,
        ppo_clip=args.ppo_clip,
        mu=1.0,
        eta=args.eta,
        budget=args.budget,
        adjust_timesteps=15,
        step_size=2,
        discount_factor=0.05,
        p_s0=0.0,
        grad_clip=None,
        verbose=args.verbose,
    )

    # Current project runner passes the ERM train-normalization explicitly.
    gas = VectorGasDRO(
        nominal_diffusion=nominal,
        standardizer=standardizer,
        predictor=predictor,
        device=device,
        config=config,
        predictor_x_mean=x_mean,
        predictor_x_std=x_std,
    )

    t1 = time.time()
    output = gas.fit(real_joint=real_joint)
    gas_seconds = time.time() - t1

    tag = safe_tag(args.run_tag)
    out_dir = args.candidate_dir.resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"gas_dro_seed{args.seed}_final.pt"

    # Save nominal diffusion together with each candidate namespace so
    # candidate runs never overwrite the preliminary official baseline.
    nominal_path = out_dir / f"gas_dro_seed{args.seed}_nominal_diffusion.pt"
    nominal.save(
        nominal_path,
        standardizer=standardizer,
        extra={
            "seed": args.seed,
            "purpose": "ood_validation_tuning_candidate",
            "iterations": 7000,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "train_samples": int(real_joint.shape[0]),
        },
    )

    final_predictor = output["predictor"]

    torch.save(
        {
            "seed": args.seed,
            "run_tag": tag,
            "erm_checkpoint": str(erm_path),
            "predictor_state_dict": final_predictor.state_dict(),
            "adversarial_diffusion_state_dict":
                output["adversarial_diffusion"].state_dict(),
            "nominal_diffusion_checkpoint": str(nominal_path),
            "predictor_normalization": {
                "x_mean": x_mean.cpu(),
                "x_std": x_std.cpu(),
                "source": "seed_matched_erm_train_only",
            },
            "gas_dro_config": vars(config),
            "final_mu": float(output["mu"]),
            "history": output["history"],
            "protocol": {
                "seed_matched_erm": True,
                "train_for_gradients": "train.csv only",
                "normalization": "ERM checkpoint TRAIN stats only",
                "ood_validation_used_in_training": False,
                "ood_test_used": False,
                "model_iterate": "final",
            },
        },
        ckpt_path,
    )

    result = {
        "seed": args.seed,
        "run_tag": tag,
        "train_csv": str(train_path),
        "erm_checkpoint": str(erm_path),
        "candidate_checkpoint": str(ckpt_path),
        "gas_dro_config": vars(config),
        "nominal_diffusion_seconds": float(diffusion_seconds),
        "gas_dro_seconds": float(gas_seconds),
        "final_mu": float(output["mu"]),
        "ood_validation_evaluation": "pending external selector",
        "ood_test_used": False,
    }

    args.results_dir.resolve().mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir.resolve() / f"{tag}_seed{args.seed}.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\n==========================================")
    print("GAS-DRO CANDIDATE COMPLETE")
    print("==========================================")
    print("Checkpoint :", ckpt_path)
    print("Result     :", json_path)
    print("OOD test   : NOT USED")
    print("==========================================\n")


if __name__ == "__main__":
    main()
