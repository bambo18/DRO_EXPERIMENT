from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"


def read_xy(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    if data.dtype.names is None:
        raise ValueError(f"{path}: CSV header not found.")

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


def build_mlp():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from models.mlp import build_mlp as _build_mlp

    return _build_mlp()


def load_predictor_checkpoint(
    method: str,
    seed: int,
    checkpoint_dir: Path,
    device: torch.device,
):
    if method == "erm":
        path = checkpoint_dir / f"erm_seed{seed}_best.pt"
        state_key = "model_state_dict"
        norm_key = "normalization"

    elif method == "gas_dro":
        path = checkpoint_dir / f"gas_dro_seed{seed}_final.pt"
        state_key = "predictor_state_dict"
        norm_key = "predictor_normalization"

    else:
        raise ValueError(f"Unsupported method: {method}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if state_key not in checkpoint:
        raise KeyError(f"{path}: missing key '{state_key}'")

    if norm_key not in checkpoint:
        raise KeyError(f"{path}: missing key '{norm_key}'")

    normalization = checkpoint[norm_key]

    if "x_mean" not in normalization or "x_std" not in normalization:
        raise KeyError(
            f"{path}: normalization must contain x_mean and x_std"
        )

    x_mean = torch.as_tensor(
        normalization["x_mean"],
        dtype=torch.float32,
    ).reshape(1, 4)

    x_std = torch.as_tensor(
        normalization["x_std"],
        dtype=torch.float32,
    ).reshape(1, 4)

    if not torch.isfinite(x_mean).all():
        raise ValueError(f"{path}: x_mean contains NaN/Inf")

    if not torch.isfinite(x_std).all() or torch.any(x_std <= 0):
        raise ValueError(f"{path}: invalid x_std")

    model = build_mlp().to(device)
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    return (
        model,
        x_mean.to(device),
        x_std.to(device),
        path,
    )


@torch.no_grad()
def evaluate_mse(
    model: torch.nn.Module,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
) -> float:
    X_raw = X_raw.to(device)
    y = y.to(device)

    # IMPORTANT:
    # ID test 자체의 mean/std를 쓰지 않고,
    # checkpoint에 저장된 TRAIN normalization만 사용.
    X = (X_raw - x_mean) / x_std

    pred = model(X)

    mse = torch.mean((pred - y) ** 2)

    return float(mse.item())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final ERM and GAS-DRO checkpoints "
            "on nominal ID test only."
        )
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=REPO_ROOT / "data/nominal/test_id.csv",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints/final_oodval",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results/id",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available.")

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"ID test CSV not found: {args.test_csv}"
        )

    X_test, y_test = read_xy(args.test_csv)

    print()
    print("==========================================")
    print("FINAL ID EVALUATION")
    print("==========================================")
    print(f"Test CSV      : {args.test_csv}")
    print(f"Samples       : {len(X_test)}")
    print(f"Device        : {device}")
    print(f"Seeds         : {args.seeds}")
    print("Training      : DISABLED")
    print("Model update  : DISABLED")
    print("Normalization : checkpoint TRAIN stats only")
    print("==========================================")
    print()

    final_results = {
        "test_csv": str(args.test_csv.resolve()),
        "num_samples": len(X_test),
        "seeds": args.seeds,
        "normalization": "checkpoint training statistics only",
        "methods": {},
    }

    for method in ["erm", "gas_dro"]:
        print()
        print("==========================================")
        print(f"METHOD: {method.upper()}")
        print("==========================================")

        seed_mses = []

        for seed in args.seeds:
            model, x_mean, x_std, ckpt_path = (
                load_predictor_checkpoint(
                    method=method,
                    seed=seed,
                    checkpoint_dir=args.checkpoint_dir,
                    device=device,
                )
            )

            mse = evaluate_mse(
                model=model,
                X_raw=X_test,
                y=y_test,
                x_mean=x_mean,
                x_std=x_std,
                device=device,
            )

            seed_mses.append(mse)

            print(
                f"seed={seed} | "
                f"ID MSE={mse:.9f} | "
                f"{ckpt_path.name}"
            )

        mean_mse = statistics.mean(seed_mses)

        if len(seed_mses) >= 2:
            std_mse = statistics.stdev(seed_mses)
        else:
            std_mse = 0.0

        print()
        print(
            f"{method.upper()} ID MSE "
            f"= {mean_mse:.9f} ± {std_mse:.9f}"
        )

        final_results["methods"][method] = {
            "seed_mse": {
                str(seed): mse
                for seed, mse in zip(args.seeds, seed_mses)
            },
            "mean_mse": mean_mse,
            "std_mse": std_mse,
        }

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.results_dir / "final_id_test_summary.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            final_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("==========================================")
    print("FINAL ID SUMMARY")
    print("==========================================")

    for method in ["erm", "gas_dro"]:
        r = final_results["methods"][method]

        print(
            f"{method.upper():8s} ID MSE = "
            f"{r['mean_mse']:.9f} ± "
            f"{r['std_mse']:.9f}"
        )

    print(f"Saved: {output_path.resolve()}")
    print("==========================================")


if __name__ == "__main__":
    main()
