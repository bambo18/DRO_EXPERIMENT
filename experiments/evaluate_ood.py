from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"
KNOWN_STRENGTHS = ("mild", "moderate", "strong")


def read_xy(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one OOD environment CSV.

    Required columns:
        X1, X2, X3, X4, Y

    Extra columns such as U1..Uy are ignored.
    """
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
    # Import lazily so this script can print helpful path errors first.
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
    """Load either an ERM or GAS-DRO predictor checkpoint.

    ERM checkpoint expected keys:
        model_state_dict
        normalization: {x_mean, x_std}

    GAS-DRO checkpoint expected keys:
        predictor_state_dict
        predictor_normalization: {x_mean, x_std}
    """
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

    # These are checkpoints created by this project.
    checkpoint = torch.load(
        path,
        map_location=device,
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

    return model, x_mean.to(device), x_std.to(device), path


@torch.no_grad()
def evaluate_mse(
    model: torch.nn.Module,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> float:
    """Evaluate MSE in the original Y scale.

    IMPORTANT:
    X is normalized only with TRAIN statistics stored in the checkpoint.
    OOD validation/test statistics are never used for normalization.
    """
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

    if count == 0:
        raise ValueError("Empty evaluation dataset.")

    return total_sq / count


def infer_strength(path: Path, split_root: Path) -> str:
    """Infer mild/moderate/strong from the relative path.

    Recommended layout:
        data/ood/validation/mild/*.csv
        data/ood/validation/moderate/*.csv
        data/ood/validation/strong/*.csv

    Also works when the strength word appears in a filename.
    """
    rel = path.relative_to(split_root)
    tokens = [part.lower() for part in rel.parts]

    for strength in KNOWN_STRENGTHS:
        if any(strength in token for token in tokens):
            return strength

    return "ungrouped"


def discover_environments(
    data_dir: Path,
    split: str,
) -> List[Tuple[Path, str]]:
    split_root = data_dir / split

    if not split_root.exists():
        raise FileNotFoundError(
            f"OOD split folder not found: {split_root}\n"
            f"Expected e.g. {data_dir}/{split}/mild/*.csv"
        )

    csv_files = sorted(split_root.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under: {split_root}"
        )

    environments = [
        (path, infer_strength(path, split_root))
        for path in csv_files
    ]

    return environments


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def summarize_seed(
    env_results: List[Dict],
) -> Dict:
    all_mse = [row["mse"] for row in env_results]

    by_strength: Dict[str, List[float]] = {}
    for row in env_results:
        by_strength.setdefault(
            row["strength"],
            [],
        ).append(row["mse"])

    strength_summary = {}
    for strength, values in sorted(by_strength.items()):
        strength_summary[strength] = {
            "num_environments": len(values),
            "mean_mse": mean(values),
            "worst_mse": max(values),
        }

    return {
        "num_environments": len(all_mse),
        "average_ood_mse": mean(all_mse),
        "worst_ood_mse": max(all_mse),
        "by_strength": strength_summary,
    }


def aggregate_seeds(seed_results: List[Dict]) -> Dict:
    average_values = [
        row["summary"]["average_ood_mse"]
        for row in seed_results
    ]
    worst_values = [
        row["summary"]["worst_ood_mse"]
        for row in seed_results
    ]

    all_strengths = sorted(
        {
            strength
            for row in seed_results
            for strength in row["summary"]["by_strength"].keys()
        }
    )

    by_strength = {}
    for strength in all_strengths:
        vals = [
            row["summary"]["by_strength"][strength]["mean_mse"]
            for row in seed_results
            if strength in row["summary"]["by_strength"]
        ]

        by_strength[strength] = {
            "mean_mse_across_seeds": mean(vals),
            "std_mse_across_seeds": sample_std(vals),
            "num_seeds": len(vals),
        }

    return {
        "average_ood_mse": {
            "mean": mean(average_values),
            "std": sample_std(average_values),
        },
        "worst_ood_mse": {
            "mean": mean(worst_values),
            "std": sample_std(worst_values),
        },
        "by_strength": by_strength,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ERM or GAS-DRO checkpoints on OOD validation/test "
            "environments. This script NEVER trains or updates the model."
        )
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=["erm", "gas_dro"],
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["validation", "test"],
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "ood",
        help=(
            "OOD root containing validation/ and test/. "
            "Recommended: data/ood/<split>/<strength>/*.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ood",
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    environments = discover_environments(
        args.data_dir.resolve(),
        args.split,
    )

    print("\n==========================================")
    print("OOD EVALUATION")
    print("==========================================")
    print(f"Method        : {args.method}")
    print(f"Split         : {args.split}")
    print(f"Device        : {device}")
    print(f"Seeds         : {args.seeds}")
    print(f"Environments  : {len(environments)}")
    print("Training      : DISABLED")
    print("Model update  : DISABLED")
    print("Normalization : checkpoint TRAIN stats only")
    print("==========================================\n")

    counts: Dict[str, int] = {}
    for _, strength in environments:
        counts[strength] = counts.get(strength, 0) + 1

    for strength, count in sorted(counts.items()):
        print(f"{strength:10s}: {count} environments")
    print()

    seed_results = []

    for seed in args.seeds:
        (
            model,
            x_mean,
            x_std,
            checkpoint_path,
        ) = load_predictor_checkpoint(
            method=args.method,
            seed=seed,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            device=device,
        )

        env_rows = []

        print(f"----- Seed {seed} -----")
        print(f"Checkpoint: {checkpoint_path}")

        for env_path, strength in environments:
            X, y = read_xy(env_path)

            mse = evaluate_mse(
                model=model,
                X_raw=X,
                y=y,
                x_mean=x_mean,
                x_std=x_std,
                device=device,
                batch_size=args.batch_size,
            )

            env_rows.append(
                {
                    "environment": str(env_path),
                    "environment_name": env_path.stem,
                    "strength": strength,
                    "num_samples": len(X),
                    "mse": float(mse),
                }
            )

            print(
                f"{strength:10s} | "
                f"{env_path.name:35s} | "
                f"MSE={mse:.9f}"
            )

        seed_summary = summarize_seed(env_rows)

        print(
            f"Seed {seed} Average OOD MSE = "
            f"{seed_summary['average_ood_mse']:.9f}"
        )
        print(
            f"Seed {seed} Worst OOD MSE   = "
            f"{seed_summary['worst_ood_mse']:.9f}\n"
        )

        seed_results.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "environments": env_rows,
                "summary": seed_summary,
            }
        )

    aggregate = aggregate_seeds(seed_results)

    result = {
        "method": args.method,
        "split": args.split,
        "device": str(device),
        "seeds": args.seeds,
        "evaluation_only": True,
        "normalization": "checkpoint train statistics only",
        "seed_results": seed_results,
        "aggregate": aggregate,
    }

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        args.results_dir
        / f"{args.method}_ood_{args.split}_summary.json"
    )

    out_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n==========================================")
    print("5-SEED OOD SUMMARY")
    print("==========================================")

    for strength, stats in aggregate["by_strength"].items():
        print(
            f"{strength:10s} MSE = "
            f"{stats['mean_mse_across_seeds']:.9f} "
            f"± {stats['std_mse_across_seeds']:.9f}"
        )

    avg = aggregate["average_ood_mse"]
    worst = aggregate["worst_ood_mse"]

    print(
        f"Average OOD MSE = "
        f"{avg['mean']:.9f} ± {avg['std']:.9f}"
    )
    print(
        f"Worst OOD MSE   = "
        f"{worst['mean']:.9f} ± {worst['std']:.9f}"
    )

    print(f"Saved           : {out_path}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
