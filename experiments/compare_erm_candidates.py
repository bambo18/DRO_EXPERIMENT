from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.erm import ERMConfig, train_erm


FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"

CANDIDATES = {
    "A": {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "weight_decay": 1e-5,
    },
    "B": {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "weight_decay": 0.0,
    },
    "C": {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "weight_decay": 1e-5,
    },
}


def read_csv_xy(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)

    missing = [
        col for col in FEATURE_COLUMNS + [TARGET_COLUMN]
        if col not in data.dtype.names
    ]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    X = np.column_stack([data[col] for col in FEATURE_COLUMNS]).astype(np.float32)
    y = np.asarray(data[TARGET_COLUMN], dtype=np.float32).reshape(-1, 1)

    return torch.from_numpy(X), torch.from_numpy(y)


def fit_train_normalizer(X_train: torch.Tensor) -> Dict[str, torch.Tensor]:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, unbiased=False, keepdim=True)

    if torch.any(std <= 0):
        raise ValueError("At least one feature has zero standard deviation.")

    return {"x_mean": mean, "x_std": std}


def apply_normalizer(
    X: torch.Tensor,
    normalization: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return (X - normalization["x_mean"]) / normalization["x_std"]


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing train.csv and validation.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "raw",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Common 5 seeds. Default: 0 1 2 3 4",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()

    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"

    if not train_path.exists():
        parser.error(f"Missing: {train_path}")
    if not val_path.exists():
        parser.error(f"Missing: {val_path}")

    # IMPORTANT: test_id.csv is intentionally never opened in this script.
    X_train, y_train = read_csv_xy(train_path)
    X_val, y_val = read_csv_xy(val_path)

    normalization = fit_train_normalizer(X_train)
    X_train = apply_normalizer(X_train, normalization)
    X_val = apply_normalizer(X_val, normalization)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print("Selection metric: mean validation MSE across seeds")
    print("test_id.csv is NOT loaded.\n")

    all_results: List[dict] = []

    for name, hp in CANDIDATES.items():
        print("=" * 72)
        print(
            f"Candidate {name}: "
            f"lr={hp['learning_rate']:g}, "
            f"batch={hp['batch_size']}, "
            f"wd={hp['weight_decay']:g}"
        )
        print("=" * 72)

        per_seed = []

        for seed in args.seeds:
            cfg = ERMConfig(
                learning_rate=hp["learning_rate"],
                batch_size=hp["batch_size"],
                max_epochs=args.max_epochs,
                weight_decay=hp["weight_decay"],
                patience=args.patience,
                min_delta=args.min_delta,
                optimizer="adam",
                seed=seed,
            )

            _, result = train_erm(
                X_train,
                y_train,
                X_val,
                y_val,
                cfg,
                device=device,
                checkpoint_path=None,
                normalization=normalization,
                verbose=args.verbose,
            )

            row = {
                "seed": seed,
                "best_epoch": int(result["best_epoch"]),
                "best_val_mse": float(result["best_val_mse"]),
            }
            per_seed.append(row)

            print(
                f"[{name}] seed={seed:>3} | "
                f"best_epoch={row['best_epoch']:>3} | "
                f"val_mse={row['best_val_mse']:.9f}"
            )

        vals = np.asarray(
            [row["best_val_mse"] for row in per_seed],
            dtype=np.float64,
        )

        candidate_result = {
            "candidate": name,
            "hyperparameters": hp,
            "per_seed": per_seed,
            "val_mse_mean": float(vals.mean()),
            "val_mse_std_ddof_1": float(vals.std(ddof=1)),
        }
        all_results.append(candidate_result)

        print(
            f"[{name}] mean ± std = "
            f"{candidate_result['val_mse_mean']:.9f} ± "
            f"{candidate_result['val_mse_std_ddof_1']:.9f}\n"
        )

    all_results.sort(key=lambda x: x["val_mse_mean"])
    best = all_results[0]

    output = {
        "selection_metric": "mean ID validation MSE across the common seeds",
        "seeds": args.seeds,
        "normalization": {
            "source": "train only",
            "target_normalized": False,
        },
        "test_used_for_selection": False,
        "best_candidate": best,
        "ranking": all_results,
    }

    output_path = results_dir / "erm_candidate_multiseed.json"
    save_json(output_path, output)

    print("\n" + "=" * 72)
    print("FINAL RANKING")
    print("=" * 72)
    for rank, row in enumerate(all_results, start=1):
        hp = row["hyperparameters"]
        print(
            f"{rank}. {row['candidate']} | "
            f"lr={hp['learning_rate']:g}, "
            f"batch={hp['batch_size']}, "
            f"wd={hp['weight_decay']:g} | "
            f"{row['val_mse_mean']:.9f} ± "
            f"{row['val_mse_std_ddof_1']:.9f}"
        )

    print("\nSelected candidate:", best["candidate"])
    print("Saved:", output_path)
    print("test_id.csv was not used.")


if __name__ == "__main__":
    main()
