from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

# Allow: python experiments/run_erm.py from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.erm import ERMConfig, evaluate_mse, train_erm


FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"


def read_csv_xy(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read only X1-X4 and Y.
    U1-U4, Uy are diagnostic columns and are never passed to the predictor.
    """
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
    """
    IMPORTANT: normalization statistics are computed from TRAIN ONLY.
    """
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, unbiased=False, keepdim=True)

    if torch.any(std <= 0):
        raise ValueError("At least one feature has zero standard deviation.")

    return {
        "x_mean": mean,
        "x_std": std,
    }


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


def load_nominal_data(
    data_dir: Path,
    include_test: bool,
):
    X_train, y_train = read_csv_xy(data_dir / "train.csv")
    X_val, y_val = read_csv_xy(data_dir / "validation.csv")

    normalization = fit_train_normalizer(X_train)

    X_train = apply_normalizer(X_train, normalization)
    X_val = apply_normalizer(X_val, normalization)

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "normalization": normalization,
    }

    # During tuning, test_id.csv is intentionally not loaded.
    if include_test:
        X_test, y_test = read_csv_xy(data_dir / "test_id.csv")
        X_test = apply_normalizer(X_test, normalization)
        result["X_test"] = X_test
        result["y_test"] = y_test

    return result


def tune(args, device: torch.device) -> None:
    """
    Hyperparameter tuning uses TRAIN + VALIDATION only.
    test_id.csv is not read in this mode.
    """
    data = load_nominal_data(args.data_dir, include_test=False)

    configs = list(itertools.product(
        args.learning_rates,
        args.batch_sizes,
        args.weight_decays,
    ))

    print(f"[TUNE] device={device}")
    print(f"[TUNE] configs={len(configs)}")
    print("[TUNE] test_id.csv is NOT loaded.")

    records: List[dict] = []

    for idx, (lr, batch_size, weight_decay) in enumerate(configs, start=1):
        cfg = ERMConfig(
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=args.max_epochs,
            weight_decay=weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            optimizer="adam",
            seed=args.tune_seed,
        )

        print(
            f"\n[TUNE {idx}/{len(configs)}] "
            f"lr={lr:g}, batch={batch_size}, wd={weight_decay:g}"
        )

        _, result = train_erm(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            cfg,
            device=device,
            checkpoint_path=None,
            normalization=data["normalization"],
            verbose=args.verbose,
        )

        records.append({
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "best_epoch": result["best_epoch"],
            "best_val_mse": result["best_val_mse"],
        })

    records.sort(key=lambda row: row["best_val_mse"])
    best = records[0]

    output = {
        "selection_metric": "ID validation MSE",
        "note": (
            "This nominal package contains ID validation. "
            "OOD validation is not used here."
        ),
        "tune_seed": args.tune_seed,
        "best": best,
        "all_results": records,
    }

    save_json(args.results_dir / "erm_tuning.json", output)

    print("\n========== ERM TUNING RESULT ==========")
    print(json.dumps(best, indent=2))
    print(f"Saved: {args.results_dir / 'erm_tuning.json'}")


def final_train(args, device: torch.device) -> None:
    """
    Train one ERM model per supplied training seed.
    Each seed gets its own best-validation checkpoint.
    ID test is evaluated only after the best checkpoint is selected.
    """
    if not args.seeds:
        raise ValueError(
            "Final mode requires --seeds with the common 5 experiment seeds."
        )

    data = load_nominal_data(args.data_dir, include_test=True)

    per_seed: List[dict] = []

    for seed in args.seeds:
        cfg = ERMConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            optimizer="adam",
            seed=seed,
        )

        checkpoint_path = args.checkpoint_dir / f"erm_seed{seed}_best.pt"

        print(f"\n========== ERM seed={seed} ==========")

        model, result = train_erm(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            cfg,
            device=device,
            checkpoint_path=checkpoint_path,
            normalization=data["normalization"],
            verbose=args.verbose,
        )

        id_test_mse = evaluate_mse(
            model,
            data["X_test"],
            data["y_test"],
            device=device,
        )

        record = {
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "validation_mse": result["best_val_mse"],
            "id_test_mse": float(id_test_mse),
            "checkpoint": str(checkpoint_path),
            "config": result["config"],
            "history": result["history"],
        }

        save_json(
            args.results_dir / f"erm_seed{seed}.json",
            record,
        )

        per_seed.append(record)

        print(
            f"[FINAL][seed={seed}] "
            f"best_val_mse={result['best_val_mse']:.6f}, "
            f"id_test_mse={id_test_mse:.6f}"
        )
        print(f"[FINAL][seed={seed}] checkpoint={checkpoint_path}")

    id_values = np.asarray(
        [row["id_test_mse"] for row in per_seed],
        dtype=np.float64,
    )

    summary = {
        "method": "ERM",
        "seeds": args.seeds,
        "normalization": {
            "source": "train only",
            "features": FEATURE_COLUMNS,
            "target_normalized": False,
        },
        "hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
        },
        "id_test_mse": {
            "per_seed": id_values.tolist(),
            "mean": float(id_values.mean()),
            "std_ddof_1": (
                float(id_values.std(ddof=1))
                if len(id_values) > 1 else None
            ),
        },
    }

    save_json(args.results_dir / "erm_summary.json", summary)

    print("\n========== ERM SUMMARY ==========")
    print(f"ID MSE mean = {summary['id_test_mse']['mean']:.6f}")
    if summary["id_test_mse"]["std_ddof_1"] is not None:
        print(
            "ID MSE std  = "
            f"{summary['id_test_mse']['std_ddof_1']:.6f}"
        )
    print(f"Saved: {args.results_dir / 'erm_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["tune", "final"],
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing train.csv, validation.csv, test_id.csv",
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

    # Common training settings.
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")

    # Tuning mode.
    parser.add_argument("--tune-seed", type=int, default=42)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[3e-4, 1e-3, 3e-3],
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[64, 128],
    )
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=[0.0, 1e-5, 1e-4],
    )

    # Final mode.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Final common training seeds. "
            "Pass the team-agreed 5 seeds explicitly."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    required = ["train.csv", "validation.csv"]
    if args.mode == "final":
        required.append("test_id.csv")

    for name in required:
        path = args.data_dir / name
        if not path.exists():
            parser.error(f"Missing required file: {path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if args.mode == "tune":
        tune(args, device)
    else:
        final_train(args, device)


if __name__ == "__main__":
    main()
