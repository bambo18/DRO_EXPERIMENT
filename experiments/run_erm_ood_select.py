from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mlp import build_mlp

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


@torch.no_grad()
def mse_on_raw(
    model: torch.nn.Module,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
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


def evaluate_ood_val(
    model: torch.nn.Module,
    val_sets: List[Tuple[Path, torch.Tensor, torch.Tensor]],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
) -> Dict:
    env_rows = []
    for path, X, y in val_sets:
        env_mse = mse_on_raw(
            model=model,
            X_raw=X,
            y=y,
            x_mean=x_mean,
            x_std=x_std,
            device=device,
        )
        env_rows.append({"path": str(path), "mse": float(env_mse)})

    values = [row["mse"] for row in env_rows]
    return {
        "envs": env_rows,
        "average_val_mse": float(mean(values)),
        "worst_val_mse": float(max(values)),
    }


def train_one(
    *,
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    val_sets: List[Tuple[Path, torch.Tensor, torch.Tensor]],
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    checkpoint_path: Path,
    verbose: bool,
) -> Dict:
    set_seed(seed)

    # IMPORTANT: normalization comes from TRAIN ONLY.
    x_mean = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)

    Xn = (X_train - x_mean) / x_std
    dataset = TensorDataset(Xn, y_train)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model = build_mlp().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.MSELoss()

    best_worst = float("inf")
    best_avg = float("inf")
    best_epoch = -1
    best_state = None
    best_envs = None
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            n = xb.shape[0]
            total += loss.item() * n
            count += n

        train_mse = total / max(count, 1)

        # OOD validation is used ONLY for model/checkpoint selection.
        # No gradients are computed from these environments.
        val = evaluate_ood_val(
            model=model,
            val_sets=val_sets,
            x_mean=x_mean,
            x_std=x_std,
            device=device,
        )

        current = val["worst_val_mse"]

        if current < best_worst - min_delta:
            best_worst = current
            best_avg = val["average_val_mse"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_envs = val["envs"]
            stale = 0
        else:
            stale += 1

        if verbose and (epoch <= 5 or epoch % 10 == 0 or stale == 0):
            print(
                f"[ERM OOD-VAL][seed={seed}] "
                f"epoch={epoch:03d} train_mse={train_mse:.9f} "
                f"worst_ood_val={current:.9f} "
                f"best={best_worst:.9f}@{best_epoch}"
            )

        if patience > 0 and stale >= patience:
            if verbose:
                print(
                    f"[ERM OOD-VAL][seed={seed}] early stop at epoch {epoch}; "
                    f"best epoch={best_epoch}"
                )
            break

    if best_state is None:
        raise RuntimeError("No ERM checkpoint was selected.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "seed": seed,
        "model_state_dict": best_state,
        "best_epoch": int(best_epoch),
        # Preserve compatibility with existing GAS-DRO loader.
        "best_val_mse": float(best_worst),
        "best_ood_val_worst_mse": float(best_worst),
        "best_ood_val_average_mse": float(best_avg),
        "ood_val_environments": best_envs,
        "selection_metric": "worst_ood_validation_mse",
        "config": {
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "min_delta": float(min_delta),
        },
        "normalization": {
            "x_mean": x_mean.cpu(),
            "x_std": x_std.cpu(),
            "source": "train_only",
        },
        "protocol": {
            "train_for_gradients": "train.csv only",
            "model_selection": "OOD validation only",
            "ood_test_used": False,
        },
    }

    torch.save(checkpoint, checkpoint_path)

    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "worst_val_mse": float(best_worst),
        "average_val_mse": float(best_avg),
    }


def config_tag(lr: float, batch: int, wd: float) -> str:
    def s(x: float) -> str:
        return f"{x:.8g}".replace(".", "p").replace("-", "m")
    return f"lr{s(lr)}_bs{batch}_wd{s(wd)}"


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final ERM model selection using OOD validation only. "
            "Training gradients use train.csv only. OOD test is never loaded."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, nargs="+", required=True)

    parser.add_argument("--learning-rates", type=float, nargs="+",
                        default=[2e-4, 3e-4, 5e-4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--weight-decays", type=float, nargs="+",
                        default=[0.0, 1e-5, 3e-5])

    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)

    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "oodval_candidates" / "erm",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "final_oodval",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "oodval_selection",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_path = args.train_csv.resolve()
    if not train_path.exists():
        parser.error(f"Missing train CSV: {train_path}")

    val_paths = [p.resolve() for p in args.val_csv]
    for p in val_paths:
        if not p.exists():
            parser.error(f"Missing OOD validation CSV: {p}")

    # Safety: this script intentionally has no OOD-test argument.
    X_train, y_train = read_xy(train_path)
    val_sets = [(p, *read_xy(p)) for p in val_paths]

    print("\n==========================================")
    print("ERM FINAL OOD-VALIDATION SELECTION")
    print("==========================================")
    print(f"Device          : {device}")
    print(f"Train CSV       : {train_path}")
    print(f"OOD val envs    : {len(val_sets)}")
    print(f"Seeds           : {args.seeds}")
    print("Selection       : WORST OOD validation MSE")
    print("Normalization   : TRAIN only")
    print("OOD test loaded : NO")
    print("==========================================\n")

    configs = list(itertools.product(
        args.learning_rates,
        args.batch_sizes,
        args.weight_decays,
    ))

    all_results = []

    for ci, (lr, batch, wd) in enumerate(configs, start=1):
        tag = config_tag(lr, batch, wd)
        print(f"\n===== ERM CONFIG {ci}/{len(configs)}: {tag} =====")

        seed_results = []
        for seed in args.seeds:
            ckpt = args.candidate_dir.resolve() / tag / f"erm_seed{seed}_best.pt"

            result = train_one(
                seed=seed,
                X_train=X_train,
                y_train=y_train,
                val_sets=val_sets,
                learning_rate=lr,
                batch_size=batch,
                weight_decay=wd,
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                device=device,
                checkpoint_path=ckpt,
                verbose=args.verbose,
            )
            seed_results.append(result)

        score = float(mean(r["worst_val_mse"] for r in seed_results))
        avg_score = float(mean(r["average_val_mse"] for r in seed_results))

        row = {
            "tag": tag,
            "config": {
                "learning_rate": lr,
                "batch_size": batch,
                "weight_decay": wd,
            },
            "seed_results": seed_results,
            "selection_score_mean_worst_ood_val_mse": score,
            "mean_average_ood_val_mse": avg_score,
        }
        all_results.append(row)

        print(f"[CONFIG SCORE] mean(seed worst OOD-val MSE) = {score:.9f}")

    # Exact same principle as confirmed WDRO/KLDRO protocol:
    # select HP by the mean across the same five seeds of each seed's worst OOD-val MSE.
    best = min(
        all_results,
        key=lambda r: r["selection_score_mean_worst_ood_val_mse"],
    )

    selected_dir = args.selected_dir.resolve()
    selected_dir.mkdir(parents=True, exist_ok=True)

    for r in best["seed_results"]:
        src = Path(r["checkpoint"])
        dst = selected_dir / f"erm_seed{r['seed']}_best.pt"
        shutil.copy2(src, dst)

    summary = {
        "method": "ERM",
        "selection_metric": "mean_across_seeds_of_worst_ood_validation_mse",
        "seeds": args.seeds,
        "train_csv": str(train_path),
        "ood_validation_csvs": [str(p) for p in val_paths],
        "normalization": "train_only",
        "ood_test_used_for_selection": False,
        "all_configs": all_results,
        "selected": best,
        "selected_checkpoint_dir": str(selected_dir),
    }

    out = args.results_dir.resolve() / "erm_oodval_selection.json"
    save_json(out, summary)

    print("\n==========================================")
    print("ERM OOD-VAL SELECTION COMPLETE")
    print("==========================================")
    print("Selected config :", best["config"])
    print(
        "Selection score : "
        f"{best['selection_score_mean_worst_ood_val_mse']:.9f}"
    )
    print("Final checkpoints:", selected_dir)
    print("Summary          :", out)
    print("OOD test         : NOT USED")
    print("==========================================\n")


if __name__ == "__main__":
    main()
