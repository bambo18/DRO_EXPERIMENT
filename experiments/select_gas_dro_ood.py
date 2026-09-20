from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mlp import build_mlp

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4"]
TARGET_COLUMN = "Y"


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
def evaluate_mse(
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


def config_key(config: Dict) -> str:
    keys = [
        "generator_lr",
        "predictor_lr",
        "ppo_clip",
        "eta",
        "budget",
        "outer_epochs",
        "generator_inner_epochs",
        "predictor_inner_epochs",
        "batch_size",
        "batch_repeat",
        "adjust_timesteps",
        "step_size",
        "discount_factor",
        "p_s0",
    ]
    kept = {k: config.get(k) for k in keys}
    return json.dumps(kept, sort_keys=True)


def sample_std(xs: List[float]) -> float:
    return 0.0 if len(xs) <= 1 else float(stdev(xs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the GAS-DRO hyperparameter configuration using ONLY OOD validation. "
            "Selection metric: mean across seeds of each seed's worst OOD-validation MSE. "
            "OOD test is never loaded."
        )
    )
    parser.add_argument("--val-csv", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "oodval_candidates" / "gas_dro",
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    val_paths = [p.resolve() for p in args.val_csv]
    for p in val_paths:
        if not p.exists():
            parser.error(f"Missing OOD validation CSV: {p}")
    val_sets = [(p, *read_xy(p)) for p in val_paths]

    candidate_paths = sorted(args.candidate_dir.resolve().rglob("gas_dro_seed*_final.pt"))
    if not candidate_paths:
        parser.error(f"No GAS-DRO candidates found under {args.candidate_dir.resolve()}")

    grouped: Dict[str, List[Dict]] = defaultdict(list)

    print("\n==========================================")
    print("GAS-DRO OOD-VALIDATION SELECTION")
    print("==========================================")
    print(f"Candidates      : {len(candidate_paths)}")
    print(f"OOD val envs    : {len(val_sets)}")
    print(f"Required seeds  : {args.seeds}")
    print("Metric          : mean(seed worst OOD-val MSE)")
    print("OOD test loaded : NO")
    print("==========================================\n")

    for path in candidate_paths:
        ckpt = torch.load(path, map_location=device)
        seed = int(ckpt["seed"])
        if seed not in args.seeds:
            continue

        config = ckpt["gas_dro_config"]
        key = config_key(config)

        norm = ckpt["predictor_normalization"]
        x_mean = torch.as_tensor(norm["x_mean"], dtype=torch.float32).reshape(1, 4)
        x_std = torch.as_tensor(norm["x_std"], dtype=torch.float32).reshape(1, 4)

        model = build_mlp().to(device)
        model.load_state_dict(ckpt["predictor_state_dict"])

        env_rows = []
        for val_path, X, y in val_sets:
            value = evaluate_mse(
                model=model,
                X_raw=X,
                y=y,
                x_mean=x_mean,
                x_std=x_std,
                device=device,
            )
            env_rows.append({"path": str(val_path), "mse": float(value)})

        values = [r["mse"] for r in env_rows]
        row = {
            "seed": seed,
            "checkpoint": str(path),
            "run_tag": ckpt.get("run_tag", path.parent.name),
            "config": config,
            "environment_mses": env_rows,
            "average_val_mse": float(mean(values)),
            "worst_val_mse": float(max(values)),
        }
        grouped[key].append(row)

        print(
            f"{row['run_tag']} | seed={seed} | "
            f"worst={row['worst_val_mse']:.9f} | "
            f"avg={row['average_val_mse']:.9f}"
        )

    complete_configs = []
    required = set(args.seeds)

    for key, rows in grouped.items():
        seeds_found = {r["seed"] for r in rows}
        if seeds_found != required:
            print(
                f"[SKIP incomplete] {rows[0]['run_tag']} "
                f"seeds={sorted(seeds_found)} required={sorted(required)}"
            )
            continue

        # Reject accidental duplicates for the same seed/config.
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        dup = {s: rr for s, rr in by_seed.items() if len(rr) != 1}
        if dup:
            raise RuntimeError(
                f"Duplicate candidate checkpoint(s) for same config/seed: "
                f"{ {s: len(rr) for s, rr in dup.items()} }"
            )

        ordered = [by_seed[s][0] for s in args.seeds]
        worsts = [r["worst_val_mse"] for r in ordered]
        avgs = [r["average_val_mse"] for r in ordered]

        complete_configs.append({
            "run_tag": ordered[0]["run_tag"],
            "config": ordered[0]["config"],
            "seed_results": ordered,
            "selection_score_mean_worst_ood_val_mse": float(mean(worsts)),
            "selection_score_std_worst_ood_val_mse": sample_std(worsts),
            "mean_average_ood_val_mse": float(mean(avgs)),
        })

    if not complete_configs:
        raise RuntimeError(
            "No complete GAS-DRO config has exactly the required seeds."
        )

    best = min(
        complete_configs,
        key=lambda r: r["selection_score_mean_worst_ood_val_mse"],
    )

    selected_dir = args.selected_dir.resolve()
    selected_dir.mkdir(parents=True, exist_ok=True)

    for row in best["seed_results"]:
        src = Path(row["checkpoint"])
        dst = selected_dir / f"gas_dro_seed{row['seed']}_final.pt"
        shutil.copy2(src, dst)

    summary = {
        "method": "GAS-DRO",
        "selection_metric": "mean_across_seeds_of_worst_ood_validation_mse",
        "seeds": args.seeds,
        "ood_validation_csvs": [str(p) for p in val_paths],
        "normalization": "seed-matched ERM TRAIN statistics only",
        "ood_test_used_for_selection": False,
        "candidate_configs": complete_configs,
        "selected": best,
        "selected_checkpoint_dir": str(selected_dir),
    }

    args.results_dir.resolve().mkdir(parents=True, exist_ok=True)
    out = args.results_dir.resolve() / "gas_dro_oodval_selection.json"
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\n==========================================")
    print("GAS-DRO OOD-VAL SELECTION COMPLETE")
    print("==========================================")
    print("Selected tag    :", best["run_tag"])
    print(
        "Selection score : "
        f"{best['selection_score_mean_worst_ood_val_mse']:.9f} "
        f"± {best['selection_score_std_worst_ood_val_mse']:.9f}"
    )
    print("Final checkpoints:", selected_dir)
    print("Summary          :", out)
    print("OOD test         : NOT USED")
    print("==========================================\n")


if __name__ == "__main__":
    main()
