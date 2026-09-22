from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
EXPECTED_OOD_VAL_ENVS = 15


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

    missing = [
        c
        for c in FEATURE_COLUMNS + [TARGET_COLUMN]
        if c not in data.dtype.names
    ]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    X = np.column_stack(
        [data[c] for c in FEATURE_COLUMNS]
    ).astype(np.float32)

    y = np.asarray(
        data[TARGET_COLUMN],
        dtype=np.float32,
    ).reshape(-1, 1)

    return torch.from_numpy(X), torch.from_numpy(y)


def build_joint(
    X: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X [N,4], got {tuple(X.shape)}")
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"Expected y [N,1], got {tuple(y.shape)}")
    return torch.cat([X, y], dim=1).float()


def load_erm_seed_checkpoint(
    path: Path,
    seed: int,
    device: torch.device,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Seed-matched final ERM checkpoint not found: {path}"
        )

    checkpoint = torch.load(path, map_location=device)

    if int(checkpoint["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch: requested {seed}, "
            f"checkpoint has {checkpoint['seed']}"
        )

    norm = checkpoint["normalization"]
    x_mean = torch.as_tensor(
        norm["x_mean"],
        dtype=torch.float32,
    ).reshape(1, 4)
    x_std = torch.as_tensor(
        norm["x_std"],
        dtype=torch.float32,
    ).reshape(1, 4)

    predictor = build_mlp().to(device)
    predictor.load_state_dict(checkpoint["model_state_dict"])

    return predictor, checkpoint, x_mean, x_std


def safe_tag(text: str) -> str:
    return "".join(
        c if (c.isalnum() or c in "-_.") else "_"
        for c in text
    )


def resolve_ood_val_csvs(args: argparse.Namespace) -> List[Path]:
    """
    Resolve exactly the 15 held-out OOD validation environments.

    Two supported interfaces:
      1) --val-root <folder>  -> recursively collect *.csv
      2) --val-csv file1 ... file15

    OOD test files must NEVER be passed here.
    """

    if args.val_root is not None:
        root = args.val_root.resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"OOD validation root not found: {root}"
            )

        csv_paths = sorted(
            p.resolve()
            for p in root.rglob("*.csv")
            if p.is_file()
        )
    else:
        csv_paths = [
            p.resolve()
            for p in args.val_csv
        ]

    missing = [
        p
        for p in csv_paths
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing OOD validation CSV(s):\n"
            + "\n".join(str(p) for p in missing)
        )

    # Remove accidental duplicates while preserving order.
    unique_paths: List[Path] = []
    seen = set()
    for path in csv_paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    if len(unique_paths) != EXPECTED_OOD_VAL_ENVS:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_OOD_VAL_ENVS} OOD validation "
            f"CSV environments, got {len(unique_paths)}.\n"
            "Check that you supplied VALIDATION only, not test/train files."
        )

    return unique_paths


def load_ood_validation_envs(
    csv_paths: List[Path],
) -> List[Dict[str, object]]:
    envs: List[Dict[str, object]] = []

    for path in csv_paths:
        X, y = read_xy(path)
        envs.append(
            {
                "path": path,
                "X": X,
                "y": y,
            }
        )

    return envs


@torch.no_grad()
def evaluate_predictor_mse(
    model: torch.nn.Module,
    X_raw: torch.Tensor,
    y_raw: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    """Evaluate raw-target MSE using the ERM TRAIN-only X normalization."""

    was_training = model.training
    model.eval()

    mean = x_mean.to(device)
    std = x_std.to(device)

    squared_error_sum = 0.0
    count = 0

    for start in range(0, X_raw.shape[0], batch_size):
        end = min(start + batch_size, X_raw.shape[0])

        X = X_raw[start:end].to(device)
        y = y_raw[start:end].to(device)

        X = (X - mean) / std
        pred = model(X)

        squared_error_sum += float(
            ((pred - y) ** 2).sum().item()
        )
        count += int(y.numel())

    if was_training:
        model.train()

    return squared_error_sum / max(count, 1)


@torch.no_grad()
def evaluate_ood_validation(
    model: torch.nn.Module,
    ood_val_envs: List[Dict[str, object]],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """
    Evaluate the SAME 15 OOD validation environments.

    Selection metric for one checkpoint:
        worst OOD-Val MSE = max(environment MSEs)
    """

    env_results: List[Dict[str, object]] = []

    for index, env in enumerate(ood_val_envs):
        mse = evaluate_predictor_mse(
            model=model,
            X_raw=env["X"],
            y_raw=env["y"],
            x_mean=x_mean,
            x_std=x_std,
            device=device,
        )

        path = env["path"]
        env_results.append(
            {
                "env_index": index,
                "file": str(path),
                "mse": float(mse),
            }
        )

    mses = [
        float(item["mse"])
        for item in env_results
    ]

    worst_mse = max(mses)
    average_mse = sum(mses) / len(mses)

    return (
        float(worst_mse),
        float(average_mse),
        env_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ONE GAS-DRO tuning candidate with unified model selection. "
            "The predictor starts from the seed-matched ERM checkpoint. "
            "ERM initialization is selection step 0. After EVERY actual "
            "predictor optimizer.step(), the same 15 OOD validation "
            "environments are evaluated. The checkpoint with minimum "
            "Worst OOD-Val MSE is selected. OOD test is never loaded here."
        )
    )

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument(
        "--erm-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "final_oodval",
        help=(
            "Must contain erm_seed{seed}_best.pt selected by OOD validation."
        ),
    )
    parser.add_argument("--run-tag", required=True)

    val_group = parser.add_mutually_exclusive_group(required=True)
    val_group.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help=(
            "Folder containing ONLY the 15 OOD-validation CSV files. "
            "CSV files are found recursively."
        ),
    )
    val_group.add_argument(
        "--val-csv",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit list of the 15 OOD-validation CSV files.",
    )

    # Official GAS-DRO defaults. Keep these unless the GAS-DRO tuning
    # protocol explicitly changes the corresponding hyperparameter.
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
        default=(
            REPO_ROOT
            / "checkpoints"
            / "oodval_candidates"
            / "gas_dro"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "oodval_candidates"
            / "gas_dro"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    train_path = args.train_csv.resolve()
    if not train_path.exists():
        parser.error(f"Missing train CSV: {train_path}")

    ood_val_paths = resolve_ood_val_csvs(args)
    ood_val_envs = load_ood_validation_envs(ood_val_paths)

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

    tag = safe_tag(args.run_tag)
    out_dir = args.candidate_dir.resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = (
        out_dir
        / f"gas_dro_seed{args.seed}_best_oodval.pt"
    )
    final_ckpt_path = (
        out_dir
        / f"gas_dro_seed{args.seed}_final.pt"
    )
    nominal_path = (
        out_dir
        / f"gas_dro_seed{args.seed}_nominal_diffusion.pt"
    )

    print("\n==========================================")
    print("GAS-DRO OOD-VAL CANDIDATE TRAINING")
    print("==========================================")
    print(f"Seed                 : {args.seed}")
    print(f"Run tag              : {tag}")
    print(f"Train samples        : {len(X_train)}")
    print(f"ERM checkpoint       : {erm_path}")
    print(f"ERM seed             : {erm_checkpoint['seed']}")
    print(f"OOD-Val environments : {len(ood_val_envs)}")
    print("GAS-DRO batch size   : 64 (official setting)")
    print("Step 0               : seed-matched ERM initialization")
    print("Selection            : min Worst OOD-Val MSE")
    print("Evaluation timing    : after EVERY predictor optimizer.step()")
    print("Normalization        : ERM TRAIN-only stats")
    print("OOD-Val gradients    : NEVER")
    print("OOD test             : NOT loaded")
    print("==========================================\n")

    # --------------------------------------------------------
    # 1) Train nominal diffusion exactly as before.
    # --------------------------------------------------------
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
        batch_size=64,  # official GAS-DRO setting
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
        batch_size=64,  # official GAS-DRO setting
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

    gas = VectorGasDRO(
        nominal_diffusion=nominal,
        standardizer=standardizer,
        predictor=predictor,
        device=device,
        config=config,
        predictor_x_mean=x_mean,
        predictor_x_std=x_std,
    )

    # Save nominal diffusion in this candidate namespace.
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

    # --------------------------------------------------------
    # 2) Unified checkpoint/model selection.
    #
    # Candidate step 0 = initial seed-matched ERM weights.
    # Then evaluate after EVERY actual predictor optimizer.step().
    # --------------------------------------------------------
    best: Dict[str, object] = {
        "worst_mse": float("inf"),
        "avg_mse": float("inf"),
        "step": None,
        "stage": None,
        "env_results": None,
    }

    selection_history: List[Dict[str, object]] = []

    def evaluate_and_maybe_save(
        predictor_step: int,
        gas_model: VectorGasDRO,
        stage: str,
    ) -> None:
        worst_mse, avg_mse, env_results = evaluate_ood_validation(
            model=gas_model.predictor,
            ood_val_envs=ood_val_envs,
            x_mean=x_mean,
            x_std=x_std,
            device=device,
        )

        record: Dict[str, object] = {
            "predictor_step": int(predictor_step),
            "stage": stage,
            "worst_ood_val_mse": float(worst_mse),
            "average_ood_val_mse": float(avg_mse),
        }
        selection_history.append(record)

        improved = worst_mse < float(best["worst_mse"])

        print("\n------------------------------------------")
        print("OOD-VALIDATION MODEL SELECTION")
        print("------------------------------------------")
        print(f"Predictor step : {predictor_step}")
        print(f"Stage          : {stage}")
        print(f"Average MSE    : {avg_mse:.9f}")
        print(f"Worst MSE      : {worst_mse:.9f}")
        print(
            "Previous best  : "
            + (
                "inf"
                if best["step"] is None
                else f"{float(best['worst_mse']):.9f} "
                     f"@ step {best['step']}"
            )
        )

        if improved:
            best["worst_mse"] = float(worst_mse)
            best["avg_mse"] = float(avg_mse)
            best["step"] = int(predictor_step)
            best["stage"] = stage
            best["env_results"] = env_results

            torch.save(
                {
                    "seed": args.seed,
                    "run_tag": tag,
                    "budget": float(config.budget),
                    "erm_checkpoint": str(erm_path),
                    "selected_predictor_step": int(predictor_step),
                    "selected_stage": stage,
                    "best_worst_ood_val_mse": float(worst_mse),
                    "best_average_ood_val_mse": float(avg_mse),
                    "ood_val_results": env_results,
                    "predictor_state_dict": (
                        gas_model.predictor.state_dict()
                    ),
                    "adversarial_diffusion_state_dict": (
                        gas_model.adversarial_diffusion.state_dict()
                    ),
                    "predictor_normalization": {
                        "x_mean": x_mean.cpu(),
                        "x_std": x_std.cpu(),
                        "source": "seed_matched_erm_train_only",
                    },
                    "gas_dro_config": vars(config),
                    "mu_at_selection": float(gas_model.mu),
                    "history_at_selection": gas_model.history,
                    "selection_metric": "worst_ood_validation_mse",
                    "step0_erm_included": True,
                    "selection_evaluation_timing": (
                        "step0_and_after_every_predictor_optimizer_step"
                    ),
                    "ood_test_used": False,
                },
                best_ckpt_path,
            )

            print("NEW BEST       : YES")
            print(f"Saved          : {best_ckpt_path}")
        else:
            print("NEW BEST       : NO")

        print("------------------------------------------\n")

    # Step 0: ERM initialization itself is a valid model-selection candidate.
    evaluate_and_maybe_save(
        predictor_step=0,
        gas_model=gas,
        stage="erm_initialization_step0",
    )

    def after_predictor_update(
        predictor_step: int,
        gas_model: VectorGasDRO,
    ) -> None:
        evaluate_and_maybe_save(
            predictor_step=predictor_step,
            gas_model=gas_model,
            stage="after_predictor_optimizer_step",
        )

    # --------------------------------------------------------
    # 3) GAS-DRO training.
    # The training objective/optimizer logic is unchanged.
    # OOD-Val is used ONLY for checkpoint/model selection.
    # --------------------------------------------------------
    t1 = time.time()
    output = gas.fit(
        real_joint=real_joint,
        after_predictor_update=after_predictor_update,
    )
    gas_seconds = time.time() - t1

    if best["step"] is None or not best_ckpt_path.exists():
        raise RuntimeError(
            "No best OOD-validation checkpoint was created."
        )

    # Reload the selected checkpoint. This is the model that must be used
    # for final held-out test evaluation, NOT the last training iterate.
    selected_checkpoint = torch.load(
        best_ckpt_path,
        map_location=device,
    )

    selected_predictor = build_mlp().to(device)
    selected_predictor.load_state_dict(
        selected_checkpoint["predictor_state_dict"]
    )

    # Keep the historical *_final.pt filename for compatibility with
    # external selectors/evaluators. It now contains the BEST OOD-Val
    # predictor, not the last iterate.
    torch.save(
        {
            "seed": args.seed,
            "run_tag": tag,
            "erm_checkpoint": str(erm_path),
            "predictor_state_dict": (
                selected_predictor.state_dict()
            ),
            "adversarial_diffusion_state_dict": (
                selected_checkpoint[
                    "adversarial_diffusion_state_dict"
                ]
            ),
            "nominal_diffusion_checkpoint": str(nominal_path),
            "predictor_normalization": {
                "x_mean": x_mean.cpu(),
                "x_std": x_std.cpu(),
                "source": "seed_matched_erm_train_only",
            },
            "gas_dro_config": vars(config),
            "selected_predictor_step": int(best["step"]),
            "selected_stage": str(best["stage"]),
            "best_worst_ood_val_mse": float(best["worst_mse"]),
            "best_average_ood_val_mse": float(best["avg_mse"]),
            "ood_val_results_at_best": best["env_results"],
            "mu_at_selection": float(
                selected_checkpoint["mu_at_selection"]
            ),
            "final_training_mu": float(output["mu"]),
            "history": output["history"],
            "selection_history": selection_history,
            "protocol": {
                "seed_matched_erm": True,
                "erm_initialization_is_step0_candidate": True,
                "gas_dro_batch_size": 64,
                "gas_dro_batch_size_source": "official GAS-DRO setting",
                "train_for_gradients": "train.csv only",
                "normalization": "ERM checkpoint TRAIN stats only",
                "ood_validation_environment_count": (
                    EXPECTED_OOD_VAL_ENVS
                ),
                "ood_validation_used_for_gradients": False,
                "ood_validation_used_for_model_selection": True,
                "selection_metric": "minimum worst OOD-Val MSE",
                "selection_evaluation_timing": (
                    "step0 and after every predictor optimizer.step()"
                ),
                "model_iterate": "best_ood_validation",
                "ood_test_used": False,
            },
        },
        final_ckpt_path,
    )

    result = {
        "seed": args.seed,
        "run_tag": tag,
        "train_csv": str(train_path),
        "ood_validation_csvs": [
            str(path)
            for path in ood_val_paths
        ],
        "erm_checkpoint": str(erm_path),
        "best_checkpoint": str(best_ckpt_path),
        "candidate_checkpoint": str(final_ckpt_path),
        "gas_dro_config": vars(config),
        "nominal_diffusion_seconds": float(diffusion_seconds),
        "gas_dro_seconds_including_oodval": float(gas_seconds),
        "predictor_update_steps": int(
            output["predictor_update_steps"]
        ),
        "selected_predictor_step": int(best["step"]),
        "selected_stage": str(best["stage"]),
        "best_worst_ood_val_mse": float(best["worst_mse"]),
        "best_average_ood_val_mse": float(best["avg_mse"]),
        "final_training_mu": float(output["mu"]),
        "selection_history": selection_history,
        "ood_test_used": False,
    }

    args.results_dir.resolve().mkdir(
        parents=True,
        exist_ok=True,
    )
    json_path = (
        args.results_dir.resolve()
        / f"{tag}_seed{args.seed}.json"
    )
    json_path.write_text(
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
    print("GAS-DRO CANDIDATE COMPLETE")
    print("==========================================")
    print(f"Total predictor steps : {output['predictor_update_steps']}")
    print(f"Selected step         : {best['step']}")
    print(f"Selected stage        : {best['stage']}")
    print(f"Best Worst OOD-Val    : {float(best['worst_mse']):.9f}")
    print(f"Best Average OOD-Val  : {float(best['avg_mse']):.9f}")
    print(f"Best checkpoint       : {best_ckpt_path}")
    print(f"Final/selected ckpt   : {final_ckpt_path}")
    print(f"Result                : {json_path}")
    print("OOD test              : NOT USED")
    print("==========================================\n")


if __name__ == "__main__":
    main()
