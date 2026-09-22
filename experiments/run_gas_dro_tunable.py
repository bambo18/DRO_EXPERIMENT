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
        raise ValueError(
            f"Expected X [N,4], got {tuple(X.shape)}"
        )

    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(
            f"Expected y [N,1], got {tuple(y.shape)}"
        )

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

    checkpoint = torch.load(
        path,
        map_location=device,
    )

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
    predictor.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return predictor, checkpoint, x_mean, x_std


def safe_tag(text: str) -> str:
    return "".join(
        c if (c.isalnum() or c in "-_.") else "_"
        for c in text
    )


def resolve_validation_paths(
    val_root: Path | None,
    val_csv: List[Path] | None,
    expected_count: int,
) -> List[Path]:
    """
    Resolve ONLY OOD-validation CSVs.

    Use either:
        --val-root <directory>
    which recursively loads every *.csv below that directory,

    or:
        --val-csv file1.csv file2.csv ...

    OOD test files must never be supplied here.
    """

    if (val_root is None) == (val_csv is None):
        raise ValueError(
            "Provide exactly one of --val-root or --val-csv."
        )

    if val_root is not None:
        root = val_root.resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"OOD validation root not found: {root}"
            )
        if not root.is_dir():
            raise ValueError(
                f"--val-root must be a directory: {root}"
            )

        paths = sorted(
            p.resolve()
            for p in root.rglob("*.csv")
            if p.is_file()
        )
    else:
        paths = []
        assert val_csv is not None

        for path in val_csv:
            resolved = path.resolve()
            if not resolved.exists():
                raise FileNotFoundError(
                    f"OOD validation CSV not found: {resolved}"
                )
            if not resolved.is_file():
                raise ValueError(
                    f"OOD validation path is not a file: {resolved}"
                )
            paths.append(resolved)

        paths = sorted(paths)

    if len(paths) == 0:
        raise RuntimeError(
            "No OOD-validation CSV files were found."
        )

    # Current experiment protocol uses 15 OOD-validation environments.
    # Set --expected-val-envs 0 only if you intentionally change that protocol.
    if expected_count > 0 and len(paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} OOD-validation environments, "
            f"but found {len(paths)}.\n"
            "Check --val-root/--val-csv. Do NOT point this at OOD test."
        )

    return paths


def load_ood_validation_envs(
    paths: List[Path],
) -> List[Dict[str, object]]:
    envs: List[Dict[str, object]] = []

    for path in paths:
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
def evaluate_worst_ood_validation(
    model: torch.nn.Module,
    environments: List[Dict[str, object]],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """
    Evaluate one checkpoint on every OOD-validation environment.

    Selection metric:
        worst OOD-val MSE = max_e MSE_e

    Validation is used for model selection only; no gradients are computed.
    """

    was_training = model.training
    model.eval()

    mean = x_mean.to(device)
    std = x_std.to(device)

    results: List[Dict[str, object]] = []

    for env in environments:
        path = env["path"]
        X = env["X"]
        y = env["y"]

        assert isinstance(path, Path)
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        X_device = X.to(device)
        y_device = y.to(device)

        X_norm = (X_device - mean) / std
        prediction = model(X_norm)

        mse = torch.mean(
            (prediction - y_device) ** 2
        ).item()

        results.append(
            {
                "environment": str(path),
                "mse": float(mse),
            }
        )

    if was_training:
        model.train()

    mse_values = [
        float(item["mse"])
        for item in results
    ]

    worst_mse = max(mse_values)
    average_mse = sum(mse_values) / len(mse_values)

    return (
        float(worst_mse),
        float(average_mse),
        results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ONE GAS-DRO tuning candidate from a seed-matched ERM "
            "checkpoint. After every GAS-DRO outer epoch, evaluate OOD "
            "validation and save the checkpoint with the lowest worst "
            "OOD-validation MSE. OOD test is never loaded."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
    )

    val_group = parser.add_mutually_exclusive_group(
        required=True
    )

    val_group.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help=(
            "Directory containing ONLY OOD-validation CSVs. "
            "All *.csv files are loaded recursively."
        ),
    )

    val_group.add_argument(
        "--val-csv",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Explicit list of OOD-validation CSV files. "
            "Do NOT pass OOD-test files."
        ),
    )

    parser.add_argument(
        "--expected-val-envs",
        type=int,
        default=15,
        help=(
            "Expected number of OOD-validation environments. "
            "Current protocol uses 15. Use 0 to disable this check."
        ),
    )

    parser.add_argument(
        "--erm-checkpoint-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "checkpoints"
            / "final_oodval"
        ),
        help=(
            "Must contain erm_seed{seed}_best.pt "
            "selected by OOD validation."
        ),
    )

    parser.add_argument(
        "--run-tag",
        required=True,
    )

    # Official defaults. Supply alternate values explicitly during the sweep.
    parser.add_argument(
        "--generator-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--predictor-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--ppo-clip",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--outer-epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--generator-inner-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--predictor-inner-epochs",
        type=int,
        default=2,
    )

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

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    train_path = args.train_csv.resolve()
    if not train_path.exists():
        parser.error(
            f"Missing train CSV: {train_path}"
        )

    # --------------------------------------------------------
    # Train data: optimization only
    # --------------------------------------------------------
    X_train, y_train = read_xy(train_path)
    real_joint = build_joint(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # OOD validation: checkpoint/model selection only
    # --------------------------------------------------------
    val_paths = resolve_validation_paths(
        val_root=args.val_root,
        val_csv=args.val_csv,
        expected_count=args.expected_val_envs,
    )

    ood_val_envs = load_ood_validation_envs(
        val_paths
    )

    # --------------------------------------------------------
    # Seed-matched ERM initialization
    # --------------------------------------------------------
    erm_path = (
        args.erm_checkpoint_dir.resolve()
        / f"erm_seed{args.seed}_best.pt"
    )

    (
        predictor,
        erm_checkpoint,
        x_mean,
        x_std,
    ) = load_erm_seed_checkpoint(
        erm_path,
        seed=args.seed,
        device=device,
    )

    # --------------------------------------------------------
    # Candidate output namespace
    # --------------------------------------------------------
    tag = safe_tag(args.run_tag)

    out_dir = (
        args.candidate_dir.resolve()
        / tag
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The existing downstream selector can keep consuming *_final.pt.
    # In this revised protocol, *_final.pt will contain the SELECTED
    # best-OOD-validation iterate, not the last training iterate.
    selected_ckpt_path = (
        out_dir
        / f"gas_dro_seed{args.seed}_final.pt"
    )

    # Audit copy saved whenever a new best checkpoint is found.
    best_ckpt_path = (
        out_dir
        / f"gas_dro_seed{args.seed}_best_oodval.pt"
    )

    nominal_path = (
        out_dir
        / f"gas_dro_seed{args.seed}_nominal_diffusion.pt"
    )

    print("\n==========================================")
    print("GAS-DRO OOD-VAL BEST-CHECKPOINT TRAINING")
    print("==========================================")
    print(f"Seed              : {args.seed}")
    print(f"Run tag           : {tag}")
    print(f"Device            : {device}")
    print(f"Train samples     : {len(X_train)}")
    print(f"OOD-val envs      : {len(ood_val_envs)}")
    print(f"ERM checkpoint    : {erm_path}")
    print(f"ERM seed          : {erm_checkpoint['seed']}")
    print("Initialization    : seed-matched ERM checkpoint")
    print("Normalization     : ERM TRAIN-only statistics")
    print("OOD validation    : model selection only, no gradients")
    print("Checkpoint metric : minimum worst OOD-val MSE")
    print("OOD test          : NOT LOADED")
    print("==========================================")

    for i, path in enumerate(val_paths, start=1):
        print(f"[OOD-VAL {i:02d}] {path}")

    print("==========================================\n")

    # --------------------------------------------------------
    # Official nominal diffusion settings
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

    diffusion_history, standardizer = (
        train_vector_diffusion_steps(
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
    )

    diffusion_seconds = time.time() - t0

    # --------------------------------------------------------
    # GAS-DRO config
    # --------------------------------------------------------
    config = GasDROConfig(
        outer_epochs=args.outer_epochs,
        generator_inner_epochs=(
            args.generator_inner_epochs
        ),
        predictor_inner_epochs=(
            args.predictor_inner_epochs
        ),
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

    gas = VectorGasDRO(
        nominal_diffusion=nominal,
        standardizer=standardizer,
        predictor=predictor,
        device=device,
        config=config,
        predictor_x_mean=x_mean,
        predictor_x_std=x_std,
    )

    # --------------------------------------------------------
    # Best OOD-validation state for THIS seed/configuration
    # --------------------------------------------------------
    best_worst_ood_val = float("inf")
    best_average_ood_val = float("inf")
    best_outer_epoch = -1
    best_env_results: List[Dict[str, object]] | None = None

    def ood_val_callback(
        outer_epoch: int,
        gas_model: VectorGasDRO,
    ) -> None:
        nonlocal best_worst_ood_val
        nonlocal best_average_ood_val
        nonlocal best_outer_epoch
        nonlocal best_env_results

        (
            worst_mse,
            average_mse,
            env_results,
        ) = evaluate_worst_ood_validation(
            model=gas_model.predictor,
            environments=ood_val_envs,
            x_mean=x_mean,
            x_std=x_std,
            device=device,
        )

        print("\n==========================================")
        print("OOD-VALIDATION CHECKPOINT EVALUATION")
        print("==========================================")
        print(f"Seed              : {args.seed}")
        print(f"Run tag           : {tag}")
        print(f"Budget            : {config.budget:.6f}")
        print(
            f"Outer epoch       : "
            f"{outer_epoch}/{config.outer_epochs}"
        )
        print(
            f"Average OOD-val   : "
            f"{average_mse:.9f}"
        )
        print(
            f"Worst OOD-val     : "
            f"{worst_mse:.9f}"
        )

        if worst_mse < best_worst_ood_val:
            previous_best = best_worst_ood_val

            best_worst_ood_val = float(worst_mse)
            best_average_ood_val = float(average_mse)
            best_outer_epoch = int(outer_epoch)
            best_env_results = env_results

            torch.save(
                {
                    "seed": args.seed,
                    "run_tag": tag,
                    "erm_checkpoint": str(erm_path),
                    "predictor_state_dict": (
                        gas_model.predictor.state_dict()
                    ),
                    "adversarial_diffusion_state_dict": (
                        gas_model.adversarial_diffusion.state_dict()
                    ),
                    "nominal_diffusion_checkpoint": str(
                        nominal_path
                    ),
                    "predictor_normalization": {
                        "x_mean": x_mean.cpu(),
                        "x_std": x_std.cpu(),
                        "source": (
                            "seed_matched_erm_train_only"
                        ),
                    },
                    "gas_dro_config": vars(config),
                    "selected_outer_epoch": int(
                        best_outer_epoch
                    ),
                    "best_worst_ood_val_mse": float(
                        best_worst_ood_val
                    ),
                    "best_average_ood_val_mse": float(
                        best_average_ood_val
                    ),
                    "ood_validation_results": (
                        best_env_results
                    ),
                    "final_mu": float(gas_model.mu),
                    "history": gas_model.history,
                    "protocol": {
                        "seed_matched_erm": True,
                        "train_for_gradients": (
                            "train.csv only"
                        ),
                        "normalization": (
                            "ERM checkpoint TRAIN stats only"
                        ),
                        "ood_validation_used_for_gradients": False,
                        "ood_validation_used_for_model_selection": True,
                        "checkpoint_selection_metric": (
                            "minimum worst OOD-validation MSE"
                        ),
                        "checkpoint_evaluation_frequency": (
                            "after every GAS-DRO outer epoch"
                        ),
                        "ood_test_used": False,
                        "model_iterate": (
                            "best_ood_validation"
                        ),
                    },
                },
                best_ckpt_path,
            )

            print("NEW BEST          : YES")
            if np.isfinite(previous_best):
                print(
                    f"Previous best     : "
                    f"{previous_best:.9f}"
                )
            print(
                f"Best outer epoch  : "
                f"{best_outer_epoch}"
            )
            print(
                f"Best checkpoint   : "
                f"{best_ckpt_path}"
            )
        else:
            print("NEW BEST          : NO")
            print(
                f"Current best      : "
                f"{best_worst_ood_val:.9f} "
                f"@ outer {best_outer_epoch}"
            )

        print("==========================================\n")

    # --------------------------------------------------------
    # GAS-DRO training + best OOD-val checkpoint selection
    # --------------------------------------------------------
    t1 = time.time()

    output = gas.fit(
        real_joint=real_joint,
        outer_epoch_callback=ood_val_callback,
    )

    gas_seconds = time.time() - t1

    if best_outer_epoch < 0 or not best_ckpt_path.exists():
        raise RuntimeError(
            "Training finished but no best OOD-validation checkpoint "
            "was saved."
        )

    # --------------------------------------------------------
    # Save nominal diffusion in this candidate namespace
    # --------------------------------------------------------
    nominal.save(
        nominal_path,
        standardizer=standardizer,
        extra={
            "seed": args.seed,
            "purpose": (
                "ood_validation_best_checkpoint_tuning_candidate"
            ),
            "iterations": 7000,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "train_samples": int(
                real_joint.shape[0]
            ),
        },
    )

    # --------------------------------------------------------
    # Preserve compatibility with the old selector:
    # gas_dro_seed{seed}_final.pt now means the SELECTED model.
    # It is copied from the best-OOD-val checkpoint, not from
    # the last outer iteration.
    # --------------------------------------------------------
    best_checkpoint = torch.load(
        best_ckpt_path,
        map_location="cpu",
    )

    best_checkpoint["nominal_diffusion_checkpoint"] = str(
        nominal_path
    )

    best_checkpoint["selected_checkpoint_source"] = str(
        best_ckpt_path
    )

    torch.save(
        best_checkpoint,
        selected_ckpt_path,
    )

    # --------------------------------------------------------
    # Result JSON
    # --------------------------------------------------------
    result = {
        "seed": args.seed,
        "run_tag": tag,
        "train_csv": str(train_path),
        "ood_validation_csvs": [
            str(path)
            for path in val_paths
        ],
        "erm_checkpoint": str(erm_path),
        "candidate_checkpoint": str(
            selected_ckpt_path
        ),
        "best_checkpoint_audit_copy": str(
            best_ckpt_path
        ),
        "gas_dro_config": vars(config),
        "nominal_diffusion_seconds": float(
            diffusion_seconds
        ),
        "gas_dro_seconds": float(
            gas_seconds
        ),
        "selected_outer_epoch": int(
            best_checkpoint["selected_outer_epoch"]
        ),
        "best_worst_ood_val_mse": float(
            best_checkpoint[
                "best_worst_ood_val_mse"
            ]
        ),
        "best_average_ood_val_mse": float(
            best_checkpoint[
                "best_average_ood_val_mse"
            ]
        ),
        "final_mu_at_selected_checkpoint": float(
            best_checkpoint["final_mu"]
        ),
        "selection": {
            "criterion": (
                "minimum worst OOD-validation MSE"
            ),
            "evaluation_frequency": (
                "after every GAS-DRO outer epoch"
            ),
            "selected_model": (
                "best OOD-validation checkpoint"
            ),
        },
        "ood_validation_used_for_gradients": False,
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
    print(f"Seed              : {args.seed}")
    print(f"Run tag           : {tag}")
    print(f"Selected outer    : {best_outer_epoch}")
    print(
        f"Best worst OOD-val: "
        f"{best_worst_ood_val:.9f}"
    )
    print(
        f"Best avg OOD-val  : "
        f"{best_average_ood_val:.9f}"
    )
    print(f"Selected checkpoint: {selected_ckpt_path}")
    print(f"Best audit copy    : {best_ckpt_path}")
    print(f"Result             : {json_path}")
    print(f"Diffusion time (s) : {diffusion_seconds:.2f}")
    print(f"GAS-DRO time (s)   : {gas_seconds:.2f}")
    print("OOD test           : NOT USED")
    print("==========================================\n")


if __name__ == "__main__":
    main()
