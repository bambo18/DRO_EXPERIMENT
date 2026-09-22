from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data/nominal/train.csv"
OOD_ROOT = ROOT / "data/ood/test"

CHECKPOINT_DIR = ROOT / "checkpoints/final_oodval"
RESULT_DIR = ROOT / "results/diagnostics"

FEATURES = ["X1", "X2", "X3", "X4"]
TARGET = "Y"
SEEDS = [0, 1, 2, 3, 4]
STRENGTHS = ["mild", "moderate", "strong"]


# ============================================================
# Data
# ============================================================

def read_csv(path: Path):
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    if data.dtype.names is None:
        raise ValueError(f"No header: {path}")

    X = np.column_stack(
        [data[col] for col in FEATURES]
    ).astype(np.float64)

    y = np.asarray(
        data[TARGET],
        dtype=np.float64,
    ).reshape(-1, 1)

    return X, y


# ============================================================
# Oracle
# ============================================================

def oracle_predict(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]

    return (
        0.7 * x2
        - 0.5 * x3
        + 0.6 * x4
        + 0.8 * np.tanh(0.5 * x1 * x4)
    ).reshape(-1, 1)


def mse_np(pred, y):
    return float(
        np.mean((pred - y) ** 2)
    )


# ============================================================
# MLP / Checkpoint
# ============================================================

def build_mlp():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from models.mlp import build_mlp as _build_mlp
    return _build_mlp()


def load_predictor(method, seed, device):
    if method == "erm":
        path = (
            CHECKPOINT_DIR
            / f"erm_seed{seed}_best.pt"
        )
        state_key = "model_state_dict"
        norm_key = "normalization"

    elif method == "gas_dro":
        path = (
            CHECKPOINT_DIR
            / f"gas_dro_seed{seed}_final.pt"
        )
        state_key = "predictor_state_dict"
        norm_key = "predictor_normalization"

    else:
        raise ValueError(method)

    if not path.exists():
        raise FileNotFoundError(path)

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = build_mlp().to(device)
    model.load_state_dict(
        ckpt[state_key]
    )
    model.eval()

    norm = ckpt[norm_key]

    x_mean = torch.as_tensor(
        norm["x_mean"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 4)

    x_std = torch.as_tensor(
        norm["x_std"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 4)

    return model, x_mean, x_std


@torch.no_grad()
def model_mse(
    model,
    x_mean,
    x_std,
    X,
    y,
    device,
):
    X_t = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    y_t = torch.as_tensor(
        y,
        dtype=torch.float32,
        device=device,
    )

    X_norm = (
        X_t - x_mean
    ) / x_std

    pred = model(X_norm)

    return float(
        torch.mean(
            (pred - y_t) ** 2
        ).item()
    )


# ============================================================
# Shift diagnostics
# ============================================================

def environment_shift_metrics(
    X,
    train_mean,
    train_std,
    train_min,
    train_max,
    inv_train_cov,
):
    env_mean = np.mean(
        X,
        axis=0,
    )

    standardized_shift = (
        env_mean - train_mean
    ) / train_std

    mean_shift_l2 = float(
        np.linalg.norm(
            standardized_shift
        )
    )

    delta = env_mean - train_mean

    mahalanobis = float(
        np.sqrt(
            max(
                delta.T
                @ inv_train_cov
                @ delta,
                0.0,
            )
        )
    )

    outside_range = (
        (X < train_min)
        | (X > train_max)
    )

    outside_3sigma = (
        (X < train_mean - 3 * train_std)
        | (X > train_mean + 3 * train_std)
    )

    outside_2sigma = (
        (X < train_mean - 2 * train_std)
        | (X > train_mean + 2 * train_std)
    )

    z = (
        0.5
        * X[:, 0]
        * X[:, 3]
    )

    tanh_val = np.tanh(z)

    return {
        "mahalanobis": mahalanobis,
        "mean_shift_l2": mean_shift_l2,

        "outside_2sigma_any": float(
            np.any(
                outside_2sigma,
                axis=1,
            ).mean()
        ),

        "outside_3sigma_any": float(
            np.any(
                outside_3sigma,
                axis=1,
            ).mean()
        ),

        "outside_train_range_any": float(
            np.any(
                outside_range,
                axis=1,
            ).mean()
        ),

        "tanh_saturation_095": float(
            np.mean(
                np.abs(tanh_val) > 0.95
            )
        ),

        "mean_shift_X1": float(
            standardized_shift[0]
        ),
        "mean_shift_X2": float(
            standardized_shift[1]
        ),
        "mean_shift_X3": float(
            standardized_shift[2]
        ),
        "mean_shift_X4": float(
            standardized_shift[3]
        ),
    }


# ============================================================
# Correlation helper
# ============================================================

def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 100)
    print("PER-ENVIRONMENT OOD DIAGNOSTIC")
    print("=" * 100)
    print(f"Device : {device}")
    print(f"Seeds  : {SEEDS}")
    print("=" * 100)

    # --------------------------------------------------------
    # Train reference
    # --------------------------------------------------------

    X_train, _ = read_csv(
        TRAIN_CSV
    )

    train_mean = np.mean(
        X_train,
        axis=0,
    )

    train_std = np.std(
        X_train,
        axis=0,
        ddof=0,
    )

    train_min = np.min(
        X_train,
        axis=0,
    )

    train_max = np.max(
        X_train,
        axis=0,
    )

    train_cov = np.cov(
        X_train,
        rowvar=False,
    )

    inv_train_cov = np.linalg.pinv(
        train_cov
    )

    # --------------------------------------------------------
    # Load ERM/GAS models only once
    # --------------------------------------------------------

    models = {
        "erm": {},
        "gas_dro": {},
    }

    print("\nLoading checkpoints...")

    for method in [
        "erm",
        "gas_dro",
    ]:
        for seed in SEEDS:
            models[method][seed] = (
                load_predictor(
                    method,
                    seed,
                    device,
                )
            )

    print("Done.\n")

    rows = []

    # --------------------------------------------------------
    # Evaluate every environment separately
    # --------------------------------------------------------

    for strength in STRENGTHS:

        folder = (
            OOD_ROOT
            / strength
        )

        files = sorted(
            folder.glob("*.csv")
        )

        if not files:
            raise FileNotFoundError(
                folder
            )

        for path in files:

            X, y = read_csv(path)

            oracle_mse = mse_np(
                oracle_predict(X),
                y,
            )

            shift = (
                environment_shift_metrics(
                    X,
                    train_mean,
                    train_std,
                    train_min,
                    train_max,
                    inv_train_cov,
                )
            )

            erm_values = []
            gas_values = []

            for seed in SEEDS:

                model, mean, std = (
                    models["erm"][seed]
                )

                erm_values.append(
                    model_mse(
                        model,
                        mean,
                        std,
                        X,
                        y,
                        device,
                    )
                )

                model, mean, std = (
                    models["gas_dro"][seed]
                )

                gas_values.append(
                    model_mse(
                        model,
                        mean,
                        std,
                        X,
                        y,
                        device,
                    )
                )

            erm_mean = float(
                np.mean(erm_values)
            )

            erm_std = float(
                np.std(
                    erm_values,
                    ddof=1,
                )
            )

            gas_mean = float(
                np.mean(gas_values)
            )

            gas_std = float(
                np.std(
                    gas_values,
                    ddof=1,
                )
            )

            row = {
                "strength": strength,
                "environment": path.name,

                **shift,

                "oracle_mse":
                    oracle_mse,

                "erm_mean_mse":
                    erm_mean,

                "erm_std_mse":
                    erm_std,

                "gas_mean_mse":
                    gas_mean,

                "gas_std_mse":
                    gas_std,

                "erm_excess":
                    erm_mean
                    - oracle_mse,

                "gas_excess":
                    gas_mean
                    - oracle_mse,

                # negative = GAS-DRO better
                # positive = ERM better
                "gas_minus_erm":
                    gas_mean
                    - erm_mean,
            }

            rows.append(row)

    # --------------------------------------------------------
    # Console table
    # --------------------------------------------------------

    print("=" * 130)
    print("ENVIRONMENT-BY-ENVIRONMENT RESULTS")
    print("=" * 130)

    print(
        f"{'Split':9s} "
        f"{'Env':11s} "
        f"{'Maha':>7s} "
        f"{'L2Shift':>8s} "
        f"{'OutRange':>9s} "
        f"{'Oracle':>9s} "
        f"{'ERM':>9s} "
        f"{'GAS':>9s} "
        f"{'ERM-Oracle':>11s} "
        f"{'GAS-ERM':>9s}"
    )

    for r in rows:
        print(
            f"{r['strength']:9s} "
            f"{r['environment']:11s} "
            f"{r['mahalanobis']:7.3f} "
            f"{r['mean_shift_l2']:8.3f} "
            f"{100*r['outside_train_range_any']:8.2f}% "
            f"{r['oracle_mse']:9.5f} "
            f"{r['erm_mean_mse']:9.5f} "
            f"{r['gas_mean_mse']:9.5f} "
            f"{r['erm_excess']:11.5f} "
            f"{r['gas_minus_erm']:+9.5f}"
        )

    # --------------------------------------------------------
    # Strength summaries
    # --------------------------------------------------------

    summary = {}

    print()
    print("=" * 100)
    print("SUMMARY BY SHIFT STRENGTH")
    print("=" * 100)

    for strength in STRENGTHS:

        subset = [
            r
            for r in rows
            if r["strength"]
            == strength
        ]

        erm = np.array([
            r["erm_mean_mse"]
            for r in subset
        ])

        gas = np.array([
            r["gas_mean_mse"]
            for r in subset
        ])

        oracle = np.array([
            r["oracle_mse"]
            for r in subset
        ])

        delta = gas - erm

        gas_wins = int(
            np.sum(delta < 0)
        )

        erm_wins = int(
            np.sum(delta > 0)
        )

        summary[strength] = {
            "num_envs":
                len(subset),

            "oracle_mean":
                float(
                    np.mean(oracle)
                ),

            "erm_mean":
                float(
                    np.mean(erm)
                ),

            "gas_mean":
                float(
                    np.mean(gas)
                ),

            "gas_minus_erm_mean":
                float(
                    np.mean(delta)
                ),

            "gas_wins":
                gas_wins,

            "erm_wins":
                erm_wins,
        }

        print()
        print(
            f"[{strength.upper()}]"
        )
        print(
            f"Oracle mean : "
            f"{np.mean(oracle):.6f}"
        )
        print(
            f"ERM mean    : "
            f"{np.mean(erm):.6f}"
        )
        print(
            f"GAS mean    : "
            f"{np.mean(gas):.6f}"
        )
        print(
            f"GAS - ERM   : "
            f"{np.mean(delta):+.6f}"
        )
        print(
            f"GAS wins    : "
            f"{gas_wins}/{len(subset)} envs"
        )
        print(
            f"ERM wins    : "
            f"{erm_wins}/{len(subset)} envs"
        )

    # --------------------------------------------------------
    # Does GAS become relatively better as shift increases?
    # --------------------------------------------------------

    maha = [
        r["mahalanobis"]
        for r in rows
    ]

    support = [
        r["outside_train_range_any"]
        for r in rows
    ]

    excess_erm = [
        r["erm_excess"]
        for r in rows
    ]

    gas_minus_erm = [
        r["gas_minus_erm"]
        for r in rows
    ]

    corr_results = {
        "mahalanobis_vs_erm_excess":
            safe_corr(
                maha,
                excess_erm,
            ),

        "support_vs_erm_excess":
            safe_corr(
                support,
                excess_erm,
            ),

        "mahalanobis_vs_gas_minus_erm":
            safe_corr(
                maha,
                gas_minus_erm,
            ),

        "support_vs_gas_minus_erm":
            safe_corr(
                support,
                gas_minus_erm,
            ),
    }

    print()
    print("=" * 100)
    print("SHIFT vs PERFORMANCE CORRELATION")
    print("=" * 100)

    print(
        "Mahalanobis vs ERM excess   : "
        f"{corr_results['mahalanobis_vs_erm_excess']:+.4f}"
    )

    print(
        "Out-of-support vs ERM excess: "
        f"{corr_results['support_vs_erm_excess']:+.4f}"
    )

    print()
    print(
        "Mahalanobis vs (GAS-ERM)    : "
        f"{corr_results['mahalanobis_vs_gas_minus_erm']:+.4f}"
    )

    print(
        "Out-of-support vs (GAS-ERM) : "
        f"{corr_results['support_vs_gas_minus_erm']:+.4f}"
    )

    print()
    print(
        "Interpretation of GAS-ERM correlation:"
    )

    print(
        "  negative -> GAS tends to become relatively better "
        "as shift increases"
    )

    print(
        "  positive -> GAS tends to become relatively worse "
        "as shift increases"
    )

    # --------------------------------------------------------
    # Biggest-shift environments
    # --------------------------------------------------------

    ranked = sorted(
        rows,
        key=lambda r:
            r["mahalanobis"],
        reverse=True,
    )

    print()
    print("=" * 100)
    print("TOP 10 ENVIRONMENTS BY MAHALANOBIS SHIFT")
    print("=" * 100)

    for i, r in enumerate(
        ranked[:10],
        start=1,
    ):
        print(
            f"{i:2d}. "
            f"{r['strength']:8s}/"
            f"{r['environment']:10s} | "
            f"Maha={r['mahalanobis']:.3f} | "
            f"OutRange="
            f"{100*r['outside_train_range_any']:.2f}% | "
            f"ERM={r['erm_mean_mse']:.5f} | "
            f"GAS={r['gas_mean_mse']:.5f} | "
            f"GAS-ERM="
            f"{r['gas_minus_erm']:+.5f}"
        )

    # --------------------------------------------------------
    # Save CSV + JSON
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        RESULT_DIR
        / "per_environment_diagnostics.csv"
    )

    json_path = (
        RESULT_DIR
        / "per_environment_diagnostics.json"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "rows": rows,
        "summary_by_strength":
            summary,
        "correlations":
            corr_results,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()