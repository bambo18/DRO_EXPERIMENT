from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data/nominal/train.csv"
ID_TEST_CSV = ROOT / "data/nominal/test_id.csv"
OOD_ROOT = ROOT / "data/ood/test"

CHECKPOINT_DIR = ROOT / "checkpoints/final_oodval"
RESULT_DIR = ROOT / "results/diagnostics"

FEATURES = ["X1", "X2", "X3", "X4"]
TARGET = "Y"
SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# Data loading
# ============================================================

def read_csv(path: Path):
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    if data.dtype.names is None:
        raise ValueError(f"No CSV header: {path}")

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in data.dtype.names
    ]

    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    X = np.column_stack(
        [data[col] for col in FEATURES]
    ).astype(np.float64)

    y = np.asarray(
        data[TARGET],
        dtype=np.float64,
    ).reshape(-1, 1)

    return X, y


def load_ood_strength(strength: str):
    folder = OOD_ROOT / strength
    files = sorted(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files: {folder}")

    Xs = []
    ys = []

    for path in files:
        X, y = read_csv(path)
        Xs.append(X)
        ys.append(y)

    return np.vstack(Xs), np.vstack(ys), files


# ============================================================
# True SCM / Oracle
# ============================================================

def oracle_predict(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]

    pred = (
        0.7 * x2
        - 0.5 * x3
        + 0.6 * x4
        + 0.8 * np.tanh(0.5 * x1 * x4)
    )

    return pred.reshape(-1, 1)


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


# ============================================================
# Linear regression baseline
# ============================================================

def fit_linear_regression(X, y):
    """
    Ordinary least squares with intercept.
    No sklearn dependency.
    """
    X_aug = np.column_stack(
        [np.ones(len(X)), X]
    )

    w, *_ = np.linalg.lstsq(
        X_aug,
        y,
        rcond=None,
    )

    return w


def predict_linear(X, w):
    X_aug = np.column_stack(
        [np.ones(len(X)), X]
    )

    return X_aug @ w


# ============================================================
# ERM checkpoint loading
# ============================================================

def build_mlp():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from models.mlp import build_mlp as _build_mlp

    return _build_mlp()


def load_erm(seed, device):
    path = CHECKPOINT_DIR / f"erm_seed{seed}_best.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"ERM checkpoint missing: {path}"
        )

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = build_mlp().to(device)
    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    model.eval()

    norm = ckpt["normalization"]

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
def erm_mse(model, x_mean, x_std, X, y, device):
    X_t = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    y_t = torch.tensor(
        y,
        dtype=torch.float32,
        device=device,
    )

    X_norm = (X_t - x_mean) / x_std

    pred = model(X_norm)

    return float(
        torch.mean(
            (pred - y_t) ** 2
        ).item()
    )


# ============================================================
# Dataset diagnostics
# ============================================================

def support_statistics(
    X,
    train_mean,
    train_std,
    train_min,
    train_max,
):
    lower2 = train_mean - 2 * train_std
    upper2 = train_mean + 2 * train_std

    lower3 = train_mean - 3 * train_std
    upper3 = train_mean + 3 * train_std

    out2 = (X < lower2) | (X > upper2)
    out3 = (X < lower3) | (X > upper3)

    out_range = (
        (X < train_min)
        | (X > train_max)
    )

    result = {
        "outside_2sigma_per_feature": (
            out2.mean(axis=0)
        ),
        "outside_3sigma_per_feature": (
            out3.mean(axis=0)
        ),
        "outside_train_range_per_feature": (
            out_range.mean(axis=0)
        ),
        "outside_2sigma_any": float(
            np.any(out2, axis=1).mean()
        ),
        "outside_3sigma_any": float(
            np.any(out3, axis=1).mean()
        ),
        "outside_train_range_any": float(
            np.any(out_range, axis=1).mean()
        ),
    }

    return result


def tanh_statistics(X):
    z = 0.5 * X[:, 0] * X[:, 3]
    t = np.tanh(z)

    return {
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "abs_z_mean": float(np.mean(np.abs(z))),
        "abs_z_gt_1": float(
            np.mean(np.abs(z) > 1.0)
        ),
        "abs_z_gt_2": float(
            np.mean(np.abs(z) > 2.0)
        ),
        "abs_z_gt_3": float(
            np.mean(np.abs(z) > 3.0)
        ),
        "tanh_abs_gt_0p95": float(
            np.mean(np.abs(t) > 0.95)
        ),
        "tanh_abs_gt_0p99": float(
            np.mean(np.abs(t) > 0.99)
        ),
    }


def correlation_with_y(X, y):
    values = {}

    y_flat = y[:, 0]

    for i, name in enumerate(FEATURES):
        corr = np.corrcoef(
            X[:, i],
            y_flat,
        )[0, 1]

        values[name] = float(corr)

    return values


def mahalanobis_mean_shift(
    X,
    train_mean,
    inv_train_cov,
):
    delta = (
        np.mean(X, axis=0)
        - train_mean
    )

    value = delta.T @ inv_train_cov @ delta

    return float(np.sqrt(max(value, 0.0)))


def standardized_mean_shift(
    X,
    train_mean,
    train_std,
):
    shift = (
        np.mean(X, axis=0)
        - train_mean
    ) / train_std

    return {
        FEATURES[i]: float(shift[i])
        for i in range(4)
    }


def std_ratio(X, train_std):
    ratio = np.std(
        X,
        axis=0,
        ddof=0,
    ) / train_std

    return {
        FEATURES[i]: float(ratio[i])
        for i in range(4)
    }


# ============================================================
# Pretty printing
# ============================================================

def pct(x):
    return f"{100 * x:7.3f}%"


def print_support(name, stats):
    print(f"\n[{name}] Support overlap")

    print(
        "outside train ±2σ (any X): "
        f"{pct(stats['outside_2sigma_any'])}"
    )
    print(
        "outside train ±3σ (any X): "
        f"{pct(stats['outside_3sigma_any'])}"
    )
    print(
        "outside train min/max (any X): "
        f"{pct(stats['outside_train_range_any'])}"
    )

    print("Per feature: outside train min/max")

    for i, feature in enumerate(FEATURES):
        value = (
            stats[
                "outside_train_range_per_feature"
            ][i]
        )

        print(
            f"  {feature}: {pct(value)}"
        )


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 70)
    print("DATASET DIAGNOSTIC")
    print("=" * 70)

    X_train, y_train = read_csv(TRAIN_CSV)
    X_id, y_id = read_csv(ID_TEST_CSV)

    X_mild, y_mild, _ = load_ood_strength(
        "mild"
    )
    X_mod, y_mod, _ = load_ood_strength(
        "moderate"
    )
    X_strong, y_strong, _ = load_ood_strength(
        "strong"
    )

    datasets = {
        "Train": (X_train, y_train),
        "ID": (X_id, y_id),
        "Mild": (X_mild, y_mild),
        "Moderate": (X_mod, y_mod),
        "Strong": (X_strong, y_strong),
    }

    print("\nDataset sizes")

    for name, (X, _) in datasets.items():
        print(
            f"{name:10s}: {len(X):,}"
        )

    # --------------------------------------------------------
    # Train reference statistics
    # --------------------------------------------------------

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
    # Linear baseline
    # --------------------------------------------------------

    linear_w = fit_linear_regression(
        X_train,
        y_train,
    )

    print()
    print("=" * 70)
    print("LINEAR REGRESSION FITTED ON NOMINAL TRAIN")
    print("=" * 70)

    print("Weights:")
    print(
        f"bias = {linear_w[0, 0]:.6f}"
    )

    for i, feature in enumerate(FEATURES):
        print(
            f"{feature:3s} = "
            f"{linear_w[i + 1, 0]:.6f}"
        )

    # --------------------------------------------------------
    # ERM models
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"ERM evaluation device: {device}")

    erm_models = {}

    erm_available = True

    try:
        for seed in SEEDS:
            erm_models[seed] = load_erm(
                seed,
                device,
            )

    except FileNotFoundError as exc:
        print()
        print("WARNING:")
        print(exc)
        print(
            "ERM excess-risk section will be skipped."
        )
        erm_available = False

    # --------------------------------------------------------
    # Main diagnostics
    # --------------------------------------------------------

    results = {}

    for name, (X, y) in datasets.items():

        support = support_statistics(
            X,
            train_mean,
            train_std,
            train_min,
            train_max,
        )

        tanh_stats = tanh_statistics(X)

        corr = correlation_with_y(
            X,
            y,
        )

        oracle = mse(
            oracle_predict(X),
            y,
        )

        linear = mse(
            predict_linear(
                X,
                linear_w,
            ),
            y,
        )

        maha = mahalanobis_mean_shift(
            X,
            train_mean,
            inv_train_cov,
        )

        mean_shift = standardized_mean_shift(
            X,
            train_mean,
            train_std,
        )

        variance_ratio = std_ratio(
            X,
            train_std,
        )

        erm_seed_mse = []
        erm_mean = None
        erm_std = None
        excess = None

        if erm_available and name != "Train":
            for seed in SEEDS:
                (
                    model,
                    x_mean,
                    x_std,
                ) = erm_models[seed]

                seed_mse = erm_mse(
                    model,
                    x_mean,
                    x_std,
                    X,
                    y,
                    device,
                )

                erm_seed_mse.append(
                    seed_mse
                )

            erm_mean = statistics.mean(
                erm_seed_mse
            )

            erm_std = statistics.stdev(
                erm_seed_mse
            )

            excess = erm_mean - oracle

        results[name] = {
            "n": len(X),
            "mean": {
                FEATURES[i]: float(
                    np.mean(X[:, i])
                )
                for i in range(4)
            },
            "std": {
                FEATURES[i]: float(
                    np.std(
                        X[:, i],
                        ddof=0,
                    )
                )
                for i in range(4)
            },
            "standardized_mean_shift":
                mean_shift,
            "std_ratio":
                variance_ratio,
            "mahalanobis_mean_shift":
                maha,
            "support": {
                k: (
                    v.tolist()
                    if isinstance(v, np.ndarray)
                    else v
                )
                for k, v
                in support.items()
            },
            "tanh": tanh_stats,
            "correlation_X_Y": corr,
            "oracle_mse": oracle,
            "linear_mse": linear,
            "erm_seed_mse": erm_seed_mse,
            "erm_mean_mse": erm_mean,
            "erm_std_mse": erm_std,
            "erm_excess_over_oracle":
                excess,
        }

    # --------------------------------------------------------
    # Print shift summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1. X-SPACE DISTRIBUTION SHIFT")
    print("=" * 70)

    for name in datasets:
        r = results[name]

        print(f"\n{name}")

        print(
            "Mahalanobis mean shift: "
            f"{r['mahalanobis_mean_shift']:.4f}"
        )

        print("Standardized mean shift:")

        for f in FEATURES:
            print(
                f"  {f}: "
                f"{r['standardized_mean_shift'][f]:+.3f}σ"
            )

        print("Std ratio vs Train:")

        for f in FEATURES:
            print(
                f"  {f}: "
                f"{r['std_ratio'][f]:.3f}x"
            )

    # --------------------------------------------------------
    # Support
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. TRAIN SUPPORT OVERLAP")
    print("=" * 70)

    for name in [
        "ID",
        "Mild",
        "Moderate",
        "Strong",
    ]:
        print_support(
            name,
            results[name]["support"],
        )

    # --------------------------------------------------------
    # tanh saturation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. NONLINEARITY / TANH SATURATION")
    print("=" * 70)

    print(
        "\n"
        f"{'Split':10s} "
        f"{'|z| mean':>10s} "
        f"{'|z|>1':>10s} "
        f"{'|z|>2':>10s} "
        f"{'|z|>3':>10s} "
        f"{'|tanh|>.95':>12s}"
    )

    for name in datasets:
        t = results[name]["tanh"]

        print(
            f"{name:10s} "
            f"{t['abs_z_mean']:10.4f} "
            f"{pct(t['abs_z_gt_1']):>10s} "
            f"{pct(t['abs_z_gt_2']):>10s} "
            f"{pct(t['abs_z_gt_3']):>10s} "
            f"{pct(t['tanh_abs_gt_0p95']):>12s}"
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. CORRELATION WITH Y")
    print("=" * 70)

    print(
        f"\n{'Split':10s} "
        f"{'X1-Y':>10s} "
        f"{'X2-Y':>10s} "
        f"{'X3-Y':>10s} "
        f"{'X4-Y':>10s}"
    )

    for name in datasets:
        c = results[name][
            "correlation_X_Y"
        ]

        print(
            f"{name:10s} "
            f"{c['X1']:10.4f} "
            f"{c['X2']:10.4f} "
            f"{c['X3']:10.4f} "
            f"{c['X4']:10.4f}"
        )

    # --------------------------------------------------------
    # Oracle / Linear / ERM
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5. PREDICTION DIFFICULTY")
    print("=" * 70)

    print(
        f"\n{'Split':10s} "
        f"{'Oracle':>12s} "
        f"{'Linear':>12s} "
        f"{'ERM':>20s} "
        f"{'ERM-Oracle':>14s}"
    )

    for name in datasets:

        r = results[name]

        if name == "Train":
            erm_text = "-"
            excess_text = "-"
        elif erm_available:
            erm_text = (
                f"{r['erm_mean_mse']:.6f}"
                f"±{r['erm_std_mse']:.6f}"
            )

            excess_text = (
                f"{r['erm_excess_over_oracle']:.6f}"
            )
        else:
            erm_text = "-"
            excess_text = "-"

        print(
            f"{name:10s} "
            f"{r['oracle_mse']:12.6f} "
            f"{r['linear_mse']:12.6f} "
            f"{erm_text:>20s} "
            f"{excess_text:>14s}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        RESULT_DIR
        / "dataset_diagnostics.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Saved: {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()