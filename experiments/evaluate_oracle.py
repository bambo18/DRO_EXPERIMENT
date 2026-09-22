from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ID_TEST = ROOT / "data/nominal/test_id.csv"
OOD_ROOT = ROOT / "data/ood/test"


def read_csv(path):
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    x1 = data["X1"]
    x2 = data["X2"]
    x3 = data["X3"]
    x4 = data["X4"]
    y = data["Y"]

    return x1, x2, x3, x4, y


def oracle_predict(x1, x2, x3, x4):
    # True conditional mean E[Y | X]
    return (
        0.7 * x2
        - 0.5 * x3
        + 0.6 * x4
        + 0.8 * np.tanh(0.5 * x1 * x4)
    )


def mse_for_file(path):
    x1, x2, x3, x4, y = read_csv(path)

    pred = oracle_predict(x1, x2, x3, x4)

    return float(np.mean((pred - y) ** 2))


def main():
    print()
    print("==========================================")
    print("ORACLE / BAYES PREDICTOR EVALUATION")
    print("==========================================")
    print("Training      : NONE")
    print("Model fitting : NONE")
    print("True SCM used : YES")
    print("Theoretical irreducible MSE = 0.3^2 = 0.09")
    print("==========================================")
    print()

    # ID
    id_mse = mse_for_file(ID_TEST)
    print(f"ID Test Oracle MSE = {id_mse:.9f}")
    print()

    all_ood_mses = []
    strength_results = {}

    for strength in ["mild", "moderate", "strong"]:
        strength_dir = OOD_ROOT / strength
        files = sorted(strength_dir.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {strength_dir}"
            )

        env_mses = []

        print(f"----- {strength.upper()} -----")

        for path in files:
            mse = mse_for_file(path)

            env_mses.append(mse)
            all_ood_mses.append(mse)

            print(
                f"{path.name:20s} | "
                f"Oracle MSE={mse:.9f}"
            )

        mean_mse = float(np.mean(env_mses))
        worst_mse = float(np.max(env_mses))

        strength_results[strength] = {
            "mean": mean_mse,
            "worst": worst_mse,
        }

        print(
            f"{strength} mean Oracle MSE  = "
            f"{mean_mse:.9f}"
        )
        print(
            f"{strength} worst Oracle MSE = "
            f"{worst_mse:.9f}"
        )
        print()

    average_ood = float(np.mean(all_ood_mses))
    worst_ood = float(np.max(all_ood_mses))

    print("==========================================")
    print("ORACLE SUMMARY")
    print("==========================================")
    print(f"ID       MSE = {id_mse:.9f}")
    print(
        f"Mild     MSE = "
        f"{strength_results['mild']['mean']:.9f}"
    )
    print(
        f"Moderate MSE = "
        f"{strength_results['moderate']['mean']:.9f}"
    )
    print(
        f"Strong   MSE = "
        f"{strength_results['strong']['mean']:.9f}"
    )
    print(f"Average OOD MSE = {average_ood:.9f}")
    print(f"Worst OOD MSE   = {worst_ood:.9f}")
    print("==========================================")


if __name__ == "__main__":
    main()