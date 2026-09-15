import time

import torch

from methods.gas_dro.vector_diffusion import (
    VectorDiffusion,
    train_vector_diffusion_steps,
)


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("==========================================")
    print("OFFICIAL GAS-DRO DIFFUSION SMOKE")
    print("==========================================")
    print("Device :", device)

    # --------------------------------------------------------
    # Temporary numerical smoke data only.
    #
    # Final experiment will use the team's exact nominal
    # [X1, X2, X3, X4, Y] training data.
    # --------------------------------------------------------

    data = torch.randn(
        2000,
        5,
    )

    # --------------------------------------------------------
    # Official GAS-DRO diffusion schedule
    # --------------------------------------------------------

    model = VectorDiffusion(
        data_dim=5,
        timesteps=500,
        beta_start=0.1,
        beta_end=0.5,
    )

    start = time.time()

    # --------------------------------------------------------
    # Official diffusion training settings
    #
    # total iterations = 7000
    # batch size       = 64
    # lr               = 1e-4
    # optimizer        = Adam
    # grad clipping    = None
    # --------------------------------------------------------

    history, standardizer = train_vector_diffusion_steps(
        model=model,
        data=data,
        device=device,
        total_iterations=7000,
        batch_size=64,
        lr=1e-4,
        grad_clip=None,
        verbose_every=500,
    )

    print("\n==========================================")
    print("TRAINING CHECK")
    print("==========================================")

    print(
        "Final loss :",
        history[-1],
    )

    print(
        "All losses finite :",
        torch.isfinite(
            torch.tensor(history)
        ).all().item(),
    )

    # --------------------------------------------------------
    # Sampling MUST be tested after training.
    # --------------------------------------------------------

    print("\n==========================================")
    print("SAMPLING CHECK")
    print("==========================================")

    model.eval()

    with torch.no_grad():

        samples = model.sample(
            num_samples=32,
            device=device,
        )

    print(
        "Shape      :",
        samples.shape,
    )

    print(
        "All finite :",
        torch.isfinite(samples)
        .all()
        .item(),
    )

    print(
        "Min        :",
        samples.min().item(),
    )

    print(
        "Max        :",
        samples.max().item(),
    )

    print(
        "Mean       :",
        samples.mean().item(),
    )

    print(
        "Std        :",
        samples.std().item(),
    )

    print(
        "Runtime    :",
        round(
            time.time() - start,
            2,
        ),
        "sec",
    )


if __name__ == "__main__":
    main()