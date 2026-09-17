"""
Common prediction MLP
=====================

Architecture:
4 -> 64 -> 64 -> 1

* ReLU after each hidden layer
* Linear output
* No Dropout or normalization layers
* Input shape:  (batch_size, 4)
* Output shape: (batch_size, 1)

- All methods use the same prediction model.
"""


import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Args:
            x: Tensor of shape (batch_size, 4)

        Returns:
            Tensor of shape (batch_size, 1)
        """

        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(
                f"Expected input shape (batch_size, 4), got {tuple(x.shape)}"
            )

        return self.net(x)


def build_mlp() -> MLP:
    
    """
    Return the fixed common benchmark predictor
    """

    return MLP()


if __name__ == "__main__":
    model = build_mlp()

    x = torch.randn(8, 4)
    y_hat = model(x)

    assert y_hat.shape == (8, 1)

    print(model)
    print("Number of trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y_hat.shape))
    print("MLP sanity check passed.")
