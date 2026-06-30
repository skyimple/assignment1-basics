"""Neural-network modules for CS336 Assignment 1."""
from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    """Linear transformation y = x @ W.T, no bias (CS336 §3.3.2).

    Weights are initialized with a truncated normal distribution
    ``N(0, 2 / (in_features + out_features))`` truncated at ``[-3, 3]``,
    per the assignment-wide initialization scheme in §3.3.1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        return x @ self.weight.T