from typing import Tuple
import torch


def _dbrms(
    x: torch.Tensor, eps: float = 1.0e-8, dim: int | Tuple[int] = -1
) -> torch.Tensor:
    return 10.0 * torch.log10(eps + torch.mean(torch.square(torch.abs(x)), dim=dim))
