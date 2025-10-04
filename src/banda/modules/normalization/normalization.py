from pydantic import BaseModel
from torch import nn
import torch


class NormalizationParams(BaseModel):
    dbrms_threshold: float | None = -60.0
    eps: float = 1.0e-6  # -120 dB


class Normalizer(nn.Module):
    def __init__(self, *, config: NormalizationParams):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self._normalize(x)
        return xn

    def _dbrms(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        rms = torch.sqrt(x.pow(2).mean(dim=-1))
        dbrms = 20 * torch.log10(rms + self.config.eps)
        return dbrms, rms

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        dbrms, rms = self._dbrms(x)

        if self.config.dbrms_threshold is not None:
            return torch.where(
                (dbrms < self.config.dbrms_threshold)[:, None, None],
                x,
                x / rms[:, None, None],
            )

        return x / rms[:, None, None]
