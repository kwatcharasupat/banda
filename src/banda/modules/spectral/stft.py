
from omegaconf import DictConfig
from pydantic import BaseModel
import torch
import torchaudio as ta
from torch import nn

from banda.data.item import SourceSeparationBatch

class STFTParams(BaseModel):
    n_fft: int | None
    win_length: int | None
    hop_length: int | None
    pad: int | None = 0
    window_fn: str | None = "hann"
    power: float | None = 2
    normalized: bool | str | None = "window"
    wkwargs: dict | None = None
    center: bool | None = True
    pad_mode: str | None = "reflect"
    onesided: bool | None = True


class Spectrogram(nn.Module):
    def __init__(self, *, config: DictConfig):
        super().__init__()
        self.config = STFTParams.model_validate(config)

        self.stft = ta.transforms.Spectrogram(
            **self.config.model_dump()
        )
        self.istft = ta.transforms.InverseSpectrogram(
            **self.config.model_dump()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stft(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.istft(x)