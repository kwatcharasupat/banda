from typing import Dict
from omegaconf import DictConfig
from banda.data.item import SourceSeparationBatch
from banda.models.masking.base import _BaseMaskingModel

import torch
from torch import nn

from banda.utils import BaseConfig


class OpenUnmixParams(DictConfig):
    in_channels: int = 2
    hidden_size: int = 512
    n_layers: int = 3
    bidirectional: bool = True
    dropout: float = 0.4
    complex: bool = True

class InnerOpenUnmix(nn.Module):
    def __init__(
        self,
        *,
        config: BaseConfig,
    ):
        super().__init__()

        n_freq = config.spectrogram.n_fft // 2 + 1
        n_channels = config.in_channels

        n_inout = n_freq * n_channels

        if config.complex:
            n_inout *= 2

        hidden_size = config.hidden_size

        self.fc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_inout,
                out_channels=hidden_size,
                kernel_size=1,
                bias=False,
            ),  
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
        )

        lstm_hidden_size = hidden_size // 2 if config.bidirectional else hidden_size

        n_layers = config.n_layers

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=n_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if n_layers > 1 else 0,
        )

        self.fc2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=1,
                bias=False,
            ),  
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=n_inout,
                kernel_size=1,
                bias=False,
            ),  
            nn.BatchNorm1d(n_inout),
            nn.ReLU() if not config.complex else nn.Identity(),
        )

        self.complex = config.complex
        self.in_channels = config.in_channels


    def forward(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, n_time = x.shape
        z = self.fc1(x) # (batch, hidden_size, n_time)
        z = torch.permute(z, (0, 2, 1)).contiguous()
        # (batch, hidden_size, n_time) -> (batch, n_time, hidden_size)

        z_lstm, _ = self.lstm(z)
        z = torch.cat([z, z_lstm], -1) # (batch, n_time, hidden_size + lstm_hidden_size*2)
        z = torch.permute(z, (0, 2, 1)).contiguous()
        # (batch, n_time, hidden_size + lstm_hidden_size*2) -> (batch, hidden_size + lstm_hidden_size*2, n_time)
        # first dense stage + batch norm
        z = self.fc2(z) # (batch, fc2_hiddensize, n_time)
        mask = self.fc3(z) # (batch, in_chan*n_freq, n_time)

        # reshape back to original dim
        if self.complex:
            mask = mask.reshape(batch_size, self.in_channels, -1, 2, n_time)
            mask = torch.permute(mask, (0, 1, 2, 4, 3)).contiguous()
            # (batch, in_chan, n_freq, 2, n_time) -> (batch, in_chan, n_freq, n_time, 2)
            mask = torch.view_as_complex(mask)
            # (batch, in_chan, n_freq, n_time, 2) -> (batch, in_chan, n_freq, n_time)
        else:
            mask = mask.reshape(batch_size, self.in_channels, -1, n_time)

        return mask


class OpenUnmix(_BaseMaskingModel):
    # adapted from https://github.com/sigsep/open-unmix-pytorch/blob/fb672c9584997c2b05e148eeaa65b4c23ed4693b/openunmix/model.py

    def __init__(
        self,
        *,
        config: BaseConfig,
    ):
        super().__init__(config=config)

        self.models = nn.ModuleDict(
            {
                stem: InnerOpenUnmix(
                    config=config,
                )
                for stem in self.config.stems
            }
        )


    def _inner_model(
        self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch
    ) -> Dict[str, torch.Tensor]:
        batch_size, in_channels, n_freq, n_time = specs_normalized.shape

        with torch.no_grad():
            if self.config.complex:
                x = torch.view_as_real(specs_normalized)
                # (batch, in_chan, n_freq, n_time, 2)
                x = x.permute(0, 1, 2, 4, 3).contiguous()
                # (batch, in_chan, n_freq, 2, n_time)
            else:
                x = specs_normalized.abs()

            x = torch.reshape(
                x, (batch_size, -1, n_time)
            )
            # (batch, in_chan, n_freq, ?, n_time) -> (batch, in_out, n_time)

            masks = {}
            for stem, model in self.models.items():
                masks[stem] = model(x)

        return masks