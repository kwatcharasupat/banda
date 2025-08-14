import ast
import math
from typing import Any, Dict, Literal, Optional, Type, TypeVar
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn
import torch

from banda.data.batch_types import SeparationBatch

from torch.nn import functional as F

import torchaudio as ta

class BaseSeparatorModel(nn.Module):
    pass

SeparatorModel = TypeVar("SeparatorModel", bound=BaseSeparatorModel)

class BaseSpectralTransform(nn.Module):
    
    def forward(self, batch: SeparationBatch) -> SeparationBatch:
        pass

    def inverse(self, batch: SeparationBatch) -> SeparationBatch:
        pass

class BaseMaskPredictor(nn.Module):
    pass

SpectralTransform = Type[BaseSpectralTransform]
MaskPredictor = Type[BaseMaskPredictor]


class STFTConfig(BaseModel):
    n_fft: int
    win_length: int | None = None
    hop_length: int | None = None
    pad: int | None = 0
    window_fn: int | None = "torch.hann_window"
    power: float | None = None
    normalized: Literal["window", "frame_length"] | bool = "window"
    wkwargs: Dict[str, Any] | None
    center: bool = True
    pad_mode: str | None = "reflect"
    onesided: bool | None  = True

class STFT(BaseSpectralTransform):
    def __init__(self, config: STFTConfig):
        super().__init__()

        self.config = config

        window_fn = ast.literal_eval(config.window_fn)

        self.stft = ta.transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            pad=config.pad,
            window_fn=window_fn,
            power=config.power,
            normalized=config.normalized,
            wkwargs=config.wkwargs,
            center=config.center,
            pad_mode=config.pad_mode,
            onesided=config.onesided,
        )

        self.inverse_stft = ta.transforms.InverseSpectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            pad=config.pad,
            window_fn=window_fn,
            power=config.power,
            normalized=config.normalized,
            wkwargs=config.wkwargs,
            center=config.center,
            pad_mode=config.pad_mode,
            onesided=config.onesided,
        )

    
    def forward(self, batch: SeparationBatch, *, enable_grad: bool = False):
        with torch.set_grad_enabled(enable_grad):
            batch.mixture.spectrogram = self.stft(batch.mixture.audio)

        return batch
    
    def inverse(self, batch: SeparationBatch,  *, enable_grad: bool = True):
        raise NotImplementedError
    
class SpectralNormalizerConfig(BaseModel):
    eps: float = 1.0e-3

    dbrms_floor: float = -120.0
    dbrms_thresh: float = -60.0


class SpectralNormalizer(nn.Module):
    def __init__(self, config: SpectralNormalizerConfig):
        super().__init__()

        self.dbrms_floor = config.dbrms_floor
        self.dbrms_floor_eps = math.pow(10.0, -self.dbrms_floor / 20.0)
        
        self.dbrms_thresh = config.dbrms_thresh

    def forward(self, x: torch.Tensor):
        
        # x.shape = (batch, channels, freq, time) or (batch, channel, time)

        std = torch.flatten(torch.square(torch.abs(x)), start_dim=1).mean().sqrt() # (batch)
        dbrms = 20.0 * torch.log10(std + self.dbrms_floor_eps) 

        # do not normalize very soft signal to prevent blowup
        std = torch.where(
            dbrms < self.dbrms_thresh,
            torch.ones_like(std),
            std,
        )

        x = x / std[:, None, None, None]

        return x

class OpenUnmixConfig(BaseModel):
    
    n_fft: int = 1024
    n_channels: int = 2
    emb_dim: int = 512
    n_layers: int = 3
    bidirectional: bool = True
    rnn_type: str = "LSTM"

class OpenUnmix(BaseMaskPredictor):
    def __init__(self, config: OpenUnmixConfig):
        super().__init__()

        self.n_fft = config.n_fft
        self.n_channels = config.n_channels
        self.emb_dim = config.emb_dim
        self.n_layers = config.n_layers
        self.bidirectional = config.bidirectional
        self.rnn_type = config.rnn_type

        self.n_freq = self.n_fft // 2

        self.input_mean = nn.Parameter(
            data=torch.zeros((self.n_freq,))
        )
        self.input_scale = nn.Parameter(
            data=torch.ones((self.n_freq,))
        )

        self.output_mean = nn.Parameter(
            data=torch.ones((self.n_freq,))
        )
        self.output_scale = nn.Parameter(
            data=torch.ones((self.n_freq,))
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_freq * self.n_channels,
                out_channels=config.emb_dim,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm1d(num_features=self.emb_dim),
            nn.Tanh()
        )

        rnn_cls = getattr(nn, config.rnn_type)
        self.rnn = rnn_cls(
            input_size=config.n_fft,
            hidden_size=config.emb_dim // (1 + int(config.bidirectional)),
            num_layers=config.n_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        self.fc2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.emb_dim * 2,
                out_channels=self.emb_dim,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )
    
        self.fc3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.emb_dim,
                out_channels=self.n_freq * self.n_channels,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm1d(self.n_freq * self.n_channels),
        )


    def forward(self, batch: SeparationBatch):

        x = batch.mixture.spectrogram # (b, c, f, t)

        xn = (x - self.input_mean[None, None, :, None]) * self.input_scale[None, None, :, None] # (b, c, f, t)

        batch, n_chan, n_freq, n_time = xn.shape

        xn = torch.reshape(xn, (batch, -1, n_time)) # (b, c*f, t)

        z1 = self.fc1(xn) # (b, d, t)
        
        z2, _, _ = self.rnn(z1) # (b, d, t)
        
        z3 = torch.concat([z1, z2], dim=1) # (b, 2*d, t)
        z4 = self.fc2(z3) # (b, d, t)

        z5 = self.fc3(z4) # (b, c*f, t)
        z5 = torch.reshape(z5, (batch, n_chan, n_freq, n_time)) # (b, c, f, t)
        z6 = (z5 * self.output_scale[None, None, :, None]) + self.output_mean[None, None, :, None] # (b, c, f, t)
        z7 = torch.relu(z6)

        return z7 * x
    
    @classmethod
    def from_config(cls, config: DictConfig):
        cfg = OpenUnmixConfig.model_validate(config)
        return cls(cfg)


class BaseSpectralMaskingSeparatorModel(BaseSeparatorModel):

    spectral_transform: SpectralTransform
    mask_predictor: MaskPredictor

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, batch: SeparationBatch):
        batch = self.spectral_transform.forward(batch)
        batch = self.mask_predictor(batch)
        batch = self.spectral_transform.inverse(batch)        
        return batch

