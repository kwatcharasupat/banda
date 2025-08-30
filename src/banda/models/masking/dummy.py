

from typing import Dict

from omegaconf import DictConfig
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.base import _BaseMaskingModel

from torch import nn

class DummyMaskingModel(_BaseMaskingModel):
    
    def __init__(self, *, config: DictConfig):
        super().__init__(config=config)

        self.re_linear = nn.ModuleDict(
            {
                stem: nn.Conv1d(in_channels=config.spectrogram.n_fft // 2 + 1, out_channels=config.spectrogram.n_fft // 2 + 1, kernel_size=(1,))
                for stem in self.config.stems
            }
        )
        self.im_linear = nn.ModuleDict(
            {
                stem: nn.Conv1d(in_channels=config.spectrogram.n_fft // 2 + 1, out_channels=config.spectrogram.n_fft // 2 + 1, kernel_size=(1,))
                for stem in self.config.stems
            }
        )

    def _inner_model(self, specs_normalized: torch.Tensor, batch: SourceSeparationBatch) -> Dict[str, torch.Tensor]:
        
        batch_size, n_channels, n_freq, n_time = specs_normalized.shape
        specs_normalized = torch.reshape(specs_normalized, (batch_size * n_channels, n_freq, n_time)).abs()
        
        masks = {}
        for stem in self.config.stems:
            re_mask = torch.reshape(self.re_linear[stem](specs_normalized), (batch_size, n_channels, n_freq, n_time))
            im_mask = torch.reshape(self.im_linear[stem](specs_normalized), (batch_size, n_channels, n_freq, n_time))
            masks[stem] = torch.complex(re_mask, im_mask)

        return masks