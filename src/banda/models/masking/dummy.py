

from typing import Dict

from omegaconf import DictConfig
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.base import _BaseMaskingModel

from torch import nn

class DummyMaskingModel(_BaseMaskingModel):
    
    def __init__(self, *, config: DictConfig):
        super().__init__(config=config)

        self.linear = nn.ModuleDict(
            {
                stem: nn.Linear(config.stft.n_fft // 2 + 1, config.stft.n_fft + 2)
                for stem in self.config.stems
            }
        )

    def _inner_model(self, specs_normalized: torch.Tensor, batch: SourceSeparationBatch) -> Dict[str, torch.Tensor]:

        masks = {stem: torch.view_as_complex(self.linear[stem](specs_normalized)) for stem in self.config.stems}

        return masks