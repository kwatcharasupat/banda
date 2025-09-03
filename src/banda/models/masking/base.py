from typing import Dict

from omegaconf import DictConfig
from banda.data.item import SourceSeparationBatch
from banda.models.base import BaseRegisteredModel

import torch

from banda.modules.normalization.normalization import Normalizer
from banda.modules.spectral.stft import Spectrogram

class _BaseMaskingModel(BaseRegisteredModel):
    
    def __init__(self, *, config: DictConfig):
        super().__init__(config=config)
        
        self.normalizer = Normalizer(config=self.config.normalization)
        self.stft = Spectrogram(config=self.config.spectrogram)

    
    def forward(self, batch: dict) -> SourceSeparationBatch:
        with torch.no_grad():
            
            batch = SourceSeparationBatch.model_validate(batch)
            assert "mixture" not in batch.sources, "Mixture should not be in sources"
            _mixture = sum(batch.sources[key]["audio"] for key in batch.sources)
            assert torch.allclose(batch.mixture["audio"], _mixture, atol=1e-5), "Mixture does not match sum of sources"

            batch : SourceSeparationBatch = self.normalizer(batch)
            specs_unnormalized : torch.Tensor = self.stft(batch.mixture["audio"])
            specs_normalized : torch.Tensor = self.stft(batch.mixture["audio/normalized"])
            
            for key in batch.sources:
                source = batch.sources[key]["audio"]
                source_specs = self.stft(source)
                batch.sources[key]["spectrogram"] = source_specs
                

        with torch.set_grad_enabled(True):
            masks : Dict[str, torch.Tensor] = self._inner_model(specs_normalized, batch=batch)
            batch = self.apply_masks(masks, specs_unnormalized, batch=batch)

        return batch

    def apply_masks(self, masks: Dict[str, torch.Tensor], specs_unnormalized: torch.Tensor, batch: SourceSeparationBatch) -> SourceSeparationBatch:
        estimates = {}
        for key, mask in masks.items():
            estimates[key] = {}
            estimates[key]["spectrogram"] = specs_unnormalized * mask
            estimates[key]["audio"] = self.stft.inverse(estimates[key]["spectrogram"], length=batch.mixture["audio"].shape[-1])
            
        batch.estimates = estimates
        return batch
    
    def _inner_model(self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError