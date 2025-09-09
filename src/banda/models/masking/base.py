from typing import Dict

from omegaconf import DictConfig
from banda.data.item import SourceSeparationBatch
from banda.models.base import BaseRegisteredModel

import torch

import structlog


from banda.modules.normalization.normalization import Normalizer
from banda.modules.spectral.stft import Spectrogram


logger = structlog.get_logger(__name__)


def _snr(
    estimate: torch.Tensor, source: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    noise = estimate - source
    return 10 * torch.log10(
        (torch.mean(torch.square(torch.abs(source))) + eps)
        / (torch.mean(torch.square(torch.abs(noise))) + eps)
    )


class _BaseMaskingModel(BaseRegisteredModel):
    def __init__(self, *, config: DictConfig):
        super().__init__(config=config)

        self.normalizer = Normalizer(config=self.config.normalization)
        self.stft = Spectrogram(config=self.config.spectrogram)

    def forward(self, batch: dict) -> SourceSeparationBatch:
        with torch.no_grad():
            batch: SourceSeparationBatch = SourceSeparationBatch.model_validate(batch)
            assert "mixture" not in batch.sources, "Mixture should not be in sources"
            # print(batch.sources.keys())
            # logger.info("Sources keys: %s", batch.sources.keys())
            # _mixture = sum(batch.sources[key]["audio"] for key in batch.sources)
            # _mixture_snr = _snr(_mixture, batch.mixture["audio"])
            # assert _mixture_snr >= 30, f"Mixture does not match sum of sources, SNR = {_mixture_snr}"

            batch.mixture["audio/normalized"] = self.normalizer(batch.mixture["audio"])
            specs_unnormalized: torch.Tensor = self.stft(batch.mixture["audio"])
            specs_normalized: torch.Tensor = self.stft(
                batch.mixture["audio/normalized"]
            )

        with torch.set_grad_enabled(True):
            masks: Dict[str, torch.Tensor] = self._inner_model(
                specs_normalized, batch=batch
            )
            batch = self.apply_masks(masks, specs_unnormalized, batch=batch)

        with torch.no_grad():
            if batch.sources:
                for key in batch.estimates:
                    # only compute this for stems with estimates to reduce the compute
                    source = batch.sources[key]["audio"]
                    source_specs = self.stft(source)
                    batch.sources[key]["spectrogram"] = source_specs

        return batch

    def apply_masks(
        self,
        masks: Dict[str, torch.Tensor],
        specs_unnormalized: torch.Tensor,
        batch: SourceSeparationBatch,
    ) -> SourceSeparationBatch:
        estimates = {}
        for key, mask in masks.items():
            estimates[key] = {}
            estimates[key]["spectrogram"] = specs_unnormalized * mask
            estimates[key]["audio"] = self.stft.inverse(
                estimates[key]["spectrogram"], length=batch.mixture["audio"].shape[-1]
            )

        batch.estimates = estimates
        return batch

    def _inner_model(
        self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
