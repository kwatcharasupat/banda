import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.bandit.base import BaseBandit

from torch import nn

import structlog
logger = structlog.get_logger(__name__)

class FixedStemBandit(BaseBandit):
    def __init__(self, *, config):
        super().__init__(config=config)
        
        self.mask_estim = nn.ModuleDict({
            stem: self._build_mask_estim()
            for stem in self.config.stems
        })

    def _inner_model(self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch):

        # logger.info("Input spectrogram: ", shape=specs_normalized.shape, dtype=specs_normalized.dtype)
        band_embs = self.bandsplit(specs_normalized)
        # logger.info("After bandsplit module: ", shape=band_embs.shape, dtype=band_embs.dtype)
        tf_outs = self.tf_model(band_embs)
        # logger.info("After TF model: ", shape=tf_outs.shape, dtype=tf_outs.dtype)
        masks = {
            stem: self.mask_estim[stem](tf_outs)
            for stem in self.config.stems
        }
        
        return masks