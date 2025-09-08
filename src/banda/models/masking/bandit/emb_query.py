import random
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.bandit.base import BaseBandit

from torch import nn

import structlog

logger = structlog.get_logger(__name__)


class AudioQueryBandit(BaseBandit):
    def __init__(self, *, config):
        super().__init__(config=config)
        self.mask_estim = self._build_mask_estim()

        self.query_encoder = self._build_query_encoder()

    def _inner_model(
        self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch
    ):
        band_embs = self.bandsplit(specs_normalized)
        tf_outs = self.tf_model(band_embs)  # (batch, n_bands, n_time, emb_dim)

        active_stems = self.get_active_stems()

        masks = {}
        for stem in active_stems:
            tf_adapted = self.adapt_query(tf_outs, stem=stem, batch=batch)
            masks[stem] = self.mask_estim(tf_adapted)

        return masks

    def adapt_query(self, tf_outs: torch.Tensor, stem: str, batch: SourceSeparationBatch) -> torch.Tensor:
        
        query = batch.queries[stem]["audio"]

        # tf_adapted = tf_outs * (1.0 + query_scale) + query_bias

        # return tf_adapted
