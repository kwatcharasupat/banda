import random
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.bandit.base import BaseBandit

from torch import nn

import structlog

logger = structlog.get_logger(__name__)


class VectorDictQueryBandit(BaseBandit):
    def __init__(self, *, config):
        super().__init__(config=config)

        self.query_bias = nn.ParameterDict(
            {
                stem: nn.Parameter(torch.randn(self.config.tf_model.emb_dim))
                for stem in self.config.stems
            }
        )

        self.query_scale = nn.ParameterDict(
            {
                stem: nn.Parameter(torch.randn(self.config.tf_model.emb_dim))
                for stem in self.config.stems
            }
        )

        self.mask_estim = self._build_mask_estim()

    def _inner_model(
        self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch
    ):
        band_embs = self.bandsplit(specs_normalized)
        tf_outs = self.tf_model(band_embs)  # (batch, n_bands, n_time, emb_dim)

        active_stems = random.choices(
            self.config.stems, k=self.config.max_simultaneous_stems
        )

        masks = {}
        for stem in active_stems:
            tf_adapted = self.adapt_query(tf_outs, stem=stem)
            masks[stem] = self.mask_estim(tf_adapted)

        return masks

    def adapt_query(self, tf_outs: torch.Tensor, stem: str) -> torch.Tensor:
        query_bias = self.query_bias[stem]
        query_scale = self.query_scale[stem]

        tf_adapted = tf_outs * (1.0 + query_scale) + query_bias

        return tf_adapted
