import random
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.bandit.base import BaseBandit
from torch.nn.utils.parametrizations import weight_norm


from torch import nn

import structlog

from banda.modules.pretrained.passt import Passt

logger = structlog.get_logger(__name__)


class FiLMAdaptor(nn.Module):
    def __init__(self, *, emb_dim: int):
        super().__init__()

        self.scale = nn.Sequential(
            weight_norm(nn.Linear(emb_dim, emb_dim)),
            nn.ELU(),
            weight_norm(nn.Linear(emb_dim, emb_dim)),
        )

        self.bias = nn.Sequential(
            weight_norm(nn.Linear(emb_dim, emb_dim)),
            nn.ELU(),
            weight_norm(nn.Linear(emb_dim, emb_dim)),
        )

    def forward(self, x: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_bands, n_time, emb_dim)
        query_emb: (batch, emb_dim)
        """
        query_scale = self.scale(query_emb)  # (batch, emb_dim)
        query_bias = self.bias(query_emb)  # (batch, emb_dim)

        x = x * (1.0 + query_scale[:, None, None, :]) + query_bias[:, None, None, :]

        return x


class EmbeddingQueryBandit(BaseBandit):
    def __init__(self, *, config):
        super().__init__(config=config)
        self.mask_estim = self._build_mask_estim()

        self._build_query_encoder()
        self._build_query_adaptor()


    def _build_query_encoder(self):
        emb_dim = self.config.tf_model.params.emb_dim
        self.query_encoder = Passt(emb_dim=emb_dim, fs=self.config.spectrogram.fs)

    def _build_query_adaptor(self):
        emb_dim = self.config.tf_model.params.emb_dim
        self.query_adaptor = FiLMAdaptor(emb_dim=emb_dim)

    def _inner_model(
        self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch
    ):
        band_embs = self.bandsplit(specs_normalized)
        tf_outs = self.pre_tf_model(band_embs)  # (batch, n_bands, n_time, emb_dim)

        active_stems = self.get_active_stems(batch=batch)

        masks = {}
        for stem in active_stems:
            tf_adapted = self.adapt_query(tf_outs, stem=stem, batch=batch)
            tf_adapted = self.post_tf_model(tf_adapted)
            masks[stem] = self.mask_estim(tf_adapted)

        return masks

    def adapt_query(
        self, tf_outs: torch.Tensor, stem: str, batch: SourceSeparationBatch
    ) -> torch.Tensor:
        query_audio = batch.queries[stem]["audio"]
        query_emb = self.query_encoder(query_audio)  # (batch, emb_dim)
        tf_adapted = self.query_adaptor(tf_outs, query_emb=query_emb)

        return tf_adapted

    def get_active_stems(self, batch: SourceSeparationBatch):
        stems = list(batch.queries.keys())

        if self.training:
            stems = random.sample(
                stems,
                k=min(self.config.max_simultaneous_stems, len(stems)),
            )

        return stems
