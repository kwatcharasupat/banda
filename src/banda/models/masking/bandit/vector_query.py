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
                stem: nn.Parameter(torch.randn(self.config.tf_model.params.emb_dim))
                for stem in self.config.stems
            }
        )

        self.query_scale = nn.ParameterDict(
            {
                stem: nn.Parameter(torch.randn(self.config.tf_model.params.emb_dim))
                for stem in self.config.stems
            }
        )

        self.mask_estim = self._build_mask_estim()


        if hasattr(self.config, "pretrained_decoder_ckpt_path"):
            logger.info(
                "Loading pretrained decoder",
                ckpt_path=self.config.pretrained_decoder_ckpt_path,
            )
            self.load_pretrained_decoder(
                ckpt_path=self.config.pretrained_decoder_ckpt_path
            )

    def _inner_model(
        self, specs_normalized: torch.Tensor, *, batch: SourceSeparationBatch
    ):
        band_embs = self.bandsplit(specs_normalized)
        tf_outs = self.pre_tf_model(band_embs)  # (batch, n_bands, n_time, emb_dim)

        active_stems = self.get_active_stems()

        masks = {}
        for stem in active_stems:
            tf_adapted = self.adapt_query(tf_outs, stem=stem)
            tf_adapted = self.post_tf_model(tf_adapted)
            masks[stem] = self.mask_estim(tf_adapted)

        return masks

    def adapt_query(self, tf_outs: torch.Tensor, stem: str) -> torch.Tensor:
        query_bias = self.query_bias[stem]
        query_scale = self.query_scale[stem]

        tf_adapted = tf_outs * (1.0 + query_scale) + query_bias

        return tf_adapted


    def load_pretrained_decoder(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        # bandsplit_dict = {
        #     k.replace("model.bandsplit.", ""): v
        #     for k, v in state_dict.items()
        #     if k.startswith("model.bandsplit.")
        # }

        # pre_tf_model_dict = {
        #     k.replace("model.pre_tf_model.", ""): v
        #     for k, v in state_dict.items()
        #     if k.startswith("model.pre_tf_model.")
        # }

        post_tf_model_dict = {
            k.replace("model.post_tf_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.post_tf_model.")
        }

        mask_estim_dict = {
            k.replace("model.mask_estim.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.mask_estim.")
        }

        self.post_tf_model.load_state_dict(post_tf_model_dict, strict=False)
        self.mask_estim.load_state_dict(mask_estim_dict, strict=True)
