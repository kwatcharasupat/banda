import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.bandit.base import BaseBandit

from torch import nn

import structlog

logger = structlog.get_logger(__name__)


class StemPrefixQueryBandit(BaseBandit):

    PREFIX_TYPES = ["stem", "mixture"]

    def __init__(self, *, config):
        super().__init__(config=config)

        self.prefix_dict = nn.ParameterDict(
            {
                prefix: nn.Parameter(torch.randn(self.config.tf_model.params.emb_dim))
                for prefix in self.PREFIX_TYPES
            }
        )

        self.stem_dict = nn.ParameterDict(
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
        band_embs = self.bandsplit(specs_normalized, batch=batch)
        tf_outs = self.pre_tf_model(band_embs)  # (batch, n_bands, n_time, emb_dim)
        # print("TF OUTS SHAPE:", tf_outs.shape)

        active_stems = self.get_active_stems()

        masks = {}
        for stem in active_stems:
            tf_adapted, prefix_idx = self.adapt_query(tf_outs, stem=stem)
            # print("TF ADAPTED SHAPE:", tf_adapted.shape)
            tf_adapted = self.post_tf_model(tf_adapted)
            # print("TF ADAPTED POST SHAPE:", tf_adapted.shape)
            tf_adapted = self.remove_prefix(tf_adapted, prefix_idx=prefix_idx)
            # print("TF ADAPTED NO PREFIX SHAPE:", tf_adapted.shape)
            masks[stem] = self.mask_estim(tf_adapted)

        return masks

    def remove_prefix(self, tf_adapted: torch.Tensor, prefix_idx: int) -> torch.Tensor:
        return tf_adapted[:, :, prefix_idx:, :]

    def adapt_query(self, tf_outs: torch.Tensor, stem: str) -> torch.Tensor:
        
        stem_prefix = self.prefix_dict["stem"]
        stem_vector = self.stem_dict[stem]
        mixture_prefix = self.prefix_dict["mixture"]

        batch, n_bands, n_time, emb_dim = tf_outs.shape
        stem_prefix_ = stem_prefix.view(1, 1, 1, emb_dim).expand(batch, n_bands, 1, -1)
        stem_vector_ = stem_vector.view(1, 1, 1, emb_dim).expand(batch, n_bands, 1, -1)
        mixture_prefix_ = mixture_prefix.view(1, 1, 1, emb_dim).expand(batch, n_bands, 1, -1)

        tf_adapted = torch.concat(
            [
                stem_prefix_,
                stem_vector_,
                mixture_prefix_,
                tf_outs
            ],
            dim=2,
        )

        prefix_idx = 3  # number of prefix tokens added

        return tf_adapted, prefix_idx


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
