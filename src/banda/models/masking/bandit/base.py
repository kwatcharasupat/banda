from typing import Dict
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.base import _BaseMaskingModel
from banda.modules.bandsplit.band_specs import MusicalBandsplitSpecification
from banda.modules.bandsplit.bandsplit import BandSplitModule
from banda.modules.maskestim.maskestim import OverlappingMaskEstimationModule
from banda.modules.tfmodels.base import TFModelRegistry
import random
from torch import nn

import structlog
logger = structlog.get_logger(__name__)


class BaseBandit(_BaseMaskingModel):
    def __init__(self, *, config):
        super().__init__(config=config)

        self._build_bandsplit()
        self._build_tfmodel()


        if hasattr(self.config, "pretrained_encoder_ckpt_path"):
            logger.info(
                "Loading pretrained encoder",
                ckpt_path=self.config.pretrained_encoder_ckpt_path,
            )
            self.load_pretrained_encoder(
                ckpt_path=self.config.pretrained_encoder_ckpt_path
            )

    def _inner_model(
        self, specs_normalized: torch.Tensor, batch: SourceSeparationBatch
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _build_bandsplit(self):
        self.band_specs = MusicalBandsplitSpecification(
            n_fft=self.config.spectrogram.n_fft,
            n_bands=self.config.bandsplit.n_bands,
            fs=self.config.spectrogram.fs,
        ).get_band_specs()

        self.bandsplit = BandSplitModule(
            band_specs=self.band_specs,
            emb_dim=self.config.bandsplit.emb_dim,
            in_channels=self.config.bandsplit.in_channels,
        )

    def _build_tfmodel(self):
        cls_str = self.config.tf_model.cls

        cls = TFModelRegistry.get_registry().get(cls_str)
        if cls is None:
            raise ValueError(
                f"TF model class '{cls_str}' not found in registry. Allowed classes are: {list(TFModelRegistry.get_registry().keys())}"
            )

        tf_model_params = self.config.tf_model.params
        if self.config.tf_model.use_split_model:
            tf_model_params["n_modules"] = tf_model_params["n_modules"] // 2

        self.pre_tf_model = cls(config=tf_model_params)

        if self.config.tf_model.use_split_model:
            self.post_tf_model = cls(config=tf_model_params)
        else:
            self.post_tf_model = nn.Identity()

    def _build_mask_estim(self):
        mask_estim = OverlappingMaskEstimationModule(
            in_channels=self.config.mask_estim.in_channels,
            band_specs=self.band_specs,
            n_freq=self.config.spectrogram.n_fft // 2 + 1,
            emb_dim=self.config.mask_estim.emb_dim,
            mlp_dim=self.config.mask_estim.mlp_dim,
        )

        return mask_estim

    def get_active_stems(self):
        if self.training:
            stems = random.sample(
                list(self.config.stems),
                k=min(self.config.max_simultaneous_stems, len(self.config.stems)),
            )
            return stems
        else:
            return self.config.stems

    def load_pretrained_encoder(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        bandsplit_dict = {
            k.replace("model.bandsplit.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.bandsplit.")
        }

        pre_tf_model_dict = {
            k.replace("model.pre_tf_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.pre_tf_model.")
        }

        self.bandsplit.load_state_dict(bandsplit_dict)
        self.pre_tf_model.load_state_dict(pre_tf_model_dict, strict=False)

    