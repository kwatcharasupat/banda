from typing import Dict
import torch
from banda.data.item import SourceSeparationBatch
from banda.models.masking.base import _BaseMaskingModel
from banda.modules.bandsplit.band_specs import MusicalBandsplitSpecification
from banda.modules.bandsplit.bandsplit import BandSplitModule
from banda.modules.bandsplit.maskestim import OverlappingMaskEstimationModule
from banda.modules.bandsplit.tfmodel import RNNSeqBandModellingModule


class BaseBandit(_BaseMaskingModel):
    def __init__(self, *, config):
        super().__init__(config=config)

        self._build_bandsplit()
        self._build_tfmodel()

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
        self.tf_model = RNNSeqBandModellingModule(
            n_modules=self.config.tf_model.n_modules,
            emb_dim=self.config.tf_model.emb_dim,
            rnn_dim=self.config.tf_model.rnn_dim,
            bidirectional=self.config.tf_model.bidirectional,
            rnn_type=self.config.tf_model.rnn_type,
        )

    def _build_mask_estim(self):
        mask_estim = OverlappingMaskEstimationModule(
            in_channels=self.config.mask_estim.in_channels,
            band_specs=self.band_specs,
            n_freq=self.config.spectrogram.n_fft // 2 + 1,
            emb_dim=self.config.mask_estim.emb_dim,
            mlp_dim=self.config.mask_estim.mlp_dim,
        )

        return mask_estim
