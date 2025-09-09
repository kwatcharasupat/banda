from omegaconf import DictConfig
import torch
from banda.data.item import SourceSeparationBatch
from banda.losses.base import BaseRegisteredLoss, LossDict

import torchaudio as ta


class L1SNRLoss(BaseRegisteredLoss):
    def __init__(
        self,
        *,
        config: DictConfig,
    ) -> None:
        super().__init__(config=config)
        self.eps = self.config.eps

        if "spectrogram/" in self.config.domain:
            self.stft = ta.transforms.Spectrogram(
                n_fft=self.config.n_fft,
                win_length=self.config.n_fft,
                hop_length=self.config.n_fft // 4,
            )
        else:
            self.stft = None

    def forward(self, batch: SourceSeparationBatch) -> LossDict:
        losses = {}

        estimates = batch.estimates
        sources = batch.sources

        for key in estimates.keys():
            if "spectrogram/" in self.config.domain:
                domain = "audio"
            else:
                domain = self.config.domain

            estimate = estimates[key][domain]
            source = sources[key][domain]

            if self.stft is not None:
                estimate = self.stft(estimate)
                source = self.stft(source)

            losses[key] = self._loss_func(estimate, source)

        if not losses:
            raise ValueError(
                f"No losses were computed. Check the keys in estimates and sources. Estimates keys: {estimates.keys()}, Sources keys: {sources.keys()}"
            )

        total_loss = sum(losses.values())

        return LossDict(total_loss=total_loss, loss_contrib=losses)

    def _loss_func(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.flatten(start_dim=1)
        y_true = y_true.flatten(start_dim=1)

        if torch.is_complex(y_pred):
            y_pred = torch.view_as_real(y_pred)
            y_true = torch.view_as_real(y_true)
        else:
            y_pred = y_pred.unsqueeze(-1)
            y_true = y_true.unsqueeze(-1)

        l1_error = torch.mean(torch.abs(y_pred - y_true), dim=-2)
        l1_true = torch.mean(torch.abs(y_true), dim=-2)

        snr = 20.0 * torch.log10((l1_true + self.eps) / (l1_error + self.eps))

        snr = snr.sum(dim=-1)

        return -torch.mean(snr)
