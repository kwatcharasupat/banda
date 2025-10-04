from omegaconf import DictConfig
import torch
from banda.data.item import SourceSeparationBatch
from banda.losses.base import BaseRegisteredLoss, LossDict

import torchaudio as ta
from banda.losses.utils import _dbrms


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
            source = sources.get(key, {}).get(domain, None)
            if source is None:
                source = torch.zeros_like(estimate, requires_grad=False)

            if "spectrogram/" in self.config.domain and self.stft is not None:
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

        return self._agg_contrib(snr, y_pred=y_pred, y_true=y_true)

    def _agg_contrib(
        self, snr: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return -torch.mean(snr)


class L1SNRLossCapSilenceContrib(L1SNRLoss):
    def __init__(self, *, config: DictConfig) -> None:
        super().__init__(config=config)

        self.max_silence_contrib = self.config.max_silence_contrib
        self.silence_thresh_db = self.config.silence_thresh_db

    def _get_weights(self, y_true: torch.Tensor):
        with torch.no_grad():
            y_true = y_true.flatten(start_dim=1)
            _db_true = _dbrms(y_true, eps=self.eps)
            silence_mask = _db_true < self.silence_thresh_db
            n_silence = torch.sum(silence_mask).item()

            if n_silence == 0:
                return None

            n_total = y_true.shape[0]
            silence_ratio = n_silence / n_total

            if silence_ratio < self.max_silence_contrib:
                return None

            # adjust the weights of silence samples to be at most max_silence_contrib out of 1.0
            silence_weight = self.max_silence_contrib * n_total / n_silence
            non_silence_weight = (
                (1.0 - self.max_silence_contrib) * n_total / (n_total - n_silence)
            )

            weights = torch.where(
                silence_mask,
                torch.tensor(silence_weight, device=y_true.device, dtype=y_true.dtype),
                torch.tensor(non_silence_weight, device=y_true.device, dtype=y_true.dtype),
            )

            assert weights.sum().item() == n_total, (
                f"weights.sum()={weights.sum().item()}, n_total={n_total}"
            )

        return weights


    def _agg_contrib(
        self, snr: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        
        weights = self._get_weights(y_true=y_true)
        if weights is None:
            return - torch.mean(snr)
        weighted_snr = weights * snr
        return -torch.mean(weighted_snr)
