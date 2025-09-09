import torchmetrics as tm
import torch

from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.functional.audio.snr import signal_noise_ratio

from banda.metrics.base import MetricRegistry


class SNR(SignalNoiseRatio):
    def __init__(self, threshold_db: float = -60.0, zero_mean: bool = False, **kwargs):
        super().__init__(zero_mean=zero_mean, **kwargs)
        self.threshold_db = threshold_db

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""

        target_ = target.flatten(start_dim=1)
        target_dbrms = 10 * torch.log10(
            torch.mean(target_**2, dim=-1, keepdim=False) + 1e-8
        )  # (batch,)
        target_filter = target_dbrms > self.threshold_db

        preds = preds[target_filter]
        target = target[target_filter]

        snr_batch = signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()


MetricRegistry.METRIC_REGISTRY["SignalNoiseRatio"] = tm.SignalNoiseRatio
MetricRegistry.METRIC_REGISTRY["ScaleInvariantSignalNoiseRatio"] = (
    tm.ScaleInvariantSignalNoiseRatio
)

MetricRegistry.METRIC_REGISTRY["SNR"] = SNR
