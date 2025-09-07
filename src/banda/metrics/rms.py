import torch
import torchmetrics as tm

from banda.metrics.base import MetricRegistry


class _Decibel(tm.Metric):
    def __init__(
        self,
    ):
        super().__init__()

        self.register_buffer("sum_db", torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0.0))

    def _db_rms(self, x: torch.Tensor) -> torch.Tensor:
        return 20 * torch.log10(torch.sqrt(torch.mean(torch.square(x))) + 1e-6)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        return self.sum_db / self.count if self.count > 0 else torch.tensor(0.0)

    def reset(self) -> None:
        self.sum_db.zero_()
        self.count.zero_()


class PredDecibel(_Decibel):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        db_rms = self._db_rms(preds)

        self.sum_db += db_rms
        self.count += 1


class TargetDecibel(_Decibel):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        db_rms = self._db_rms(target)

        self.sum_db += db_rms
        self.count += 1


MetricRegistry.METRIC_REGISTRY["TargetDecibel"] = TargetDecibel
MetricRegistry.METRIC_REGISTRY["PredDecibel"] = PredDecibel
