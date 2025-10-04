from omegaconf import DictConfig
import torch
from banda.data.item import SourceSeparationBatch
from banda.losses.base import BaseRegisteredLoss, LossDict

import structlog

from banda.losses.utils import _dbrms

logger = structlog.get_logger(__name__)


class DecibelMatchLoss(BaseRegisteredLoss):
    def __init__(
        self,
        *,
        config: DictConfig,
    ) -> None:
        super().__init__(config=config)

        self.eps = self.config.eps
        self.adaptive = self.config.adaptive
        self.min_weight = self.config.min_weight
        self.max_weight = self.config.max_weight
        self.min_db = self.config.min_db
        self.max_below_true = self.config.max_below_true

        self.domain = "audio"

    def compute_weight(self, db_pred: torch.Tensor, db_true: torch.Tensor):
        with torch.no_grad():
            db_true_to_floor = torch.minimum(
                db_true - self.min_db,
                torch.tensor(self.max_below_true, device=db_true.device),
            )
            db_pred_to_true = db_true - db_pred

            ratio = torch.clamp(
                db_pred_to_true / (db_true_to_floor + 1.0e-12), 0.0, 1.0
            )

            weight = self.min_weight + (self.max_weight - self.min_weight) * ratio

            # the closer db_pred is to db_true, the lower the weight

            weight = torch.where(
                (db_pred > db_true) | (db_true < self.min_db),
                torch.tensor(self.min_weight),
                weight,
            )

            # assert torch.all(weight >= self.min_weight)
            # assert torch.all(weight <= self.max_weight)

            if not torch.all(weight >= self.min_weight) or not torch.all(
                weight <= self.max_weight
            ):
                # logger.warning("Some weights are below the minimum weight.")
                logger.error(
                    "Some weights are below the minimum weight.",
                    db_true=db_true,
                    db_pred=db_pred,
                    weight=weight,
                )
                raise ValueError()

        return weight

    def forward(self, batch: SourceSeparationBatch) -> LossDict:
        losses = {}

        estimates = batch.estimates
        sources = batch.sources

        for key in estimates.keys():
            estimate = estimates[key]["audio"]
            source = sources.get(key, {}).get("audio", None)
            if source is None:
                source = torch.zeros_like(estimate, requires_grad=False)
            losses[key] = self._loss_func(estimate, source)

        total_loss = sum(losses.values())

        return LossDict(total_loss=total_loss, loss_contrib=losses)

    def _loss_func(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        db_pred = _dbrms(y_pred, eps=self.eps)

        with torch.no_grad():
            y_true = y_true.reshape(batch_size, -1)
            db_true = _dbrms(y_true, eps=self.eps)

            weights = self.compute_weight(db_pred, db_true)

        diff = torch.abs(db_pred - db_true)
        loss = torch.mean(weights * diff)

        return loss
