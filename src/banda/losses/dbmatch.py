
from omegaconf import DictConfig
import torch
from banda.data.item import SourceSeparationBatch
from banda.losses.base import BaseRegisteredLoss, LossDict


class DecibelMatchLoss(BaseRegisteredLoss):
    def __init__(
        self,
        *,
        config: DictConfig ,
    ) -> None:
        super().__init__(config=config)

        self.eps = self.config.eps
        self.adaptive = self.config.adaptive
        self.min_weight = self.config.min_weight
        self.max_weight = self.config.max_weight
        self.min_db = self.config.min_db
        self.max_below_true = self.config.max_below_true

    def compute_weight(self, db_pred: torch.Tensor, db_true: torch.Tensor):
        with torch.no_grad():
            db_true_to_floor = torch.minimum(
                db_true - self.min_db,
                torch.tensor(self.max_below_true, device=db_true.device),
            )
            db_pred_to_true = db_true - db_pred

            ratio = torch.clamp(db_pred_to_true / db_true_to_floor, 0.0, 1.0)

            weight = self.min_weight + (self.max_weight - self.min_weight) * ratio

            # the closer db_pred is to db_true, the lower the weight

            weight = torch.where(
                (db_pred > db_true) | (db_true < self.min_db),
                torch.tensor(self.min_weight),
                weight,
            )

            assert torch.all(weight >= self.min_weight)
            assert torch.all(weight <= self.max_weight)

        return weight
    
    def forward(self, batch: SourceSeparationBatch) -> LossDict:

        losses = {}
        
        estimates = batch.estimates
        sources = batch.sources

        for key in estimates.keys():
            estimate = estimates[key]["audio"]
            source = sources[key]["audio"]
            losses[key] = self._loss_func(estimate, source)

        total_loss = sum(losses.values())

        return LossDict(
            total_loss=total_loss,
            loss_contrib=losses
        )

    def _loss_func(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        db_true = 10.0 * torch.log10(
            self.eps + torch.mean(torch.square(torch.abs(y_true)), dim=-1)
        )
        db_pred = 10.0 * torch.log10(
            self.eps + torch.mean(torch.square(torch.abs(y_pred)), dim=-1)
        )
        
        if self.adaptive:
            weights = self.compute_weight(db_pred, db_true)
        else:
            weights = 1.0

        diff = torch.abs(db_pred - db_true)

        loss = torch.mean(weights * diff)

        return loss
