

from typing import Literal
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn

from banda.data.item import SourceSeparationBatch
from banda.losses.base import LossDict
from banda.losses.handler import LossHandler
from banda.models.base import BaseRegisteredModel

class SourceSeparationSystem(pl.LightningModule):
    def __init__(self, 
        *,
        model: BaseRegisteredModel,
        loss_handler: LossHandler,
        metric_handler: nn.Module | None = None,
        optimizer_config: DictConfig | None = None
    ):
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_config = optimizer_config

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.common_step(batch, mode="train")

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.common_step(batch, mode="val")

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        raise NotImplementedError("Test step is not implemented yet.")
    
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        raise NotImplementedError("Predict step is not implemented yet.")

    def common_step(self, batch: dict, *, mode: Literal["train", "val", "test", "predict"]):
        batch = self.forward(batch)
        loss_dict = self.loss_handler(batch)
        metric_dict = self.metric_handler(batch) if self.metric_handler else {}
        self._log(loss_dict=loss_dict, metric_dict=metric_dict, mode=mode)

        return loss_dict.total_loss

    def forward(self, batch: dict):
        batch = SourceSeparationBatch.model_validate(batch)
        return self.model(batch)

    def configure_optimizers(self):
        # Define the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_config.optimizer.params.init_lr)
        return [optimizer], []

    def _log(self, loss_dict: LossDict, metric_dict: dict, mode: Literal["train", "val", "test", "predict"]):

        prefix = f"{mode}/"
        on_step = mode == "train"
        on_epoch = mode != "train"

        loss_contrib_dict = {f"{prefix}loss/{key}": value for key, value in loss_dict.loss_contrib.items()}
        metric_dict = {f"{prefix}metric/{key}": value for key, value in metric_dict.items()}

        self.log_dict(loss_contrib_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False)
        self.log_dict(metric_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False)
        
        self.log(name="loss", value=loss_dict.total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log(name=f"{mode}/loss", value=loss_dict.total_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)