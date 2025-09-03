

from typing import Literal
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn

from banda.data.item import SourceSeparationBatch
from banda.losses.base import LossDict
from banda.losses.handler import LossHandler
from banda.metrics.handler import MetricHandler
from banda.models.base import BaseRegisteredModel

class SourceSeparationSystem(pl.LightningModule):
    def __init__(self, 
        *,
        model: BaseRegisteredModel,
        loss_handler: LossHandler,
        metric_handler: MetricHandler,
        optimizer_config: DictConfig | None = None
    ):
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_config = optimizer_config
        
    def on_train_batch_start(self, batch, batch_idx):
        self.metric_handler.reset()
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.metric_handler.reset()

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch, total_loss =  self.common_step(batch, mode="train")

        self.metric_handler.update(batch)
        metric_dict = self.metric_handler.compute()
        self._log_metric(metric_dict=metric_dict, mode="train")

        return total_loss
    
    def on_validation_epoch_start(self):
        self.metric_handler.reset()

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch, total_loss =  self.common_step(batch, mode="val")

        self.metric_handler.update(batch)

        return total_loss
    
    def on_validation_epoch_end(self):
        metric_dict = self.metric_handler.compute()
        self._log_metric(metric_dict=metric_dict, mode="val")
        self.metric_handler.reset()

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        raise NotImplementedError("Test step is not implemented yet.")
    
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        raise NotImplementedError("Predict step is not implemented yet.")

    def common_step(self, batch: dict, *, mode: Literal["train", "val", "test", "predict"]):
        batch = self.forward(batch)
        loss_dict = self.loss_handler(batch)
        self._log_loss(loss_dict=loss_dict, mode=mode)

        return batch, loss_dict.total_loss

    def forward(self, batch: dict):
        batch = SourceSeparationBatch.model_validate(batch)
        return self.model(batch)

    def configure_optimizers(self):
        # Define the optimizer and learning rate scheduler
        
        optimizer_config = self.optimizer_config.optimizer
        cls = getattr(torch.optim, optimizer_config.cls)
        params = optimizer_config.params

        # optimizer = cls(self.model.parameters(), **params)

        no_decay_keywords = ["bias", "LayerNorm", "GroupNorm", "original1"]
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if any(keyword in name for keyword in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        params = {
            k: v for k, v in params.items() if k != "weight_decay"
        }

        optimizer = cls([
                {"params": decay_params, "weight_decay": params.get("weight_decay", 1.0e-2)},
                {"params": no_decay_params, "weight_decay": 0.0}
            ],
            **params
        )

        scheduler_config = self.optimizer_config.scheduler
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config.cls)
        scheduler_kwargs = scheduler_config.params

        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

        return [optimizer], [scheduler]

    def _log_loss(self, loss_dict: LossDict, mode: Literal["train", "val", "test", "predict"]):

        prefix = f"{mode}"
        on_step = mode == "train"
        on_epoch = mode != "train"
        
        prog_bar_loss = {
            k: v
            for k, v in loss_dict.loss_contrib.items()
            if "/" not in k
        }
        
        self.log_dict(prog_bar_loss, on_step=True, prog_bar=True, logger=False)

        loss_contrib_dict = {f"{prefix}/loss/{key}": value for key, value in loss_dict.loss_contrib.items()}

        self.log_dict(loss_contrib_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False)
        
        self.log(name="loss", value=loss_dict.total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log(name=f"{mode}/loss", value=loss_dict.total_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    def _log_metric(self, metric_dict: dict, mode: Literal["train", "val", "test", "predict"]):
        prefix = f"{mode}"
        on_step = mode == "train"
        on_epoch = mode != "train"
        
        if mode == "train":
            self.log_dict(metric_dict, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        metric_dict = {f"{prefix}/metric/{key}": value for key, value in metric_dict.items()}

        self.log_dict(metric_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False)