from typing import Literal
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from banda.data.item import SourceSeparationBatch
from banda.inference.handler import InferenceHandler
from banda.losses.base import LossDict
from banda.losses.handler import LossHandler
from banda.metrics.handler import MetricHandler
from banda.models.base import BaseRegisteredModel
from pytorch_lightning.loggers import WandbLogger


class SourceSeparationSystem(pl.LightningModule):
    def __init__(
        self,
        *,
        model: BaseRegisteredModel,
        loss_handler: LossHandler,
        metric_handler: MetricHandler,
        inference_handler: InferenceHandler | None = None,
        optimizer_config: DictConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.inference_handler = inference_handler
        self.optimizer_config = optimizer_config

    def on_train_epoch_start(self):
        self.metric_handler.reset()

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch, total_loss = self.common_step(batch, mode="train")

        self.metric_handler.reset()
        self.metric_handler.update(batch)
        metric_dict = self.metric_handler.compute()
        self._log_metric(metric_dict=metric_dict, mode="train")

        return total_loss

    def on_validation_epoch_start(self):
        self.metric_handler.reset()

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch, total_loss = self.common_step(batch, mode="val")
        self.metric_handler.update(batch)
        return total_loss

    def on_validation_epoch_end(self):
        metric_dict = self.metric_handler.compute()
        self._log_metric(metric_dict=metric_dict, mode="val")

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        reconstructed_batch = self.inference_step(batch, batch_idx, dataloader_idx)

        self.metric_handler.reset()
        self.metric_handler.update(reconstructed_batch)
        metric_dict = self.metric_handler.compute()

        logger: WandbLogger = self.logger
        logger.log_table(
            key=f"test/metrics/{batch_idx}",
            columns=list(metric_dict.keys()),
            data=[list(metric_dict.values())],
        )

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        return self.inference_step(batch, batch_idx, dataloader_idx)

    def inference_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int
    ) -> SourceSeparationBatch:
        batch = SourceSeparationBatch.model_validate(batch)
        assert self.inference_handler is not None, "Inference handler is not set."

        chunked_batches = self.inference_handler.chunk_batch(batch)

        chunked_outputs = []
        for chunked_batch in chunked_batches:
            chunked_batch.mixture["audio"] = chunked_batch.mixture["audio"].to(
                self.device
            )
            chunked_output = self.forward(chunked_batch)
            for source in chunked_output.estimates:
                chunked_output.estimates[source]["audio"] = chunked_output.estimates[
                    source
                ]["audio"].cpu()
            chunked_outputs.append(chunked_output)

        reconstructed_batch = self.inference_handler.chunked_reconstruct(
            chunked_outputs
        )

        return reconstructed_batch

    def common_step(
        self, batch: dict, *, mode: Literal["train", "val", "test", "predict"]
    ):
        batch = self.forward(batch)
        loss_dict = self.loss_handler(batch)
        self._log_loss(loss_dict=loss_dict, mode=mode)

        return batch, loss_dict.total_loss

    def forward(self, batch: dict | SourceSeparationBatch) -> SourceSeparationBatch:
        batch = SourceSeparationBatch.model_validate(batch)
        return self.model(batch)

    def configure_optimizers(self):
        # Define the optimizer and learning rate scheduler

        optimizer_config = self.optimizer_config.optimizer
        cls = getattr(torch.optim, optimizer_config.cls)
        params = optimizer_config.params
        optimizer = cls(self.parameters(), **params)

        scheduler_config = self.optimizer_config.scheduler
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config.cls)
        scheduler_kwargs = scheduler_config.params
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

        return [optimizer], [scheduler]

    def _log_loss(
        self, loss_dict: LossDict, mode: Literal["train", "val", "test", "predict"]
    ):
        prefix = f"{mode}"
        on_step = mode == "train"
        on_epoch = mode != "train"

        prog_bar_loss = {
            k: v for k, v in loss_dict.loss_contrib.items() if "/" not in k
        }

        self.log_dict(prog_bar_loss, on_step=True, prog_bar=True, logger=False)

        loss_contrib_dict = {
            f"{prefix}/loss/{key}": value
            for key, value in loss_dict.loss_contrib.items()
        }

        self.log_dict(
            loss_contrib_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False
        )

        self.log(
            name="loss",
            value=loss_dict.total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{mode}/loss",
            value=loss_dict.total_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=False,
            logger=True,
        )

    def _log_metric(
        self, metric_dict: dict, mode: Literal["train", "val", "test", "predict"]
    ):
        prefix = f"{mode}"
        on_step = mode == "train"
        on_epoch = mode != "train"

        if mode == "train":
            self.log_dict(
                metric_dict, on_step=True, on_epoch=False, prog_bar=True, logger=False
            )

        metric_dict = {
            f"{prefix}/metric/{key}": value for key, value in metric_dict.items()
        }

        self.log_dict(metric_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False)
