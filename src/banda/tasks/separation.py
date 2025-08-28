import torch
import pytorch_lightning as pl
from typing import Any, Dict, Union, Optional
from banda.configs.train_configs import OptimizerConfig
# from banda.models.bandit_separator import Bandit # Import Bandit
from banda.losses.loss_handler import LossHandler
from banda.metrics.metric_handler import MetricHandler
from banda.models.separator import SeparatorModel
import structlog
import logging
import hydra.utils # Import hydra.utils

from pydantic import BaseModel
# Removed: from banda.configs.train_configs import OptimizerConfig # Import OptimizerConfig from train_configs

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

class SeparationTask(pl.LightningModule):
    """
    A PyTorch Lightning module for source separation tasks.

    This module encapsulates the model, loss function, and metric calculation
    for training and evaluating source separation models.
    """

    def __init__(
        self,
        model: SeparatorModel,
        loss_handler: LossHandler,
        metric_handler: MetricHandler,
        optimizer_config: OptimizerConfig,
    ):
        """
        Initializes the SeparationTask.

        Args:
            model (Separator): The source separation model.
            loss_handler (SeparationLossHandler): The handler for calculating loss.
            metric_handler (MetricHandler): The handler for calculating metrics.
            optimizer_config (OptimizerConfig): Configuration for the optimizer.
        """
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_config = optimizer_config
        self.save_hyperparameters(ignore=["model", "loss_handler", "metric_handler"])

    def setup(self, stage: str) -> None:
        """
        Setup hook called before training, validation, or testing.
        Used to move loss and metric handlers to the correct device.
        """
        if self.trainer.accelerator == "mps":
            logger.debug(f"setup: Moving loss_handler to {self.device}")
            self.loss_handler.to(self.device)
            # Removed explicit metric_handler.to(self.device) as it's now an nn.Module
            # and will be moved by Lightning automatically.
            # Removed explicit metric.to(self.device) calls as well.


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
        Returns:
            torch.Tensor: The calculated loss for the current step.
        """
        mix_audio = batch.mixture.audio
        query = batch.query_class.class_label if batch.query_class and batch.query_class.class_label is not None else None
        predictions = self.forward(mix_audio, query)
        loss = self.loss_handler.calculate_loss(predictions, batch)
        if torch.isnan(loss):
            logger.error("Detected NaN in training loss. Terminating training.")
            raise ValueError("NaN detected in training loss.")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
        """
        mix_audio = batch.mixture.audio
        query = batch.query_class.class_label if batch.query_class and batch.query_class.class_label is not None else None
        predictions = self.forward(mix_audio, query)
        loss = self.loss_handler.calculate_loss(predictions, batch)
        if torch.isnan(loss):
            logger.error("Detected NaN in validation loss. Terminating training.")
            raise ValueError("NaN detected in validation loss.")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "val")

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        Computes and logs epoch-level metrics.
        """
        self.metric_handler.compute_and_log_epoch_metrics(self, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        Performs a single test step.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
        """
        mix_audio = batch.mixture.audio
        query = batch.query_class.class_label if batch.query_class and batch.query_class.class_label is not None else None
        predictions = self.forward(mix_audio, query)
        loss = self.loss_handler.calculate_loss(predictions, batch)
        if torch.isnan(loss):
            logger.error("Detected NaN in test loss. Terminating training.")
            raise ValueError("NaN detected in test loss.")
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "test")

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.
        Computes and logs epoch-level metrics.
        """
        self.metric_handler.compute_and_log_epoch_metrics(self, "test")

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer_class = hydra.utils.get_class(self.optimizer_config.target_)
        optimizer_params = {k: v for k, v in self.optimizer_config.model_dump().items() if k != "target_"}
        optimizer = optimizer_class(self.parameters(), **optimizer_params)
        return optimizer
