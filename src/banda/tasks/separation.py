import torch
import pytorch_lightning as pl
from typing import Any, Dict, Union
from banda.models.core_models.banquet_separator import Banquet
from banda.losses.separation_loss_handler import SeparationLossHandler
from banda.metrics.metric_handler import MetricHandler
from banda.utils.registry import TASKS_REGISTRY
import structlog
import logging
import hydra.utils # Import hydra.utils

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


@TASKS_REGISTRY.register("separation_task")
class SeparationTask(pl.LightningModule):
    """
    A PyTorch Lightning module for source separation tasks.

    This module encapsulates the model, loss function, and metric calculation
    for training and evaluating source separation models.
    """

    def __init__(
        self,
        model: Banquet,
        loss_handler: SeparationLossHandler,
        metric_handler: MetricHandler,
        optimizer: Dict[str, Any],
    ):
        """
        Initializes the SeparationTask.

        Args:
            model (Separator): The source separation model.
            loss_handler (SeparationLossHandler): The handler for calculating loss.
            metric_handler (MetricHandler): The handler for calculating metrics.
            optimizer (Dict[str, Any]): Configuration for the optimizer.
        """
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_config = optimizer
        self.save_hyperparameters(ignore=["model", "loss_handler", "metric_handler"])

    def setup(self, stage: str) -> None:
        """
        Setup hook called before training, validation, or testing.
        Used to move loss and metric handlers to the correct device.
        """
        print(f"SeparationTask.setup called with stage: {stage}")
        if self.trainer.accelerator == "mps":
            logger.debug(f"setup: Moving loss_handler to {self.device}")
            self.loss_handler.to(self.device)
            # Removed explicit metric_handler.to(self.device) as it's now an nn.Module
            # and will be moved by Lightning automatically.
            # Removed explicit metric.to(self.device) calls as well.


    def forward(self, audio: torch.Tensor, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            audio (torch.Tensor): Input audio tensor.
            query (torch.Tensor): Input query tensor.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated audio tensors.
        """
        return self.model(audio, query)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss for the current step.
        """
        mix_audio = batch.mixture # Changed from batch.mix_audio
        query = batch.query # Extract query from batch
        predictions = self.forward(mix_audio, query) # Pass query to forward
        loss = self.loss_handler.calculate_loss(predictions, batch)
        if torch.isnan(loss):
            logger.error("Detected NaN in training loss. Terminating training.")
            raise ValueError("NaN detected in training loss.")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "train") # Re-enabled
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
        """
        mix_audio = batch.mixture # Changed from batch.mix_audio
        query = batch.query # Extract query from batch
        predictions = self.forward(mix_audio, query) # Pass query to forward
        loss = self.loss_handler.calculate_loss(predictions, batch)
        if torch.isnan(loss):
            logger.error("Detected NaN in validation loss. Terminating training.")
            raise ValueError("NaN detected in validation loss.")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "val") # Re-enabled

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        Computes and logs epoch-level metrics.
        """
        self.metric_handler.compute_and_log_epoch_metrics(self, "val") # Re-enabled

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        Performs a single test step.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
        """
        mix_audio = batch.mixture # Changed from batch.mix_audio
        query = batch.query # Extract query from batch
        predictions = self.forward(mix_audio, query) # Pass query to forward
        loss = self.loss_handler.calculate_loss(predictions, batch)
        if torch.isnan(loss):
            logger.error("Detected NaN in test loss. Terminating training.")
            raise ValueError("NaN detected in test loss.")
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "test") # Re-enabled

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.
        Computes and logs epoch-level metrics.
        """
        self.metric_handler.compute_and_log_epoch_metrics(self, "test") # Re-enabled

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer = hydra.utils.instantiate(self.optimizer_config, self.parameters())
        return optimizer
