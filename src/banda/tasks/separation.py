#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import torch
import pytorch_lightning as pl
import torch
import torchmetrics
from typing import Dict, Any, Union

from banda.models.separator import Separator
from banda.data.batch_types import SeparationBatch, FixedStemSeparationBatch
from banda.losses.base import LossHandler


class SeparationTask(pl.LightningModule):
    """
    PyTorch Lightning module for source separation.
    """

    def __init__(
        self,
        model: Separator,
        loss_handler: LossHandler,
        optimizer_config: Dict[str, Any],
        metric_fn: Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]],
    ) -> None:
        """
        Args:
            model (Separator): The source separation model.
            loss_handler (LossHandler): The loss handler to use for calculating loss.
            optimizer_config (Dict[str, Any]): Configuration for the optimizer.
            metric_fn (Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]]):
                The metric function(s) to use for evaluation. Can be a single metric or a dictionary of metrics.
        """
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.optimizer_config = optimizer_config
        self.metric_fn = metric_fn
        self.save_hyperparameters(ignore=["model", "loss_handler", "metric_fn"])

        # If metric_fn is a single metric, wrap it in a MetricCollection for consistent handling
        if isinstance(self.metric_fn, torchmetrics.Metric):
            self.metrics = torchmetrics.MetricCollection({"default_metric": self.metric_fn})
        elif isinstance(self.metric_fn, dict):
            self.metrics = torchmetrics.MetricCollection(self.metric_fn)
        else:
            raise TypeError("metric_fn must be a torchmetrics.Metric or a dict of torchmetrics.Metric.")


    def forward(self, batch: SeparationBatch) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            batch (SeparationBatch): The input batch containing the mixture and other data.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                                     Each tensor has shape: (batch_size, channels, samples)
        """
        # The model's forward method should be adapted to handle different batch types
        # For now, assuming it takes the mixture from the batch.
        # This will need to be refined based on the specific model architecture.
        if isinstance(batch, FixedStemSeparationBatch):
            mixture = batch.audio.mixture
        else:
            # Handle other batch types or raise an error if not supported by the model
            raise NotImplementedError(f"Model forward pass not implemented for batch type: {type(batch)}")
        
        return self.model(mixture)

    def training_step(self, batch: SeparationBatch, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (SeparationBatch): The input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The calculated loss for the current training step.
        """
        predictions = self.forward(batch)
        loss = self.loss_handler.calculate_loss(predictions, batch)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: SeparationBatch, batch_idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            batch (SeparationBatch): The input batch.
            batch_idx (int): Index of the current batch.
        """
        predictions = self.forward(batch)
        loss = self.loss_handler.calculate_loss(predictions, batch)
        
        self.log("val_loss", loss)

        # Update metrics
        if isinstance(batch, FixedStemSeparationBatch):
            true_sources = batch.audio.sources
            for source_name, sep_audio in predictions.items():
                if source_name in true_sources:
                    # Assuming metric_fn expects (preds, target)
                    # Need to ensure sep_audio and true_sources[source_name] have compatible shapes
                    # For SI-SNR, it's (preds, target)
                    # The metric collection will handle logging for each metric
                    self.metrics.update(sep_audio, true_sources[source_name])
        else:
            # Handle metrics for other batch types
            pass # For now, skip metrics for other batch types

    def on_validation_epoch_end(self) -> None:
        """
        Logs metrics at the end of the validation epoch.
        """
        metrics_dict = self.metrics.compute()
        for metric_name, metric_value in metrics_dict.items():
            self.log(f"val_{metric_name}", metric_value)
        self.metrics.reset()

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        # Use hydra.utils.instantiate to create the optimizer
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer