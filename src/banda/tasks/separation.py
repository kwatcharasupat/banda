#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#


import torch
import pytorch_lightning as pl
import torch
import torchmetrics
from typing import Dict, Any, Union
from omegaconf import DictConfig
import importlib

from banda.models.separator import Separator
from banda.data.batch_types import SeparationBatch, FixedStemSeparationBatch
from banda.losses.base import LossHandler
from banda.losses.multi_resolution_l1_snr import MultiResolutionSTFTLoss


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
        optimizer_class_path = self.optimizer_config.target_
        optimizer_kwargs = {k: v for k, v in self.optimizer_config.items() if k != 'target_'}

        module_name, class_name = optimizer_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        optimizer_cls = getattr(module, class_name)

        optimizer = optimizer_cls(params=self.parameters(), **optimizer_kwargs)
        return optimizer

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Instantiates a SeparationTask from a DictConfig.
        """
        model = Separator.from_config(config.model)
        # Corrected: Directly instantiate MultiResolutionSTFTLoss from config.loss.loss_fn
        loss_handler = MultiResolutionSTFTLoss.from_config(config.loss.loss_fn)
        optimizer_config = config.optimizer
        
        # Instantiate metric_fn
        metric_config = config.metrics
        if isinstance(metric_config, DictConfig) and 'target_' in metric_config:
            class_path = metric_config.target_
            kwargs = {k: v for k, v in metric_config.items() if k != 'target_'}
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            metric_cls = getattr(module, class_name)
            metric_fn = metric_cls(**kwargs)
        elif isinstance(metric_config, DictConfig): # Handle multiple metrics
            metric_fn = {}
            for metric_name, sub_config in metric_config.items():
                class_path = sub_config.target_
                kwargs = {k: v for k, v in sub_config.items() if k != 'target_'}
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                metric_cls = getattr(module, class_name)
                metric_fn[metric_name] = metric_cls(**kwargs)
        else:
            raise ValueError(f"Unsupported metric configuration type: {type(metric_config)}")

        return cls(
            model=model,
            loss_handler=loss_handler,
            optimizer_config=optimizer_config,
            metric_fn=metric_fn,
        )