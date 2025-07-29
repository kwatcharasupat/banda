#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import pytorch_lightning as pl
import torch
import torch.nn as nn
from banda.data.batch_types import QueryAudioSeparationBatch, QueryClassSeparationBatch, FixedStemSeparationBatch
import torchmetrics
from typing import Dict, Any, Union
from omegaconf import DictConfig
import importlib
import hydra.utils


from banda.data.batch_types import SeparationBatch
from banda.losses.base import LossHandler
from banda.losses.multi_resolution_l1_snr import MultiResolutionSTFTLoss
from banda.losses.separation_loss_handler import SeparationLossHandler
from banda.losses.time_domain_loss import TimeDomainLoss # Import TimeDomainLoss
from banda.utils.inference import OverlapAddProcessor
from banda.utils.inference_handler import InferenceHandler
from banda.metrics.metric_handler import MetricHandler # Import MetricHandler


class SeparationTask(pl.LightningModule):
    """
    PyTorch Lightning module for source separation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_handler: LossHandler,
        optimizer_config: Dict[str, Any],
        metric_fn: Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]],
        inference_config: DictConfig,
    ) -> None:
        """
        Args:
            model (Separator): The source separation model.
            loss_handler (LossHandler): The loss handler to use for calculating loss.
            optimizer_config (Dict[str, Any]): Configuration for the optimizer.
            metric_fn (Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]]):
                The metric function(s) to use for evaluation. Can be a single metric or a dictionary of metrics.
            inference_config (DictConfig): Configuration for inference, including overlap-add parameters.
        """
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.optimizer_config = optimizer_config
        self.save_hyperparameters(ignore=["model", "loss_handler", "metric_fn", "inference_config"])

        self.metric_handler = MetricHandler(metric_fn)
        self.inference_handler = InferenceHandler(model, inference_config)


    def forward(self, batch: SeparationBatch) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            batch (FixedStemSeparationBatch): The input batch containing the mixture and other data.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                                     Each tensor has shape: (batch_size, channels, samples)
        """
        mixture = batch.audio.mixture
        
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
        
        self.log("train_loss", loss, prog_bar=True)
        self.metric_handler.update_step_metrics(self, predictions, batch, "train")

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
        self.metric_handler.update_step_metrics(self, predictions, batch, "val")


    def on_validation_epoch_end(self) -> None:
        """
        Logs metrics at the end of the validation epoch.
        """
        self.metric_handler.compute_and_log_epoch_metrics(self, "val")

    def test_step(self, batch: SeparationBatch, batch_idx: int, batch_size_cut_factor: int = 1) -> None:
        """
        Performs a single test step using overlap-add inference.
        """
        full_length_mixture_audio = batch.audio.mixture.squeeze(0) # Remove batch dimension for single audio
        
        # Get true sources using the polymorphic method
        true_sources = {k: v.squeeze(0) for k, v in batch.audio.sources.items()} # Remove batch dimension

        # Get query for inference using the polymorphic method
        query_for_inference = batch.get_inference_query()

        separated_audio = self.inference_handler.perform_inference(
            full_length_mixture_audio,
            query_for_inference,
            self.device,
            batch_size_cut_factor,
        )

        # Calculate loss and metrics on the full-length separated audio
        # Need to re-add batch dimension for loss calculation and move to correct device
        predictions_batched = {k: v.unsqueeze(0).to(self.device) for k, v in separated_audio.items()}
        true_sources_batched = {k: v.unsqueeze(0).to(self.device) for k, v in true_sources.items()}
        
        # Create a dummy batch for loss calculation
        dummy_batch = FixedStemSeparationBatch(
            audio=batch.audio.__class__(
                mixture=batch.audio.mixture, # Original mixture with batch dim
                sources=true_sources_batched
            ),
            identifier=batch.identifier
        )

        loss = self.loss_handler.calculate_loss(predictions_batched, dummy_batch)
        self.log("test_loss", loss)

        # Use the metric handler for updating test metrics
        self.metric_handler.update_step_metrics(self, predictions_batched, dummy_batch, "test")


    def on_test_epoch_end(self) -> None:
        """
        Logs metrics at the end of the test epoch.
        """
        self.metric_handler.compute_and_log_epoch_metrics(self, "test")

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of training.
        Ensures that the loss function and metrics are on the correct device.
        """
        # Check if the accelerator is MPS and move the loss_handler and metrics to the MPS device
        if self.trainer.accelerator == "mps":
            self.loss_handler.to(self.trainer.device) # Move the entire loss_handler module
            self.metric_handler.to(self.trainer.device) # Move the entire metrics module

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer_class_path = self.optimizer_config._target_
        optimizer_kwargs = {k: v for k, v in self.optimizer_config.items() if k != '_target_'}

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
        model = hydra.utils.instantiate(config.model)
        
        # Manually instantiate the STFT loss function
        stft_loss_fn_config = config.loss.stft_loss_fn
        stft_loss_fn_class_path = stft_loss_fn_config._target_
        stft_loss_fn_kwargs = {k: v for k, v in stft_loss_fn_config.items() if k != '_target_'}

        module_name, class_name = stft_loss_fn_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        stft_loss_fn_cls = getattr(module, class_name)
        actual_stft_loss_fn = stft_loss_fn_cls(**stft_loss_fn_kwargs)

        # Manually instantiate the Time-Domain loss function
        time_loss_fn_config = config.loss.time_loss_fn
        time_loss_fn_class_path = time_loss_fn_config._target_
        time_loss_fn_kwargs = {k: v for k, v in time_loss_fn_config.items() if k != '_target_'}

        module_name, class_name = time_loss_fn_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        time_loss_fn_cls = getattr(module, class_name)
        actual_time_loss_fn = time_loss_fn_cls(**time_loss_fn_kwargs)

        # Manually instantiate the LossHandler with both loss functions
        loss_handler = SeparationLossHandler(stft_loss_fn=actual_stft_loss_fn, time_loss_fn=actual_time_loss_fn)

        optimizer_config = config.optimizer
        
        # Instantiate metric_fn
        metric_config = config.metrics
        
        # Instantiate metric_fn using hydra.utils.instantiate
        metric_fn = hydra.utils.instantiate(metric_config)

        # Pass inference config
        inference_config = config.inference

        return cls(
            model=model,
            loss_handler=loss_handler,
            optimizer_config=optimizer_config,
            metric_fn=metric_fn,
            inference_config=inference_config,
        )