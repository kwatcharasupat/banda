import torch
import torch.nn as nn # Import nn
import torchmetrics
import logging
from typing import Dict, Any, Union, List
import pytorch_lightning as pl
from banda.utils.registry import METRICS_REGISTRY
import structlog
logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


@METRICS_REGISTRY.register("metric_handler")
class MetricHandler(nn.Module): # Inherit from nn.Module
    """
    Handles metric calculation, updating, and logging for source separation tasks.

    This class encapsulates the logic for updating metrics, making the PyTorch LightningModule
    more generic and focused on the core training loop. It supports single or multiple metrics.
    """
    def __init__(self, metric_fn: Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]]) -> None:
        """
        Initializes the MetricHandler.

        Args:
            metric_fn (Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]]):
                The metric function(s) to use for evaluation. Can be a single torchmetrics.Metric
                instance or a dictionary of metric instances (e.g., {"sdr": SDRMetric(), "sir": SIRMetric()}).
        """
        super().__init__() # Call super constructor
        if isinstance(metric_fn, dict):
            self.metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection(metric_fn)
        else:
            # Assume it's a single metric if not a dict
            self.metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection({"default_metric": metric_fn})

    def update_step_metrics(self, pl_module: pl.LightningModule, predictions: Dict[str, torch.Tensor], batch: Any, stage: str) -> None:
        """
        Updates metrics for a single step (training, validation, or test).

        For training, metrics are computed and logged per step. For validation/test,
        the metric state is updated for epoch-end computation.

        Args:
            pl_module (pl.LightningModule): The LightningModule instance for logging and device access.
            predictions (Dict[str, torch.Tensor]): A dictionary of separated audio predictions.
                Keys are source names (e.g., "vocals", "bass"), values are torch.Tensor.
                Shape of each tensor: (batch_size, channels, samples)
            batch (Any): The input batch containing true sources. It is expected to have
                         a `get_true_sources()` method returning a dictionary of true sources.
            stage (str): The current stage ('train', 'val', 'test').
        """
        true_sources: Dict[str, torch.Tensor] = batch.sources
        if true_sources: # Only calculate/update metrics if true sources are available
            if stage == "train":
                # For training, compute and log metrics per step for each metric in the collection
                for metric_name, metric_instance in self.metrics.items():
                    current_step_metric_values: List[torch.Tensor] = []
                    for source_name, sep_audio in predictions.items():
                        if source_name in true_sources:
                            # Call the metric instance directly. This will update its internal state
                            # and return the computed value for the current input.
                            value: torch.Tensor = metric_instance(sep_audio.to(pl_module.device), true_sources[source_name].to(pl_module.device))
                            current_step_metric_values.append(value)
                    if current_step_metric_values:
                        avg_metric_value: torch.Tensor = torch.stack(current_step_metric_values).mean()
                        pl_module.log(f"{stage}_{metric_name}", avg_metric_value, prog_bar=True)
            else: # validation or test
                # For validation/test, update the metric state for epoch-end computation
                for source_name, sep_audio in predictions.items():
                    if source_name in true_sources:
                        logger.debug(f"MetricHandler.update_step_metrics: sep_audio device: {sep_audio.device}, true_sources device: {true_sources[source_name].device}")
                        logger.debug(f"MetricHandler.update_step_metrics: pl_module.device: {pl_module.device}")
                        for metric_name, metric_instance in self.metrics.items():
                            logger.debug(f"MetricHandler.update_step_metrics: Metric '{metric_name}' device: {metric_instance.device}")
                        # Ensure inputs to metrics are on the correct device
                        # Update all metrics in the collection
                        self.metrics.update(sep_audio.to(pl_module.device), true_sources[source_name].to(pl_module.device))

    def compute_and_log_epoch_metrics(self, pl_module: pl.LightningModule, stage: str) -> None:
        """
        Computes and logs metrics at the end of an epoch (validation or test).
        Resets the metric state after computation.

        Args:
            pl_module (pl.LightningModule): The LightningModule instance for logging.
            stage (str): The current stage ('val' or 'test').
        """
        metrics_dict: Dict[str, torch.Tensor] = self.metrics.compute()
        for metric_name, metric_value in metrics_dict.items():
            pl_module.log(f"{stage}_{metric_name}", metric_value)
        self.metrics.reset()

    # Removed the custom to() method as it will be handled by nn.Module