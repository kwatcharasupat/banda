import torch
import torchmetrics
from typing import Dict, Any, Union
import pytorch_lightning as pl

class MetricHandler:
    """
    Handles metric calculation, updating, and logging for source separation tasks.
    Encapsulates the logic for updating metrics, making the LightningModule more generic.
    """
    def __init__(self, metric_fn: Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]]):
        """
        Args:
            metric_fn (Union[torchmetrics.Metric, Dict[str, torchmetrics.Metric]]):
                The metric function(s) to use for evaluation. Can be a single metric or a dictionary of metrics.
        """
        if isinstance(metric_fn, dict):
            self.metrics = torchmetrics.MetricCollection(metric_fn)
        else:
            # Assume it's a single metric if not a dict
            self.metrics = torchmetrics.MetricCollection({"default_metric": metric_fn})

    def update_step_metrics(self, pl_module: pl.LightningModule, predictions: Dict[str, torch.Tensor], batch: Any, stage: str):
        """
        Updates metrics for a single step (training, validation, or test).
        Logs training metrics directly. For validation/test, updates the metric state.

        Args:
            pl_module (pl.LightningModule): The LightningModule instance for logging and device access.
            predictions (Dict[str, torch.Tensor]): Separated audio predictions.
            batch (Any): The input batch containing true sources.
            stage (str): The current stage ('train', 'val', 'test').
        """
        true_sources = batch.get_true_sources()
        if true_sources: # Only calculate/update metrics if true sources are available
            if stage == "train":
                # For training, compute and log metrics per step for each metric in the collection
                for metric_name, metric_instance in self.metrics.items():
                    current_step_metric_values = []
                    for source_name, sep_audio in predictions.items():
                        if source_name in true_sources:
                            # Call the metric instance directly. This will update its internal state
                            # and return the computed value for the current input.
                            value = metric_instance(sep_audio.to(pl_module.device), true_sources[source_name].to(pl_module.device))
                            current_step_metric_values.append(value)
                    if current_step_metric_values:
                        avg_metric_value = torch.stack(current_step_metric_values).mean()
                        pl_module.log(f"{stage}_{metric_name}", avg_metric_value, prog_bar=True)
            else: # validation or test
                # For validation/test, update the metric state for epoch-end computation
                for source_name, sep_audio in predictions.items():
                    if source_name in true_sources:
                        # Ensure inputs to metrics are on the correct device
                        # Update all metrics in the collection
                        self.metrics.update(sep_audio.to(pl_module.device), true_sources[source_name].to(pl_module.device))

    def compute_and_log_epoch_metrics(self, pl_module: pl.LightningModule, stage: str):
        """
        Computes and logs metrics at the end of an epoch (validation or test).
        Resets the metric state after computation.

        Args:
            pl_module (pl.LightningModule): The LightningModule instance for logging.
            stage (str): The current stage ('val' or 'test').
        """
        metrics_dict = self.metrics.compute()
        for metric_name, metric_value in metrics_dict.items():
            pl_module.log(f"{stage}_{metric_name}", metric_value)
        self.metrics.reset()

    def to(self, device: torch.device):
        """
        Moves all internal metrics to the specified device.

        Args:
            device (torch.device): The target device.
        """
        self.metrics.to(device)