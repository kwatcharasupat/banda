import torch
import torch.nn as nn
from typing import Dict, List, Union
import structlog
import hydra.utils

logger = structlog.get_logger(__name__)



from banda.data.batch_types import (
    SeparationBatch,
    FixedStemSeparationBatch,
    QueryAudioSeparationBatch,
    QueryClassSeparationBatch,
)
from banda.losses.base import LossHandler
from banda.losses.loss_configs import LossCollectionConfig


class SeparationLossHandler(LossHandler):
    """
    Handles loss calculation for various source separation tasks based on batch type.
    """

    def __init__(self, loss_config: LossCollectionConfig):
        """
        Args:
            loss_config (LossCollectionConfig): A Pydantic configuration object containing
                                                all necessary parameters for the loss functions.
        """
        # The base LossHandler expects a single primary loss_fn. We'll use the first one provided.
        first_loss_name = next(iter(loss_config.losses))
        super().__init__(hydra.utils.instantiate(loss_config.losses[first_loss_name].fn))
        
        self.loss_fns = nn.ModuleDict()
        self.loss_weights = {}
        for name, config in loss_config.losses.items():
            self.loss_fns[name] = hydra.utils.instantiate(config.fn)
            self.loss_weights[name] = config.weight


    def calculate_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: SeparationBatch,
    ) -> torch.Tensor:
        """
        Calculates the loss based on the type of SeparationBatch.

        Args:
            predictions (Dict[str, torch.Tensor]): A dictionary of predicted source tensors.
            batch (SeparationBatch): The input batch containing true sources and other data.

        Returns:
            torch.Tensor: The calculated loss.
        """
        total_loss = torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device) if predictions else torch.tensor(0.0)

        if isinstance(batch, FixedStemSeparationBatch):
            true_sources = batch.sources
            num_sources_with_predictions = 0
            for source_name, sep_audio in predictions.items():
                if source_name in true_sources:
                    num_sources_with_predictions += 1
                    for loss_name, loss_fn in self.loss_fns.items():
                        weight = self.loss_weights.get(loss_name, 1.0)
                        loss = loss_fn(sep_audio, true_sources[source_name])
                        total_loss += loss * weight
            
            if num_sources_with_predictions > 0:
                return total_loss / num_sources_with_predictions
            else:
                return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device) if predictions else torch.tensor(0.0)

        elif isinstance(batch, QueryAudioSeparationBatch):
            true_sources = batch.sources
            if predictions:
                target_source_name = next(iter(predictions))
                if target_source_name in true_sources:
                    for loss_name, loss_fn in self.loss_fns.items():
                        weight = self.loss_weights.get(loss_name, 1.0)
                        loss = loss_fn(predictions[target_source_name], true_sources[target_source_name])
                        total_loss += loss * weight
                    return total_loss
                else:
                    raise ValueError(f"Target source '{target_source_name}' not found in true sources for QueryAudioSeparationBatch.")
            else:
                return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device) if predictions else torch.tensor(0.0)

        elif isinstance(batch, QueryClassSeparationBatch):
            true_sources = batch.sources
            query_class = batch.query_class
            if query_class in predictions and query_class in true_sources:
                for loss_name, loss_fn in self.loss_fns.items():
                    weight = self.loss_weights.get(loss_name, 1.0)
                    loss = loss_fn(predictions[query_class], true_sources[query_class])
                    total_loss += loss * weight
                return total_loss
            else:
                raise ValueError(f"Query class '{query_class}' not found in predictions or true sources for QueryClassSeparationBatch.")

        else:
            raise TypeError(f"Unsupported batch type for loss calculation: {type(batch)}")

    def to(self, device: torch.device) -> None:
        """
        Moves all internal loss functions to the specified device.

        Args:
            device (torch.device): The target device.
        """
        logger.debug(f"SeparationLossHandler.to: Moving loss functions to device {device}")
        for loss_fn in self.loss_fns.values():
            loss_fn.to(device)