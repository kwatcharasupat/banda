#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from typing import Dict
import torch
import torch.nn as nn

from banda.data.batch_types import (
    SeparationBatch,
    FixedStemSeparationBatch,
    QueryAudioSeparationBatch,
    QueryClassSeparationBatch,
)
from banda.losses.base import LossHandler


class SeparationLossHandler(LossHandler):
    """
    Handles loss calculation for various source separation tasks based on batch type.
    """

    def __init__(self, stft_loss_fn: nn.Module, time_loss_fn: nn.Module):
        """
        Args:
            stft_loss_fn (nn.Module): The STFT-domain loss function.
            time_loss_fn (nn.Module): The time-domain loss function.
        """
        super().__init__(stft_loss_fn) # Keep stft_loss_fn as the primary loss_fn for base class
        self.stft_loss_fn = stft_loss_fn
        self.time_loss_fn = time_loss_fn

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
        # Initialize total_loss as a list to collect batch-wise losses
        losses_per_sample = []

        if isinstance(batch, FixedStemSeparationBatch):
            true_sources = batch.audio.sources
            for source_name, sep_audio in predictions.items():
                if source_name in true_sources:
                    # Calculate STFT loss (batch-wise)
                    stft_loss = self.stft_loss_fn(sep_audio, true_sources[source_name])
                    # Calculate Time-domain loss (batch-wise)
                    time_loss = self.time_loss_fn(sep_audio, true_sources[source_name])
                    losses_per_sample.append(stft_loss + time_loss)
            
            # Stack and average the losses across the batch
            if losses_per_sample:
                return torch.mean(torch.stack(losses_per_sample))
            else:
                return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)

        elif isinstance(batch, QueryAudioSeparationBatch):
            true_sources = batch.audio.sources
            target_source_name = "vocals" # This would come from the query logic
            if target_source_name in predictions and target_source_name in true_sources:
                stft_loss = self.stft_loss_fn(predictions[target_source_name], true_sources[target_source_name])
                time_loss = self.time_loss_fn(predictions[target_source_name], true_sources[target_source_name])
                return torch.mean(stft_loss + time_loss) # Assuming these are batch-wise
            else:
                raise NotImplementedError(f"Loss calculation for {type(batch).__name__} with query audio is not fully implemented.")

        elif isinstance(batch, QueryClassSeparationBatch):
            true_sources = batch.audio.sources
            query_class = batch.query_class
            if query_class in predictions and query_class in true_sources:
                stft_loss = self.stft_loss_fn(predictions[query_class], true_sources[query_class])
                time_loss = self.time_loss_fn(predictions[query_class], true_sources[query_class])
                return torch.mean(stft_loss + time_loss) # Assuming these are batch-wise
            else:
                raise NotImplementedError(f"Loss calculation for {type(batch).__name__} with query class is not fully implemented.")

        else:
            raise TypeError(f"Unsupported batch type for loss calculation: {type(batch)}")