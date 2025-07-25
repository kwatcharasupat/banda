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

    def __init__(self, loss_fn: nn.Module):
        """
        Args:
            loss_fn (nn.Module): The base loss function to use (e.g., L1Loss, MSELoss).
        """
        super().__init__(loss_fn)

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
        total_loss = torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)

        if isinstance(batch, FixedStemSeparationBatch):
            true_sources = batch.audio.sources
            for source_name, sep_audio in predictions.items():
                if source_name in true_sources:
                    total_loss += self.loss_fn(sep_audio, true_sources[source_name])
            return total_loss

        elif isinstance(batch, QueryAudioSeparationBatch):
            # Example: Loss for query audio separation might involve comparing
            # the separated source corresponding to the query with the true source.
            # This is a placeholder and needs specific implementation based on the model's output.
            true_sources = batch.audio.sources
            # Assuming the model predicts a single source based on the query
            # Or predicts all sources and we select the relevant one
            # For now, let's assume 'vocals' is the target for simplicity
            target_source_name = "vocals" # This would come from the query logic
            if target_source_name in predictions and target_source_name in true_sources:
                total_loss += self.loss_fn(predictions[target_source_name], true_sources[target_source_name])
            else:
                raise NotImplementedError(f"Loss calculation for {type(batch).__name__} with query audio is not fully implemented.")
            return total_loss

        elif isinstance(batch, QueryClassSeparationBatch):
            # Example: Loss for query class separation might involve comparing
            # the separated source corresponding to the query class with the true source.
            true_sources = batch.audio.sources
            query_class = batch.query_class
            # Assuming query_class directly maps to a source name
            if query_class in predictions and query_class in true_sources:
                total_loss += self.loss_fn(predictions[query_class], true_sources[query_class])
            else:
                raise NotImplementedError(f"Loss calculation for {type(batch).__name__} with query class is not fully implemented.")
            return total_loss

        else:
            raise TypeError(f"Unsupported batch type for loss calculation: {type(batch)}")