#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn

from banda.data.batch_types import SeparationBatch


class LossHandler(ABC):
    """
    Abstract base class for handling loss calculations for different separation tasks.
    """

    def __init__(self, loss_fn: nn.Module):
        """
        Args:
            loss_fn (nn.Module): The base loss function to use (e.g., L1Loss, MSELoss).
        """
        self.loss_fn = loss_fn

    @abstractmethod
    def calculate_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: SeparationBatch,
    ) -> torch.Tensor:
        """
        Calculates the loss based on the predictions and the batch data.

        Args:
            predictions (Dict[str, torch.Tensor]): A dictionary of predicted source tensors.
            batch (SeparationBatch): The input batch containing true sources and other data.

        Returns:
            torch.Tensor: The calculated loss.
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Instantiates a LossHandler from a configuration dictionary.
        This method should be overridden by subclasses if they require more complex
        instantiation logic than just passing kwargs to __init__.
        """
        return cls(**config)