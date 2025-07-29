#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from abc import ABC
from typing import Dict, Optional, Union
import torch
from pydantic import BaseModel, ConfigDict

from banda.data.types import TorchInputAudioDict, Identifier


class BaseBatch(BaseModel, ABC):
    """
    Base Pydantic model for a batch of data.
    All specific batch types should inherit from this.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_true_sources(self) -> Dict[str, torch.Tensor]:
        """
        Returns ground truth sources for metric calculation.
        Default implementation returns an empty dictionary.
        """
        return {}

    def get_inference_query(self) -> Optional[Union[torch.Tensor, int, str]]:
        """
        Returns the query for inference, if applicable.
        Default implementation returns None.
        """
        return None


class FixedStemSeparationBatch(BaseBatch):
    """
    Batch structure for fixed-stem source separation tasks (e.g., vocals, bass, drums, other).

    Attributes:
        audio (TorchInputAudioDict): Contains the mixture and ground truth sources.
        identifier (Identifier): Identifier for the current track/chunk.
    """
    audio: TorchInputAudioDict
    identifier: Identifier

    def to_device(self, device: torch.device) -> 'FixedStemSeparationBatch':
        """
        Transfers all torch.Tensor attributes of the batch to the specified device.
        """
        if self.audio.mixture is not None:
            self.audio.mixture = self.audio.mixture.to(device)
        for stem, audio_tensor in self.audio.sources.items():
            self.audio.sources[stem] = audio_tensor.to(device)
        return self

    def get_true_sources(self) -> Dict[str, torch.Tensor]:
        """
        Returns ground truth sources for metric calculation for fixed-stem batches.
        """
        return self.audio.sources


class QueryAudioSeparationBatch(BaseBatch):
    """
    Batch structure for query-based source separation tasks where the query is an audio signal.

    Attributes:
        audio (TorchInputAudioDict): Contains the mixture and ground truth sources.
        query_audio (torch.Tensor): The audio signal used as a query.
                                    Shape: (batch_size, channels, samples)
        identifier (Identifier): Identifier for the current track/chunk.
    """
    audio: TorchInputAudioDict
    query_audio: torch.Tensor
    identifier: Identifier

    def get_inference_query(self) -> Optional[Union[torch.Tensor, int, str]]:
        """
        Returns the query audio for inference.
        """
        return self.query_audio.squeeze(0)


class QueryClassSeparationBatch(BaseBatch):
    """
    Batch structure for query-based source separation tasks where the query is a class label.

    Attributes:
        audio (TorchInputAudioDict): Contains the mixture and ground truth sources.
        query_class (Union[int, str]): The class label used as a query.
        identifier (Identifier): Identifier for the current track/chunk.
    """
    audio: TorchInputAudioDict
    query_class: Union[int, str]
    identifier: Identifier

    def get_inference_query(self) -> Optional[Union[torch.Tensor, int, str]]:
        """
        Returns the query class for inference.
        """
        return self.query_class


# Union type for all possible batch types
SeparationBatch = Union[
    FixedStemSeparationBatch,
    QueryAudioSeparationBatch,
    QueryClassSeparationBatch,
]