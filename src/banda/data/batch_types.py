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


class FixedStemSeparationBatch(BaseBatch):
    """
    Batch structure for fixed-stem source separation tasks (e.g., vocals, bass, drums, other).

    Attributes:
        audio (TorchInputAudioDict): Contains the mixture and ground truth sources.
        identifier (Identifier): Identifier for the current track/chunk.
    """
    audio: TorchInputAudioDict
    identifier: Identifier


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


# Union type for all possible batch types
SeparationBatch = Union[
    FixedStemSeparationBatch,
    QueryAudioSeparationBatch,
    QueryClassSeparationBatch,
]