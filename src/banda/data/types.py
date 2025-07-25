#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from abc import abstractmethod
from typing import Dict, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict

TorchSourceDict = Dict[str, torch.Tensor]
"""Dictionary mapping source names to PyTorch tensors."""

NumPySourceDict = Dict[str, np.ndarray]
"""Dictionary mapping source names to NumPy arrays."""


class TorchInputAudioDict(BaseModel):
    """
    Data structure for input audio in PyTorch format.

    Attributes:
        mixture (torch.Tensor): Mixture audio tensor of shape (channels, samples).
        sources (TorchSourceDict): Dictionary of source audio tensors.
    """

    mixture: Optional[torch.Tensor]
    sources: TorchSourceDict

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_numpy(
            cls,
            *,
            mixture: Optional[np.ndarray],
            sources: NumPySourceDict,
            float_dtype: torch.dtype = torch.float32,
    ) -> "TorchInputAudioDict":
        """
        Create a TorchInputAudioDict from NumPy arrays.

        Args:
            mixture (np.ndarray): Mixture audio array of shape (channels, samples).
            sources (NumPySourceDict): Dictionary of source audio arrays.

        Returns:
            TorchInputAudioDict: Instance with tensors converted from NumPy arrays.
        """
        return cls(
            mixture=torch.from_numpy(mixture).to(dtype=float_dtype)
            if mixture is not None
            else None,
            sources={
                key: torch.from_numpy(value).to(dtype=float_dtype)
                for key, value in sources.items()
            },
        )


class NumPyInputAudioDict(BaseModel):
    """
    Data structure for input audio in NumPy format.

    Attributes:
        mixture (np.ndarray): Mixture audio array of shape (channels, samples).
        sources (NumPySourceDict): Dictionary of source audio arrays.
    """

    mixture: np.ndarray
    sources: NumPySourceDict

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChunkIdentifier(BaseModel):
    """
    Chunk identifier model.

    Attributes:
        chunk_index (int): Index of the chunk.
        track_id (str): Unique track identifier.
    """

    chunk_index: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Identifier(BaseModel):
    """
    Identifier model that allows any fields.

    Attributes:
        Any fields can be added dynamically.
    """

    chunk_identifier: Optional[ChunkIdentifier] = None

    model_config = ConfigDict(extra="allow")

    @property
    @abstractmethod
    def track_id(self) -> str:
        """
        Returns a unique track identifier.

        Returns:
            str: Unique track identifier.
        """


GenericIdentifier = TypeVar("GenericIdentifier", bound=Identifier)

NPIntArray = npt.NDArray[np.integer]
"""Type alias for NumPy arrays with integer elements."""

NPFloatArray = npt.NDArray[np.floating]
"""Type alias for NumPy arrays with floating-point elements."""

NPComplexArray = npt.NDArray[np.complexfloating]
"""Type alias for NumPy arrays with complex floating-point elements."""

# Common source separation stems
__VDBO__ = ["vocals", "bass", "drums", "other"]