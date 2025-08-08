"""
This module defines Pydantic models for standardizing data batch structures
used throughout the banda codebase.
"""

from typing import Union, Dict, Any, Optional, List, Set
from pydantic import BaseModel
import torch



class TorchInputAudioDict(BaseModel):
    """
    Represents a dictionary of audio data (mixture and sources) as PyTorch tensors.

    Attributes:
        mixture (Optional[torch.Tensor]): The mixed audio. Shape: (channels, samples)
        sources (Dict[str, torch.Tensor]): A dictionary mapping source names to their audio data.
            Shape of each source: (channels, samples)
    """
    mixture: Optional[torch.Tensor] = None
    sources: Dict[str, torch.Tensor]

    class Config:

        """Pydantic configuration for TorchInputAudioDict."""

        arbitrary_types_allowed = True # Allow torch.Tensor

class AudioBatch(BaseModel):
    """
    Represents a batch of audio data.

    Attributes:
        audio (TorchInputAudioDict): A TorchInputAudioDict containing the mixture and sources.
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    audio: TorchInputAudioDict
    metadata: Dict[str, Any]

    class Config:
        """Pydantic configuration for AudioBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor

class FixedStemSeparationBatch(BaseModel):
    """
    Represents a batch of data for fixed-stem source separation.

    Attributes:
        mixture (torch.Tensor): The mixed audio. Shape: (batch_size, channels, samples)
        sources (Dict[str, torch.Tensor]): A dictionary of source audios.
            Shape of each source: (batch_size, channels, samples)
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
        query (Optional[torch.Tensor]): An optional query tensor for query-based models.
            Shape: (batch_size, query_features)
    """
    mixture: torch.Tensor
    sources: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    query: Optional[torch.Tensor] = None # Added query field

    class Config:
        """Pydantic configuration for FixedStemSeparationBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor

    def to_device(self, device: torch.device) -> "FixedStemSeparationBatch":
        """
        Moves all torch.Tensor attributes of the batch to the specified device.
        """
        new_mixture = self.mixture.to(device) if self.mixture is not None else None
        new_sources = {k: v.to(device) for k, v in self.sources.items()}
        new_query = self.query.to(device) if self.query is not None else None # Move query to device
        # Assuming metadata does not contain tensors that need device transfer
        return FixedStemSeparationBatch(
            mixture=new_mixture,
            sources=new_sources,
            metadata=self.metadata,
            query=new_query # Pass query to new batch
        )

class SpectrogramBatch(BaseModel):
    """
    Represents a batch of spectrogram data.

    Attributes:
        spectrogram (torch.Tensor): A torch.Tensor containing the spectrogram data.
            Shape: (batch_size, channels, frequency_bins, time_frames)
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    spectrogram: torch.Tensor
    metadata: Dict[str, Any]

    class Config:
        """Pydantic configuration for SpectrogramBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor

class MaskBatch(BaseModel):
    """
    Represents a batch of mask data, typically used for source separation.

    Attributes:
        mask (torch.Tensor): A torch.Tensor containing the mask data.
            Shape: (batch_size, channels, frequency_bins, time_frames)
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    mask: torch.Tensor
    metadata: Dict[str, Any]

    class Config:
        """Pydantic configuration for MaskBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor

class QueryBatch(BaseModel):
    """
    Represents a batch of query data, e.g., for query-based separation.

    Attributes:
        query (torch.Tensor): A torch.Tensor containing the query data.
            Shape: (batch_size, query_features) or (batch_size, channels, samples)
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    query: torch.Tensor
    metadata: Dict[str, Any]

    class Config:
        """Pydantic configuration for QueryBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor

class QueryAudioSeparationBatch(BaseModel):
    """
    Represents a batch of data for query-based source separation with audio queries.

    Attributes:
        mixture (torch.Tensor): The mixed audio. Shape: (batch_size, channels, samples)
        query_audio (torch.Tensor): The audio query. Shape: (batch_size, channels, samples)
        sources (Dict[str, torch.Tensor]): A dictionary of source audios.
            Shape of each source: (batch_size, channels, samples)
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    mixture: torch.Tensor
    query_audio: torch.Tensor
    sources: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]

    class Config:
        """Pydantic configuration for QueryAudioSeparationBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor

class QueryClassSeparationBatch(BaseModel):
    """
    Represents a batch of data for query-based source separation with class queries.

    Attributes:
        mixture (torch.Tensor): The mixed audio. Shape: (batch_size, channels, samples)
        query_class (torch.Tensor): The class query (e.g., one-hot encoded). Shape: (batch_size, num_classes)
        sources (Dict[str, torch.Tensor]): A dictionary of source audios.
            Shape of each source: (batch_size, channels, samples)
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    mixture: torch.Tensor
    query_class: torch.Tensor
    sources: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]

    class Config:
        """Pydantic configuration for QueryClassSeparationBatch."""
        arbitrary_types_allowed = True # Allow torch.Tensor
SeparationBatch = Union[FixedStemSeparationBatch, QueryAudioSeparationBatch, QueryClassSeparationBatch]