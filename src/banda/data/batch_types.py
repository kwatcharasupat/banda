"""
This module defines Pydantic models for standardizing data batch structures
used throughout the banda codebase.
"""

from typing import Type, TypeVar, Union, Dict, Any, Optional, List, Set
from pydantic import BaseModel, ConfigDict
import torch
import numpy as np
import numpy.typing as npt


class ArbitraryTypesAllowedBaseModel(BaseModel):
    """
    A Pydantic BaseModel that allows arbitrary types by default.
    This is useful for models that need to store PyTorch tensors.
    """
    class Config:
        arbitrary_types_allowed = True


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_numpy(
            cls,
            *,
            mixture: Optional[np.ndarray],
            sources: Dict[str, np.ndarray],
            float_dtype: torch.dtype = torch.float32,
    ) -> "TorchInputAudioDict":
        """
        Create a TorchInputAudioDict from NumPy arrays.

        Args:
            mixture (np.ndarray): Mixture audio array of shape (channels, samples).
            sources (Dict[str, np.ndarray]): Dictionary of source audio arrays.

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


class AudioBatch(ArbitraryTypesAllowedBaseModel):
    """
    Represents a batch of audio data.

    Attributes:
        audio (TorchInputAudioDict): A TorchInputAudioDict containing the mixture and sources.
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    audio: TorchInputAudioDict
    metadata: Dict[str, Any]


class AudioSignal(ArbitraryTypesAllowedBaseModel):
    """
    Represents a signal abstraction that can have time-domain and STFT-domain components.

    Attributes:
        audio (Optional[torch.Tensor]): The time-domain audio signal. Shape: (batch_size, channels, samples)
        spectrogram (Optional[torch.Tensor]): The STFT-domain spectrogram. Shape: (batch_size, channels, freq_bins, time_frames)
    """
    audio: Optional[torch.Tensor] = None
    spectrogram: Optional[torch.Tensor] = None

    def to_device(self, device: torch.device) -> "AudioSignal":
        """
        Moves all torch.Tensor attributes of the AudioSignal to the specified device.
        """
        new_audio = self.audio.to(device) if self.audio is not None else None
        new_spectrogram = self.spectrogram.to(device) if self.spectrogram is not None else None
        return AudioSignal(audio=new_audio, spectrogram=new_spectrogram)

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the underlying tensor.
        """
        if self.audio is not None:
            return self.audio.device
        elif self.spectrogram is not None:
            return self.spectrogram.device
        else:
            raise ValueError("AudioSignal has no audio or spectrogram to determine device.")


class QuerySignal(ArbitraryTypesAllowedBaseModel):
    """
    Represents a query abstraction that can have various data types.

    Attributes:
        audio (Optional[AudioSignal]): Audio query.
        embedding (Optional[torch.Tensor]): Embedding query. Shape: (batch_size, embedding_dim)
        class_label (Optional[torch.Tensor]): Class label query (e.g., one-hot encoded). Shape: (batch_size, num_classes)
        one_hot (Optional[torch.Tensor]): One-hot encoded query. Shape: (batch_size, num_classes)
        data_dict (Optional[Dict[str, torch.Tensor]]): Dictionary of arbitrary tensor data.
    """
    audio: Optional[AudioSignal] = None
    embedding: Optional[torch.Tensor] = None
    class_label: Optional[torch.Tensor] = None
    one_hot: Optional[torch.Tensor] = None
    data_dict: Optional[Dict[str, torch.Tensor]] = None

    def to_device(self, device: torch.device) -> "QuerySignal":
        """
        Moves all torch.Tensor attributes of the QuerySignal to the specified device.
        """
        new_audio = self.audio.to_device(device) if self.audio is not None else None
        new_embedding = self.embedding.to(device) if self.embedding is not None else None
        new_class_label = self.class_label.to(device) if self.class_label is not None else None
        new_one_hot = self.one_hot.to(device) if self.one_hot is not None else None
        new_data_dict = {k: v.to(device) for k, v in self.data_dict.items()} if self.data_dict is not None else None
        return QuerySignal(
            audio=new_audio,
            embedding=new_embedding,
            class_label=new_class_label,
            one_hot=new_one_hot,
            data_dict=new_data_dict
        )


class BaseSeparationBatch(ArbitraryTypesAllowedBaseModel):
    """
    Base class for all separation batches, encapsulating common attributes.

    Attributes:
        mixture (AudioSignal): The mixed audio signal, potentially with time-domain and STFT-domain components.
        sources (Dict[str, AudioSignal]): A dictionary of source audio signals.
        metadata (Dict[str, Any]): A dictionary for arbitrary metadata associated with the batch.
    """
    mixture: AudioSignal
    sources: Dict[str, AudioSignal]
    metadata: Dict[str, Any]

    estimates: Dict[str, AudioSignal]

    def to_device(self, device: torch.device) -> "BaseSeparationBatch":
        """
        Moves all torch.Tensor attributes of the batch to the specified device.
        """
        new_mixture = self.mixture.to_device(device)
        new_sources = {k: v.to_device(device) for k, v in self.sources.items()}
        # Assuming metadata does not contain tensors that need device transfer
        return self.__class__(
            mixture=new_mixture,
            sources=new_sources,
            metadata=self.metadata
        )

class QueryAudioSeparationBatch(BaseSeparationBatch):
    """
    Represents a batch of data for query-based source separation with audio queries.

    Attributes:
        query_audio (QuerySignal): The audio query.
    """
    query_audio: QuerySignal

    def to_device(self, device: torch.device) -> "QueryAudioSeparationBatch":
        """
        Moves all torch.Tensor attributes of the batch to the specified device.
        """
        base_batch = super().to_device(device)
        new_query_audio = self.query_audio.to_device(device) if self.query_audio is not None else None
        return QueryAudioSeparationBatch(
            mixture=base_batch.mixture,
            sources=base_batch.sources,
            metadata=base_batch.metadata,
            query_audio=new_query_audio
        )

class QueryClassSeparationBatch(BaseSeparationBatch):
    """
    Represents a batch of data for query-based source separation with class queries.

    Attributes:
        query_class (Optional[QuerySignal]): The class query. Made optional to support non-query models.
    """
    query_class: Optional[QuerySignal] = None # Made optional

    def to_device(self, device: torch.device) -> "QueryClassSeparationBatch":
        """
        Moves all torch.Tensor attributes of the batch to the specified device.
        """
        base_batch = super().to_device(device)
        new_query_class = self.query_class.to_device(device) if self.query_class is not None else None
        return QueryClassSeparationBatch(
            mixture=base_batch.mixture,
            sources=base_batch.sources,
            metadata=base_batch.metadata,
            query_class=new_query_class
        )

class FixedStemSeparationBatch(QueryClassSeparationBatch):
    """
    Represents a batch of data for fixed-stem source separation.
    Inherits from QueryClassSeparationBatch, where the query_class is implicitly
    derived from the fixed stems.
    """
    # No additional fields needed, as 'query_class' from parent handles the fixed stems.
    # The 'query' field from previous iterations is now handled by 'query_class' in QueryClassSeparationBatch.

    def to_device(self, device: torch.device) -> "FixedStemSeparationBatch":
        """
        Moves all torch.Tensor attributes of the batch to the specified device.
        """
        # Call the parent's to_device, which handles mixture, sources, metadata, and query_class
        return super().to_device(device)


SeparationBatch = TypeVar("SeparationBatch", bound=BaseSeparationBatch)