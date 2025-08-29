import numpy as np
import torch

from ..typing import Identifier, NumPySourceDict, IdentifierWithChunkIndex
from .base import PreMixTransform


class ChunkingTransform(PreMixTransform):
    """
    ChunkingTransform is a transform that chunks the input audio into smaller segments.
    """

    def __init__(self, *, chunk_size_seconds: float, fs: int) -> None:
        """
        Args:
            chunk_size_seconds (float): The size of each chunk in seconds.
        """
        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = int(round(chunk_size_seconds * fs))
        self.fs = fs

    def chunk(self, *, source: np.ndarray, start_sample: int) -> np.ndarray:
        """
        Chunk the input audio into smaller segments.
        Args:
            source (np.ndarray): The input audio.
            start_sample (int): The start sample for chunking.
        Returns:
            np.ndarray: The chunked audio.
        """
        _, n_samples = source.shape
        end_sample = start_sample + self.chunk_size_samples

        if start_sample > n_samples:
            raise ValueError(
                f"start_sample {start_sample} is greater than n_samples {n_samples}"
            )

        if end_sample > n_samples:
            raise ValueError(
                f"end_sample {end_sample} is greater than n_samples {n_samples}"
            )

        return source[start_sample:end_sample]


class RandomChunkingTransform(ChunkingTransform):
    """
    RandomChunkingTransform is a transform that randomly chunks the input audio into smaller segments.
    """

    def __init__(
        self, *, chunk_size_seconds: float, fs: int, align_sources: bool
    ) -> None:
        """
        Args:
            chunk_size (int): The size of each chunk in samples.
        """
        super().__init__(chunk_size_seconds=chunk_size_seconds, fs=fs)
        self.align_sources = align_sources

    def __call__(
        self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        if self.align_sources:
            return self._aligned_random_chunk(sources=sources)
        else:
            return self._unaligned_random_chunk(sources=sources)

    def _aligned_random_chunk(self, *, sources: NumPySourceDict) -> NumPySourceDict:
        """
        Aligned chunking of the input audio.
        Args:
            sources (NumPySourceDict): The input audio.
        Returns:
            NumPySourceDict: The chunked audio.
        """
        n_samples = min([source.shape[-1] for source in sources.values()])
        start = self.get_random_start(n_samples=n_samples)
        return {
            key: self.chunk(source=source, start_sample=start)
            for key, source in sources.items()
        }

    def _unaligned_random_chunk(self, *, sources: NumPySourceDict) -> NumPySourceDict:
        """
        Unaligned chunking of the input audio.
        Args:
            sources (NumPySourceDict): The input audio.
        Returns:
            NumPySourceDict: The chunked audio.
        """
        return {
            key: self.random_chunk(source=source) for key, source in sources.items()
        }

    def random_chunk(self, *, source: np.ndarray) -> np.ndarray:
        """
        Randomly chunk the input audio into smaller segments.
        Args:
            source (np.ndarray): The input audio.
        Returns:
            np.ndarray: The chunked audio.
        """
        _, n_samples = source.shape
        start = self.get_random_start(n_samples=n_samples)
        return self.chunk(source=source, start_sample=start)

    def get_random_start(self, *, n_samples: int) -> int:
        """
        Get the start index for chunking.
        """

        if n_samples < self.chunk_size_samples:
            raise ValueError(
                f"n_samples {n_samples} is less than chunk_size_samples {self.chunk_size_samples}"
            )

        if n_samples == self.chunk_size_samples:
            return 0

        start = int(
            torch.randint(
                low=0,
                high=n_samples - self.chunk_size_samples,
                size=(1,),
            ).item()
        )
        return start


class DeterministicChunkingTransform(ChunkingTransform):
    """
    DeterministicChunkingTransform is a transform that deterministically chunks the input audio into smaller segments.
    """

    def __init__(
        self, *, chunk_size_seconds: float, hop_size_seconds: float, fs: int
    ) -> None:
        super().__init__(chunk_size_seconds=chunk_size_seconds, fs=fs)

        self.hop_size_seconds = hop_size_seconds
        self.hop_size_samples = int(round(hop_size_seconds * fs))

    def __call__(
        self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        """
        Args:
            sources (NumPySourceDict): The input audio.
            chunk_index (int): The index of the chunk to extract.
        Returns:
            NumPySourceDict: The chunked audio.
        """
        identifier = IdentifierWithChunkIndex(**identifier.model_dump())
        chunk_index: int = identifier.chunk_index
        start_sample = chunk_index * self.hop_size_samples
        return {
            key: self.chunk(source=source, start_sample=start_sample)
            for key, source in sources.items()
        }
