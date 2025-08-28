#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from tracemalloc import start
import numpy as np
import torch
import structlog

logger = structlog.get_logger(__name__)

from banda.data.augmentations.base import PreMixTransform
from banda.data.types import Identifier, NumPySourceDict


class ChunkingTransform(PreMixTransform):
    """
    ChunkingTransform is a transform that chunks the input audio into smaller segments.
    """

    def __init__(self, *, chunk_size_seconds: float, fs: int) -> None:
        """
        Args:
            chunk_size_seconds (float): The size of each chunk in seconds.
            fs (int): The sample rate of the audio.
        """
        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = int(round(chunk_size_seconds * fs))
        self.fs = fs

    def chunk(self, *, source: np.ndarray, start_sample: int) -> np.ndarray:
        """
        Chunk the input audio into smaller segments.
        Args:
            source (np.ndarray): The input audio. Shape (channels, samples).
            start_sample (int): The start sample for chunking.
        Returns:
            np.ndarray: The chunked audio. Shape (channels, chunk_size_samples).
        """
        # Ensure source is at least 2D (channels, samples)
        if source.ndim == 1:
            source = source[np.newaxis, :]

        _, n_samples = source.shape
        end_sample = start_sample + self.chunk_size_samples

        if start_sample > n_samples:
            raise ValueError(
                f"start_sample {start_sample} is greater than n_samples {n_samples}"
            )

        # Pad if the chunk extends beyond the audio length
        if end_sample > n_samples:
            padding_needed = end_sample - n_samples
            # Pad with zeros at the end
            chunk = np.pad(source[:, start_sample:], ((0, 0), (0, padding_needed)), mode='constant')
        else:
            chunk = source[:, start_sample:end_sample]

        return chunk


class RandomChunkingTransform(ChunkingTransform):
    """
    RandomChunkingTransform is a transform that randomly chunks the input audio into smaller segments.
    """

    def __init__(
            self, *, chunk_size_seconds: float, fs: int, align_sources: bool = False
    ) -> None:
        """
        Args:
            chunk_size_seconds (float): The size of each chunk in seconds.
            fs (int): The sample rate of the audio.
            align_sources (bool): If True, all sources will be chunked from the same random start point.
                                  If False, each source will be chunked from its own random start point.
        """
        super().__init__(chunk_size_seconds=chunk_size_seconds, fs=fs)
        self.align_sources = align_sources

    def __call__(
            self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        """
        Apply random chunking to the source data.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            identifier (Identifier): Identifier for the source data.

        Returns:
            NumPySourceDict: Transformed source data with random chunks.
        """
        if self.align_sources:
            return self._aligned_random_chunk(sources=sources)
        else:
            return self._unaligned_random_chunk(sources=sources)

    def _aligned_random_chunk(self, *, sources: NumPySourceDict) -> NumPySourceDict:
        """
        Aligned chunking of the input audio. All sources are chunked from the same random start point.

        Args:
            sources (NumPySourceDict): The input audio.
        Returns:
            NumPySourceDict: The chunked audio.
        """
        # Find the minimum length among all sources
        min_n_samples = min([source.shape[-1] for source in sources.values()])
        logger.debug(f"min_n_samples: {min_n_samples}, chunk_size_samples: {self.chunk_size_samples}")
        
        # Calculate the maximum possible start sample to ensure a full chunk can be extracted
        max_start_sample = max(0, min_n_samples - self.chunk_size_samples)

        if max_start_sample == 0 and min_n_samples < self.chunk_size_samples:
            # If all sources are shorter than the chunk size, start at 0 and chunk will pad
            start = 0
        elif max_start_sample == 0:
            # If sources are exactly chunk_size_samples long, start at 0
            start = 0
        else:
            start = int(
                torch.randint(
                    low=0,
                    high=max_start_sample + 1, # +1 because randint is exclusive of high
                    size=(1,),
                ).item()
            )
        logger.debug(f"Calculated start sample: {start}")

        return {
            key: self.chunk(source=source, start_sample=start)
            for key, source in sources.items()
        }

    def _unaligned_random_chunk(self, *, sources: NumPySourceDict) -> NumPySourceDict:
        """
        Unaligned chunking of the input audio. Each source is chunked from its own random start point.

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
            source (np.ndarray): The input audio. Shape (channels, samples).
        Returns:
            np.ndarray: The chunked audio. Shape (channels, chunk_size_samples).
        """
        # Ensure source is at least 2D (channels, samples)
        if source.ndim == 1:
            source = source[np.newaxis, :]

        _, n_samples = source.shape
        
        # Calculate the maximum possible start sample
        max_start_sample = max(0, n_samples - self.chunk_size_samples)

        if max_start_sample == 0 and n_samples < self.chunk_size_samples:
            # If source is shorter than the chunk size, start at 0 and chunk will pad
            start = 0
        elif max_start_sample == 0:
            # If source is exactly chunk_size_samples long, start at 0
            start = 0
        else:
            start = int(
                torch.randint(
                    low=0,
                    high=max_start_sample + 1, # +1 because randint is exclusive of high
                    size=(1,),
                ).item()
            )
        return self.chunk(source=source, start_sample=start)
    
    
    
class DeterministicChunkingTransform(ChunkingTransform):
    """
    DeterministicChunkingTransform is a transform that chunks the input audio into smaller segments
    starting from a fixed position (e.g., the beginning of the audio).
    """

    def __init__(
            self, *, chunk_size_seconds: float, 
            hop_size_seconds: float,
            fs: int
    ) -> None:
        """
        Args:
            chunk_size_seconds (float): The size of each chunk in seconds.
            fs (int): The sample rate of the audio.
            align_sources (bool): If True, all sources will be chunked from the same random start point.
                                  If False, each source will be chunked from its own random start point.
        """
        super().__init__(chunk_size_seconds=chunk_size_seconds, fs=fs)
        self.hop_size_seconds = hop_size_seconds
        self.hop_size_samples = int(round(hop_size_seconds * fs))

    def __call__(
            self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        """
        Apply deterministic chunking to the source data.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            identifier (Identifier): Identifier for the source data.

        Returns:
            NumPySourceDict: Transformed source data with deterministic chunks.
        """

        _, n_samples = sources[list(sources.keys())[0]].shape

        output = []

        for start_sample in range(0, n_samples - self.chunk_size_samples + 1, self.hop_size_samples):
            output.append({
                key: self.chunk(source=source, start_sample=start_sample)
                for key, source in sources.items()
            })

        return output
