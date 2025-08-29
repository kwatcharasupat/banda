from abc import ABC, abstractmethod
from pydantic_tensor import NumpyNDArray

from .mixin import FixedStemMixin
from ..transforms.chunking import DeterministicChunkingTransform
from ..typing import ChunkIdentifier, GenericIdentifier, NPIntArray, NumPySourceDict
import numpy as np
import numpy.typing as npt
from ..connectors.base import DatasetConnector
from ..transforms.base import ComposePreMixTransforms, PostMixTransform, PreMixTransform
from ..transforms.chunking import RandomChunkingTransform
from .base import SourceSeparationDataset


from typing import Callable, Generic, List, Optional, Tuple, TypeVar


def dbrms(x: np.ndarray) -> float:
    return 10.0 * np.log10(np.mean(np.square(np.abs(x))) + 1e-8)


class ChunkFilter(ABC):
    @abstractmethod
    def __call__(self, chunk_audio: NumPySourceDict) -> bool:
        """
        Determines whether to keep or discard a chunk based on its audio data.

        Args:
            chunk_audio (NumpyNDArray[np.floating]): The audio data of the chunk.

        Returns:
            bool: True if the chunk should be kept, False otherwise.
        """


class NonsilentChunkFilter(ChunkFilter):
    def __init__(self, dbrms_threshold: float = -60.0) -> None:
        self.dbrms_threshold = dbrms_threshold

    def __call__(self, sources: NumPySourceDict) -> bool:
        for s in sources.values():
            if dbrms(s) < self.dbrms_threshold:
                return False
        return True


class ChunkDataset(
    SourceSeparationDataset[GenericIdentifier], Generic[GenericIdentifier]
):
    """
    Base class for chunked datasets.

    Args:
        split (str): The dataset split (e.g., "train", "val", "test").
        dataset_connector (DatasetConnector): The dataset connector.
        fs (int): The sampling rate for audio data.
        recompute_mixture (bool): Whether to recompute the mixture.
        premix_transform (PreMixTransform): The premix transform.
        postmix_transform (PostMixTransform): The postmix transform.
        auto_load_mixture (bool): Whether to automatically load the mixture.
        allowed_extensions (List[str]): List of allowed file extensions.
        allow_resampling (bool): Whether to allow resampling.
        mixture_stem_key (str): The key for the mixture stem.

    Methods:
        get_identifier(index: int) -> GenericIdentifier:
            Retrieves the identifier for a given index.
    """

    def __init__(
        self,
        *,
        split: str,
        dataset_connector: DatasetConnector[GenericIdentifier],
        fs: int = 44100,
        recompute_mixture: bool = True,
        chunk_filter: Optional[ChunkFilter] = None,
        premix_transform: Optional[PreMixTransform] = None,
        postmix_transform: Optional[PostMixTransform] = None,
        auto_load_mixture: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        allow_resampling: bool = False,
        mixture_stem_key: Optional[str] = "mixture",
    ) -> None:
        super().__init__(
            split=split,
            dataset_connector=dataset_connector,
            fs=fs,
            recompute_mixture=recompute_mixture,
            premix_transform=premix_transform,
            postmix_transform=postmix_transform,
            auto_load_mixture=auto_load_mixture,
            allowed_extensions=allowed_extensions,
            allow_resampling=allow_resampling,
            mixture_stem_key=mixture_stem_key,
        )


class RandomChunkDataset(ChunkDataset[GenericIdentifier], Generic[GenericIdentifier]):
    """
    MedleyDB dataset that returns random chunks.

    Args:
        chunk_size_seconds (float): The size of each chunk in seconds.
        align_sources (bool, optional): Whether to align sources. Defaults to True.
        target_length (int): The target number of chunks.

    Methods:
        _make_premix_transform(bandx.data.) -> PreMixTransform:
            Creates a premix transform for random chunking.
        __len__() -> int:
            Returns the target number of chunks.
    """

    def __init__(
        self,
        *,
        split: str,
        dataset_connector: DatasetConnector[GenericIdentifier],
        chunk_size_seconds: float,
        target_length: int,
        fs: int = 44100,
        chunk_filter: Optional[ChunkFilter] = None,
        prechunk_premix_transform: Optional[PreMixTransform] = None,
        postchunk_premix_transform: Optional[PreMixTransform] = None,
        postmix_transform: Optional[PostMixTransform] = None,
        recompute_mixture: bool = True,
        auto_load_mixture: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        allow_resampling: bool = False,
        mixture_stem_key: Optional[str] = "mixture",
        align_sources: bool = True,
    ) -> None:
        premix_transform = self._make_premix_transform(
            chunk_size_seconds=chunk_size_seconds,
            fs=fs,
            align_sources=align_sources,
            prechunk_premix_transform=prechunk_premix_transform,
            postchunk_premix_transform=postchunk_premix_transform,
        )
        super().__init__(
            split=split,
            dataset_connector=dataset_connector,
            fs=fs,
            recompute_mixture=recompute_mixture,
            chunk_filter=chunk_filter,
            premix_transform=premix_transform,
            postmix_transform=postmix_transform,
            auto_load_mixture=auto_load_mixture,
            allowed_extensions=allowed_extensions,
            allow_resampling=allow_resampling,
            mixture_stem_key=mixture_stem_key,
        )
        self.target_length = target_length

    def _make_premix_transform(
        self,
        *,
        chunk_size_seconds: float,
        fs: int,
        align_sources: bool = True,
        prechunk_premix_transform: Optional[PreMixTransform],
        postchunk_premix_transform: Optional[PreMixTransform],
    ) -> PreMixTransform:
        """
        Creates a premix transform for random chunking.

        Args:
            chunk_size_seconds (float): The size of each chunk in seconds.
            fs (int): The sampling rate for audio data.
            align_sources (bool, optional): Whether to align sources. Defaults to True.
            prechunk_premix_transform (Optional[PreMixTransform]): Transformation before chunking.
            postchunk_premix_transform (Optional[PreMixTransform]): Transformation after chunking.

        Returns:
            PreMixTransform: The composed premix transform.
        """
        transforms = []
        if prechunk_premix_transform:
            transforms.append(prechunk_premix_transform)
        transforms.append(
            RandomChunkingTransform(
                chunk_size_seconds=chunk_size_seconds,
                fs=fs,
                align_sources=align_sources,
            )
        )
        if postchunk_premix_transform:
            transforms.append(postchunk_premix_transform)
        return ComposePreMixTransforms(transforms)

    def __len__(self) -> int:
        """
        Returns the target number of chunks.

        Returns:
            int: The target number of chunks.
        """
        return self.target_length


class DeterministicChunkDataset(
    ChunkDataset[GenericIdentifier], Generic[GenericIdentifier]
):
    """
    MedleyDB dataset that returns deterministic chunks.

    Args:
        chunk_size_seconds (float): The size of each chunk in seconds.
        hop_size_seconds (float): The hop size between chunks in seconds.
        target_length (Optional[int], optional): The target number of chunks. Defaults to None.

    Methods:
        n_chunks() -> int:
            Returns the total number of chunks in the dataset.
        n_chunks_per_track() -> List[int]:
            Returns the number of chunks per track.
        _get_n_chunks_for_identifier(identifier: MedleyDBIdentifier) -> int:
            Calculates the number of chunks for a given identifier.
        _make_premix_transform(bandx.data.) -> Tuple[PreMixTransform, DeterministicChunkingTransform]:
            Creates a premix transform for deterministic chunking.
        get_identifier(index: int) -> MedleyDBIdentifierWithChunk:
            Retrieves the identifier for a given chunk index.
        get_track_and_chunk_index(index: int) -> Tuple[int, int]:
            Maps a chunk index to a track index and chunk index.
    """

    def __init__(
        self,
        *,
        split: str,
        dataset_connector: DatasetConnector[GenericIdentifier],
        chunk_size_seconds: float,
        hop_size_seconds: float,
        target_length: Optional[int] = None,
        fs: int = 44100,
        chunk_filter: Optional[ChunkFilter] = None,
        prechunk_premix_transform: Optional[PreMixTransform] = None,
        postchunk_premix_transform: Optional[PreMixTransform] = None,
        postmix_transform: Optional[PostMixTransform] = None,
        recompute_mixture: bool = True,
        auto_load_mixture: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        allow_resampling: bool = False,
        mixture_stem_key: Optional[str] = "mixture",
    ) -> None:
        premix_transform, self.chunking_transform = self._make_premix_transform(
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            fs=fs,
            prechunk_premix_transform=prechunk_premix_transform,
            postchunk_premix_transform=postchunk_premix_transform,
        )
        super().__init__(
            split=split,
            dataset_connector=dataset_connector,
            fs=fs,
            recompute_mixture=recompute_mixture,
            chunk_filter=chunk_filter,
            premix_transform=premix_transform,
            postmix_transform=postmix_transform,
            auto_load_mixture=auto_load_mixture,
            allowed_extensions=allowed_extensions,
            allow_resampling=allow_resampling,
            mixture_stem_key=mixture_stem_key,
        )
        self.target_length = target_length or self.n_chunks

    @property
    def n_chunks(self) -> int:
        """
        Returns the total number of chunks in the dataset.

        Returns:
            int: The total number of chunks.
        """
        return sum(self.n_chunks_per_track)

    @property
    def n_chunks_per_track(self) -> List[int]:
        """
        Returns the number of chunks per track.

        Returns:
            List[int]: The number of chunks per track.
        """
        n_chunks_per_track = []
        for identifier in self.identifiers:
            n_chunks_per_track.append(self._get_n_chunks_for_identifier(identifier))
        return n_chunks_per_track

    def _get_n_chunks_for_identifier(self, identifier: GenericIdentifier) -> int:
        """
        Calculates the number of chunks for a given identifier.

        Args:
            identifier (MedleyDBIdentifier): The track identifier.

        Returns:
            int: The number of chunks for the track.
        """
        stem_path = self._get_stem_path(
            stem=self.mixture_stem_key,
            identifier=identifier,
        )
        data = np.load(stem_path, mmap_mode="r")
        _, n_samples = data["audio"].shape
        chunk_size_samples = self.chunking_transform.chunk_size_samples
        hop_size_samples = self.chunking_transform.hop_size_samples
        n_chunks = int(np.ceil((n_samples - chunk_size_samples) / hop_size_samples)) + 1
        return n_chunks

    def _make_premix_transform(
        self,
        *,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        fs: int,
        prechunk_premix_transform: Optional[PreMixTransform],
        postchunk_premix_transform: Optional[PreMixTransform],
    ) -> Tuple[PreMixTransform, DeterministicChunkingTransform]:
        """
        Creates a premix transform for deterministic chunking.

        Args:
            chunk_size_seconds (float): The size of each chunk in seconds.
            hop_size_seconds (float): The hop size between chunks in seconds.
            fs (int): The sampling rate for audio data.
            prechunk_premix_transform (Optional[PreMixTransform]): Transformation before chunking.
            postchunk_premix_transform (Optional[PreMixTransform]): Transformation after chunking.

        Returns:
            Tuple[PreMixTransform, DeterministicChunkingTransform]: The composed premix transform and chunking transform.
        """
        transforms = []
        if prechunk_premix_transform:
            transforms.append(prechunk_premix_transform)
        chunking_transform = DeterministicChunkingTransform(
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            fs=fs,
        )
        transforms.append(chunking_transform)
        if postchunk_premix_transform:
            transforms.append(postchunk_premix_transform)
        return ComposePreMixTransforms(transforms), chunking_transform

    def __len__(self) -> int:
        """
        Returns the target number of chunks.

        Returns:
            int: The target number of chunks.
        """
        return self.target_length

    def get_identifier(self, index: int) -> GenericIdentifier:
        """
        Retrieves the identifier for a given chunk index.

        Args:
            index (int): The index of the chunk.

        Returns:
            MedleyDBIdentifierWithChunk: The identifier for the chunk.
        """
        track_index, chunk_index = self.get_track_and_chunk_index(index)
        track_identifier = super().get_identifier(track_index)
        track_identifier.chunk_identifier = ChunkIdentifier(
            chunk_index=chunk_index,
        )

        return track_identifier

    def get_track_and_chunk_index(self, index: int) -> Tuple[int, int]:
        """
        Maps a chunk index to a track index and chunk index.

        Args:
            index (int): The index of the chunk.

        Returns:
            Tuple[int, int]: The track index and chunk index.
        """
        cumulative_chunks: NPIntArray = np.cumsum([0] + self.n_chunks_per_track)
        track_index = int(np.searchsorted(cumulative_chunks, index, side="right") - 1)
        chunk_index: int = index - cumulative_chunks[track_index]
        return track_index, chunk_index


class FixedStemRandomChunkDataset(
    FixedStemMixin[GenericIdentifier],
    RandomChunkDataset[GenericIdentifier],
    Generic[GenericIdentifier],
):
    """
    Fixed stem dataset with random chunking.
    """

    def __init__(
        self,
        *,
        allowed_stems: List[str],
        **kwargs,
    ) -> None:
        """
        Args:
            allowed_stems (List[str]): List of allowed stems.
            **kwargs: Additional arguments for the parent class.
        """
        super().__init__(allowed_stems=allowed_stems, **kwargs)


class FixedStemDeterministicChunkDataset(
    FixedStemMixin[GenericIdentifier],
    DeterministicChunkDataset[GenericIdentifier],
    Generic[GenericIdentifier],
):
    """
    Fixed stem dataset with deterministic chunking.
    """

    def __init__(
        self,
        *,
        allowed_stems: List[str],
        **kwargs,
    ) -> None:
        """
        Args:
            allowed_stems (List[str]): List of allowed stems.
            **kwargs: Additional arguments for the parent class.
        """
        super().__init__(allowed_stems=allowed_stems, **kwargs)
