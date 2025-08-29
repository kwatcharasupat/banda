from abc import ABC, abstractmethod
import os
from typing import (
    Generic,
    List,
    Optional,
    TypeVar,
)
import warnings

import numpy as np
from torch.utils import data


from ..connectors.base import DatasetConnector


from ..typing import (
    GenericIdentifier,
    NumPySourceDict,
    TorchInputAudioDict,
)

from ..transforms.base import (
    PostMixTransform,
    PreMixTransform,
    IdentityPostMixTransform,
    IdentityPreMixTransform,
)


import torchaudio as ta  # type: ignore


class ResamplingWarning(Warning):
    """
    Warning for resampling operations.
    """


class NonWavFileWarning(Warning):
    """
    Warning for non-WAV file extensions.
    """


class SourceSeparationDataset(data.Dataset, ABC, Generic[GenericIdentifier]):
    """
    Abstract base class for source separation datasets.
    """

    __DEFAULT_ALLOWED_EXTENSIONS__ = [".npz", ".wav"]
    _warned_npy_sample_rate = False
    _warned_non_wav = False
    _warned_resampling = False
    __DEFAULT_MIXTURE_STEM_KEY__ = "mixture"

    def __init__(
        self,
        *,
        split: str,
        dataset_connector: DatasetConnector[GenericIdentifier],
        fs: int,
        premix_transform: Optional[PreMixTransform] = None,
        postmix_transform: Optional[PostMixTransform] = None,
        recompute_mixture: bool = True,
        auto_load_mixture: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        allow_resampling: bool = False,
        mixture_stem_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            split (str): Dataset split (e.g., "train", "val", "test").
            data_root (str): Root directory of the dataset.
            fs (int): Target sample rate.
            premix_transform (Optional[PreMixTransform]): Transform applied before mixing.
            postmix_transform (Optional[PostMixTransform]): Transform applied after mixing.
            recompute_mixture (bool): Whether to recompute the mixture.
            auto_load_mixture (bool): Whether to automatically load the mixture.
            allowed_extensions (Optional[List[str]]): Allowed file extensions.
            allow_resampling (bool): Whether to allow resampling.
            mixture_stem_key (Optional[str]): Key for the mixture stem.
        """
        self.split = split
        self.dataset_connector = dataset_connector
        self.fs = fs
        self.recompute_mixture = recompute_mixture
        self.allowed_extensions = (
            allowed_extensions or self.__DEFAULT_ALLOWED_EXTENSIONS__
        )
        self.allow_resampling = allow_resampling
        self.mixture_stem_key = mixture_stem_key or self.__DEFAULT_MIXTURE_STEM_KEY__
        self.auto_load_mixture = auto_load_mixture

        self.premix_transform: PreMixTransform = (
            premix_transform or IdentityPreMixTransform()
        )
        self.postmix_transform: PostMixTransform = (
            postmix_transform or IdentityPostMixTransform()
        )

    def get_stem(self, *, stem: str, identifier: GenericIdentifier) -> np.ndarray:
        """
        Get the audio data for a specific stem.

        Args:
            stem (str): Stem name.
            identifier (Identifier): Track identifier.

        Returns:
            np.ndarray: Audio data for the stem.
        """
        path = self._get_stem_path(stem=stem, identifier=identifier)
        return self._get_stem_from_path(path)

    def _get_stem_from_path(self, path: str) -> np.ndarray:
        """
        Load stem audio data from a file.

        Args:
            path (str): Path to the stem file.

        Returns:
            np.ndarray: Loaded audio data.
        """
        ext = os.path.splitext(path)[-1]
        if self.allowed_extensions and ext not in self.allowed_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")

        if ext == ".npz":
            return self._load_npz(path)
        elif ext == ".npy":
            return self._load_npy(path)
        else:
            if not self._warned_non_wav:
                msg = (
                    f"File {path} has a non-wav extension {ext}. "
                    "If this is a compressed audio file, the audio loading may be slow. "
                    "If this is a non-audio file, unexpected behavior may occur. "
                    "This warning will only be shown once."
                )
                warnings.warn(msg, NonWavFileWarning)
                self._warned_non_wav = True
            return self._load_audio(path)

    def _load_npz(self, path: str) -> np.ndarray:
        """
        Load audio data from a .npz file.

        Args:
            path (str): Path to the .npz file.

        Returns:
            np.ndarray: Resampled audio data.
        """
        data = np.load(path, mmap_mode="r")
        audio = data["audio"]
        fs = data["fs"]

        return self._resample(audio, original_fs=fs)

    def _load_npy(self, path: str) -> np.ndarray:
        """
        Load audio data from a .npy file.

        Args:
            path (str): Path to the .npy file.

        Returns:
            np.ndarray: Resampled audio data.

        Raises:
            UserWarning: If sampling rates are not checked.
        """
        if not self._warned_npy_sample_rate:
            msg = (
                "Sampling rates are not checked for .npy files. Use with caution."
                "This warning will only be shown once."
            )
            warnings.warn(msg, UserWarning)
            self._warned_npy_sample_rate = True

        audio = np.load(path, mmap_mode="r")
        return self._resample(audio, original_fs=self.fs)

    def _load_audio(self, path: str) -> np.ndarray:
        """
        Load audio data from a file.

        Args:
            path (str): Path to the audio file.

        Returns:
            np.ndarray: Resampled audio data.
        """
        audio, fs = ta.load(path)
        return self._resample(audio, original_fs=fs)

    def _resample(self, audio: np.ndarray, *, original_fs: int) -> np.ndarray:
        """
        Resample audio data to the target sample rate.

        Args:
            audio (np.ndarray): Input audio data.
            original_fs (int): Original sample rate.

        Returns:
            np.ndarray: Resampled audio data.

        Raises:
            ValueError: If resampling is not allowed.
            NotImplementedError: If resampling is not implemented.
        """
        if original_fs == self.fs:
            return audio

        if not self.allow_resampling:
            raise ValueError(
                f"Expected sample rate {self.fs}, got {original_fs}. "
                "Use allow_resampling=True to resample."
            )

        if not self._warned_resampling:
            msg = (
                f"Resampling from {original_fs} to {self.fs}. "
                "Consider resampling beforehand to avoid performance issues. "
                "This warning will only be shown once."
            )
            warnings.warn(msg, ResamplingWarning)
            self._warned_resampling = True

        raise NotImplementedError("Resampling is not implemented.")

    def _get_stem_path(self, *, stem: str, identifier: GenericIdentifier) -> str:
        """
        Abstract method to get the file path for a stem.

        Args:
            stem (str): Stem name.
            identifier (Identifier): Track identifier.

        Returns:
            str: Path to the stem file.
        """
        return self.dataset_connector._get_stem_path(stem=stem, identifier=identifier)

    def get_identifier(self, index: int) -> GenericIdentifier:
        """
        Abstract method to get the identifier for a track.

        Args:
            index (int): Index of the track.

        Returns:
            Identifier: Track identifier.
        """
        return self.dataset_connector.get_identifier(index=index)

    def _compute_mixture(self, *, sources: NumPySourceDict) -> np.ndarray:
        """
        Compute the mixture from source stems.

        Args:
            sources (NumPySourceDict): Dictionary of source stems.

        Returns:
            np.ndarray: Mixture audio data.
        """
        return sum((sources[stem] for stem in sources), np.array(0.0))

    def get_stems(
        self, *, stems: List[str], identifier: GenericIdentifier
    ) -> NumPySourceDict:
        """
        Get audio data for multiple stems.

        Args:
            stems (List[str]): List of stem names.
            identifier (Identifier): Track identifier.

        Returns:
            NumPySourceDict: Dictionary of stem audio data.
        """
        return {stem: self.get_stem(stem=stem, identifier=identifier) for stem in stems}

    def get_audio_dict(
        self, *, stems: List[str], identifier: GenericIdentifier
    ) -> TorchInputAudioDict:
        """
        Get a dictionary containing mixture and source audio data.

        Args:
            stems (List[str]): List of stem names.
            identifier (Identifier): Track identifier.

        Returns:
            TorchInputAudioDict: Dictionary of mixture and source audio data.
        """
        sources = self.get_stems(stems=stems, identifier=identifier)
        sources = self.premix_transform(sources=sources, identifier=identifier)
        if self.recompute_mixture:
            mixture = self._compute_mixture(sources=sources)
        elif self.auto_load_mixture:
            mixture = self.get_stem(stem=self.mixture_stem_key, identifier=identifier)
        else:
            mixture = None

        return self.postmix_transform(
            audio_dict=TorchInputAudioDict.from_numpy(
                mixture=mixture,
                sources=sources,
            ),
            identifier=identifier,
        )

    @property
    def identifiers(self) -> List[GenericIdentifier]:
        """
        Get the list of identifiers for the dataset.

        Returns:
            List[Identifier]: List of track identifiers.
        """
        return self.dataset_connector.identifiers

    @property
    def n_tracks(self) -> int:
        """
        Get the number of tracks in the dataset.

        Returns:
            int: Number of tracks.
        """
        return self.dataset_connector.n_tracks
