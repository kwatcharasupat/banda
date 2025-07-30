#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import os
import warnings
from abc import abstractmethod
from abc import ABC
from typing import (
    Generic,
    List,
    Optional,
    Dict,
    Any,
)

import numpy as np
import structlog
import torchaudio as ta
from pydantic import BaseModel, Field
from omegaconf import DictConfig
import importlib

from banda.data.augmentations.base import (
    PostMixTransform,
    PreMixTransform,
    IdentityPostMixTransform,
    IdentityPreMixTransform,
    TransformConfig,
    Transform
)
from banda.data.types import (
    GenericIdentifier,
    NumPySourceDict,
    TorchInputAudioDict,
    Identifier,
)
from banda.core.interfaces import BaseDataset
from banda.data.batch_types import AudioBatch

logger = structlog.get_logger(__name__)

class ResamplingWarning(Warning):
    """
    Warning for resampling operations.
    """

class NonWavFileWarning(Warning):
    """
    Warning for non-WAV file extensions.
    """

class DatasetConnectorConfig(BaseModel):
    """
    Configuration model for a dataset connector.

    Attributes:
        target_ (str): The full path to the dataset connector class.
        config (Dict[str, Any]): A dictionary of configuration parameters for the connector.
    """
    model_config = {'arbitrary_types_allowed': True}
    target_: str = Field(...)
    config: Dict[str, Any] = Field({})

class DatasetConnector(ABC, Generic[GenericIdentifier]):
    """
    Abstract base class for dataset connectors.

    Dataset connectors are responsible for providing metadata and file paths
    for a specific dataset split.
    """

    def __init__(self, *, split: str, data_root: str) -> None:
        """
        Initializes the DatasetConnector.

        Args:
            split (str): The dataset split (e.g., "train", "test", "validation").
            data_root (str): The root directory where the dataset files are stored.
        """
        self.split: str = split
        self.data_root: str = data_root

    @abstractmethod
    def _get_stem_path(self, *, stem: str, identifier: GenericIdentifier) -> str:
        """
        Abstract method to get the file path for a specific stem of a track.

        Args:
            stem (str): The name of the stem (e.g., "vocals", "bass", "mixture").
            identifier (GenericIdentifier): The unique identifier for the track.

        Returns:
            str: The absolute path to the stem's audio file.
        """
        raise NotImplementedError

    def get_identifier(self, index: int) -> GenericIdentifier:
        """
        Abstract method to get the identifier for a track at a given index.

        Args:
            index (int): The index of the track.

        Returns:
            GenericIdentifier: The unique identifier for the track.
        """
        return self.identifiers[index]

    @property
    @abstractmethod
    def identifiers(self) -> List[GenericIdentifier]:
        """
        Abstract property to get the list of identifiers for the dataset.

        Returns:
            List[GenericIdentifier]: A list of unique identifiers for all tracks in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_tracks(self) -> int:
        """
        Abstract property to get the number of tracks in the dataset.

        Returns:
            int: The total number of tracks in the dataset.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: DictConfig) -> "DatasetConnector":
        """
        Create a dataset connector instance from a configuration object.

        This method dynamically loads the specified dataset connector class
        and initializes it with the provided configuration.

        Args:
            config (DictConfig): A DictConfig object containing the
                                 `target_` (class path) and `config` (parameters)
                                 for the dataset connector.

        Returns:
            DatasetConnector: An instance of the specified dataset connector.

        Raises:
            AttributeError: If the specified class or module cannot be found.
            TypeError: If the configuration parameters do not match the connector's constructor.
        """
        class_path: str = config.target_
        
        if hasattr(config, 'config') and isinstance(config.config, DictConfig):
            kwargs: Dict[str, Any] = {k: v for k, v in config.config.items()}
        else:
            kwargs: Dict[str, Any] = {k: v for k, v in config.items() if k != 'target_'}

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        connector_cls: Type["DatasetConnector"] = getattr(module, class_name)

        return connector_cls(**kwargs)

class SourceSeparationDataset(BaseDataset, Generic[GenericIdentifier]):
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
        Initializes the SourceSeparationDataset.

        Args:
            split (str): The dataset split (e.g., "train", "test", "validation").
            dataset_connector (DatasetConnector[GenericIdentifier]): An instance of a
                DatasetConnector to retrieve track identifiers and stem paths.
            fs (int): The target sampling rate for the audio data.
            premix_transform (Optional[PreMixTransform]): An optional transform to apply
                to individual stems before mixing. Defaults to IdentityPreMixTransform.
            postmix_transform (Optional[PostMixTransform]): An optional transform to apply
                to the mixed audio and sources after mixing. Defaults to IdentityPostMixTransform.
            recompute_mixture (bool): If True, the mixture will be recomputed from sources.
                If False, the mixture will be loaded from the dataset if available.
            auto_load_mixture (bool): If True, attempts to load the mixture stem if
                `recompute_mixture` is False.
            allowed_extensions (Optional[List[str]]): A list of allowed file extensions
                for audio files. Defaults to `__DEFAULT_ALLOWED_EXTENSIONS__`.
            allow_resampling (bool): If True, audio will be resampled to `fs` if
                the original sample rate differs. If False, a ValueError is raised.
            mixture_stem_key (Optional[str]): The key used to identify the mixture stem
                in the dataset. Defaults to `__DEFAULT_MIXTURE_STEM_KEY__`.
        """
        super().__init__()
 
        self.split: str = split
        self.dataset_connector: DatasetConnector[GenericIdentifier] = dataset_connector
        self.fs: int = fs
        self.recompute_mixture: bool = recompute_mixture
        self.allowed_extensions: List[str] = (
            allowed_extensions or self.__DEFAULT_ALLOWED_EXTENSIONS__
        )
        self.allow_resampling: bool = allow_resampling
        self.mixture_stem_key: str = mixture_stem_key or self.__DEFAULT_MIXTURE_STEM_KEY__
        self.auto_load_mixture: bool = auto_load_mixture
 
        self.premix_transform: PreMixTransform = (
            premix_transform or IdentityPreMixTransform()
        )
        self.post_mix_transform: PostMixTransform = (
            postmix_transform or IdentityPostMixTransform()
        )
 
    def __len__(self) -> int:
        """
        Returns the total number of tracks in the dataset.

        Returns:
            int: The number of tracks.
        """
        return self.n_tracks
 
    def __getitem__(self, index: int) -> AudioBatch:
        """
        Retrieves a processed audio batch for a given index.

        Args:
            index (int): The index of the track to retrieve.

        Returns:
            AudioBatch: A Pydantic model containing the mixture audio and metadata.
                The `audio` field will have shape (batch_size, channels, samples).

        Raises:
            ValueError: If the mixture audio cannot be obtained.
        """
        identifier: GenericIdentifier = self.get_identifier(index)
        
        stems_to_load: List[str] = ["vocals", "bass", "drums", "other"]
 
        audio_dict: TorchInputAudioDict = self.get_audio_dict(stems=stems_to_load, identifier=identifier)
 
        if audio_dict.mixture is None:
            raise ValueError("Mixture audio cannot be None for AudioBatch.")
 
        return AudioBatch(
            audio=audio_dict.model_dump(),
            metadata={"identifier": identifier.model_dump(), "sources": audio_dict.sources}
        )
 
    def get_stem(self, *, stem: str, identifier: GenericIdentifier) -> np.ndarray:
        """
        Retrieves the audio data for a specific stem of a track.

        Args:
            stem (str): The name of the stem to retrieve (e.g., "vocals", "bass").
            identifier (GenericIdentifier): The unique identifier for the track.

        Returns:
            np.ndarray: The audio data for the requested stem.
                Shape: (channels, samples)
        """
        path: str = self._get_stem_path(stem=stem, identifier=identifier)
        return self._get_stem_from_path(path, requested_stem=stem)
 
    def _get_stem_from_path(self, path: str, requested_stem: str) -> np.ndarray:
        """
        Loads audio data from a given file path.

        Args:
            path (str): The path to the audio file.
            requested_stem (str): The name of the stem being loaded (used for .npz files).

        Returns:
            np.ndarray: The loaded audio data.
                Shape: (channels, samples)

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext: str = os.path.splitext(path)[-1]
        if self.allowed_extensions and ext not in self.allowed_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")
 
        if ext == ".npz":
            return self._load_npz(path, requested_stem=requested_stem)
        elif ext == ".npy":
            return self._load_npy(path)
        else:
            if not self._warned_non_wav:
                msg: str = (
                    f"File {path} has a non-wav extension {ext}. "
                    "If this is a compressed audio file, the audio loading may be slow. "
                    "If this is a non-audio file, unexpected behavior may occur. "
                    "This warning will only be shown once."
                )
                warnings.warn(msg, NonWavFileWarning)
                self._warned_non_wav = True
            return self._load_audio(path)
 
    def _load_npz(self, path: str, requested_stem: str) -> np.ndarray:
        """
        Loads audio data from a .npz file.

        Args:
            path (str): The path to the .npz file.
            requested_stem (str): The key for the requested stem within the .npz file.

        Returns:
            np.ndarray: The loaded and potentially resampled audio data.
                Shape: (channels, samples)
        """
        data: Any = np.load(path, mmap_mode="r")
        audio: np.ndarray = data[requested_stem] # Shape: (channels, samples)
        fs: int = data["fs"]
 
        return self._resample(audio, original_fs=fs)
 
    def _load_npy(self, path: str) -> np.ndarray:
        """
        Loads audio data from a .npy file.

        Args:
            path (str): The path to the .npy file.

        Returns:
            np.ndarray: The loaded and potentially resampled audio data.
                Shape: (channels, samples)
        """
        if not self._warned_npy_sample_rate:
            msg: str = (
                "Sampling rates are not checked for .npy files. Use with caution."
                "This warning will only be shown once."
            )
            warnings.warn(msg, UserWarning)
            self._warned_npy_sample_rate = True
 
        audio: np.ndarray = np.load(path, mmap_mode="r") # Shape: (channels, samples)
        return self._resample(audio, original_fs=self.fs)
 
    def _load_audio(self, path: str) -> np.ndarray:
        """
        Loads audio data from a general audio file (e.g., .wav).

        Args:
            path (str): The path to the audio file.

        Returns:
            np.ndarray: The loaded and potentially resampled audio data.
                Shape: (channels, samples)
        """
        audio: torch.Tensor # Shape: (channels, samples)
        fs: int
        audio, fs = ta.load(path)
        return self._resample(audio.numpy(), original_fs=fs)
 
    def _resample(self, audio: np.ndarray, *, original_fs: int) -> np.ndarray:
        """
        Resamples audio data to the target sampling rate `self.fs`.

        Args:
            audio (np.ndarray): The input audio data.
                Shape: (channels, samples)
            original_fs (int): The original sampling rate of the audio.

        Returns:
            np.ndarray: The resampled audio data.
                Shape: (channels, new_samples)

        Raises:
            ValueError: If resampling is not allowed and the original sample rate
                does not match the target sample rate.
        """
        if original_fs == self.fs:
            return audio
 
        if not self.allow_resampling:
            raise ValueError(
                f"Expected sample rate {self.fs}, got {original_fs}. "
                "Use allow_resampling=True to resample."
            )
 
        if not self._warned_resampling:
            msg: str = (
                f"Resampling from {original_fs} to {self.fs}. "
                "Consider resampling beforehand to avoid performance issues. "
                "This warning will only be shown once."
            )
            warnings.warn(msg, ResamplingWarning)
            self._warned_resampling = True
        
        resampler = ta.transforms.Resample(orig_freq=original_fs, new_freq=self.fs)
        return resampler(torch.from_numpy(audio)).numpy()
 
    def _get_stem_path(self, *, stem: str, identifier: GenericIdentifier) -> str:
        """
        Retrieves the file path for a specific stem using the dataset connector.

        Args:
            stem (str): The name of the stem.
            identifier (GenericIdentifier): The unique identifier for the track.

        Returns:
            str: The file path to the stem's audio.
        """
        return self.dataset_connector._get_stem_path(stem=stem, identifier=identifier)
 
    def get_identifier(self, index: int) -> GenericIdentifier:
        """
        Retrieves the track identifier for a given index using the dataset connector.

        Args:
            index (int): The index of the track.

        Returns:
            GenericIdentifier: The unique identifier for the track.
        """
        return self.dataset_connector.get_identifier(index=index)
 
    def _compute_mixture(self, *, sources: NumPySourceDict) -> np.ndarray:
        """
        Computes the mixture audio by summing the individual source audios.

        Args:
            sources (NumPySourceDict): A dictionary where keys are source names
                and values are NumPy arrays of audio data.
                Shape of each source: (channels, samples)

        Returns:
            np.ndarray: The computed mixture audio.
                Shape: (channels, min_samples) where min_samples is the minimum
                length among all sources.
        """
        min_len: int = min(source.shape[-1] for source in sources.values())
        aligned_sources: List[np.ndarray] = [source[..., :min_len] for source in sources.values()]
        return sum(aligned_sources) # Shape: (channels, min_samples)
 
    def get_stems(
        self, *, stems: List[str], identifier: GenericIdentifier
    ) -> NumPySourceDict:
        """
        Retrieves audio data for multiple specified stems of a track.

        Args:
            stems (List[str]): A list of stem names to retrieve.
            identifier (GenericIdentifier): The unique identifier for the track.

        Returns:
            NumPySourceDict: A dictionary mapping stem names to their audio data.
                Shape of each audio data: (channels, samples)
        """
        return {stem: self.get_stem(stem=stem, identifier=identifier) for stem in stems}
 
    def get_audio_dict(
        self, *, stems: List[str], identifier: GenericIdentifier
    ) -> TorchInputAudioDict:
        """
        Retrieves and processes a dictionary of audio data (sources and mixture).

        This method handles loading stems, applying pre-mix transforms,
        computing or loading the mixture, and applying post-mix transforms.

        Args:
            stems (List[str]): A list of stem names to retrieve.
            identifier (GenericIdentifier): The unique identifier for the track.

        Returns:
            TorchInputAudioDict: A Pydantic model containing the mixture and source
                audio data as PyTorch tensors.
                Shape of mixture/sources: (channels, samples)
        """
        sources: NumPySourceDict = self.get_stems(stems=stems, identifier=identifier)
        sources = self.premix_transform(sources=sources, identifier=identifier)
        mixture: Optional[np.ndarray] = None
        if self.recompute_mixture:
            mixture = self._compute_mixture(sources=sources) # Shape: (channels, samples)
        elif self.auto_load_mixture:
            mixture = self.get_stem(stem=self.mixture_stem_key, identifier=identifier) # Shape: (channels, samples)
        else:
            mixture = None
 
        return self.post_mix_transform(
            audio_dict=TorchInputAudioDict.from_numpy(
                mixture=mixture,
                sources=sources,
            ),
            identifier=identifier,
        )
 
    @property
    def identifiers(self) -> List[GenericIdentifier]:
        """
        Returns the list of track identifiers from the dataset connector.

        Returns:
            List[GenericIdentifier]: A list of unique identifiers for all tracks.
        """
        return self.dataset_connector.identifiers
 
    @property
    def n_tracks(self) -> int:
        """
        Returns the total number of tracks from the dataset connector.

        Returns:
            int: The total number of tracks.
        """
        return self.dataset_connector.n_tracks
 
    @classmethod
    def from_config(
        cls,
        config: DictConfig,
    ) -> "SourceSeparationDataset":
        """
        Creates a SourceSeparationDataset instance from a configuration object.

        Args:
            config (DictConfig): A DictConfig object containing the dataset configuration.

        Returns:
            SourceSeparationDataset: An instance of the configured dataset.
        """
        dataset_connector: DatasetConnector[GenericIdentifier] = DatasetConnector.from_config(config['dataset_connector'])
 
        premix_transform: Optional[Transform] = None
        if config.get("premix_transform") is not None:
            premix_transform = Transform.from_config(config.premix_transform)
 
        postmix_transform: Optional[Transform] = None
        if config.get("postmix_transform") is not None:
            postmix_transform = Transform.from_config(config.postmix_transform)
 
        split: str = config.get("split")
        fs: int = config.get("fs")
        recompute_mixture: bool = config.get("recompute_mixture", True)
        auto_load_mixture: bool = config.get("auto_load_mixture", True)
        allowed_extensions: Optional[List[str]] = config.get("allowed_extensions", None)
        allow_resampling: bool = config.get("allow_resampling", False)
        mixture_stem_key: Optional[str] = config.get("mixture_stem_key", None)
 
        kwargs: Dict[str, Any] = {
            "split": split,
            "dataset_connector": dataset_connector,
            "fs": fs,
            "premix_transform": premix_transform,
            "postmix_transform": postmix_transform,
            "recompute_mixture": recompute_mixture,
            "auto_load_mixture": auto_load_mixture,
            "allowed_extensions": allowed_extensions,
            "allow_resampling": allow_resampling,
            "mixture_stem_key": mixture_stem_key,
        }
 
        return cls(**kwargs)