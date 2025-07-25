#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import os
import warnings
from abc import ABC
from typing import (
    Generic,
    List,
    Optional,
)

import numpy as np
import structlog
import torchaudio as ta  # type: ignore
from pydantic import BaseModel, Extra
from torch.utils import data

from banda.data.augmentations.base import (
    PostMixTransform,
    PreMixTransform,
    IdentityPostMixTransform,
    IdentityPreMixTransform,
    TransformConfig,
)
from banda.data.types import (
    GenericIdentifier,
    NumPySourceDict,
    TorchInputAudioDict,
    Identifier,
    __VDBO__
)
from banda.utils.config import ConfigWithTarget


logger = structlog.get_logger(__name__)


class ResamplingWarning(Warning):
    """
    Warning for resampling operations.
    """


class NonWavFileWarning(Warning):
    """
    Warning for non-WAV file extensions.
    """


class DatasetConnectorConfig(ConfigWithTarget):
    pass


class DatasetConnector(ABC, Generic[GenericIdentifier]):
    """
    Abstract base class for dataset connectors.
    """

    def __init__(self, *, split: str, data_root: str) -> None:
        self.split = split
        self.data_root = data_root

    def _get_stem_path(self, *, stem: str, identifier: GenericIdentifier) -> str:
        """
        Abstract method to get the file path for a stem.
        """
        raise NotImplementedError

    def get_identifier(self, index: int) -> GenericIdentifier:
        """
        Abstract method to get the identifier for a track.
        """
        raise NotImplementedError

    @property
    def identifiers(self) -> List[GenericIdentifier]:
        """
        Abstract property to get the list of identifiers for the dataset.
        """
        raise NotImplementedError

    @property
    def n_tracks(self) -> int:
        """
        Abstract property to get the number of tracks in the dataset.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: BaseModel):
        """
        Create a dataset connector instance from a configuration object.
        """
        return cls(**config.model_dump())


class _DatasetConfig(BaseModel, extra=Extra.allow):
    split: str
    dataset_connector: DatasetConnectorConfig
    fs: int
    premix_transform: Optional[TransformConfig] = None
    postmix_transform: Optional[TransformConfig] = None
    recompute_mixture: bool = True
    auto_load_mixture: bool = True
    allowed_extensions: Optional[List[str]] = None
    allow_resampling: bool = False
    mixture_stem_key: Optional[str] = None


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
            dataset_connector (DatasetConnector): Connector for accessing the dataset.
            fs (int): Target sample rate.
            premix_transform (Optional[PreMixTransform]): Transform applied before mixing.
            postmix_transform (Optional[PostMixTransform]): Transform applied after mixing.
            recompute_mixture (bool): Whether to recompute the mixture.
            auto_load_mixture (bool): Whether to automatically load the mixture.
            allowed_extensions (Optional[List[str]]): Allowed file extensions.
            allow_resampling (bool): Whether to allow resampling.
            mixture_stem_key (Optional[str]): Key for the mixture stem.
        """

        super().__init__()

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

        # Placeholder for actual resampling implementation
        # You would typically use torchaudio.transforms.Resample or similar here
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
        # Ensure all sources have the same length before summing
        min_len = min(source.shape[-1] for source in sources.values())
        aligned_sources = [source[..., :min_len] for source in sources.values()]
        return sum(aligned_sources)

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

    @classmethod
    def from_config(
        cls,
        config: _DatasetConfig,
    ):
        """
        Create a dataset instance from a configuration object.

        Args:
            config (_DatasetConfig): Configuration object for the dataset.

        Returns:
            SourceSeparationDataset: Instance of the dataset.
        """

        dataset_connector_cls = config.dataset_connector.target_
        dataset_connector = dataset_connector_cls.from_config(
            config.dataset_connector.config
        )

        other_kwargs = config.model_dump(exclude={"dataset_connector"})

        logger.info(
            f"Creating dataset {cls.__name__} with split '{config.split}' and connector {dataset_connector_cls.__name__}",
        )

        logger.debug(
            f"Dataset configuration: {config.model_dump(exclude={'dataset_connector'})}",
        )

        return cls(
            dataset_connector=dataset_connector,
            **other_kwargs,
        )


class DatasetConfig(ConfigWithTarget):
    _superclass_ = SourceSeparationDataset
    """
    DatasetConfig is a configuration model for defining parameters used in a dataset.

    Attributes:
        _target_ (str): Specifies the target class or function to be used.
        params (Dict[str, Any]): Parameters for initializing the dataset.
    """

    _target_: str
    config: _DatasetConfig


class MUSDB18Identifier(Identifier):
    song_name: Optional[str] = None
    artist_name: Optional[str] = None

    @property
    def track_id(self) -> str:
        return f"{self.artist_name} - {self.song_name}"


class MUSDB18Connector(DatasetConnector[MUSDB18Identifier], ABC):
    """
    Connector for the MUSDB18 dataset.
    """

    __ALLOWED_STEMS__ = __VDBO__
    __EXPECTED_NUM_FILES__ = {
        "train": 86,
        "val": 14,
        "test": 50,
    }

    def __init__(
        self,
        *,
        split: str,
        data_root: str = "$DATA_ROOT/musdb18hq/intermediates/npz",
    ) -> None:
        """
        Args:
            split (str): The split to use.
            data_root (str): The root directory of the dataset.
        """
        super().__init__(split=split, data_root=data_root)
        self.stem_path = os.path.join(self.data_root, split, "{track_id}", "{stem}.npz")

        self._identifiers = self._load_identifiers(data_root=data_root, split=split)

    def _load_identifiers(
        self, *, data_root: str, split: str
    ) -> List[MUSDB18Identifier]:
        """
        Load the identifiers for the dataset.

        Args:
            data_root (str): The root directory of the dataset.
            split (str): The split to use.

        Returns:
            List[MUSDB18Identifier]: The list of identifiers for the dataset.
        """
        # Replace $DATA_ROOT with actual path if it's an environment variable
        data_root = os.path.expandvars(data_root)

        track_ids = os.listdir(os.path.join(data_root, split))

        identifiers = []
        for track_id in track_ids:
            # Assuming track_id format "artist - song_name"
            if " - " in track_id:
                artist, song_name = track_id.split(" - ", 1)
            else:
                # Handle cases where track_id might just be the song name or a single identifier
                artist = "unknown"
                song_name = track_id
                logger.warning(f"Track ID '{track_id}' does not contain ' - '. Assuming artist is 'unknown'.")

            identifier = MUSDB18Identifier(
                artist_name=artist,
                song_name=song_name,
            )
            identifiers.append(identifier)

        if len(identifiers) != self.__EXPECTED_NUM_FILES__[split]:
            logger.warning(
                f"Expected {self.__EXPECTED_NUM_FILES__[split]} files in {os.path.join(data_root, split)}, but found {len(identifiers)}. This might indicate an incomplete dataset."
            )

        return identifiers

    def _get_stem_path(self, *, stem: str, identifier: MUSDB18Identifier) -> str:
        """
        Get the path to a stem file.

        Args:
            stem (str): The stem to get the path for.
            identifier (MUSDB18Identifier): The identifier for the track.

        Returns:
            str: The path to the stem file.
        """
        # Ensure identifier is a MUSDB18Identifier instance
        if not isinstance(identifier, MUSDB18Identifier):
            identifier = MUSDB18Identifier(**identifier.model_dump())

        track_id = identifier.track_id
        # Replace $DATA_ROOT with actual path if it's an environment variable
        stem_path_formatted = os.path.expandvars(self.stem_path.format(track_id=track_id, stem=stem))
        return stem_path_formatted

    @property
    def identifiers(
        self,
    ) -> List[MUSDB18Identifier]:
        return self._identifiers

    @property
    def n_tracks(self) -> int:
        return len(self._identifiers)


class MUSDB18HQDataset(SourceSeparationDataset):
    """
    Dataset for the MUSDB18HQ dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Additional MUSDB18HQ specific initialization if needed