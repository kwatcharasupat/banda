import os
import warnings
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
import torchaudio as ta  # type: ignore
from pydantic import BaseModel, Field
from torch.utils import data
from omegaconf import DictConfig, OmegaConf
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
    __VDBO__
)
from banda.data.datasets.musdb18hq import SourceSeparationDataset, DatasetConnector, DatasetConnectorConfig


logger = structlog.get_logger(__name__)


class MedleyDBIdentifier(Identifier):
    """
    Identifier for MedleyDB tracks.
    """
    filename: str
    song_name: Optional[str] = None
    artist_name: Optional[str] = None

    @property
    def track_id(self) -> str:
        return self.filename


class MedleyDBConnector(DatasetConnector[MedleyDBIdentifier]):
    """
    Connector for the MedleyDB dataset.
    """
    __ALLOWED_STEMS__ = __VDBO__ # Assuming similar stems as MUSDB18
    __EXPECTED_NUM_FILES__ = {
        "train": 0, # Placeholder, actual numbers would be determined by dataset
        "val": 0,
        "test": 0,
    }

    def __init__(
        self,
        *,
        split: str,
        data_root: str = "$DATA_ROOT/medleydb/intermediates/npz", # Placeholder path
    ) -> None:
        super().__init__(split=split, data_root=data_root)
        self.stem_path = os.path.join(self.data_root, split, "{track_filename}")
        self._identifiers = self._load_identifiers(data_root=data_root, split=split)

    def _load_identifiers(
        self, *, data_root: str, split: str
    ) -> List[MedleyDBIdentifier]:
        data_root = os.path.expandvars(data_root)
        split_dir = os.path.join(data_root, split)
        
        if not os.path.exists(split_dir):
            logger.warning(f"MedleyDB split directory not found: {split_dir}. Returning empty list of identifiers.")
            return []

        track_filenames = [f for f in os.listdir(split_dir) if f.endswith(".npz")] # Assuming .npz format

        identifiers = []
        for track_filename in track_filenames:
            base_name = os.path.splitext(track_filename)[0]
            # Placeholder for actual parsing logic based on MedleyDB file naming convention
            artist = "unknown"
            song_name = base_name
            
            identifier = MedleyDBIdentifier(
                filename=track_filename,
                artist_name=artist,
                song_name=song_name,
            )
            identifiers.append(identifier)

        if len(identifiers) != self.__EXPECTED_NUM_FILES__.get(split, len(identifiers)):
            logger.warning(
                f"Expected {self.__EXPECTED_NUM_FILES__.get(split, 'unknown')} files in {split_dir}, but found {len(identifiers)}. This might indicate an incomplete dataset."
            )

        return identifiers

    def _get_stem_path(self, *, stem: str, identifier: MedleyDBIdentifier) -> str:
        if not isinstance(identifier, MedleyDBIdentifier):
            identifier = MedleyDBIdentifier(**identifier.model_dump())
        track_filename = identifier.filename
        stem_path_formatted = os.path.expandvars(self.stem_path.format(track_filename=track_filename))
        return stem_path_formatted

    @property
    def identifiers(self) -> List[MedleyDBIdentifier]:
        return self._identifiers

    @property
    def n_tracks(self) -> int:
        return len(self._identifiers)


class MedleyDBDataset(SourceSeparationDataset):
    """
    Dataset for the MedleyDB dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Additional MedleyDB specific initialization if needed