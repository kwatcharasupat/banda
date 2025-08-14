import os
from typing import (
    List,
    Optional,
    Dict,
    Any,
)

from banda.utils.registry import DatasetRegistry
import structlog

from banda.data.types import (
    Identifier,
    __VDBO__
)
from banda.data.datasets.base import SourceSeparationDataset, DatasetConnector

logger = structlog.get_logger(__name__)

class MoisesDBIdentifier(Identifier):
    """
    Pydantic model for uniquely identifying a track in the MoisesDB dataset.

    Attributes:
        filename (str): The filename of the track (e.g., "track_name.npz").
        song_name (Optional[str]): The name of the song.
        artist_name (Optional[str]): The name of the artist.
    """
    filename: str
    song_name: Optional[str] = None
    artist_name: Optional[str] = None

    @property
    def track_id(self) -> str:
        """
        Returns a unique identifier for the track.

        Returns:
            str: The filename of the track, serving as its unique ID.
        """
        return self.filename

class MoisesDBConnector(DatasetConnector[MoisesDBIdentifier]):
    """
    Connector for the MoisesDB dataset.

    This class handles loading track identifiers and providing paths to stem files
    for the MoisesDB dataset.
    """
    __ALLOWED_STEMS__: List[str] = __VDBO__
    __EXPECTED_NUM_FILES__: Dict[str, int] = {
        "train": 0,  # MoisesDB typically used for validation/test, not train
        "val": 0,
        "test": 0,
    }

    def __init__(
        self,
        *,
        split: str,
        data_root: str = "$DATA_ROOT/moisesdb/intermediates/npz",
    ) -> None:
        """
        Initializes the MoisesDBConnector.

        Args:
            split (str): The dataset split (e.g., "train", "test", "val").
            data_root (str): The root directory where the MoisesDB dataset
                             (specifically the intermediates/npz folder) is located.
        """
        super().__init__(split=split, data_root=data_root)
        self.stem_path: str = os.path.join(self.data_root, split, "{track_filename}")
        self._identifiers: List[MoisesDBIdentifier] = self._load_identifiers(data_root=data_root, split=split)

    def _load_identifiers(
        self, *, data_root: str, split: str
    ) -> List[MoisesDBIdentifier]:
        """
        Loads track identifiers from the specified data root and split.

        Args:
            data_root (str): The root directory of the dataset.
            split (str): The dataset split.

        Returns:
            List[MoisesDBIdentifier]: A list of MoisesDBIdentifier objects.
        """
        data_root = os.path.expandvars(data_root)
        split_dir: str = os.path.join(data_root, split)
        
        if not os.path.exists(split_dir):
            logger.warning(f"MoisesDB split directory not found: {split_dir}. Returning empty list of identifiers.")
            return []

        track_filenames: List[str] = [f for f in os.listdir(split_dir) if f.endswith(".npz")]

        identifiers: List[MoisesDBIdentifier] = []
        for track_filename in track_filenames:
            base_name: str = os.path.splitext(track_filename)[0]
            artist: str = "unknown"
            song_name: str = base_name
            
            identifier: MoisesDBIdentifier = MoisesDBIdentifier(
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

    def _get_stem_path(self, *, stem: str, identifier: MoisesDBIdentifier) -> str:
        """
        Constructs the file path for a given stem and track identifier.

        Args:
            stem (str): The name of the stem (e.g., "vocals", "mixture").
            identifier (MoisesDBIdentifier): The identifier for the track.

        Returns:
            str: The full path to the stem's audio file.
        """
        if not isinstance(identifier, MoisesDBIdentifier):
            identifier = MoisesDBIdentifier(**identifier.model_dump())
        track_filename: str = identifier.filename
        stem_path_formatted: str = os.path.expandvars(self.stem_path.format(track_filename=track_filename))
        return stem_path_formatted

    @property
    def identifiers(self) -> List[MoisesDBIdentifier]:
        """
        Returns the list of all track identifiers for the current split.

        Returns:
            List[MoisesDBIdentifier]: A list of MoisesDBIdentifier objects.
        """
        return self._identifiers

    @property
    def n_tracks(self) -> int:
        """
        Returns the total number of tracks in the current split.

        Returns:
            int: The number of tracks.
        """
        return len(self._identifiers)

@DatasetRegistry.register()
class MoisesDBDataset(SourceSeparationDataset):
    """
    Dataset for the MoisesDB dataset.

    This class extends SourceSeparationDataset to provide specific
    data loading and processing for the MoisesDB dataset.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the MoisesDBDataset.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the
                      SourceSeparationDataset constructor.
        """
        super().__init__(**kwargs)