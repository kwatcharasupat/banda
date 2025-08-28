#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import os
from typing import (
    List,
    Optional,
    Dict,
    Any,
)

import structlog

from banda.data.connectors.base import DatasetConnector
from banda.utils.registry import DatasetRegistry
from banda.data.types import (
    Identifier,
    __VDBO__
)
from banda.data.datasets.base import SourceSeparationDataset

logger = structlog.get_logger(__name__)

class MUSDB18Identifier(Identifier):
    """
    Pydantic model for uniquely identifying a track in the MUSDB18HQ dataset.

    Attributes:
        filename (str): The filename of the track (e.g., "artist - song.npz").
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

class MUSDB18Connector(DatasetConnector[MUSDB18Identifier]):
    """
    Connector for the MUSDB18HQ dataset.

    This class handles loading track identifiers and providing paths to stem files
    for the MUSDB18HQ dataset.
    """

    __ALLOWED_STEMS__: List[str] = __VDBO__
    __EXPECTED_NUM_FILES__: Dict[str, int] = {
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
        Initializes the MUSDB18Connector.

        This class handles loading track identifiers and providing paths to stem files
        for the MUSDB18HQ dataset.
        """
        super().__init__(split=split, data_root=data_root)
        self.stem_path: str = os.path.join(self.data_root, split, "{track_filename}")
        self._identifiers: List[MUSDB18Identifier] = self._load_identifiers(data_root=data_root, split=split)

    def _load_identifiers(
        self, *, data_root: str, split: str
    ) -> List[MUSDB18Identifier]:
        """
        Loads track identifiers from the specified data root and split.

        Args:
            data_root (str): The root directory of the dataset.
            split (str): The dataset split.

        Returns:
            List[MUSDB18Identifier]: A list of MUSDB18Identifier objects.
        """
        data_root = os.path.expanduser(os.path.expandvars(data_root))
        track_filenames: List[str] = [f for f in os.listdir(os.path.join(data_root, split)) if f.endswith(".npz")]

        identifiers: List[MUSDB18Identifier] = []
        for track_filename in track_filenames:
            base_name: str = os.path.splitext(track_filename)[0]
            artist: str
            song_name: str
            if " - " in base_name:
                artist, song_name = base_name.split(" - ", 1)
            else:
                artist = "unknown"
                song_name = base_name
                logger.warning(f"Track filename '{track_filename}' does not contain ' - '. Assuming artist is 'unknown'.")

            identifier: MUSDB18Identifier = MUSDB18Identifier(
                filename=track_filename,
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
        Constructs the file path for a given stem and track identifier.

        Args:
            stem (str): The name of the stem (e.g., "vocals", "mixture").
            identifier (MUSDB18Identifier): The identifier for the track.

        Returns:
            str: The full path to the stem's audio file.
        """
        if not isinstance(identifier, MUSDB18Identifier):
            identifier = MUSDB18Identifier(**identifier.model_dump())

        track_filename: str = identifier.filename
        stem_path_formatted: str = os.path.expandvars(self.stem_path.format(track_filename=track_filename))
        return stem_path_formatted

    @property
    def identifiers(
        self,
    ) -> List[MUSDB18Identifier]:
        """
        Returns the list of all track identifiers for the current split.

        Returns:
            List[MUSDB18Identifier]: A list of MUSDB18Identifier objects.
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
