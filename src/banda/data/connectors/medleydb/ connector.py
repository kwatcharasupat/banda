import structlog


logger = structlog.get_logger(__name__)

import os
from typing import List, Optional

from ...typing import Identifier
from ..base import DatasetConnector


class MedleyDBIdentifier(Identifier):
    """
    Identifier for MedleyDB tracks.

    Attributes:
        song_name (Optional[str]): Name of the song.
        artist_name (Optional[str]): Name of the artist.
        chunk_index (Optional[int]): Index of the chunk, if applicable.

    Methods:
        track_id() -> str:
            Returns the track identifier in the format "artist_name - song_name".
    """

    song_name: Optional[str] = None
    artist_name: Optional[str] = None

    @property
    def track_id(self) -> str:
        return f"{self.artist_name} - {self.song_name}"


class MedleyDBConnector(DatasetConnector[MedleyDBIdentifier]):
    def __init__(
        self,
        *,
        split: str,
        data_root: str = "$DATA_ROOT/medleydb/intermediates/npz",
    ) -> None:
        """
        Initializes the MedleyDBConnector with the root directory of the dataset.

        Args:
            data_root (str): The root directory of the dataset.
        """
        super().__init__(
            split=split,
            data_root=data_root,
        )

        self.stem_path = os.path.join(self.data_root, split, "{track_id}", "{stem}.npz")

    @property
    def identifiers(
        self,
    ) -> List[MedleyDBIdentifier]:
        """
        Loads track identifiers from the dataset directory.

        Args:
            data_root (str): The root directory of the dataset.
            split (str): The dataset split (e.g., 'train', 'test').

        Returns:
            List[MedleyDBIdentifier]: A list of track identifiers.
        """
        track_ids = os.listdir(os.path.join(self.data_root, self.split))
        identifiers = []
        for track_id in track_ids:
            artist, song_name = track_id.split(" - ")
            identifier = MedleyDBIdentifier(
                artist_name=artist,
                song_name=song_name,
            )
            identifiers.append(identifier)
        return identifiers

    def _get_stem_path(self, *, stem: str, identifier: MedleyDBIdentifier) -> str:
        """
        Returns the file path for a specific stem and identifier.

        Args:
            stem (str): The stem name.
            identifier (Identifier): The track identifier.

        Returns:
            str: The file path for the stem.
        """
        identifier = MedleyDBIdentifier(**identifier.model_dump())
        track_id = identifier.track_id
        return self.stem_path.format(track_id=track_id, stem=stem)
