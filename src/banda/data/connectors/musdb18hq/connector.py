import os
from typing import List, Optional

import structlog

from bandx.data.typing import __VDBO__

from ..base import DatasetConnector


from ...datasets.base import Identifier


logger = structlog.get_logger(__name__)


class MUSDB18Identifier(Identifier):
    song_name: Optional[str] = None
    artist_name: Optional[str] = None

    @property
    def track_id(self) -> str:
        return f"{self.artist_name} - {self.song_name}"


class MUSDB18Connector(DatasetConnector[MUSDB18Identifier]):
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
        track_ids = os.listdir(os.path.join(data_root, split))

        identifiers = []
        for track_id in track_ids:
            artist, song_name = track_id.split(" - ")
            identifier = MUSDB18Identifier(
                artist_name=artist,
                song_name=song_name,
            )
            identifiers.append(identifier)

        if len(identifiers) != self.__EXPECTED_NUM_FILES__[split]:
            raise ValueError(
                f"Expected {self.__EXPECTED_NUM_FILES__[split]} files, but found {len(identifiers)}."
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
        identifier = MUSDB18Identifier(**identifier.model_dump())

        track_id = identifier.track_id
        return self.stem_path.format(track_id=track_id, stem=stem)
