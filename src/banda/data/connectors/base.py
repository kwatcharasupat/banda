from abc import abstractmethod
import os
from typing import Generic, List, TypeVar

from ..typing import GenericIdentifier, Identifier


class DatasetConnector(Generic[GenericIdentifier]):
    stem_path: str

    def __init__(
        self,
        *,
        split: str,
        data_root: str,
    ) -> None:
        super().__init__()

        self.split = split
        self.data_root = os.path.expanduser(os.path.expandvars(data_root))

    @property
    @abstractmethod
    def identifiers(
        self,
    ) -> List[GenericIdentifier]:
        pass

    @property
    def n_tracks(self) -> int:
        """
        Returns the number of tracks in the dataset.

        Returns:
            int: The number of tracks.
        """
        return len(self.identifiers)

    @abstractmethod
    def _get_stem_path(self, *, stem: str, identifier: GenericIdentifier) -> str:
        """
        Returns the file path for a specific stem and identifier.

        Args:
            stem (str): The stem name.
            identifier (Identifier): The track identifier.

        Returns:
            str: The file path for the stem.
        """

    def get_identifier(self, index: int) -> GenericIdentifier:
        return self.identifiers[index % self.n_tracks]
