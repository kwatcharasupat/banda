from banda.data.datamodule import DatasetConnectorConfig


import importlib
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Type

from banda.data.types import GenericIdentifier


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
    def from_config(cls, config: DatasetConnectorConfig) -> "DatasetConnector":
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


        if isinstance(config, dict):
            config = DatasetConnectorConfig(**config)

        assert isinstance(config, DatasetConnectorConfig), type(config)

        class_path: str = config.target_ # Kept _target_ here as it's from Hydra config        
        kwargs: Dict[str, Any] = config.model_dump(exclude=["target_"])['config']

        # Expand environment variables in data_root if present
        if 'data_root' in kwargs:
            kwargs['data_root'] = os.path.expandvars(kwargs['data_root'])
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        connector_cls: Type["DatasetConnector"] = getattr(module, class_name)


        return connector_cls(**kwargs)