from typing import List
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from banda.data.augmentation.compose import Compose, CompositionConfig
from banda.data.datasources.base import BaseRegisteredDatasource, TrackIdentifier
from banda.data.item import SourceSeparationItem
from banda.utils import WithClassConfig, BaseConfig

import structlog 

logger = structlog.get_logger(__name__)

class DatasetParams(BaseConfig):
    n_channels: int
    fs: int
    
    allow_autolooping: bool
    
    premix_augmentation: CompositionConfig | None = None
    variable_sources: bool = False

class DatasetConfig(WithClassConfig[DatasetParams]):
    pass

class DatasetRegistry(type):
    
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    DATASET_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.DATASET_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.DATASET_REGISTRY)

class BaseRegisteredDataset(Dataset, metaclass=DatasetRegistry):
    def __init__(self, *, 
                 datasources: list[BaseRegisteredDatasource],
                 config: DictConfig):
        super().__init__()

        self.datasources = datasources
        self.config = config
        
        if not self.datasources:
            raise ValueError("At least one datasource must be provided to the dataset.")
        
        self._cache_sizes()
        self._build_index_cache()
        self._build_augmentation()
        
    def _cache_sizes(self):
        raise NotImplementedError()
        
    def _build_index_cache(self):
        datasource_sizes = [len(ds) for ds in self.datasources]
        self.cumulative_sizes = np.cumsum(datasource_sizes)
        self.total_size = self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0

        if self.total_size == 0:
            raise ValueError("Dataset is empty.")

    def _resolve_index(self, index: int) -> tuple[int, int]:
        
        if index >= self.total_size:
            if not self.config.allow_autolooping:
                raise IndexError("Index out of bounds")
            index = index % self.total_size

        datasource_index = np.searchsorted(self.cumulative_sizes, index, side='right')
        base_index = self.cumulative_sizes[datasource_index - 1] if datasource_index > 0 else 0
        item_index = index - base_index
        
        # logger.info(f"Flat idx = {index}, total size = {self.total_size}, Datasource idx = {datasource_index}, item idx = {item_index}")
        
        return datasource_index, item_index
    
    
    def _load_audio(self, track_identifier: TrackIdentifier):
        
        npz_data = np.load(track_identifier.full_path, mmap_mode='r')

        audio_data = SourceSeparationItem(mixture=None, sources={}, estimates={})

        for source, source_components in track_identifier.sources.items():
            if source == "mixture":
                logger.warning("Mixture is NOT being loaded into sources. Are you sure?.")
                
            if source not in npz_data and not self.config.variable_sources:
                raise ValueError(f"Source '{source}' not found in audio data. Turn on `variable_sources` to deal with datasets with inconsistent sources.")

            source_audios = []
            for source_component in source_components:
                source_audio = npz_data.get(source_component, None)
                if source_audio is None and not self.config.variable_sources:
                    raise ValueError(f"Source component '{source_component}' not found in audio data.")
                source_audios.append(source_audio)

            audio_data.sources[source] = {"audio": source_audios}

        return audio_data

    def _build_augmentation(self):
        augmentation_config = self.config.premix_augmentation
        if augmentation_config is not None:
            self.premix_augmentation = Compose(config=augmentation_config)
        else:
            self.premix_augmentation = None
        